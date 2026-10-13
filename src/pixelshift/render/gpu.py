"""GPU renderer (OpenGL 3.3 via moderngl): the same look as the soft CPU renderer, in real time.

Every source pixel is one GL point. The vertex shader repeats the motion math of
:class:`pixelshift.animate.Animator` (delay, easing, wobble, swirl), so the CPU only uploads the
particle data once and then sends a single number, the time ``t``, per frame.

A frame is produced in up to three passes:

1. **Soft pass** (when ``0 < mix``): particles are drawn as Gaussian blobs with additive blending
   into a float texture. Several time samples trailing ``t`` give motion blur.
2. **Crisp pass** (near ``t = 0`` and ``t = 1``): particles are drawn as hard squares, which makes
   the first and last frames exactly the source and the rearranged image.
3. **Resolve pass**: divides accumulated color by accumulated density, fades sparse regions into
   the background, fills tiny gaps with a mip-mapped local average, and cross-fades to the crisp
   image near the ends.

Install the extra dependencies with ``pip install -e ".[gpu]"``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pixelshift.animate import Animator

if TYPE_CHECKING:
    import moderngl


class GpuUnavailableError(RuntimeError):
    """Raised when no usable OpenGL context can be created."""


EASING_IDS = {"linear": 0, "smoothstep": 1, "smootherstep": 2, "cubic": 3}

_MOTION = """
uniform float t;
uniform float delay_scale;
uniform float wobble;
uniform float swirl;
uniform float swirl_falloff;
uniform float radius;
uniform vec2 center;
uniform int easing;

in vec2 in_p0;
in vec2 in_p1;
in vec3 in_color;
in float in_delay;
in vec2 in_phase;
in float in_freq;

float ease(float u) {
    if (easing == 0) return u;
    if (easing == 1) return u * u * (3.0 - 2.0 * u);
    if (easing == 2) return u * u * u * (u * (6.0 * u - 15.0) + 10.0);
    return u < 0.5 ? 4.0 * u * u * u : 1.0 - pow(-2.0 * u + 2.0, 3.0) / 2.0;
}

vec2 position() {
    float u = clamp((t - in_delay) / delay_scale, 0.0, 1.0);
    vec2 pos = in_p0 + (in_p1 - in_p0) * ease(u);
    float env = 4.0 * u * (1.0 - u);
    if (wobble > 0.0) {
        pos += wobble * env * sin(6.28318530718 * (in_freq * t + in_phase));
    }
    if (swirl != 0.0) {
        vec2 rel = pos - center;
        float r = clamp(length(rel) / radius, 0.0, 1.0);
        float theta = swirl * env * (1.0 - swirl_falloff * r);
        float c = cos(theta);
        float s = sin(theta);
        pos = center + vec2(rel.x * c - rel.y * s, rel.x * s + rel.y * c);
    }
    return pos;
}
"""

_SOFT_VERTEX = (
    "#version 330\n"
    + _MOTION
    + """
uniform float size_px;
uniform float scale;
uniform float point_diameter;
out vec3 v_color;

void main() {
    vec2 px = (position() + 0.5) * scale;
    gl_Position = vec4(px / size_px * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = point_diameter;
    v_color = in_color;
}
"""
)

_SOFT_FRAGMENT = """
#version 330
in vec3 v_color;
uniform float sigma;
uniform float point_diameter;
uniform float sample_weight;
out vec4 frag;

void main() {
    vec2 d = (gl_PointCoord - 0.5) * point_diameter;
    float w = exp(-dot(d, d) / (2.0 * sigma * sigma)) * sample_weight;
    frag = vec4(v_color * w, w);
}
"""

_CRISP_VERTEX = (
    "#version 330\n"
    + _MOTION
    + """
uniform float size_px;
uniform float scale;
out vec3 v_color;

void main() {
    vec2 px = floor(position() * scale + 0.5) + 0.5 * scale;
    gl_Position = vec4(px / size_px * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = scale;
    v_color = in_color;
}
"""
)

_CRISP_FRAGMENT = """
#version 330
in vec3 v_color;
out vec4 frag;

void main() {
    frag = vec4(v_color, 1.0);
}
"""

_FULLSCREEN_VERTEX = """
#version 330
void main() {
    vec2 p = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_RESOLVE_FRAGMENT = """
#version 330
uniform sampler2D acc;
uniform sampler2D crisp;
uniform float size_px;
uniform float cover_k;
uniform float fill_k;
uniform float wide_lod;
uniform float soft_mix;
uniform vec3 background;
out vec4 frag;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a = texelFetch(acc, p, 0);
    vec3 color = a.rgb / max(a.a, 1e-9);
    float coverage = 1.0 - exp(-cover_k * a.a);
    vec4 w = textureLod(acc, gl_FragCoord.xy / size_px, wide_lod);
    vec3 fill_color = w.rgb / max(w.a, 1e-9);
    float fill = 1.0 - exp(-fill_k * w.a);
    vec3 under = mix(background, fill_color, fill);
    vec3 soft = mix(under, color, coverage);
    vec3 hard = texelFetch(crisp, p, 0).rgb;
    frag = vec4(mix(hard, soft, soft_mix), 1.0);
}
"""

_BLIT_FRAGMENT = """
#version 330
uniform sampler2D image;
uniform vec2 viewport_size;
uniform vec2 viewport_origin;
out vec4 frag;

void main() {
    vec2 uv = (gl_FragCoord.xy - viewport_origin) / viewport_size;
    frag = vec4(texture(image, vec2(uv.x, 1.0 - uv.y)).rgb, 1.0);
}
"""


def create_headless_context() -> moderngl.Context:
    """Create an off-screen OpenGL context, or raise :class:`GpuUnavailableError`."""
    try:
        import moderngl
    except ImportError as exc:
        raise GpuUnavailableError(
            'moderngl is not installed; run: pip install -e ".[gpu]"'
        ) from exc
    try:
        return moderngl.create_standalone_context()
    except Exception as exc:  # driver and platform errors are not a fixed type
        raise GpuUnavailableError(f"cannot create an OpenGL context: {exc}") from exc


class GpuRenderer:
    """Real-time soft-particle renderer. Same interface as the CPU renderers."""

    def __init__(
        self,
        animator: Animator,
        *,
        scale: int = 3,
        background: tuple[int, int, int] = (12, 12, 16),
        softness: float = 0.55,
        motion_blur: float = 2.5,
        samples: int = 7,
        edge: float = 0.04,
        context: moderngl.Context | None = None,
        accumulation: str = "f2",
    ) -> None:
        if scale < 1 or samples < 1 or softness <= 0 or motion_blur < 0 or not 0 < edge < 0.5:
            raise ValueError("invalid renderer settings")
        try:
            import moderngl
        except ImportError as exc:
            raise GpuUnavailableError(
                'moderngl is not installed; run: pip install -e ".[gpu]"'
            ) from exc

        self._mgl = moderngl
        self.ctx = context if context is not None else create_headless_context()
        self.animator = animator
        self.scale = scale
        self.samples = samples
        self.edge = edge
        if accumulation not in ("f2", "f4"):
            raise ValueError("accumulation must be 'f2' (half) or 'f4' (float)")
        self._acc_dtype = accumulation
        self.size = animator.mapping.n * scale
        cfg = animator.config
        frames = cfg.frame_count
        self._trail = motion_blur / max(frames - 1, 1)
        self._background = tuple(c / 255.0 for c in background)

        sigma = softness * scale
        self._sigma = sigma
        self._diameter = float(math.ceil(6.0 * sigma) + 1)
        mass = 2.0 * math.pi * sigma * sigma
        self._cover_k = 6.0 * scale * scale / mass
        self._fill_k = 4.0 * scale * scale / mass
        self._wide_lod = max(math.log2(6.0 * scale), 0.0)

        self._build_particles(animator)
        self._build_programs()
        self._build_targets()
        self._set_motion_uniforms(cfg.easing)

    # -- setup ------------------------------------------------------------------------------

    def _build_particles(self, animator: Animator) -> None:
        m = animator.mapping
        colors = m.source_rgb[m.perm].astype(np.float32) / 255.0
        data = np.concatenate(
            [
                animator.start.astype(np.float32),
                animator.end.astype(np.float32),
                colors,
                animator.delay.astype(np.float32)[:, None],
                animator.phase.astype(np.float32),
                animator.freq.astype(np.float32)[:, None],
            ],
            axis=1,
        )
        self._particles = self.ctx.buffer(np.ascontiguousarray(data).tobytes())
        self._layout = "2f 2f 3f 1f 2f 1f"
        self._attributes = ["in_p0", "in_p1", "in_color", "in_delay", "in_phase", "in_freq"]

    def _build_programs(self) -> None:
        ctx = self.ctx
        self._soft = ctx.program(vertex_shader=_SOFT_VERTEX, fragment_shader=_SOFT_FRAGMENT)
        self._crisp = ctx.program(vertex_shader=_CRISP_VERTEX, fragment_shader=_CRISP_FRAGMENT)
        self._resolve = ctx.program(
            vertex_shader=_FULLSCREEN_VERTEX, fragment_shader=_RESOLVE_FRAGMENT
        )
        self._blit = ctx.program(vertex_shader=_FULLSCREEN_VERTEX, fragment_shader=_BLIT_FRAGMENT)
        self._soft_vao = ctx.vertex_array(
            self._soft, [(self._particles, self._layout, *self._attributes)]
        )
        self._crisp_vao = ctx.vertex_array(
            self._crisp, [(self._particles, self._layout, *self._attributes)]
        )
        self._quad_resolve = ctx.vertex_array(self._resolve, [])
        self._quad_blit = ctx.vertex_array(self._blit, [])

    def _build_targets(self) -> None:
        ctx, size = self.ctx, (self.size, self.size)
        self._acc = ctx.texture(size, 4, dtype=self._acc_dtype)
        self._acc.filter = (self._mgl.LINEAR_MIPMAP_LINEAR, self._mgl.LINEAR)
        self._acc.repeat_x = self._acc.repeat_y = False
        self._acc_fbo = ctx.framebuffer(color_attachments=[self._acc])
        self._crisp_tex = ctx.texture(size, 4, dtype="f1")
        self._crisp_tex.filter = (self._mgl.NEAREST, self._mgl.NEAREST)
        self._crisp_fbo = ctx.framebuffer(color_attachments=[self._crisp_tex])
        self._out_tex = ctx.texture(size, 4, dtype="f1")
        self._out_tex.filter = (self._mgl.LINEAR, self._mgl.LINEAR)
        self._out_fbo = ctx.framebuffer(color_attachments=[self._out_tex])

    def _set_motion_uniforms(self, easing: str) -> None:
        cfg = self.animator.config
        values: dict[str, Any] = {
            "delay_scale": 1.0 - cfg.delay_strength,
            "wobble": cfg.wobble,
            "swirl": cfg.swirl,
            "swirl_falloff": cfg.swirl_falloff,
            "radius": float(self.animator.radius),
            "center": tuple(float(c) for c in self.animator.center),
            "easing": EASING_IDS[easing],
            "size_px": float(self.size),
            "scale": float(self.scale),
        }
        for program in (self._soft, self._crisp):
            for name, value in values.items():
                if name in program:
                    program[name].value = value  # type: ignore[union-attr]
        self._soft["sigma"].value = self._sigma  # type: ignore[union-attr]
        self._soft["point_diameter"].value = self._diameter  # type: ignore[union-attr]
        self._resolve["size_px"].value = float(self.size)  # type: ignore[union-attr]
        self._resolve["cover_k"].value = self._cover_k  # type: ignore[union-attr]
        self._resolve["fill_k"].value = self._fill_k  # type: ignore[union-attr]
        self._resolve["wide_lod"].value = self._wide_lod  # type: ignore[union-attr]
        self._resolve["background"].value = self._background  # type: ignore[union-attr]
        self._resolve["acc"].value = 0  # type: ignore[union-attr]
        self._resolve["crisp"].value = 1  # type: ignore[union-attr]
        self._blit["image"].value = 0  # type: ignore[union-attr]

    # -- rendering --------------------------------------------------------------------------

    def _mix(self, t: float) -> float:
        x = min(1.0, min(t, 1.0 - t) / self.edge)
        return x * x * (3 - 2 * x)

    def render(self, t: float) -> None:
        """Render time ``t`` in [0, 1] into the internal output texture."""
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0, 1]")
        ctx, mgl = self.ctx, self._mgl
        mix = self._mix(t)
        ctx.enable(mgl.PROGRAM_POINT_SIZE)
        ctx.viewport = (0, 0, self.size, self.size)

        if mix > 0.0:
            self._acc_fbo.use()
            self._acc_fbo.clear(0.0, 0.0, 0.0, 0.0)
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.ONE, mgl.ONE
            if self.samples == 1:
                times = [t]
            else:
                times = [
                    min(max(t - self._trail * k / (self.samples - 1), 0.0), 1.0)
                    for k in range(self.samples)
                ]
            self._soft["sample_weight"].value = 1.0 / len(times)  # type: ignore[union-attr]
            for ts in times:
                self._soft["t"].value = float(ts)  # type: ignore[union-attr]
                self._soft_vao.render(mgl.POINTS)
            ctx.disable(mgl.BLEND)
            self._acc.build_mipmaps()

        if mix < 1.0:
            self._crisp_fbo.use()
            r, g, b = self._background
            self._crisp_fbo.clear(color=(r, g, b, 1.0))
            self._crisp["t"].value = float(t)  # type: ignore[union-attr]
            self._crisp_vao.render(mgl.POINTS)

        self._out_fbo.use()
        self._acc.use(location=0)
        self._crisp_tex.use(location=1)
        self._resolve["soft_mix"].value = float(mix)  # type: ignore[union-attr]
        self._quad_resolve.render(mgl.TRIANGLE_STRIP, vertices=4)

    def frame(self, t: float) -> NDArray[np.uint8]:
        """Render and read back an (H, W, 3) uint8 frame."""
        self.render(t)
        raw = self._out_fbo.read(components=3)
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.size, self.size, 3).copy()

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        count = self.animator.config.frame_count
        for k in range(count):
            yield self.frame(k / (count - 1))

    def release(self) -> None:
        """Free this renderer's GPU objects. The OpenGL context itself is left alone."""
        for name in (
            "_acc_fbo",
            "_crisp_fbo",
            "_out_fbo",
            "_acc",
            "_crisp_tex",
            "_out_tex",
            "_soft_vao",
            "_crisp_vao",
            "_quad_resolve",
            "_quad_blit",
            "_soft",
            "_crisp",
            "_resolve",
            "_blit",
            "_particles",
        ):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.release()
                setattr(self, name, None)

    def blit(self, target: moderngl.Framebuffer, width: int, height: int) -> None:
        """Draw the last rendered frame into ``target``, centered and upright, keeping aspect."""
        side = min(width, height)
        origin = ((width - side) // 2, (height - side) // 2)
        target.use()
        self.ctx.viewport = (0, 0, width, height)
        target.clear(0.04, 0.04, 0.05, 1.0)
        self.ctx.viewport = (origin[0], origin[1], side, side)
        self._out_tex.use(location=0)
        self._blit["viewport_size"].value = (float(side), float(side))  # type: ignore[union-attr]
        self._blit["viewport_origin"].value = (float(origin[0]), float(origin[1]))  # type: ignore[union-attr]
        self._quad_blit.render(self._mgl.TRIANGLE_STRIP, vertices=4)
        self.ctx.viewport = (0, 0, width, height)
