import numpy as np
import pytest

from pixelshift.animate import AnimationConfig, Animator
from pixelshift.pipeline import solve_images
from pixelshift.render import GpuRenderer, SoftRenderer
from pixelshift.render.gpu import GpuUnavailableError, create_headless_context
from pixelshift.solvers import ExactSolver


@pytest.fixture(scope="module")
def ctx():
    try:
        context = create_headless_context()
    except GpuUnavailableError as exc:
        pytest.skip(f"no OpenGL available: {exc}")
    yield context
    context.release()


def smooth_image(seed: int, n: int):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.zeros((n, n, 3))
    for _ in range(5):
        cx, cy, s = rng.random(), rng.random(), 0.1 + 0.3 * rng.random()
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))[..., None] * rng.random(3)
    return (255 * img / img.max()).astype(np.uint8)


@pytest.fixture(scope="module")
def mapping():
    return solve_images(smooth_image(1, 16), smooth_image(2, 16), ExactSolver(), lam=0.001)


def config(**kw):
    base = dict(duration=1.0, fps=10, swirl=1.0, delay_wave=0.5, delay_strength=0.5, wobble=0.0)
    return AnimationConfig(**{**base, **kw})


def multiset(img):
    return sorted(map(tuple, img.reshape(-1, 3).tolist()))


@pytest.mark.parametrize("easing", ["linear", "smoothstep", "smootherstep", "cubic"])
def test_first_and_last_frames_match_the_cpu_exactly(ctx, mapping, easing):
    anim = Animator(mapping, config(easing=easing, wobble=0.7))
    gpu = GpuRenderer(anim, scale=3, context=ctx)
    cpu = SoftRenderer(anim, scale=3)
    assert np.array_equal(gpu.frame(0.0), cpu.frame(0.0))
    assert np.array_equal(gpu.frame(1.0), cpu.frame(1.0))


def test_last_frame_contains_exactly_the_source_pixels(ctx, mapping):
    gpu = GpuRenderer(Animator(mapping, config()), scale=2, context=ctx)
    last = gpu.frame(1.0)
    expected = np.kron(mapping.source_rgb.reshape(16, 16, 3), np.ones((2, 2, 1), dtype=np.uint8))
    assert multiset(last) == multiset(expected)


@pytest.mark.parametrize("t", [0.2, 0.5, 0.8])
def test_middle_frames_are_close_to_the_cpu_soft_renderer(ctx, mapping, t):
    anim = Animator(mapping, config(wobble=0.5))
    gpu = GpuRenderer(anim, scale=3, context=ctx, accumulation="f4").frame(t)
    cpu = SoftRenderer(anim, scale=3).frame(t)
    assert np.abs(gpu.astype(int) - cpu.astype(int)).mean() < 3.0


def test_frames_are_deterministic_and_well_formed(ctx, mapping):
    gpu = GpuRenderer(Animator(mapping, config()), scale=3, context=ctx)
    a, b = gpu.frame(0.5), gpu.frame(0.5)
    assert a.shape == (48, 48, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, gpu.frame(0.0))


def test_frames_iterator_yields_the_configured_count(ctx, mapping):
    gpu = GpuRenderer(Animator(mapping, config()), scale=2, context=ctx)
    assert len(list(gpu.frames())) == 11


def test_blit_is_upright_and_pixel_exact(ctx, mapping):
    gpu = GpuRenderer(Animator(mapping, config()), scale=3, context=ctx)
    gpu.render(0.0)
    tex = ctx.texture((48, 48), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    gpu.blit(fbo, 48, 48)
    shown = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(48, 48, 3)
    assert np.array_equal(np.flipud(shown), gpu.frame(0.0))  # GL rows are bottom-up


def test_invalid_settings_are_rejected(ctx, mapping):
    anim = Animator(mapping, config())
    for bad in (
        dict(scale=0),
        dict(samples=0),
        dict(softness=0.0),
        dict(edge=0.9),
        dict(accumulation="f8"),
    ):
        with pytest.raises(ValueError):
            GpuRenderer(anim, context=ctx, **bad)
    with pytest.raises(ValueError):
        GpuRenderer(anim, context=ctx).render(1.5)


def test_release_frees_a_renderer_and_a_new_one_can_replace_it(ctx, mapping):
    first = GpuRenderer(Animator(mapping, config()), scale=3, context=ctx)
    before = first.frame(0.0)
    first.release()
    first.release()  # releasing twice is harmless
    second = GpuRenderer(Animator(mapping, config()), scale=3, context=ctx)
    assert np.array_equal(second.frame(0.0), before)
