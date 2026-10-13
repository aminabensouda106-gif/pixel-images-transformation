"""Soft renderer: glowing particles with motion blur, for a smooth, fluid look.

Each frame is built in three steps:

1. **Motion blur.** Pixel positions are sampled at several instants trailing the frame time and
   every sample is splatted (bilinear weights) into one accumulation buffer, so fast pixels leave
   soft streaks.
2. **Soft particles.** The accumulated color and density are blurred with a small Gaussian, then
   divided, so neighboring particles blend into a continuous surface instead of showing gaps.
3. **Coverage.** Where density is low the result fades into the background, which gives the
   dissolving-dust edges.

Near ``t = 0`` and ``t = 1`` the frame is cross-faded to the crisp :class:`CpuRenderer` frame, so
the first and last frames are exactly the source and the rearranged image.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from pixelshift.animate import Animator
from pixelshift.render.cpu import CpuRenderer

Float32 = NDArray[np.float32]


class SoftRenderer:
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
    ) -> None:
        """
        softness:    particle radius (Gaussian sigma) as a fraction of one grid pixel.
        motion_blur: length of the trail in frames (0 = no blur).
        samples:     time samples per frame used for the blur.
        edge:        fraction of the timeline at each end that cross-fades to the crisp image.
        """
        if scale < 1 or samples < 1 or softness <= 0 or motion_blur < 0 or not 0 < edge < 0.5:
            raise ValueError("invalid renderer settings")
        self.animator = animator
        self.scale = scale
        self.softness = softness
        self.motion_blur = motion_blur
        self.samples = samples
        self.edge = edge
        self.crisp = CpuRenderer(animator, scale=scale, background=background)
        self.size = self.crisp.size
        self._background = np.array(background, dtype=np.float32)
        self._colors = self.crisp._colors.astype(np.float32)
        frames = animator.config.frame_count
        self._trail = motion_blur / max(frames - 1, 1)  # trail length in normalized time

    def _mix(self, t: float) -> float:
        """0 at the ends (crisp image), 1 in the middle (soft particles)."""
        x = min(1.0, min(t, 1.0 - t) / self.edge)
        return x * x * (3 - 2 * x)

    def _soft_frame(self, t: float) -> Float32:
        s, size = self.scale, self.size
        num = np.zeros((size * size, 3), dtype=np.float32)
        den = np.zeros(size * size, dtype=np.float32)
        times = [t] if self.samples == 1 else t - self._trail * np.linspace(0, 1, self.samples)
        weight_each = np.float32(1.0 / len(times))
        for ts in times:
            pos = self.animator.positions(float(np.clip(ts, 0.0, 1.0)))
            cx = (pos[:, 0] + 0.5) * s - 0.5
            cy = (pos[:, 1] + 0.5) * s - 0.5
            x0, y0 = np.floor(cx).astype(np.int64), np.floor(cy).astype(np.int64)
            fx, fy = (cx - x0).astype(np.float32), (cy - y0).astype(np.float32)
            for dx, wx in ((0, 1 - fx), (1, fx)):
                for dy, wy in ((0, 1 - fy), (1, fy)):
                    xi, yi = x0 + dx, y0 + dy
                    ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
                    w = (wx * wy)[ok] * weight_each
                    idx = yi[ok] * size + xi[ok]
                    den += np.bincount(idx, weights=w, minlength=size * size).astype(np.float32)
                    for c in range(3):
                        num[:, c] += np.bincount(
                            idx, weights=w * self._colors[ok, c], minlength=size * size
                        ).astype(np.float32)
        sigma = self.softness * s
        den_img = gaussian_filter(den.reshape(size, size), sigma)
        num_img = gaussian_filter(num.reshape(size, size, 3), (sigma, sigma, 0))
        color = num_img / np.maximum(den_img, 1e-9)[..., None]
        coverage = 1.0 - np.exp(-6.0 * den_img * s * s)  # density of 1 particle per cell ~ opaque
        # Tiny gaps inside the swarm are filled with the local average color, so that only the
        # outer edge of the swarm dissolves into the background.
        wide = 3.0 * s
        den_w = gaussian_filter(den.reshape(size, size), wide)
        num_w = gaussian_filter(num.reshape(size, size, 3), (wide, wide, 0))
        fill_color = num_w / np.maximum(den_w, 1e-9)[..., None]
        fill = (1.0 - np.exp(-4.0 * den_w * s * s))[..., None]
        under = fill * fill_color + (1.0 - fill) * self._background
        out = coverage[..., None] * color + (1.0 - coverage[..., None]) * under
        return np.asarray(out, dtype=np.float32)

    def frame(self, t: float) -> NDArray[np.uint8]:
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0, 1]")
        m = self._mix(t)
        if m <= 0.0:
            return self.crisp.frame(t)
        soft = self._soft_frame(t)
        if m < 1.0:
            soft = m * soft + (1.0 - m) * self.crisp.frame(t).astype(np.float32)
        return np.clip(np.rint(soft), 0, 255).astype(np.uint8)

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        count = self.animator.config.frame_count
        for k in range(count):
            yield self.frame(k / (count - 1))
