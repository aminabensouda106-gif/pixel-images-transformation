"""CPU renderer: scatter each pixel as a small square into a frame buffer.

Every source pixel becomes a ``scale x scale`` square, so at ``t = 0`` and ``t = 1`` the frame is
exactly the upscaled source / result image. In between, clumping of moving pixels leaves gaps that
show the background, which gives the speckled "dust" look of pixels in flight.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from pixelshift.animate import Animator


class CpuRenderer:
    def __init__(
        self,
        animator: Animator,
        *,
        scale: int = 3,
        background: tuple[int, int, int] = (12, 12, 16),
    ) -> None:
        if scale < 1:
            raise ValueError("scale must be >= 1")
        self.animator = animator
        self.scale = scale
        self.background = np.array(background, dtype=np.uint8)
        m = animator.mapping
        self.size = m.n * scale
        self._colors = m.source_rgb[m.perm]  # color of the pixel assigned to each slot

    def frame(self, t: float) -> NDArray[np.uint8]:
        s, size = self.scale, self.size
        pos = np.rint(self.animator.positions(t) * s).astype(np.int64)
        x = np.clip(pos[:, 0], 0, size - s)
        y = np.clip(pos[:, 1], 0, size - s)
        out = np.empty((size, size, 3), dtype=np.uint8)
        out[:] = self.background
        for dy in range(s):
            for dx in range(s):
                out[y + dy, x + dx] = self._colors
        return out

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        count = self.animator.config.frame_count
        for k in range(count):
            yield self.frame(k / (count - 1))
