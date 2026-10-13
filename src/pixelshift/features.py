"""Per-pixel feature vectors: perceptual color plus weighted position.

With features ``f = (c1, c2, c3, w*x, w*y)`` and ``w = sqrt(lam)``, the squared Euclidean distance
between two feature vectors equals the project cost::

    ||color_a - color_b||^2 + lam * ||pos_a - pos_b||^2

Positions are normalized to [0, 1] so ``lam`` means the same thing at every resolution.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pixelshift import color

ColorSpace = Literal["oklab", "lab"]

Float = NDArray[np.float64]


def build_features(
    rgb: NDArray[np.uint8], *, lam: float, color_space: ColorSpace = "oklab"
) -> Float:
    """Return a (P, 5) feature array for an (n, n, 3) uint8 image, in raster order."""
    if rgb.ndim != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[2] != 3:
        raise ValueError("expected an (n, n, 3) image")
    if lam < 0:
        raise ValueError("lam must be >= 0")
    n = rgb.shape[0]
    rgb_f = color.rgb_u8_to_float(rgb).reshape(-1, 3)
    if color_space == "oklab":
        col = color.rgb_to_oklab(rgb_f)
    elif color_space == "lab":
        col = color.rgb_to_lab(rgb_f) / 100.0  # bring L to roughly the OKLab scale
    else:
        raise ValueError(f"unknown color space: {color_space}")
    ys, xs = np.divmod(np.arange(n * n), n)
    scale = np.sqrt(lam) / max(n - 1, 1)
    pos = np.stack([xs * scale, ys * scale], axis=-1)
    return np.concatenate([col, pos], axis=1)
