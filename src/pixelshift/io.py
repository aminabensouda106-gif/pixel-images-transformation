"""Image loading, square cropping and saving."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_square(path: str | Path, n: int) -> NDArray[np.uint8]:
    """Load an image, center-crop to a square, resize to ``n x n``. Returns (n, n, 3) uint8."""
    if n < 1:
        raise ValueError("n must be >= 1")
    with Image.open(path) as opened:
        img = opened.convert("RGB")
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((n, n), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.uint8).copy()


def save_image(path: str | Path, rgb: NDArray[np.uint8]) -> None:
    """Save an (H, W, 3) uint8 array."""
    Image.fromarray(rgb, mode="RGB").save(path)
