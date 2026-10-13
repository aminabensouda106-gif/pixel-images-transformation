"""The hand-off object between the solver and everything downstream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def is_permutation(perm: NDArray[np.integer[Any]]) -> bool:
    """True if ``perm`` contains each of 0..len(perm)-1 exactly once."""
    if perm.ndim != 1:
        return False
    seen = np.zeros(perm.size, dtype=bool)
    if perm.size and (perm.min() < 0 or perm.max() >= perm.size):
        return False
    seen[perm] = True
    return bool(seen.all())


@dataclass(frozen=True)
class Mapping:
    """Result of a solve.

    ``perm[i]`` is the index (raster order) of the source pixel placed at target slot ``i``.
    """

    n: int
    perm: NDArray[np.int64]
    source_rgb: NDArray[np.uint8]  # (P, 3) source pixels in raster order
    cost: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = self.n * self.n
        if self.perm.shape != (p,) or not is_permutation(self.perm):
            raise ValueError("perm must be a permutation of range(n*n)")
        if self.source_rgb.shape != (p, 3):
            raise ValueError("source_rgb must have shape (n*n, 3)")

    def start_positions(self) -> NDArray[np.float64]:
        """(P, 2) x, y where the pixel assigned to each target slot starts."""
        ys, xs = np.divmod(self.perm, self.n)
        return np.stack([xs, ys], axis=-1).astype(np.float64)

    def end_positions(self) -> NDArray[np.float64]:
        """(P, 2) x, y of each target slot."""
        ys, xs = np.divmod(np.arange(self.n * self.n), self.n)
        return np.stack([xs, ys], axis=-1).astype(np.float64)

    def result_image(self) -> NDArray[np.uint8]:
        """(n, n, 3) uint8 image made of the source pixels rearranged by ``perm``."""
        return self.source_rgb[self.perm].reshape(self.n, self.n, 3)
