"""Luminance-rank baseline.

In one dimension the optimal transport map is sorting (Brenier). We use only the lightness channel
(feature 0), so this ignores color and position. It is a fast, deliberately weak reference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float64]


class SortSolver:
    name = "sort"

    def solve(self, src_feat: Float, tgt_feat: Float, *, seed: int = 0) -> NDArray[np.int64]:
        if src_feat.shape != tgt_feat.shape:
            raise ValueError("source and target features must have the same shape")
        src_order = np.argsort(src_feat[:, 0], kind="stable")
        tgt_order = np.argsort(tgt_feat[:, 0], kind="stable")
        perm = np.empty(src_feat.shape[0], dtype=np.int64)
        perm[tgt_order] = src_order
        return perm
