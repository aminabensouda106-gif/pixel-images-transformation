"""Recursive median-split matching: a fast, good initial permutation.

Both pixel sets always have equal size inside a group. We pick the feature axis with the largest
variance over the union of the two sets, sort both sets along it, split each at the median, and
match lower half with lower half and upper with upper. Recursing down to single pixels yields a
bijection by construction. Cost is O(P log^2 P).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

Float = NDArray[np.float64]


@njit(cache=True)
def _median_split(src: Float, tgt: Float) -> NDArray[np.int64]:
    p, d = src.shape
    s_idx = np.arange(p)
    t_idx = np.arange(p)
    perm = np.empty(p, dtype=np.int64)
    stack = np.empty((128, 2), dtype=np.int64)
    top = 0
    stack[0, 0] = 0
    stack[0, 1] = p
    top = 1
    while top > 0:
        top -= 1
        lo = stack[top, 0]
        hi = stack[top, 1]
        m = hi - lo
        if m == 1:
            perm[t_idx[lo]] = s_idx[lo]
            continue
        axis = 0
        best = -1.0
        for a in range(d):
            mean = 0.0
            for k in range(lo, hi):
                mean += src[s_idx[k], a] + tgt[t_idx[k], a]
            mean /= 2 * m
            var = 0.0
            for k in range(lo, hi):
                u = src[s_idx[k], a] - mean
                v = tgt[t_idx[k], a] - mean
                var += u * u + v * v
            if var > best:
                best = var
                axis = a
        vals = np.empty(m)
        for k in range(m):
            vals[k] = src[s_idx[lo + k], axis]
        order = np.argsort(vals)
        tmp = s_idx[lo:hi].copy()
        for k in range(m):
            s_idx[lo + k] = tmp[order[k]]
        for k in range(m):
            vals[k] = tgt[t_idx[lo + k], axis]
        order = np.argsort(vals)
        tmp = t_idx[lo:hi].copy()
        for k in range(m):
            t_idx[lo + k] = tmp[order[k]]
        mid = lo + m // 2
        stack[top, 0] = lo
        stack[top, 1] = mid
        stack[top + 1, 0] = mid
        stack[top + 1, 1] = hi
        top += 2
    return perm


class MedianSplitSolver:
    name = "mediansplit"

    def solve(self, src_feat: Float, tgt_feat: Float, *, seed: int = 0) -> NDArray[np.int64]:
        if src_feat.shape != tgt_feat.shape:
            raise ValueError("source and target features must have the same shape")
        result: NDArray[np.int64] = _median_split(
            np.ascontiguousarray(src_feat), np.ascontiguousarray(tgt_feat)
        )
        return result
