"""Exact solver: optimal assignment via SciPy (a Jonker-Volgenant variant).

Builds the dense P x P cost matrix, so it is limited to small grids. It serves as ground truth for
scoring the fast solvers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

Float = NDArray[np.float64]

MAX_PIXELS = 4096  # 64 x 64; the dense float64 matrix is then 134 MB


class ExactSolver:
    name = "exact"

    def solve(self, src_feat: Float, tgt_feat: Float, *, seed: int = 0) -> NDArray[np.int64]:
        if src_feat.shape != tgt_feat.shape:
            raise ValueError("source and target features must have the same shape")
        p = src_feat.shape[0]
        if p > MAX_PIXELS:
            raise ValueError(f"exact solver is limited to {MAX_PIXELS} pixels, got {p}")
        cost = cdist(tgt_feat, src_feat, metric="sqeuclidean")
        rows, cols = linear_sum_assignment(cost)
        perm = np.empty(p, dtype=np.int64)
        perm[rows] = cols
        return perm


def _best_columns(tgt: Float, src: Float) -> NDArray[np.int64]:
    """Optimal source column for each target row."""
    cols: NDArray[np.int64] = linear_sum_assignment(cdist(tgt, src, metric="sqeuclidean"))[1]
    return cols


class ChunkPolish:
    """Exact re-assignment inside groups of target slots with similar features.

    Each sweep draws a random direction in feature space, sorts the target slots along it and cuts
    the order into chunks of ``chunk`` slots. Within a chunk the pixels currently assigned are
    optimally redistributed. Every chunk solve can only lower the cost or keep it.
    """

    name = "polish"

    def __init__(self, sweeps: int = 8, chunk: int = 256, workers: int | None = None) -> None:
        self.sweeps = sweeps
        self.chunk = chunk
        self.workers = workers

    def refine(
        self, src_feat: Float, tgt_feat: Float, perm: NDArray[np.int64], *, seed: int = 0
    ) -> NDArray[np.int64]:
        rng = np.random.default_rng(seed)
        out = perm.copy()
        p, d = tgt_feat.shape
        active = np.flatnonzero(tgt_feat.std(axis=0) > 0)
        for _ in range(self.sweeps):
            direction = np.zeros(d)
            direction[active] = rng.normal(size=active.size)
            order = np.argsort(tgt_feat @ direction, kind="stable")
            # a random offset moves the chunk borders between sweeps
            shift = int(rng.integers(0, self.chunk)) if self.chunk < p else 0
            cuts = list(range(shift, p, self.chunk))
            bounds = [0, *cuts, p] if shift else [*cuts, p]
            groups = [
                order[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:], strict=False) if hi - lo > 1
            ]
            # chunks are disjoint sets of slots, so they can be solved concurrently
            pixels = [out[slots] for slots in groups]

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(
                    pool.map(
                        lambda slots, pix: _best_columns(tgt_feat[slots], src_feat[pix]),
                        groups,
                        pixels,
                    )
                )
            for slots, pix, cols in zip(groups, pixels, results, strict=True):
                out[slots] = pix[cols]
        return out
