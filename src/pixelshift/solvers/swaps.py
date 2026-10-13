"""Parallel pairwise-swap refinement.

A swap exchanges the source pixels assigned to two target slots and is kept only if it lowers the
cost. Per round we pick one shift ``(dx, dy)`` and pair every eligible slot with its shifted one.
Eligibility uses a parity pattern (``floor((x + ox) / |dx|)`` even, same for y) that guarantees the
pairs are disjoint, so all swaps of a round run in parallel with no locks and the result does not
depend on thread scheduling.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from pixelshift.solvers.mediansplit import MedianSplitSolver

Float = NDArray[np.float64]


@njit(parallel=True, cache=True, fastmath=True)
def _swap_rounds(
    src: Float, tgt: Float, perm: NDArray[np.int64], n: int, moves: NDArray[np.int64]
) -> None:
    d = src.shape[1]
    for r in range(moves.shape[0]):
        dx, dy, ox, oy = moves[r, 0], moves[r, 1], moves[r, 2], moves[r, 3]
        adx = abs(dx)
        ady = abs(dy)
        for i in prange(n * n):
            x = i % n
            y = i // n
            if adx > 0 and ((x + ox) // adx) % 2 != 0:
                continue
            if ady > 0 and ((y + oy) // ady) % 2 != 0:
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            j = ny * n + nx
            a = perm[i]
            b = perm[j]
            cur = 0.0
            swp = 0.0
            for k in range(d):
                ia = src[a, k] - tgt[i, k]
                jb = src[b, k] - tgt[j, k]
                ib = src[b, k] - tgt[i, k]
                ja = src[a, k] - tgt[j, k]
                cur += ia * ia + jb * jb
                swp += ib * ib + ja * ja
            if swp < cur:
                perm[i] = b
                perm[j] = a


def make_moves(n: int, rounds: int, seed: int) -> NDArray[np.int64]:
    """Random shift schedule: the maximum radius shrinks from n/2 to 2 over the rounds."""
    rng = np.random.default_rng(seed)
    moves = np.zeros((rounds, 4), dtype=np.int64)
    r_start, r_end = max(n // 2, 2), 2
    for t in range(rounds):
        frac = t / max(rounds - 1, 1)
        r_max = max(r_end, round(r_start + (r_end - r_start) * frac))
        r = max(1, int(round(math.exp(rng.uniform(0.0, math.log(r_max))))))
        dx, dy = 0, 0
        while dx == 0 and dy == 0:
            dx, dy = int(rng.integers(-r, r + 1)), int(rng.integers(-r, r + 1))
        ox = int(rng.integers(0, 2 * max(abs(dx), 1)))
        oy = int(rng.integers(0, 2 * max(abs(dy), 1)))
        moves[t] = (dx, dy, ox, oy)
    return moves


def grid_side(p: int) -> int:
    n = math.isqrt(p)
    if n * n != p:
        raise ValueError(f"{p} pixels is not a square grid")
    return n


class SwapSolver:
    """Median-split initial guess followed by parallel swap refinement."""

    name = "swaps"

    def __init__(self, rounds: int = 600) -> None:
        self.rounds = rounds

    def refine(
        self, src_feat: Float, tgt_feat: Float, perm: NDArray[np.int64], *, seed: int = 0
    ) -> NDArray[np.int64]:
        n = grid_side(src_feat.shape[0])
        out = perm.copy()
        _swap_rounds(
            np.ascontiguousarray(src_feat),
            np.ascontiguousarray(tgt_feat),
            out,
            n,
            make_moves(n, self.rounds, seed),
        )
        return out

    def solve(self, src_feat: Float, tgt_feat: Float, *, seed: int = 0) -> NDArray[np.int64]:
        init = MedianSplitSolver().solve(src_feat, tgt_feat, seed=seed)
        return self.refine(src_feat, tgt_feat, init, seed=seed)
