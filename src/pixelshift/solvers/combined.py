"""The production solver: median-split guess, parallel swaps, then chunked exact polish."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pixelshift.solvers.exact import ChunkPolish
from pixelshift.solvers.mediansplit import MedianSplitSolver
from pixelshift.solvers.swaps import SwapSolver

Float = NDArray[np.float64]


class PipelineSolver:
    name = "pipeline"

    def __init__(self, swap_rounds: int = 600, sweeps: int = 12, chunk: int = 256) -> None:
        self.swaps = SwapSolver(rounds=swap_rounds)
        self.polish = ChunkPolish(sweeps=sweeps, chunk=chunk)

    def solve(self, src_feat: Float, tgt_feat: Float, *, seed: int = 0) -> NDArray[np.int64]:
        perm = MedianSplitSolver().solve(src_feat, tgt_feat, seed=seed)
        perm = self.swaps.refine(src_feat, tgt_feat, perm, seed=seed)
        return self.polish.refine(src_feat, tgt_feat, perm, seed=seed)
