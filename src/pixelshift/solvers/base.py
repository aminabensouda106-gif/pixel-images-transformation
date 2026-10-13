"""Solver interface and registry."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float64]


class Solver(Protocol):
    """A solver returns ``perm`` with ``perm[i]`` = source index placed at target slot ``i``."""

    name: str

    def solve(self, src_feat: Float, tgt_feat: Float, *, seed: int = 0) -> NDArray[np.int64]: ...


def get_solver(name: str) -> Solver:
    """Look up a solver by name."""
    from pixelshift.solvers.combined import PipelineSolver
    from pixelshift.solvers.exact import ExactSolver
    from pixelshift.solvers.mediansplit import MedianSplitSolver
    from pixelshift.solvers.sort1d import SortSolver
    from pixelshift.solvers.swaps import SwapSolver

    registry: dict[str, type[Solver]] = {
        "sort": SortSolver,
        "exact": ExactSolver,
        "mediansplit": MedianSplitSolver,
        "swaps": SwapSolver,
        "pipeline": PipelineSolver,
    }
    try:
        return registry[name]()
    except KeyError:
        raise ValueError(f"unknown solver {name!r}; available: {sorted(registry)}") from None
