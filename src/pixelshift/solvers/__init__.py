"""Solvers: each turns two feature arrays into a permutation."""

from __future__ import annotations

from pixelshift.solvers.base import Solver, get_solver
from pixelshift.solvers.combined import PipelineSolver
from pixelshift.solvers.exact import ChunkPolish, ExactSolver
from pixelshift.solvers.mediansplit import MedianSplitSolver
from pixelshift.solvers.sort1d import SortSolver
from pixelshift.solvers.swaps import SwapSolver

__all__ = [
    "ChunkPolish",
    "ExactSolver",
    "MedianSplitSolver",
    "PipelineSolver",
    "SortSolver",
    "Solver",
    "SwapSolver",
    "get_solver",
]
