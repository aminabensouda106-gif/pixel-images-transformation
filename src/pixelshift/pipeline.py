"""High-level wiring: images in, ``Mapping`` out."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pixelshift.features import ColorSpace, build_features
from pixelshift.mapping import Mapping
from pixelshift.metrics import total_cost
from pixelshift.solvers import Solver


def solve_images(
    source: NDArray[np.uint8],
    target: NDArray[np.uint8],
    solver: Solver,
    *,
    lam: float = 0.001,
    color_space: ColorSpace = "oklab",
    seed: int = 0,
) -> Mapping:
    """Rearrange the pixels of ``source`` to resemble ``target`` (both (n, n, 3) uint8)."""
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    src_feat = build_features(source, lam=lam, color_space=color_space)
    tgt_feat = build_features(target, lam=lam, color_space=color_space)
    t0 = time.perf_counter()
    perm = solver.solve(src_feat, tgt_feat, seed=seed)
    runtime = time.perf_counter() - t0
    return Mapping(
        n=source.shape[0],
        perm=perm,
        source_rgb=source.reshape(-1, 3),
        cost=total_cost(src_feat, tgt_feat, perm),
        meta={
            "solver": solver.name,
            "lam": lam,
            "color_space": color_space,
            "seed": seed,
            "runtime_s": runtime,
        },
    )
