"""Quality metrics for a permutation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pixelshift import color
from pixelshift.mapping import Mapping

Float = NDArray[np.float64]


def total_cost(src_feat: Float, tgt_feat: Float, perm: NDArray[np.integer[Any]]) -> float:
    """Objective value: sum over target slots of ||src_feat[perm[i]] - tgt_feat[i]||^2."""
    diff = src_feat[perm] - tgt_feat
    return float(np.einsum("ij,ij->", diff, diff))


def mean_delta_e(result_rgb: NDArray[np.uint8], target_rgb: NDArray[np.uint8]) -> float:
    """Mean CIELAB ΔE76 between two images of the same shape (lower is better)."""
    a = color.rgb_to_lab(color.rgb_u8_to_float(result_rgb).reshape(-1, 3))
    b = color.rgb_to_lab(color.rgb_u8_to_float(target_rgb).reshape(-1, 3))
    return float(np.linalg.norm(a - b, axis=1).mean())


def mean_displacement(mapping: Mapping) -> float:
    """Average travel distance in pixels, as a fraction of the grid side."""
    d = np.linalg.norm(mapping.end_positions() - mapping.start_positions(), axis=1)
    return float(d.mean() / max(mapping.n - 1, 1))


def cost_gap(cost: float, reference_cost: float) -> float:
    """Relative excess cost over a reference (usually the exact optimum)."""
    return (cost - reference_cost) / reference_cost if reference_cost > 0 else 0.0
