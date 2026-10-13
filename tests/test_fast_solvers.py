import numpy as np
import pytest

from pixelshift.features import build_features
from pixelshift.mapping import is_permutation
from pixelshift.metrics import cost_gap, total_cost
from pixelshift.pipeline import solve_images
from pixelshift.solvers import (
    ChunkPolish,
    ExactSolver,
    MedianSplitSolver,
    PipelineSolver,
    SortSolver,
    SwapSolver,
    get_solver,
)


def smooth_image(seed: int, n: int):
    """Smooth random blobs: closer to a photo than white noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.zeros((n, n, 3))
    for _ in range(5):
        cx, cy, s = rng.random(), rng.random(), 0.1 + 0.3 * rng.random()
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        img += blob[..., None] * rng.random(3)
    return (255 * img / img.max()).astype(np.uint8)


def feats(a, b, lam=0.2):
    return build_features(a, lam=lam), build_features(b, lam=lam)


@pytest.mark.parametrize("p", [1, 2, 7, 64, 100, 1000])
def test_median_split_is_a_permutation_for_any_size(p):
    rng = np.random.default_rng(p)
    perm = MedianSplitSolver().solve(rng.random((p, 5)), rng.random((p, 5)))
    assert is_permutation(perm)


def test_median_split_beats_luminance_sort():
    a, b = smooth_image(0, 32), smooth_image(1, 32)
    sf, tf = feats(a, b)
    assert total_cost(sf, tf, MedianSplitSolver().solve(sf, tf)) < total_cost(
        sf, tf, SortSolver().solve(sf, tf)
    )


@pytest.mark.parametrize("n", [8, 17, 32])
def test_swap_refine_is_valid_and_never_increases_cost(n):
    a, b = smooth_image(2, n), smooth_image(3, n)
    sf, tf = feats(a, b)
    start = MedianSplitSolver().solve(sf, tf)
    out = SwapSolver(rounds=200).refine(sf, tf, start, seed=3)
    assert is_permutation(out)
    assert total_cost(sf, tf, out) <= total_cost(sf, tf, start)


def test_swap_refine_does_not_modify_input_and_is_deterministic():
    a, b = smooth_image(4, 16), smooth_image(5, 16)
    sf, tf = feats(a, b)
    start = MedianSplitSolver().solve(sf, tf)
    keep = start.copy()
    s = SwapSolver(rounds=100)
    one, two = s.refine(sf, tf, start, seed=9), s.refine(sf, tf, start, seed=9)
    assert np.array_equal(start, keep)
    assert np.array_equal(one, two)


def test_swaps_reject_non_square_grid():
    x = np.zeros((10, 5))
    with pytest.raises(ValueError):
        SwapSolver().refine(x, x, np.arange(10))


def test_chunk_polish_never_increases_cost_and_is_deterministic():
    a, b = smooth_image(6, 20), smooth_image(7, 20)
    sf, tf = feats(a, b)
    start = SwapSolver(rounds=50).solve(sf, tf)
    polish = ChunkPolish(sweeps=3, chunk=64)
    out = polish.refine(sf, tf, start, seed=1)
    assert is_permutation(out)
    assert total_cost(sf, tf, out) <= total_cost(sf, tf, start)
    assert np.array_equal(out, polish.refine(sf, tf, start, seed=1))


def test_chunk_polish_with_one_chunk_is_exact():
    a, b = smooth_image(8, 10), smooth_image(9, 10)
    sf, tf = feats(a, b)
    start = MedianSplitSolver().solve(sf, tf)
    out = ChunkPolish(sweeps=1, chunk=1000).refine(sf, tf, start)
    best = ExactSolver().solve(sf, tf)
    assert total_cost(sf, tf, out) == pytest.approx(total_cost(sf, tf, best))


@pytest.mark.parametrize("lam", [0.0, 0.2])
def test_pipeline_is_close_to_exact(lam):
    a, b = smooth_image(10, 32), smooth_image(11, 32)
    exact = solve_images(a, b, ExactSolver(), lam=lam)
    fast = solve_images(a, b, PipelineSolver(), lam=lam)
    assert fast.cost >= exact.cost - 1e-9
    assert cost_gap(fast.cost, exact.cost) < 0.05


def test_pipeline_preserves_pixels_and_is_deterministic():
    a, b = smooth_image(12, 24), smooth_image(13, 24)
    m1 = solve_images(a, b, PipelineSolver(), seed=5)
    m2 = solve_images(a, b, PipelineSolver(), seed=5)
    assert np.array_equal(m1.perm, m2.perm)
    key = lambda x: sorted(map(tuple, x.reshape(-1, 3).tolist()))  # noqa: E731
    assert key(m1.result_image()) == key(a)


def test_registry_has_new_solvers():
    for name in ("mediansplit", "swaps", "pipeline"):
        assert get_solver(name).name == name
