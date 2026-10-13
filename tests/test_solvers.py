import itertools

import numpy as np
import pytest

from pixelshift.features import build_features
from pixelshift.mapping import Mapping, is_permutation
from pixelshift.metrics import cost_gap, mean_delta_e, mean_displacement, total_cost
from pixelshift.pipeline import solve_images
from pixelshift.solvers import ExactSolver, SortSolver, get_solver


def random_image(seed: int, n: int):
    return np.random.default_rng(seed).integers(0, 256, (n, n, 3), dtype=np.uint8)


def gradient_image(n: int):
    g = np.linspace(0, 255, n).astype(np.uint8)
    return np.stack([np.tile(g, (n, 1)), np.tile(g[:, None], (1, n)), np.full((n, n), 90)], -1)


PAIRS = [
    (random_image(0, 8), random_image(1, 8)),
    (random_image(2, 12), gradient_image(12)),
    (gradient_image(10), random_image(3, 10)),
]


@pytest.mark.parametrize("solver", [SortSolver(), ExactSolver()], ids=lambda s: s.name)
@pytest.mark.parametrize("pair", range(len(PAIRS)))
def test_solvers_return_valid_permutation_preserving_pixels(solver, pair):
    src, tgt = PAIRS[pair]
    m = solve_images(src, tgt, solver)
    assert is_permutation(m.perm)
    result = m.result_image()
    key = lambda a: sorted(map(tuple, a.reshape(-1, 3).tolist()))  # noqa: E731
    assert key(result) == key(src)


@pytest.mark.parametrize("lam", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("pair", range(len(PAIRS)))
def test_exact_never_worse_than_sort(pair, lam):
    src, tgt = PAIRS[pair]
    exact = solve_images(src, tgt, ExactSolver(), lam=lam)
    sort = solve_images(src, tgt, SortSolver(), lam=lam)
    assert exact.cost <= sort.cost + 1e-9


def test_exact_beats_sort_on_delta_e_without_spatial_term():
    for src, tgt in PAIRS:
        exact = solve_images(src, tgt, ExactSolver(), lam=0.0)
        sort = solve_images(src, tgt, SortSolver(), lam=0.0)
        assert mean_delta_e(exact.result_image(), tgt) < mean_delta_e(sort.result_image(), tgt)


def test_exact_matches_brute_force():
    src, tgt = random_image(4, 2), random_image(5, 2)
    sf, tf = build_features(src, lam=0.3), build_features(tgt, lam=0.3)
    best = min(total_cost(sf, tf, np.array(p)) for p in itertools.permutations(range(4)))
    exact = solve_images(src, tgt, ExactSolver(), lam=0.3)
    assert exact.cost == pytest.approx(best)


def test_identical_images_have_zero_cost_and_displacement():
    img = random_image(6, 6)
    m = solve_images(img, img, ExactSolver(), lam=0.5)
    assert m.cost == pytest.approx(0.0, abs=1e-12)
    assert mean_displacement(m) == 0.0


def test_larger_lambda_reduces_travel_distance():
    src, tgt = PAIRS[0]
    near = solve_images(src, tgt, ExactSolver(), lam=5.0)
    far = solve_images(src, tgt, ExactSolver(), lam=0.0)
    assert mean_displacement(near) < mean_displacement(far)


def test_solvers_are_deterministic():
    src, tgt = PAIRS[0]
    for name in ("sort", "exact"):
        a = solve_images(src, tgt, get_solver(name), seed=1).perm
        b = solve_images(src, tgt, get_solver(name), seed=1).perm
        assert np.array_equal(a, b)


def test_exact_rejects_large_grids():
    big = np.zeros((65 * 65, 5))
    with pytest.raises(ValueError):
        ExactSolver().solve(big, big)


def test_mapping_rejects_invalid_perm():
    rgb = np.zeros((4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        Mapping(n=2, perm=np.array([0, 0, 1, 2]), source_rgb=rgb, cost=0.0)


def test_mapping_positions_and_result_image():
    src = random_image(7, 3)
    m = solve_images(src, random_image(8, 3), SortSolver())
    assert m.start_positions().shape == (9, 2)
    assert m.end_positions().shape == (9, 2)
    assert np.array_equal(m.result_image().reshape(-1, 3), src.reshape(-1, 3)[m.perm])


def test_cost_gap():
    assert cost_gap(1.1, 1.0) == pytest.approx(0.1)
    assert cost_gap(0.0, 0.0) == 0.0


def test_unknown_solver_and_color_space():
    with pytest.raises(ValueError):
        get_solver("nope")
    with pytest.raises(ValueError):
        build_features(random_image(0, 4), lam=0.1, color_space="hsv")  # type: ignore[arg-type]
