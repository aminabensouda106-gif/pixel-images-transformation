import numpy as np
import pytest

from pixelshift.animate import (
    TRANSITIONS,
    WAVE_SHAPES,
    AnimationConfig,
    Animator,
    transition_options,
)
from pixelshift.pipeline import solve_images
from pixelshift.render import GpuRenderer, SoftRenderer
from pixelshift.render.gpu import GpuUnavailableError, create_headless_context
from pixelshift.solvers import ExactSolver, MedianSplitSolver


def smooth_image(seed: int, n: int):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.zeros((n, n, 3))
    for _ in range(5):
        cx, cy, s = rng.random(), rng.random(), 0.1 + 0.3 * rng.random()
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))[..., None] * rng.random(3)
    return (255 * img / img.max()).astype(np.uint8)


@pytest.fixture(scope="module")
def mapping():
    return solve_images(smooth_image(1, 12), smooth_image(2, 12), ExactSolver(), lam=0.001)


@pytest.fixture(scope="module")
def identity():
    img = smooth_image(3, 9)
    return solve_images(img, img, MedianSplitSolver(), lam=1.0)  # pixels stay where they are


def config_for(name: str, **kw):
    return AnimationConfig(duration=1.0, fps=10, **{**transition_options(name), **kw})


@pytest.mark.parametrize("name", sorted(TRANSITIONS))
def test_every_preset_builds_and_keeps_exact_endpoints(mapping, name):
    anim = Animator(mapping, config_for(name))
    assert np.array_equal(anim.positions(0.0), mapping.start_positions())
    assert np.array_equal(anim.positions(1.0), mapping.end_positions())
    assert not np.array_equal(anim.positions(0.5), anim.positions(0.0))


def test_unknown_transition_lists_the_choices():
    with pytest.raises(ValueError, match="spiral"):
        transition_options("teleport")


def test_presets_are_copies_so_callers_cannot_change_them():
    options = transition_options("spiral")
    options["swirl"] = 99.0
    assert transition_options("spiral")["swirl"] != 99.0


def test_burst_pushes_pixels_outward_mid_flight_and_back(mapping):
    calm = Animator(mapping, config_for("spiral", swirl=0.0, burst=0.0))
    burst = Animator(mapping, config_for("spiral", swirl=0.0, burst=1.0))
    center = calm.center
    spread = lambda a: np.linalg.norm(a.positions(0.5) - center, axis=1).mean()  # noqa: E731
    assert spread(burst) > spread(calm) * 1.2
    assert np.array_equal(burst.positions(1.0), mapping.end_positions())


def test_horizontal_and_vertical_waves_order_departures_by_column_and_row(identity):
    n = identity.n
    base = dict(delay_strength=0.5, delay_jitter=0.0, delay_wave=1.0)
    horizontal = Animator(identity, AnimationConfig(wave_shape="horizontal", **base))
    vertical = Animator(identity, AnimationConfig(wave_shape="vertical", **base))
    h = horizontal.delay.reshape(n, n)
    v = vertical.delay.reshape(n, n)
    assert np.all(np.diff(h, axis=1) > 0) and np.allclose(h, h[0])  # same for every row
    assert np.all(np.diff(v, axis=0) > 0) and np.allclose(v.T, v.T[0])  # same for every column


def test_radial_waves_start_at_the_center_or_the_edge(identity):
    n = identity.n
    base = dict(delay_strength=0.5, delay_jitter=0.0, delay_wave=1.0)
    out = Animator(identity, AnimationConfig(wave_shape="radial", **base)).delay.reshape(n, n)
    inward = Animator(identity, AnimationConfig(wave_shape="radial_in", **base)).delay.reshape(n, n)
    mid = n // 2
    assert out[mid, mid] == out.min() and out[0, 0] == out.max()
    assert inward[mid, mid] == inward.max() and inward[0, 0] == inward.min()


def test_invalid_shape_and_burst_are_rejected():
    with pytest.raises(ValueError):
        AnimationConfig(wave_shape="zigzag")
    with pytest.raises(ValueError):
        AnimationConfig(burst=-0.1)
    with pytest.raises(ValueError):
        AnimationConfig(burst=5.0)
    assert set(WAVE_SHAPES) == {"diagonal", "horizontal", "vertical", "radial", "radial_in"}


@pytest.fixture(scope="module")
def ctx():
    try:
        context = create_headless_context()
    except GpuUnavailableError as exc:
        pytest.skip(f"no OpenGL available: {exc}")
    yield context
    context.release()


@pytest.mark.parametrize("name", sorted(TRANSITIONS))
def test_gpu_matches_cpu_for_every_preset(ctx, mapping, name):
    anim = Animator(mapping, config_for(name))
    gpu = GpuRenderer(anim, scale=3, context=ctx, accumulation="f4")
    cpu = SoftRenderer(anim, scale=3)
    assert np.array_equal(gpu.frame(0.0), cpu.frame(0.0))
    assert np.array_equal(gpu.frame(1.0), cpu.frame(1.0))
    for t in (0.3, 0.6):
        diff = np.abs(gpu.frame(t).astype(int) - cpu.frame(t).astype(int)).mean()
        assert diff < 3.0, f"{name} at t={t}: mean difference {diff:.2f}"
