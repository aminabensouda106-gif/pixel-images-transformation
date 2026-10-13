import numpy as np
import pytest
from PIL import Image

from pixelshift.animate import EASINGS, AnimationConfig, Animator
from pixelshift.export import save_gif
from pixelshift.pipeline import solve_images
from pixelshift.render import CpuRenderer
from pixelshift.solvers import ExactSolver

PALETTE = np.array([[200, 30, 30], [30, 200, 30], [30, 30, 200], [220, 220, 40]], dtype=np.uint8)


def few_color_image(seed: int, n: int):
    idx = np.random.default_rng(seed).integers(0, len(PALETTE), (n, n))
    return PALETTE[idx]


@pytest.fixture(scope="module")
def mapping():
    return solve_images(few_color_image(1, 8), few_color_image(2, 8), ExactSolver())


def multiset(img):
    return sorted(map(tuple, img.reshape(-1, 3).tolist()))


@pytest.mark.parametrize("easing", sorted(EASINGS))
def test_easings_fix_endpoints_and_are_monotonic(easing):
    u = np.linspace(0, 1, 101)
    e = EASINGS[easing](u)
    assert e[0] == pytest.approx(0) and e[-1] == pytest.approx(1)
    assert np.all(np.diff(e) >= -1e-12)


@pytest.mark.parametrize("wobble", [0.0, 1.5])
def test_endpoints_are_exact_grid_positions(mapping, wobble):
    anim = Animator(mapping, AnimationConfig(wobble=wobble))
    assert np.array_equal(anim.positions(0.0), mapping.start_positions())
    assert np.array_equal(anim.positions(1.0), mapping.end_positions())


def test_positions_are_deterministic_and_seeded(mapping):
    a = Animator(mapping, AnimationConfig(wobble=1.0, seed=3)).positions(0.4)
    b = Animator(mapping, AnimationConfig(wobble=1.0, seed=3)).positions(0.4)
    c = Animator(mapping, AnimationConfig(wobble=1.0, seed=4)).positions(0.4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_pixels_are_in_flight_midway_and_delays_spread_departures(mapping):
    anim = Animator(mapping, AnimationConfig(delay_strength=0.5))
    mid = anim.positions(0.5)
    moved = np.linalg.norm(mid - mapping.start_positions(), axis=1) > 0
    assert 0 < moved.sum() <= moved.size
    flat = Animator(mapping, AnimationConfig(delay_strength=0.0, delay_jitter=0.0))
    assert np.allclose(flat.delay, 0)


def test_invalid_config_and_time(mapping):
    with pytest.raises(ValueError):
        AnimationConfig(easing="bounce")
    with pytest.raises(ValueError):
        AnimationConfig(delay_strength=1.0)
    with pytest.raises(ValueError):
        Animator(mapping).positions(1.5)


def test_first_and_last_frames_are_exactly_source_and_result(mapping):
    anim = Animator(mapping, AnimationConfig(wobble=1.0))
    r = CpuRenderer(anim, scale=3)
    src = mapping.source_rgb.reshape(8, 8, 3)
    up = lambda img: np.kron(img, np.ones((3, 3, 1), dtype=np.uint8))  # noqa: E731
    assert np.array_equal(r.frame(0.0), up(src))
    assert np.array_equal(r.frame(1.0), up(mapping.result_image()))
    assert multiset(r.frame(1.0)) == multiset(up(src))  # only source colors, same counts


def test_middle_frame_has_expected_shape_and_background_gaps(mapping):
    r = CpuRenderer(Animator(mapping, AnimationConfig(delay_strength=0.6)), scale=3)
    f = r.frame(0.5)
    assert f.shape == (24, 24, 3) and f.dtype == np.uint8
    assert (f == r.background).all(axis=2).any()  # pixels in flight leave gaps


def test_frames_count_matches_config(mapping):
    cfg = AnimationConfig(duration=1.0, fps=10)
    frames = list(CpuRenderer(Animator(mapping, cfg), scale=2).frames())
    assert len(frames) == cfg.frame_count == 11


def test_gif_last_frame_is_the_rearranged_image(mapping, tmp_path):
    r = CpuRenderer(Animator(mapping, AnimationConfig(duration=0.5, fps=10)), scale=3)
    path = tmp_path / "out.gif"
    count = save_gif(path, r.frames(), fps=10)
    with Image.open(path) as gif:
        assert 2 <= gif.n_frames <= count  # Pillow merges identical consecutive frames
        gif.seek(gif.n_frames - 1)
        last = np.asarray(gif.convert("RGB"))
    up = np.ones((3, 3, 1), dtype=np.uint8)
    assert np.array_equal(last, np.kron(mapping.result_image(), up))
    assert multiset(last) == multiset(np.kron(mapping.source_rgb.reshape(8, 8, 3), up))


def test_save_gif_rejects_empty():
    with pytest.raises(ValueError):
        save_gif("x.gif", [])


# --- flow style: wave timing, swirl, soft renderer ---------------------------------------------


def test_swirl_keeps_exact_endpoints_and_bends_paths(mapping):
    anim = Animator(mapping, AnimationConfig(swirl=1.2, delay_wave=0.5))
    assert np.array_equal(anim.positions(0.0), mapping.start_positions())
    assert np.array_equal(anim.positions(1.0), mapping.end_positions())
    straight = Animator(mapping, AnimationConfig(swirl=0.0, delay_wave=0.5))
    assert not np.allclose(anim.positions(0.5), straight.positions(0.5))


def test_wave_delay_is_a_smooth_sweep_across_the_image():
    n = 10
    src = few_color_image(1, n)
    m = solve_images(src, src, ExactSolver(), lam=1.0)  # identity: pixels stay put
    cfg = AnimationConfig(delay_strength=0.5, delay_jitter=0.0, delay_wave=1.0)
    delay = Animator(m, cfg).delay.reshape(n, n)
    assert delay[0, 0] == pytest.approx(0.0) and delay[-1, -1] == pytest.approx(0.5)
    assert np.all(np.diff(delay, axis=1) > 0) and np.all(np.diff(delay, axis=0) > 0)


def test_wave_and_jitter_shares_must_fit():
    with pytest.raises(ValueError):
        AnimationConfig(delay_wave=0.7, delay_jitter=0.6)
    with pytest.raises(ValueError):
        AnimationConfig(swirl_falloff=2.0)


def make_soft(mapping, **kw):
    from pixelshift.render import SoftRenderer

    cfg = AnimationConfig(duration=1.0, fps=10, swirl=1.0, delay_wave=0.5, delay_strength=0.5)
    return SoftRenderer(Animator(mapping, cfg), scale=3, **kw)


def test_soft_renderer_first_and_last_frames_are_exact(mapping):
    r = make_soft(mapping)
    up = np.ones((3, 3, 1), dtype=np.uint8)
    assert np.array_equal(r.frame(0.0), np.kron(mapping.source_rgb.reshape(8, 8, 3), up))
    assert np.array_equal(r.frame(1.0), np.kron(mapping.result_image(), up))


def test_soft_renderer_middle_frames_are_valid_and_deterministic(mapping):
    r = make_soft(mapping)
    f = r.frame(0.5)
    assert f.shape == (24, 24, 3) and f.dtype == np.uint8
    assert np.array_equal(f, make_soft(mapping).frame(0.5))
    assert not np.array_equal(f, r.crisp.frame(0.5))  # actually soft, not the pixel render


def test_soft_renderer_color_stays_inside_the_palette_hull(mapping):
    f = make_soft(mapping).frame(0.5).astype(int)
    lo = np.minimum(PALETTE.min(axis=0), np.array([12, 12, 16]))
    hi = np.maximum(PALETTE.max(axis=0), np.array([12, 12, 16]))
    assert (f >= lo - 1).all() and (f <= hi + 1).all()


def test_soft_renderer_frames_count_and_no_blur_option(mapping):
    r = make_soft(mapping, motion_blur=0.0, samples=1)
    assert len(list(r.frames())) == 11
    with pytest.raises(ValueError):
        make_soft(mapping, softness=0.0)
    with pytest.raises(ValueError):
        make_soft(mapping, edge=0.9)


def test_soft_frames_can_be_written_to_gif(mapping, tmp_path):
    r = make_soft(mapping)
    path = tmp_path / "soft.gif"
    save_gif(path, r.frames(), fps=10)
    with Image.open(path) as gif:
        gif.seek(gif.n_frames - 1)
        last = np.asarray(gif.convert("RGB"))
    assert np.array_equal(last, np.kron(mapping.result_image(), np.ones((3, 3, 1), dtype=np.uint8)))
