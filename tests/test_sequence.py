import numpy as np
import pytest

from pixelshift.animate import AnimationConfig, Animator
from pixelshift.pipeline import solve_images
from pixelshift.render import SoftRenderer
from pixelshift.sequence import (
    chain_frames,
    crossfade,
    frame_times,
    loop_frames,
    release_renderer,
    upscale,
)
from pixelshift.solvers import MedianSplitSolver


class FakeRenderer:
    """Frames are a flat image whose value encodes t, so sequences are easy to check."""

    size = 4

    def __init__(self):
        self.released = False

    def frame(self, t):
        return np.full((4, 4, 3), round(t * 200), dtype=np.uint8)

    def release(self):
        self.released = True


def test_frame_times_are_even_and_inclusive():
    assert frame_times(5) == [0.0, 0.25, 0.5, 0.75, 1.0]
    with pytest.raises(ValueError):
        frame_times(1)


def test_loop_goes_there_holds_and_comes_back_to_the_start():
    frames = [int(f[0, 0, 0]) for f in loop_frames(FakeRenderer(), 5, hold_frames=2)]
    assert frames == [0, 50, 100, 150, 200, 200, 200, 150, 100, 50, 0]
    assert frames[0] == frames[-1]  # seamless: it ends exactly where it began


def test_loop_without_hold():
    frames = [int(f[0, 0, 0]) for f in loop_frames(FakeRenderer(), 3)]
    assert frames == [0, 100, 200, 100, 0]


def test_upscale_and_crossfade():
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert upscale(img, 6).shape == (6, 6, 3)
    assert np.array_equal(upscale(img, 4)[:2, :2], np.broadcast_to(img[0, 0], (2, 2, 3)))
    with pytest.raises(ValueError):
        upscale(img, 5)
    a, b = np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2, 3), 200, dtype=np.uint8)
    fade = list(crossfade(a, b, 4))
    assert len(fade) == 4 and np.array_equal(fade[-1], b)
    values = [int(f[0, 0, 0]) for f in fade]
    assert values == sorted(values) and 0 < values[0] < 200


def test_crossfade_with_zero_steps_is_empty():
    z = np.zeros((2, 2, 3), dtype=np.uint8)
    assert list(crossfade(z, z, 0)) == []


def tiny_images(n=8, count=3):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (n, n, 3), dtype=np.uint8) for _ in range(count)]


def build(images, **kw):
    cfg = AnimationConfig(duration=0.4, fps=10, swirl=1.0, delay_wave=0.5, delay_strength=0.5)
    made = []

    def make_renderer(mapping):
        r = SoftRenderer(Animator(mapping, cfg), scale=2)
        made.append(r)
        return r

    solves = []

    def solve(src, tgt):
        solves.append((src, tgt))
        return solve_images(src, tgt, MedianSplitSolver())

    args = dict(frame_count=cfg.frame_count, hold_frames=2, settle_frames=3)
    frames = list(chain_frames(images, solve, make_renderer, **{**args, **kw}))
    return frames, solves, made, cfg


def test_chain_has_the_expected_frame_count_and_starts_and_ends_on_the_real_pictures():
    imgs = tiny_images()
    frames, solves, _, cfg = build(imgs)
    per_morph = cfg.frame_count + 2 + 3 + 2  # morph, hold, settle, hold
    assert len(frames) == 2 + 2 * per_morph  # leading hold + two morphs
    assert np.array_equal(frames[0], upscale(imgs[0], 16))
    assert np.array_equal(frames[-1], upscale(imgs[-1], 16))
    assert [np.array_equal(s, i) for (s, _), i in zip(solves, imgs[:2], strict=True)] == [True] * 2


def test_chain_has_no_jump_between_morphs():
    imgs = tiny_images()
    frames, _, _, cfg = build(imgs)
    settled = 2 + (cfg.frame_count + 2 + 3 + 2) - 1  # index of the last frame of morph one
    # the settled picture is exactly the real second image ...
    assert np.array_equal(frames[settled], upscale(imgs[1], 16))
    # ... and the next morph starts from exactly that picture
    assert np.array_equal(frames[settled + 1], upscale(imgs[1], 16))


def test_chain_can_close_the_loop_back_to_the_first_picture():
    imgs = tiny_images()
    frames, solves, _, _ = build(imgs, close_loop=True)
    assert len(solves) == 3
    assert np.array_equal(solves[-1][1], imgs[0])
    assert np.array_equal(frames[-1], upscale(imgs[0], 16))


def test_chain_without_settling_skips_the_crossfade():
    frames, _, _, cfg = build(tiny_images(count=2), settle_frames=0)
    assert len(frames) == 2 + cfg.frame_count + 2


def test_chain_releases_every_renderer_and_validates_input():
    renderers = []

    def make_renderer(mapping):
        renderers.append(FakeRenderer())
        return renderers[-1]

    imgs = tiny_images(count=3)
    list(
        chain_frames(
            imgs,
            lambda s, t: solve_images(s, t, MedianSplitSolver()),
            make_renderer,
            frame_count=3,
            hold_frames=0,
            settle_frames=0,
        )
    )
    assert len(renderers) == 2 and all(r.released for r in renderers)
    with pytest.raises(ValueError):
        list(chain_frames(imgs[:1], None, None, frame_count=3, hold_frames=0, settle_frames=0))


def test_release_renderer_ignores_renderers_without_release():
    release_renderer(object())  # no error
