import time

import numpy as np
import pytest
from PIL import Image

from pixelshift.drop import DropHandler
from pixelshift.pipeline import solve_images
from pixelshift.solvers import MedianSplitSolver

N = 8


def make_image(tmp_path, name, seed):
    arr = np.random.default_rng(seed).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(arr).save(path)
    return str(path)


def solve(src, tgt):
    return solve_images(src, tgt, MedianSplitSolver())


def wait(handler, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        result = handler.poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("drop job did not finish")


@pytest.fixture
def handler():
    h = DropHandler(N, solve, np.zeros((N, N, 3), dtype=np.uint8))
    yield h
    h.close()


def test_two_files_become_source_and_target(handler, tmp_path):
    a, b = make_image(tmp_path, "a.png", 1), make_image(tmp_path, "b.png", 2)
    assert "solving" in handler.submit([a, b])
    result = wait(handler)
    assert result.ok and result.mapping.n == N and result.target.shape == (N, N, 3)
    assert not handler.busy
    from pixelshift.io import load_square

    assert np.array_equal(result.mapping.source_rgb.reshape(N, N, 3), load_square(a, N))
    assert np.array_equal(result.target, load_square(b, N))


def test_one_file_chains_from_the_previous_target(handler, tmp_path):
    a, b, c = (make_image(tmp_path, f"{x}.png", s) for x, s in (("a", 1), ("b", 2), ("c", 3)))
    handler.submit([a, b])
    first = wait(handler)
    handler.submit([c])
    second = wait(handler)
    # the previous target (b) is the new source
    assert np.array_equal(second.mapping.source_rgb.reshape(N, N, 3), first.target)
    from pixelshift.io import load_square

    assert np.array_equal(second.target, load_square(c, N))


def test_first_single_drop_uses_the_initial_target_as_source(tmp_path):
    initial = np.random.default_rng(9).integers(0, 256, (N, N, 3), dtype=np.uint8)
    h = DropHandler(N, solve, initial)
    try:
        h.submit([make_image(tmp_path, "x.png", 4)])
        result = wait(h)
        assert np.array_equal(result.mapping.source_rgb.reshape(N, N, 3), initial)
    finally:
        h.close()


def test_extra_files_are_ignored_with_a_note(handler, tmp_path):
    files = [make_image(tmp_path, f"{i}.png", i) for i in range(3)]
    assert "extra files ignored" in handler.submit(files)
    assert wait(handler).ok


def test_busy_handler_rejects_a_second_drop(tmp_path):
    release = []

    def slow(src, tgt):
        while not release:
            time.sleep(0.01)
        return solve(src, tgt)

    h = DropHandler(N, slow, np.zeros((N, N, 3), dtype=np.uint8))
    try:
        a, b = make_image(tmp_path, "a.png", 1), make_image(tmp_path, "b.png", 2)
        h.submit([a, b])
        assert h.busy and h.poll() is None
        assert "still solving" in h.submit([a])
        release.append(True)
        assert wait(h).ok
    finally:
        release.append(True)
        h.close()


def test_unreadable_file_reports_an_error_and_keeps_the_old_target(handler, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")
    handler.submit([str(bad)])
    result = wait(handler)
    assert not result.ok and "cannot read image" in result.message
    assert not handler.busy
    good = make_image(tmp_path, "g.png", 5)
    handler.submit([good])  # still works after a failure
    assert wait(handler).ok


def test_solver_failure_does_not_crash_the_handler(tmp_path):
    def boom(src, tgt):
        raise RuntimeError("kaput")

    h = DropHandler(N, boom, np.zeros((N, N, 3), dtype=np.uint8))
    try:
        h.submit([make_image(tmp_path, "a.png", 1)])
        result = wait(h)
        assert not result.ok and "kaput" in result.message
    finally:
        h.close()


def test_empty_drop_and_idle_poll(handler):
    assert "nothing" in handler.submit([])
    assert handler.poll() is None and not handler.busy
