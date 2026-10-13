import time

import pytest

from pixelshift.live import HOLD_SECONDS, MODES, Playback, auto_scale


def run_until(play: Playback, predicate, step=0.05, limit=200.0):
    elapsed = 0.0
    while not predicate() and elapsed < limit:
        play.update(step)
        elapsed += step
    return elapsed


def test_playback_holds_at_start_then_advances():
    p = Playback(duration=2.0)
    p.update(HOLD_SECONDS / 2)
    assert p.t == 0.0
    run_until(p, lambda: p.t > 0.0)
    assert 0.0 < p.t < 0.1


def test_speed_scales_playback():
    slow, fast = Playback(duration=4.0), Playback(duration=4.0)
    fast.speed = 2.0
    for p in (slow, fast):
        p._hold = 0.0
        p.update(1.0)
    assert fast.t == pytest.approx(2 * slow.t) and slow.t == pytest.approx(0.25)


def test_pingpong_reverses_after_a_hold_at_each_end():
    p = Playback(duration=1.0, mode="pingpong")
    run_until(p, lambda: p.t >= 1.0)
    assert p.t == 1.0
    run_until(p, lambda: p.direction == -1)
    assert p.direction == -1 and p.t == 1.0
    run_until(p, lambda: p.t <= 0.0)
    assert p.t == 0.0


def test_loop_wraps_to_the_start():
    p = Playback(duration=1.0, mode="loop")
    run_until(p, lambda: p.t >= 1.0)
    run_until(p, lambda: p.t < 0.5)
    assert p.t == 0.0 and p.direction == 1


def test_once_stops_at_the_end():
    p = Playback(duration=1.0, mode="once")
    run_until(p, lambda: p.paused)
    assert p.paused and p.t == 1.0
    p.update(5.0)
    assert p.t == 1.0


def test_pause_scrub_reverse_restart_speed_and_mode():
    p = Playback(duration=2.0)
    p._hold = 0.0
    p.toggle_pause()
    p.update(1.0)
    assert p.t == 0.0
    p.scrub(0.4)
    assert p.t == pytest.approx(0.4) and p.paused
    p.scrub(5.0)
    assert p.t == 1.0
    p.scrub(-5.0)
    assert p.t == 0.0
    p.restart()
    assert not p.paused and p.t == 0.0
    p.reverse()
    assert p.direction == -1
    for _ in range(40):
        p.faster()
    assert p.speed == 8.0
    for _ in range(80):
        p.slower()
    assert p.speed == pytest.approx(0.1)
    seen = {p.cycle_mode() for _ in MODES}
    assert seen == set(MODES)


def test_invalid_playback_arguments():
    with pytest.raises(ValueError):
        Playback(duration=0)
    with pytest.raises(ValueError):
        Playback(duration=1.0, mode="bounce")


def test_auto_scale_targets_the_window_size():
    assert auto_scale(128) == 6 and auto_scale(512) == 2 and auto_scale(2000) == 1


def test_run_live_smoke_with_a_hidden_window():
    import numpy as np

    from pixelshift.animate import AnimationConfig
    from pixelshift.live import run_live
    from pixelshift.pipeline import solve_images
    from pixelshift.render.gpu import GpuUnavailableError
    from pixelshift.solvers import MedianSplitSolver

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    m = solve_images(a, b, MedianSplitSolver())
    try:
        stats = run_live(
            m,
            AnimationConfig(duration=1.0),
            max_frames=5,
            vsync=False,
            visible=False,
            window_size=200,
        )
    except GpuUnavailableError as exc:
        pytest.skip(f"no window/OpenGL available: {exc}")
    assert stats.frames == 5 and stats.fps > 0


def test_a_real_drop_event_swaps_in_a_new_animation(tmp_path):
    """Drive the real window: simulate a file drop and check the new pair is loaded."""
    import threading

    import numpy as np
    from PIL import Image

    from pixelshift.animate import AnimationConfig
    from pixelshift.live import run_live
    from pixelshift.pipeline import solve_images
    from pixelshift.render.gpu import GpuUnavailableError
    from pixelshift.solvers import MedianSplitSolver

    try:
        import glfw
    except ImportError:
        pytest.skip("glfw not installed")

    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    dropped = tmp_path / "dropped.png"
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(dropped)

    solved: list[tuple[np.ndarray, np.ndarray]] = []

    def solve(src, tgt):
        solved.append((src, tgt))
        return solve_images(src, tgt, MedianSplitSolver())

    m = solve_images(a, b, MedianSplitSolver())

    # glfw has no public API to inject a drop, so call the registered callback ourselves from a
    # helper that runs once the window loop is up.
    real_set = glfw.set_drop_callback

    def capture(window, callback):
        def fire():
            time.sleep(0.2)
            callback(window, [str(dropped)])

        threading.Thread(target=fire, daemon=True).start()
        return real_set(window, callback)

    glfw.set_drop_callback = capture
    try:
        stats = run_live(
            m,
            AnimationConfig(duration=1.0),
            max_frames=400,
            vsync=True,
            visible=False,
            window_size=200,
            solve=solve,
            target_image=b,
        )
    except GpuUnavailableError as exc:
        pytest.skip(f"no window/OpenGL available: {exc}")
    finally:
        glfw.set_drop_callback = real_set
    assert stats.frames == 400
    assert len(solved) == 1
    assert np.array_equal(solved[0][0], b)  # the old target became the new source
