"""Live window: watch the transformation in real time and control playback.

Keys: SPACE pause/play, R restart, B reverse direction, M cycle loop mode (ping-pong / loop /
once), LEFT / RIGHT scrub, UP / DOWN speed, S save a screenshot, F fullscreen, ESC or Q quit.

The playback logic lives in :class:`Playback` (plain Python, no graphics) so it can be tested
without a window. :func:`run_live` needs the optional GPU dependencies:
``pip install -e ".[gpu]"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pixelshift.animate import AnimationConfig, Animator
from pixelshift.mapping import Mapping
from pixelshift.render.gpu import GpuRenderer, GpuUnavailableError

MODES = ("pingpong", "loop", "once")
HOLD_SECONDS = 0.8  # pause at each end so the viewer can look at the picture


class Playback:
    """Normalized animation time ``t`` in [0, 1] driven by wall-clock steps."""

    def __init__(self, duration: float, *, mode: str = "pingpong") -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.duration = duration
        self.mode = mode
        self.t = 0.0
        self.direction = 1
        self.speed = 1.0
        self.paused = False
        self._hold = HOLD_SECONDS
        self._pending: str | None = None

    def update(self, dt: float) -> float:
        """Advance by ``dt`` real seconds and return the new ``t``."""
        if self.paused or dt <= 0:
            return self.t
        if self._hold > 0:
            self._hold -= dt
            if self._hold <= 0:
                self._apply_pending()
            return self.t
        self.t += self.direction * self.speed * dt / self.duration
        if self.t >= 1.0 or self.t <= 0.0:
            self.t = min(max(self.t, 0.0), 1.0)
            self._hold = HOLD_SECONDS
            self._pending = self.mode
        return self.t

    def _apply_pending(self) -> None:
        if self._pending == "pingpong":
            self.direction = -self.direction
        elif self._pending == "loop":
            self.t = 0.0 if self.direction > 0 else 1.0
        elif self._pending == "once":
            self.paused = True
        self._pending = None

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def restart(self) -> None:
        self.t, self.direction, self.paused = 0.0, 1, False
        self._hold, self._pending = HOLD_SECONDS, None

    def reverse(self) -> None:
        self.direction = -self.direction
        self._hold, self._pending = 0.0, None

    def cycle_mode(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    def scrub(self, delta: float) -> None:
        """Move by ``delta`` of the full animation and pause, for frame-by-frame inspection."""
        self.t = min(max(self.t + delta, 0.0), 1.0)
        self.paused = True
        self._hold, self._pending = 0.0, None

    def faster(self) -> None:
        self.speed = min(self.speed * 1.25, 8.0)

    def slower(self) -> None:
        self.speed = max(self.speed / 1.25, 0.1)


@dataclass
class LiveStats:
    frames: int = 0
    seconds: float = 0.0
    screenshots: list[Path] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def auto_scale(n: int, target_px: int = 768) -> int:
    """Pick an integer output scale so the picture is about ``target_px`` wide."""
    return max(1, round(target_px / n))


def run_live(
    mapping: Mapping,
    config: AnimationConfig,
    *,
    scale: int | None = None,
    window_size: int = 768,
    vsync: bool = True,
    fullscreen: bool = False,
    mode: str = "pingpong",
    max_frames: int | None = None,
    visible: bool = True,
    title: str = "pixelshift",
) -> LiveStats:
    """Open a window and play the transformation until it is closed (or ``max_frames``)."""
    try:
        import glfw
        import moderngl
    except ImportError as exc:
        raise GpuUnavailableError(
            'the live window needs extra packages; run: pip install -e ".[gpu]"'
        ) from exc

    if not glfw.init():
        raise GpuUnavailableError("could not initialize GLFW (no display available?)")
    stats = LiveStats()
    window = None
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, visible)
        window = glfw.create_window(window_size, window_size, title, None, None)
        if not window:
            raise GpuUnavailableError("could not create an OpenGL 3.3 window")
        glfw.make_context_current(window)
        glfw.swap_interval(1 if vsync else 0)

        ctx = moderngl.create_context()
        renderer = GpuRenderer(
            Animator(mapping, config),
            scale=scale or auto_scale(mapping.n, window_size),
            context=ctx,
        )
        play = Playback(config.duration, mode=mode)
        windowed_box = glfw.get_window_pos(window) + glfw.get_window_size(window)

        def on_key(win: Any, key: int, _scan: int, action: int, _mods: int) -> None:
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE and action == glfw.PRESS:
                play.toggle_pause()
            elif key == glfw.KEY_R and action == glfw.PRESS:
                play.restart()
            elif key == glfw.KEY_B and action == glfw.PRESS:
                play.reverse()
            elif key == glfw.KEY_M and action == glfw.PRESS:
                play.cycle_mode()
            elif key == glfw.KEY_LEFT:
                play.scrub(-0.01)
            elif key == glfw.KEY_RIGHT:
                play.scrub(0.01)
            elif key == glfw.KEY_UP and action == glfw.PRESS:
                play.faster()
            elif key == glfw.KEY_DOWN and action == glfw.PRESS:
                play.slower()
            elif key == glfw.KEY_S and action == glfw.PRESS:
                path = Path(f"pixelshift_{int(time.time())}.png")
                from pixelshift.io import save_image

                save_image(path, renderer.frame(play.t))
                stats.screenshots.append(path)
                print(f"saved {path}")
            elif key == glfw.KEY_F and action == glfw.PRESS:
                if glfw.get_window_monitor(win):
                    x, y, w, h = windowed_box
                    glfw.set_window_monitor(win, None, x, y, w, h, 0)  # type: ignore[arg-type]
                else:
                    monitor = glfw.get_primary_monitor()
                    mode_ = glfw.get_video_mode(monitor)
                    glfw.set_window_monitor(
                        win, monitor, 0, 0, mode_.size.width, mode_.size.height, mode_.refresh_rate
                    )

        glfw.set_key_callback(window, on_key)
        if fullscreen:
            on_key(window, glfw.KEY_F, 0, glfw.PRESS, 0)

        start = last = last_title = time.perf_counter()
        since_title = 0
        while not glfw.window_should_close(window):
            glfw.poll_events()
            now = time.perf_counter()
            t = play.update(now - last)
            last = now
            renderer.render(t)
            width, height = glfw.get_framebuffer_size(window)
            renderer.blit(ctx.screen, max(width, 1), max(height, 1))
            glfw.swap_buffers(window)
            stats.frames += 1
            since_title += 1
            if now - last_title >= 0.5:
                fps = since_title / (now - last_title)
                state = "paused" if play.paused else f"x{play.speed:.2f}"
                glfw.set_window_title(
                    window, f"{title} | t={t:.2f} | {fps:.0f} fps | {state} | {play.mode}"
                )
                last_title, since_title = now, 0
            if max_frames is not None and stats.frames >= max_frames:
                break
        ctx.finish()
        stats.seconds = time.perf_counter() - start
        return stats
    finally:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()
