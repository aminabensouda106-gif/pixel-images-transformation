"""Frame sequences built from renderers: seamless loops and photo chains.

* :func:`loop_frames` plays A to B and back to A, so a GIF or video loops with no jump.
* :func:`chain_frames` morphs through a list of pictures in order (A to B to C ...). After each
  morph the picture has the previous picture's pixels, so a short cross-fade restores the real
  picture before the next morph starts from it. Both ends of every morph are exact, so the
  sequence has no visual jumps.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pixelshift.mapping import Mapping

Image = NDArray[np.uint8]


class FrameRenderer(Protocol):
    size: int

    def frame(self, t: float) -> Image: ...


def frame_times(count: int) -> list[float]:
    """``count`` evenly spaced times from 0 to 1 inclusive."""
    if count < 2:
        raise ValueError("count must be at least 2")
    return [k / (count - 1) for k in range(count)]


def loop_frames(renderer: FrameRenderer, count: int, *, hold_frames: int = 0) -> Iterator[Image]:
    """A to B, a hold on B, then B back to A. Starts and ends on A, so it loops seamlessly."""
    times = frame_times(count)
    for t in times:
        yield renderer.frame(t)
    last = renderer.frame(1.0)
    for _ in range(hold_frames):
        yield last
    for t in reversed(times[:-1]):
        yield renderer.frame(t)


def upscale(img: Image, size: int) -> Image:
    """Nearest-neighbor upscale of an (n, n, 3) image to (size, size, 3)."""
    n = img.shape[0]
    if size % n:
        raise ValueError("size must be a multiple of the image side")
    return np.kron(img, np.ones((size // n, size // n, 1), dtype=np.uint8))


def crossfade(a: Image, b: Image, steps: int) -> Iterator[Image]:
    """``steps`` frames fading from ``a`` (exclusive) to ``b`` (inclusive)."""
    for k in range(1, steps + 1):
        w = k / steps
        w = w * w * (3 - 2 * w)
        yield np.clip(np.rint((1 - w) * a + w * b), 0, 255).astype(np.uint8)


def chain_frames(
    images: Sequence[Image],
    solve: Callable[[Image, Image], Mapping],
    make_renderer: Callable[[Mapping], FrameRenderer],
    *,
    frame_count: int,
    hold_frames: int,
    settle_frames: int,
    close_loop: bool = False,
) -> Iterator[Image]:
    """Frames that morph through ``images`` in order (and back to the first if ``close_loop``)."""
    if len(images) < 2:
        raise ValueError("a chain needs at least two images")
    pairs = list(zip(images[:-1], images[1:], strict=True))
    if close_loop:
        pairs.append((images[-1], images[0]))

    for index, (source, target) in enumerate(pairs):
        mapping = solve(source, target)
        renderer = make_renderer(mapping)
        try:
            if index == 0:
                start = renderer.frame(0.0)
                for _ in range(hold_frames):
                    yield start
            for t in frame_times(frame_count):
                yield renderer.frame(t)
            end = renderer.frame(1.0)
            for _ in range(hold_frames):
                yield end
            if settle_frames > 0:
                truth = upscale(target, renderer.size)
                yield from crossfade(end, truth, settle_frames)
                for _ in range(hold_frames):
                    yield truth
        finally:
            release_renderer(renderer)


def release_renderer(renderer: Any) -> None:
    """Free GPU resources if the renderer holds any."""
    release = getattr(renderer, "release", None)
    if callable(release):
        release()
