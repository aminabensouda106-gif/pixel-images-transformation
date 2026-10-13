"""Writing images and animations to disk."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def save_mp4(
    path: str | Path,
    frames: Iterable[NDArray[np.uint8]],
    *,
    fps: int = 30,
    hold_start: float = 0.5,
    hold_end: float = 1.5,
    quality: int = 8,
) -> int:
    """Write frames as an H.264 MP4 (yuv420p, plays everywhere) and return the frame count.

    Frames are streamed, so memory stays flat. Odd frame sizes are padded by one pixel because
    yuv420p needs even dimensions. ``quality`` runs from 0 (worst) to 10 (best).
    """
    import imageio.v2 as imageio

    written = 0
    last: NDArray[np.uint8] | None = None
    video: Any = imageio.get_writer(  # imageio's stubs don't expose append_data
        str(path),
        fps=fps,
        codec="libx264",
        quality=quality,
        pixelformat="yuv420p",
        macro_block_size=2,
    )
    with video as writer:
        for frame in frames:
            frame = _even(frame)
            if last is None:
                for _ in range(round(hold_start * fps)):
                    writer.append_data(frame)
                    written += 1
            writer.append_data(frame)
            written += 1
            last = frame
        if last is None:
            raise ValueError("no frames to write")
        for _ in range(round(hold_end * fps)):
            writer.append_data(last)
            written += 1
    return written


def _even(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    h, w = frame.shape[:2]
    return np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")


def save_gif(
    path: str | Path,
    frames: Iterable[NDArray[np.uint8]],
    *,
    fps: int = 25,
    hold_start: float = 0.5,
    hold_end: float = 1.5,
    loop: int = 0,
) -> int:
    """Write frames as an animated GIF and return the number of frames written.

    ``hold_start`` / ``hold_end`` (seconds) freeze the first and last frame. GIF stores colors in a
    256-entry palette per frame, so photographic frames are quantized (a limit of the format).
    """
    images = [Image.fromarray(f, mode="RGB") for f in frames]
    if not images:
        raise ValueError("no frames to write")
    frame_ms = round(1000 / fps)
    durations = [frame_ms] * len(images)
    durations[0] += round(hold_start * 1000)
    durations[-1] += round(hold_end * 1000)
    palettized = [im.quantize(colors=256, dither=Image.Dither.NONE) for im in images]
    palettized[0].save(
        path,
        save_all=True,
        append_images=palettized[1:],
        duration=durations,
        loop=loop,
        optimize=False,
        disposal=1,
    )
    return len(images)
