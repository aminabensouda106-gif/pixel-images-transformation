"""Drag-and-drop handling for the live window, independent of any graphics code.

Dropping **two** files makes the first the source and the second the target. Dropping **one** file
makes it the new target and the current target the new source, so dropping photo after photo
morphs through them in order.

Solving takes a second or more, so it runs on a worker thread. The window keeps playing and calls
:meth:`DropHandler.poll` every frame to pick up a finished result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pixelshift.io import load_square
from pixelshift.mapping import Mapping

Image = NDArray[np.uint8]
SolveFn = Callable[[Image, Image], Mapping]


@dataclass(frozen=True)
class DropResult:
    """Outcome of one drop: a new mapping and target image, or an error message."""

    mapping: Mapping | None
    target: Image | None
    message: str

    @property
    def ok(self) -> bool:
        return self.mapping is not None


class DropHandler:
    def __init__(self, size: int, solve: SolveFn, target: Image) -> None:
        self.size = size
        self._solve = solve
        self._target = target
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pixelshift-drop")
        self._job: Future[DropResult] | None = None

    @property
    def busy(self) -> bool:
        return self._job is not None

    def submit(self, paths: Sequence[str]) -> str:
        """Start solving for the dropped files. Returns a short status message."""
        if self._job is not None:
            return "still solving the previous drop, please wait"
        if not paths:
            return "nothing was dropped"
        if len(paths) >= 2:
            source_path: str | None = paths[0]
            target_path = paths[1]
            note = " (extra files ignored)" if len(paths) > 2 else ""
        else:
            source_path, target_path, note = None, paths[0], ""
        self._job = self._pool.submit(self._work, source_path, target_path)
        return f"solving{note} ..."

    def _work(self, source_path: str | None, target_path: str) -> DropResult:
        try:
            source = self._target if source_path is None else load_square(source_path, self.size)
            target = load_square(target_path, self.size)
        except OSError as exc:
            return DropResult(None, None, f"cannot read image: {exc}")
        try:
            mapping = self._solve(source, target)
        except Exception as exc:  # a failed solve must never kill the window
            return DropResult(None, None, f"solve failed: {exc}")
        return DropResult(mapping, target, "done")

    def poll(self) -> DropResult | None:
        """Return the finished result (once), or ``None`` if still running or nothing pending."""
        job = self._job
        if job is None or not job.done():
            return None
        self._job = None
        result = job.result()
        if result.target is not None:
            self._target = result.target
        return result

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
