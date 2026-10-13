"""Renderer interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Renderer(Protocol):
    def frame(self, t: float) -> NDArray[np.uint8]:
        """Return an (H, W, 3) uint8 frame for normalized time ``t`` in [0, 1]."""
        ...
