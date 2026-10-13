"""Renderers turn animator positions into frames."""

from __future__ import annotations

from pixelshift.render.base import Renderer
from pixelshift.render.cpu import CpuRenderer

__all__ = ["CpuRenderer", "Renderer"]
