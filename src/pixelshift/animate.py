"""Animation engine: where is every pixel at normalized time ``t``?

Pixel ``i`` (the one assigned to target slot ``i``) travels from ``start[i]`` to ``end[i]``::

    u = clip((t - delay_i) / (1 - delay_strength), 0, 1)
    p(t) = start + (end - start) * ease(u) + wobble

Delays are at most ``delay_strength``, so every pixel has arrived by ``t = 1``. The wobble is scaled
by ``4u(1 - u)``, so it vanishes at both ends and pixels start and finish exactly on the grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pixelshift.mapping import Mapping

Float = NDArray[np.float64]


def _linear(u: Float) -> Float:
    return u


def _smoothstep(u: Float) -> Float:
    return u * u * (3 - 2 * u)


def _smootherstep(u: Float) -> Float:
    return u * u * u * (u * (6 * u - 15) + 10)


def _ease_in_out_cubic(u: Float) -> Float:
    return np.where(u < 0.5, 4 * u**3, 1 - (-2 * u + 2) ** 3 / 2)


EASINGS: dict[str, Callable[[Float], Float]] = {
    "linear": _linear,
    "smoothstep": _smoothstep,
    "smootherstep": _smootherstep,
    "cubic": _ease_in_out_cubic,
}


WAVE_SHAPES = ("diagonal", "horizontal", "vertical", "radial", "radial_in")

# Named looks. Each entry overrides the matching AnimationConfig fields.
TRANSITIONS: dict[str, dict[str, float | str]] = {
    "spiral": {
        "swirl": 1.0,
        "burst": 0.0,
        "wave_shape": "diagonal",
        "delay_wave": 0.6,
        "delay_jitter": 0.1,
        "delay_strength": 0.5,
    },
    "wave": {
        "swirl": 0.0,
        "burst": 0.0,
        "wave_shape": "horizontal",
        "delay_wave": 0.9,
        "delay_jitter": 0.05,
        "delay_strength": 0.6,
    },
    "burst": {
        "swirl": 0.35,
        "burst": 0.7,
        "wave_shape": "radial",
        "delay_wave": 0.3,
        "delay_jitter": 0.3,
        "delay_strength": 0.35,
    },
    "iris": {
        "swirl": 0.0,
        "burst": 0.0,
        "wave_shape": "radial",
        "delay_wave": 0.9,
        "delay_jitter": 0.05,
        "delay_strength": 0.6,
    },
}


def transition_options(name: str) -> dict[str, float | str]:
    """Config overrides for a named transition (``spiral``, ``wave``, ``burst``, ``iris``)."""
    try:
        return dict(TRANSITIONS[name])
    except KeyError:
        raise ValueError(f"unknown transition {name!r}; available: {sorted(TRANSITIONS)}") from None


@dataclass(frozen=True)
class AnimationConfig:
    """Timing and motion parameters.

    delay_strength: fraction of the timeline over which departures are spread (0 = all at once).
    delay_jitter:   share of the delay that is random.
    delay_wave:     share of the delay that is a smooth sweep across the image, so neighboring
                    pixels leave together. The remaining share follows travel distance.
    wave_shape:     direction of that sweep: ``diagonal``, ``horizontal``, ``vertical``,
                    ``radial`` (center leaves first, like an opening iris) or ``radial_in``
                    (edges leave first).
    wobble:         amplitude of a random sideways shake, in grid pixels (0 = none).
    swirl:          peak rotation in radians of the whole flow around the image center. Each
                    pixel's path bends into a spiral; 0 gives straight lines.
    swirl_falloff:  how much weaker the swirl is at the image edge than at the center (0 to 1).
    burst:          how far the flow expands outward from the center mid-flight, as a fraction of
                    each pixel's distance from the center (0 = none). Pixels burst out and come
                    back in.
    """

    duration: float = 3.0
    fps: int = 30
    easing: str = "smoothstep"
    delay_strength: float = 0.4
    delay_jitter: float = 0.5
    delay_wave: float = 0.0
    wave_shape: str = "diagonal"
    wobble: float = 0.0
    swirl: float = 0.0
    swirl_falloff: float = 0.6
    burst: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.easing not in EASINGS:
            raise ValueError(f"unknown easing {self.easing!r}; available: {sorted(EASINGS)}")
        if not 0 <= self.delay_strength < 1:
            raise ValueError("delay_strength must be in [0, 1)")
        if not 0 <= self.delay_jitter <= 1:
            raise ValueError("delay_jitter must be in [0, 1]")
        if not 0 <= self.delay_wave <= 1 or self.delay_jitter + self.delay_wave > 1:
            raise ValueError("delay_wave must be in [0, 1] and delay_jitter + delay_wave <= 1")
        if self.wave_shape not in WAVE_SHAPES:
            raise ValueError(f"unknown wave_shape {self.wave_shape!r}; available: {WAVE_SHAPES}")
        if not 0 <= self.swirl_falloff <= 1:
            raise ValueError("swirl_falloff must be in [0, 1]")
        if not 0 <= self.burst <= 3:
            raise ValueError("burst must be in [0, 3]")
        if self.duration <= 0 or self.fps < 1:
            raise ValueError("duration and fps must be positive")

    @property
    def frame_count(self) -> int:
        return max(2, round(self.duration * self.fps) + 1)


class Animator:
    def __init__(self, mapping: Mapping, config: AnimationConfig | None = None) -> None:
        self.mapping = mapping
        self.config = config or AnimationConfig()
        cfg = self.config
        self.start = mapping.start_positions()
        self.end = mapping.end_positions()
        rng = np.random.default_rng(cfg.seed)
        dist = np.linalg.norm(self.end - self.start, axis=1)
        dist_norm = dist / dist.max() if dist.max() > 0 else np.zeros_like(dist)
        side = max(mapping.n - 1, 1)
        wave = self._wave_field(side)
        dist_share = 1 - cfg.delay_jitter - cfg.delay_wave
        mix = (
            dist_share * dist_norm
            + cfg.delay_wave * wave
            + cfg.delay_jitter * rng.random(dist.size)
        )
        self.delay = cfg.delay_strength * np.clip(mix, 0.0, 1.0)
        self.center = np.full(2, (mapping.n - 1) / 2)
        self.radius = max(mapping.n / np.sqrt(2), 1.0)
        self.phase = rng.random((dist.size, 2))
        self.freq = 1.0 + 2.0 * rng.random(dist.size)
        self._ease = EASINGS[cfg.easing]

    def _wave_field(self, side: int) -> Float:
        """Departure order in [0, 1] for each pixel, from its starting position."""
        x, y = self.start[:, 0], self.start[:, 1]
        shape = self.config.wave_shape
        if shape == "horizontal":
            return x / side
        if shape == "vertical":
            return y / side
        if shape in ("radial", "radial_in"):
            dist: Float = np.hypot(x - side / 2, y - side / 2)
            r: Float = dist / max(float(dist.max()), 1e-9)
            return r if shape == "radial" else 1.0 - r
        return (x + y) / (2 * side)  # diagonal: 0 top-left .. 1 bottom-right

    def positions(self, t: float) -> Float:
        """(P, 2) x, y positions (grid units) of every pixel at normalized time ``t`` in [0, 1]."""
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0, 1]")
        scale = 1.0 - self.config.delay_strength
        u = np.clip((t - self.delay) / scale, 0.0, 1.0)
        pos = self.start + (self.end - self.start) * self._ease(u)[:, None]
        if self.config.wobble > 0:
            angle = 2 * np.pi * (self.freq[:, None] * t + self.phase)
            envelope = 4.0 * u * (1.0 - u)  # exactly 0 at u = 0 and u = 1
            pos = pos + (self.config.wobble * envelope)[:, None] * np.sin(angle)
        if self.config.burst != 0:
            envelope = 4.0 * u * (1.0 - u)
            pos = self.center + (pos - self.center) * (1.0 + self.config.burst * envelope)[:, None]
        if self.config.swirl != 0:
            rel = pos - self.center
            r = np.clip(np.linalg.norm(rel, axis=1) / self.radius, 0.0, 1.0)
            envelope = 4.0 * u * (1.0 - u)
            theta = self.config.swirl * envelope * (1.0 - self.config.swirl_falloff * r)
            cos, sin = np.cos(theta), np.sin(theta)
            pos = self.center + np.stack(
                [rel[:, 0] * cos - rel[:, 1] * sin, rel[:, 0] * sin + rel[:, 1] * cos], axis=1
            )
        return pos
