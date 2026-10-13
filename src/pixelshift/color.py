"""Vectorized color-space conversions.

All functions take and return float arrays whose last axis has length 3.
RGB inputs/outputs are sRGB in [0, 1]. Convert from uint8 with ``rgb_u8_to_float``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float64]

# Linear sRGB -> LMS (Ottosson, 2020) and LMS' -> OKLab.
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)

# Linear sRGB -> XYZ (D65) and the D65 white point.
_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_WHITE = _RGB2XYZ @ np.ones(3)

_DELTA = 6 / 29


def rgb_u8_to_float(rgb: NDArray[np.uint8]) -> Float:
    """uint8 [0, 255] -> float64 [0, 1]."""
    return rgb.astype(np.float64) / 255.0


def float_to_rgb_u8(rgb: Float) -> NDArray[np.uint8]:
    """float [0, 1] -> uint8 [0, 255], clipped and rounded."""
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def srgb_to_linear(c: Float) -> Float:
    """Undo the sRGB gamma curve."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: Float) -> Float:
    """Apply the sRGB gamma curve."""
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb_to_oklab(rgb: Float) -> Float:
    """sRGB [0, 1] -> OKLab (L in [0, 1], a and b roughly in [-0.4, 0.4])."""
    lms = srgb_to_linear(rgb) @ _M1.T
    lab: Float = np.cbrt(lms) @ _M2.T
    return lab


def oklab_to_rgb(lab: Float) -> Float:
    """OKLab -> sRGB [0, 1] (not clipped)."""
    lms = (lab @ _M2_INV.T) ** 3
    return linear_to_srgb(lms @ _M1_INV.T)


def _lab_f(t: Float) -> Float:
    return np.where(t > _DELTA**3, np.cbrt(t), t / (3 * _DELTA**2) + 4 / 29)


def _lab_f_inv(t: Float) -> Float:
    return np.where(t > _DELTA, t**3, 3 * _DELTA**2 * (t - 4 / 29))


def rgb_to_lab(rgb: Float) -> Float:
    """sRGB [0, 1] -> CIELAB (D65), L in [0, 100]."""
    xyz = (srgb_to_linear(rgb) @ _RGB2XYZ.T) / _WHITE
    f = _lab_f(xyz)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: Float) -> Float:
    """CIELAB (D65) -> sRGB [0, 1] (not clipped)."""
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    xyz = _lab_f_inv(np.stack([fx, fy, fz], axis=-1)) * _WHITE
    return linear_to_srgb(xyz @ _XYZ2RGB.T)
