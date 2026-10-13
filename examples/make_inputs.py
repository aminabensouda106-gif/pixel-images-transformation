"""Generate the example input images (no external assets, safe to publish).

Run from the repository root:  python examples/make_inputs.py
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "inputs"
N = 256


def palette() -> Image.Image:
    """Hue across, brightness down: a source that contains many colors."""
    h = np.linspace(0, 1, N, endpoint=False)
    v = np.linspace(1, 0.05, N)
    img = np.zeros((N, N, 3), dtype=np.uint8)
    for y, value in enumerate(v):
        for x, hue in enumerate(h):
            img[y, x] = [round(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.95, value)]
    return Image.fromarray(img)


def smiley() -> Image.Image:
    img = Image.new("RGB", (N, N), (20, 40, 120))
    d = ImageDraw.Draw(img)
    d.ellipse((28, 28, N - 28, N - 28), fill=(250, 210, 30), outline=(20, 20, 20), width=6)
    d.ellipse((80, 90, 112, 130), fill=(20, 20, 20))
    d.ellipse((N - 112, 90, N - 80, 130), fill=(20, 20, 20))
    d.arc((70, 80, N - 70, N - 60), start=20, end=160, fill=(20, 20, 20), width=10)
    return img


def sunset() -> Image.Image:
    """Sky gradient from deep blue to orange, a sun, dark hills."""
    sky_top, sky_mid, sky_low = (
        np.array([30, 20, 90]),
        np.array([220, 80, 70]),
        np.array([255, 190, 80]),
    )
    t = np.linspace(0, 1, N)[:, None]
    col = np.where(
        t < 0.6,
        sky_top + (sky_mid - sky_top) * (t / 0.6),
        sky_mid + (sky_low - sky_mid) * ((t - 0.6) / 0.4),
    )
    img = Image.fromarray(np.repeat(col[:, None, :], N, axis=1).astype(np.uint8).reshape(N, N, 3))
    d = ImageDraw.Draw(img)
    d.ellipse((N // 2 - 40, 130, N // 2 + 40, 210), fill=(255, 240, 150))
    xs = np.arange(N)
    hills = (190 + 25 * np.sin(xs / 25) + 15 * np.sin(xs / 9)).astype(int)
    for x, y in zip(xs, hills, strict=True):
        d.line((x, y, x, N), fill=(25, 15, 40))
    return img


def heart() -> Image.Image:
    img = Image.new("RGB", (N, N), (15, 10, 25))
    yy, xx = np.mgrid[0:N, 0:N]
    x = (xx - N / 2) / (N / 2.6)
    y = -(yy - N / 2.2) / (N / 2.6)
    mask = (x**2 + y**2 - 1) ** 3 - x**2 * y**3 <= 0
    arr = np.asarray(img).copy()
    arr[mask] = (230, 40, 70)
    return Image.fromarray(arr)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (
        ("palette", palette),
        ("smiley", smiley),
        ("sunset", sunset),
        ("heart", heart),
    ):
        make().save(OUT / f"{name}.png")
        print(f"wrote {OUT / name}.png")


if __name__ == "__main__":
    main()
