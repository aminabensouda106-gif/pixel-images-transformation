"""Command-line interface: ``pixelshift run --source a.jpg --target b.jpg --out result.mp4``."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
import typer
from numpy.typing import NDArray

from pixelshift import __version__
from pixelshift.animate import EASINGS, AnimationConfig, Animator
from pixelshift.export import save_gif, save_mp4
from pixelshift.features import ColorSpace
from pixelshift.io import load_square, save_image
from pixelshift.metrics import mean_delta_e, mean_displacement
from pixelshift.pipeline import solve_images
from pixelshift.render import CpuRenderer, SoftRenderer
from pixelshift.solvers import get_solver

app = typer.Typer(
    help="Rebuild one picture using only the pixels of another.",
    no_args_is_help=True,
    add_completion=False,
)

ANIMATION_SUFFIXES = {".mp4", ".gif"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=2)


def _upscale(img: NDArray[np.uint8], scale: int) -> NDArray[np.uint8]:
    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))


@app.callback()
def main() -> None:
    """pixelshift: pixel-rearrangement morphs."""


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def run(
    source: Path = typer.Option(..., "--source", "-s", help="Image whose pixels are used."),
    target: Path = typer.Option(..., "--target", "-t", help="Image to rebuild."),
    out: Path = typer.Option(
        Path("result.mp4"), "--out", "-o", help=".mp4 or .gif (animation), or .png (final image)."
    ),
    size: int = typer.Option(128, min=2, help="Grid side N; both images become N x N pixels."),
    lam: float = typer.Option(
        0.001, min=0.0, help="Travel penalty: 0 = best likeness, larger = shorter movement."
    ),
    solver: str = typer.Option("pipeline", help="sort | exact | mediansplit | swaps | pipeline"),
    color_space: str = typer.Option("oklab", help="oklab | lab"),
    seed: int = typer.Option(0, help="Random seed (output is deterministic for a given seed)."),
    duration: float = typer.Option(3.5, min=0.1, help="Animation length in seconds."),
    fps: int = typer.Option(30, min=1, max=120, help="Frames per second."),
    easing: str = typer.Option("smootherstep", help="linear | smoothstep | smootherstep | cubic"),
    style: str = typer.Option("soft", help="soft (fluid, motion blur) | pixel (crisp dust)"),
    swirl: float = typer.Option(1.0, help="Peak spiral rotation in radians (0 = straight paths)."),
    blur: float = typer.Option(2.5, min=0.0, help="Motion-blur trail length in frames (soft)."),
    delay: float = typer.Option(0.5, min=0.0, max=0.95, help="Spread of departure times (0-0.95)."),
    wave: float = typer.Option(0.6, min=0.0, max=1.0, help="Share of departure order that sweeps."),
    jitter: float = typer.Option(0.1, min=0.0, max=1.0, help="Share of departure order at random."),
    wobble: float = typer.Option(0.0, min=0.0, help="Random sideways shake in grid pixels."),
    scale: int = typer.Option(4, min=1, help="Output pixels per grid pixel."),
    png: Path | None = typer.Option(None, help="Also save the final rearranged image here."),
) -> None:
    """Rearrange the pixels of SOURCE to look like TARGET and render the transition."""
    suffix = out.suffix.lower()
    if suffix not in ANIMATION_SUFFIXES | {".png"}:
        raise _fail(f"unsupported output {out.name!r}; use .mp4, .gif or .png")
    if color_space not in ("oklab", "lab"):
        raise _fail("color-space must be 'oklab' or 'lab'")
    if easing not in EASINGS:
        raise _fail(f"unknown easing {easing!r}; choose from {sorted(EASINGS)}")
    if style not in ("soft", "pixel"):
        raise _fail("style must be 'soft' or 'pixel'")
    try:
        solver_obj = get_solver(solver)
    except ValueError as exc:
        raise _fail(str(exc)) from None
    for path in (source, target):
        if not path.is_file():
            raise _fail(f"file not found: {path}")

    typer.echo(f"Loading images at {size}x{size} ...")
    try:
        src_img, tgt_img = load_square(source, size), load_square(target, size)
    except OSError as exc:
        raise _fail(f"cannot read image: {exc}") from None

    typer.echo(f"Solving with '{solver}' (lam={lam}) ...")
    try:
        mapping = solve_images(
            src_img,
            tgt_img,
            solver_obj,
            lam=lam,
            color_space=cast(ColorSpace, color_space),
            seed=seed,
        )
    except ValueError as exc:
        raise _fail(str(exc)) from None
    result = mapping.result_image()
    typer.echo(
        f"  solved in {mapping.meta['runtime_s']:.2f}s | "
        f"color error dE={mean_delta_e(result, tgt_img):.2f} | "
        f"avg travel={mean_displacement(mapping):.1%} of the grid"
    )

    if png is not None:
        save_image(png, _upscale(result, scale))
        typer.echo(f"Saved final image: {png}")

    if suffix == ".png":
        save_image(out, _upscale(result, scale))
        typer.echo(f"Saved final image: {out}")
        return

    try:
        config = AnimationConfig(
            duration=duration,
            fps=fps,
            easing=easing,
            delay_strength=delay,
            delay_jitter=jitter,
            delay_wave=wave,
            wobble=wobble,
            swirl=swirl,
            seed=seed,
        )
    except ValueError as exc:
        raise _fail(str(exc)) from None
    animator = Animator(mapping, config)
    renderer: CpuRenderer | SoftRenderer
    if style == "soft":
        renderer = SoftRenderer(animator, scale=scale, motion_blur=blur)
    else:
        renderer = CpuRenderer(animator, scale=scale)
    typer.echo(f"Rendering {config.frame_count} frames ({renderer.size}x{renderer.size}) ...")
    t0 = time.perf_counter()
    writer = save_mp4 if suffix == ".mp4" else save_gif
    writer(out, renderer.frames(), fps=fps)
    typer.echo(f"Saved animation: {out} ({time.perf_counter() - t0:.1f}s)")
