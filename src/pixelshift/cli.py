"""Command-line interface.

``pixelshift run``   renders one transformation to a file.
``pixelshift chain`` morphs through a list of pictures into one video.
``pixelshift live``  opens a window.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import typer
from numpy.typing import NDArray

from pixelshift import __version__
from pixelshift.animate import (
    EASINGS,
    TRANSITIONS,
    AnimationConfig,
    Animator,
    transition_options,
)
from pixelshift.export import save_gif, save_mp4
from pixelshift.features import ColorSpace
from pixelshift.io import load_square, save_image
from pixelshift.mapping import Mapping
from pixelshift.metrics import mean_delta_e, mean_displacement
from pixelshift.pipeline import solve_images
from pixelshift.render import CpuRenderer, GpuRenderer, GpuUnavailableError, SoftRenderer
from pixelshift.sequence import chain_frames, loop_frames
from pixelshift.solvers import get_solver

app = typer.Typer(
    help="Rebuild one picture using only the pixels of another.",
    no_args_is_help=True,
    add_completion=False,
)

ANIMATION_SUFFIXES = {".mp4", ".gif"}
TRANSITION_NAMES = " | ".join(sorted(TRANSITIONS))


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=2)


def _upscale(img: NDArray[np.uint8], scale: int) -> NDArray[np.uint8]:
    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))


def _check_choice(value: str, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise _fail(f"{name} must be one of {list(allowed)}")


def _solver_fn(*, solver: str, lam: float, color_space: str, seed: int) -> tuple[Any, Any]:
    """Validate solver options. Returns (solver object, function solving an image pair)."""
    _check_choice(color_space, ("oklab", "lab"), "color-space")
    try:
        solver_obj = get_solver(solver)
    except ValueError as exc:
        raise _fail(str(exc)) from None

    def solve_pair(src: NDArray[np.uint8], tgt: NDArray[np.uint8]) -> Mapping:
        return solve_images(
            src,
            tgt,
            solver_obj,
            lam=lam,
            color_space=cast(ColorSpace, color_space),
            seed=seed,
        )

    return solver_obj, solve_pair


def _load(paths: list[Path], size: int) -> list[NDArray[np.uint8]]:
    for path in paths:
        if not path.is_file():
            raise _fail(f"file not found: {path}")
    typer.echo(f"Loading {len(paths)} images at {size}x{size} ...")
    try:
        return [load_square(p, size) for p in paths]
    except OSError as exc:
        raise _fail(f"cannot read image: {exc}") from None


def _solve_with_report(
    solve_pair: Any, src: NDArray[np.uint8], tgt: NDArray[np.uint8], label: str
) -> Mapping:
    typer.echo(f"Solving {label} ...")
    try:
        mapping: Mapping = solve_pair(src, tgt)
    except ValueError as exc:
        raise _fail(str(exc)) from None
    typer.echo(
        f"  solved in {mapping.meta['runtime_s']:.2f}s | "
        f"color error dE={mean_delta_e(mapping.result_image(), tgt):.2f} | "
        f"avg travel={mean_displacement(mapping):.1%} of the grid"
    )
    return mapping


def _config(
    *,
    transition: str,
    duration: float,
    fps: int,
    easing: str,
    delay: float | None,
    jitter: float | None,
    wave: float | None,
    wave_shape: str | None,
    wobble: float,
    swirl: float | None,
    burst: float | None,
    seed: int,
) -> AnimationConfig:
    """Build the animation settings: a named transition, then any explicit overrides."""
    try:
        options: dict[str, Any] = dict(transition_options(transition))
    except ValueError as exc:
        raise _fail(str(exc)) from None
    overrides = {
        "delay_strength": delay,
        "delay_jitter": jitter,
        "delay_wave": wave,
        "wave_shape": wave_shape,
        "swirl": swirl,
        "burst": burst,
    }
    options.update({k: v for k, v in overrides.items() if v is not None})
    if easing not in EASINGS:
        raise _fail(f"unknown easing {easing!r}; choose from {sorted(EASINGS)}")
    try:
        return AnimationConfig(
            duration=duration, fps=fps, easing=easing, wobble=wobble, seed=seed, **options
        )
    except ValueError as exc:
        raise _fail(str(exc)) from None


def _make_renderer(
    animator: Animator,
    *,
    style: str,
    renderer: str,
    scale: int,
    blur: float,
    quiet: bool = False,
) -> CpuRenderer | SoftRenderer | GpuRenderer:
    def say(message: str) -> None:
        if not quiet:
            typer.echo(message)

    if style == "pixel":
        if renderer == "gpu":
            raise _fail("the gpu renderer only supports --style soft")
        return CpuRenderer(animator, scale=scale)
    if renderer in ("auto", "gpu"):
        try:
            gpu = GpuRenderer(animator, scale=scale, motion_blur=blur)
        except GpuUnavailableError as exc:
            if renderer == "gpu":
                raise _fail(str(exc)) from None
            say(f"  (no GPU renderer available, using the CPU: {exc})")
        else:
            say(f"Renderer: GPU ({gpu.ctx.info['GL_RENDERER']})")
            return gpu
    say("Renderer: CPU")
    return SoftRenderer(animator, scale=scale, motion_blur=blur)


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
    transition: str = typer.Option("spiral", help=f"Motion preset: {TRANSITION_NAMES}"),
    loop: bool = typer.Option(False, help="Play A to B and back to A, so it loops seamlessly."),
    hold: float = typer.Option(0.8, min=0.0, help="Seconds to hold on the finished picture."),
    style: str = typer.Option("soft", help="soft (fluid, motion blur) | pixel (crisp dust)"),
    renderer: str = typer.Option("auto", help="auto | gpu | cpu (gpu needs the [gpu] extra)"),
    swirl: float | None = typer.Option(None, help="Spiral rotation in radians (overrides preset)."),
    burst: float | None = typer.Option(None, min=0.0, max=3.0, help="Outward burst (0-3)."),
    blur: float = typer.Option(2.5, min=0.0, help="Motion-blur trail length in frames (soft)."),
    delay: float | None = typer.Option(None, min=0.0, max=0.95, help="Departure spread (0-0.95)."),
    wave: float | None = typer.Option(None, min=0.0, max=1.0, help="Share of departure sweep."),
    wave_shape: str | None = typer.Option(
        None, help="diagonal | horizontal | vertical | radial | radial_in"
    ),
    jitter: float | None = typer.Option(None, min=0.0, max=1.0, help="Random departure share."),
    wobble: float = typer.Option(0.0, min=0.0, help="Random sideways shake in grid pixels."),
    scale: int = typer.Option(4, min=1, help="Output pixels per grid pixel."),
    png: Path | None = typer.Option(None, help="Also save the final rearranged image here."),
) -> None:
    """Rearrange the pixels of SOURCE to look like TARGET and render the transition."""
    suffix = out.suffix.lower()
    if suffix not in ANIMATION_SUFFIXES | {".png"}:
        raise _fail(f"unsupported output {out.name!r}; use .mp4, .gif or .png")
    _check_choice(style, ("soft", "pixel"), "style")
    _check_choice(renderer, ("auto", "gpu", "cpu"), "renderer")
    config = _config(
        transition=transition,
        duration=duration,
        fps=fps,
        easing=easing,
        delay=delay,
        jitter=jitter,
        wave=wave,
        wave_shape=wave_shape,
        wobble=wobble,
        swirl=swirl,
        burst=burst,
        seed=seed,
    )
    _, solve_pair = _solver_fn(solver=solver, lam=lam, color_space=color_space, seed=seed)
    src_img, tgt_img = _load([source, target], size)
    mapping = _solve_with_report(solve_pair, src_img, tgt_img, f"with '{solver}' (lam={lam})")
    result = mapping.result_image()

    if png is not None:
        save_image(png, _upscale(result, scale))
        typer.echo(f"Saved final image: {png}")
    if suffix == ".png":
        save_image(out, _upscale(result, scale))
        typer.echo(f"Saved final image: {out}")
        return

    frames = _make_renderer(
        Animator(mapping, config), style=style, renderer=renderer, scale=scale, blur=blur
    )
    writer = save_mp4 if suffix == ".mp4" else save_gif
    t0 = time.perf_counter()
    if loop:
        sequence = loop_frames(frames, config.frame_count, hold_frames=round(hold * fps))
        typer.echo(f"Rendering a seamless loop ({frames.size}x{frames.size}) ...")
        writer(out, sequence, fps=fps, hold_start=hold, hold_end=0.0)
    else:
        typer.echo(f"Rendering {config.frame_count} frames ({frames.size}x{frames.size}) ...")
        writer(out, frames.frames(), fps=fps, hold_start=0.5, hold_end=hold + 0.7)
    typer.echo(f"Saved animation: {out} ({time.perf_counter() - t0:.1f}s)")


@app.command()
def chain(
    images: list[Path] = typer.Argument(..., help="Two or more images, in order."),
    out: Path = typer.Option(Path("chain.mp4"), "--out", "-o", help=".mp4 or .gif."),
    size: int = typer.Option(128, min=2, help="Grid side N; every image becomes N x N pixels."),
    lam: float = typer.Option(0.001, min=0.0, help="Travel penalty (see `run --help`)."),
    solver: str = typer.Option("pipeline", help="sort | exact | mediansplit | swaps | pipeline"),
    color_space: str = typer.Option("oklab", help="oklab | lab"),
    seed: int = typer.Option(0, help="Random seed."),
    duration: float = typer.Option(3.5, min=0.1, help="Length of each morph in seconds."),
    fps: int = typer.Option(30, min=1, max=120, help="Frames per second."),
    easing: str = typer.Option("smootherstep", help="linear | smoothstep | smootherstep | cubic"),
    transition: str = typer.Option("spiral", help=f"Motion preset: {TRANSITION_NAMES}"),
    hold: float = typer.Option(0.6, min=0.0, help="Seconds to hold on each picture."),
    settle: float = typer.Option(
        0.6, min=0.0, help="Seconds of cross-fade that restore each real picture."
    ),
    loop: bool = typer.Option(
        False, help="Close the loop: morph the last image back to the first."
    ),
    style: str = typer.Option("soft", help="soft (fluid, motion blur) | pixel (crisp dust)"),
    renderer: str = typer.Option("auto", help="auto | gpu | cpu (gpu needs the [gpu] extra)"),
    swirl: float | None = typer.Option(None, help="Spiral rotation in radians (overrides preset)."),
    burst: float | None = typer.Option(None, min=0.0, max=3.0, help="Outward burst (0-3)."),
    blur: float = typer.Option(2.5, min=0.0, help="Motion-blur trail length in frames (soft)."),
    delay: float | None = typer.Option(None, min=0.0, max=0.95, help="Departure spread (0-0.95)."),
    wave: float | None = typer.Option(None, min=0.0, max=1.0, help="Share of departure sweep."),
    wave_shape: str | None = typer.Option(
        None, help="diagonal | horizontal | vertical | radial | radial_in"
    ),
    jitter: float | None = typer.Option(None, min=0.0, max=1.0, help="Random departure share."),
    wobble: float = typer.Option(0.0, min=0.0, help="Random sideways shake in grid pixels."),
    scale: int = typer.Option(4, min=1, help="Output pixels per grid pixel."),
) -> None:
    """Morph through IMAGES in order (A to B to C ...) and write one video.

    Each morph is built from the previous picture's own pixels. A short cross-fade then restores
    the real picture before the next morph starts, so the video has no jumps.
    """
    suffix = out.suffix.lower()
    if suffix not in ANIMATION_SUFFIXES:
        raise _fail(f"unsupported output {out.name!r}; use .mp4 or .gif")
    if len(images) < 2:
        raise _fail("give at least two images")
    _check_choice(style, ("soft", "pixel"), "style")
    _check_choice(renderer, ("auto", "gpu", "cpu"), "renderer")
    config = _config(
        transition=transition,
        duration=duration,
        fps=fps,
        easing=easing,
        delay=delay,
        jitter=jitter,
        wave=wave,
        wave_shape=wave_shape,
        wobble=wobble,
        swirl=swirl,
        burst=burst,
        seed=seed,
    )
    _, solve_pair = _solver_fn(solver=solver, lam=lam, color_space=color_space, seed=seed)
    pictures = _load(images, size)
    total = len(pictures) if loop else len(pictures) - 1
    step = {"n": 0}
    first_renderer = {"announced": False}

    def solve_step(src: NDArray[np.uint8], tgt: NDArray[np.uint8]) -> Mapping:
        step["n"] += 1
        return _solve_with_report(
            solve_pair, src, tgt, f"morph {step['n']} of {total} with '{solver}'"
        )

    def make_renderer(mapping: Mapping) -> CpuRenderer | SoftRenderer | GpuRenderer:
        quiet = first_renderer["announced"]
        first_renderer["announced"] = True
        return _make_renderer(
            Animator(mapping, config),
            style=style,
            renderer=renderer,
            scale=scale,
            blur=blur,
            quiet=quiet,
        )

    sequence = chain_frames(
        pictures,
        solve_step,
        make_renderer,
        frame_count=config.frame_count,
        hold_frames=round(hold * fps),
        settle_frames=round(settle * fps),
        close_loop=loop,
    )
    writer = save_mp4 if suffix == ".mp4" else save_gif
    t0 = time.perf_counter()
    writer(out, sequence, fps=fps, hold_start=0.0, hold_end=0.0)
    typer.echo(f"Saved animation: {out} ({time.perf_counter() - t0:.1f}s)")


@app.command()
def live(
    source: Path = typer.Option(..., "--source", "-s", help="Image whose pixels are used."),
    target: Path = typer.Option(..., "--target", "-t", help="Image to rebuild."),
    size: int = typer.Option(128, min=2, help="Grid side N; both images become N x N pixels."),
    lam: float = typer.Option(0.001, min=0.0, help="Travel penalty (see `run --help`)."),
    solver: str = typer.Option("pipeline", help="sort | exact | mediansplit | swaps | pipeline"),
    color_space: str = typer.Option("oklab", help="oklab | lab"),
    seed: int = typer.Option(0, help="Random seed."),
    duration: float = typer.Option(3.5, min=0.1, help="Animation length in seconds."),
    easing: str = typer.Option("smootherstep", help="linear | smoothstep | smootherstep | cubic"),
    transition: str = typer.Option("spiral", help=f"Motion preset: {TRANSITION_NAMES}"),
    swirl: float | None = typer.Option(None, help="Spiral rotation in radians (overrides preset)."),
    burst: float | None = typer.Option(None, min=0.0, max=3.0, help="Outward burst (0-3)."),
    delay: float | None = typer.Option(None, min=0.0, max=0.95, help="Departure spread (0-0.95)."),
    wave: float | None = typer.Option(None, min=0.0, max=1.0, help="Share of departure sweep."),
    wave_shape: str | None = typer.Option(
        None, help="diagonal | horizontal | vertical | radial | radial_in"
    ),
    jitter: float | None = typer.Option(None, min=0.0, max=1.0, help="Random departure share."),
    wobble: float = typer.Option(0.0, min=0.0, help="Random sideways shake in grid pixels."),
    window: int = typer.Option(768, min=128, help="Window size in pixels."),
    mode: str = typer.Option("pingpong", help="pingpong | loop | once"),
    fullscreen: bool = typer.Option(False, help="Start in fullscreen."),
    vsync: bool = typer.Option(True, help="Sync to the display refresh rate."),
) -> None:
    """Open a window and watch SOURCE turn into TARGET. Needs the [gpu] extra.

    Drag and drop: drop two images on the window to use them as source and target, or one image
    to morph the current target into it.

    Keys: SPACE pause, R restart, B reverse, M loop mode, LEFT/RIGHT scrub, UP/DOWN speed,
    S screenshot, F fullscreen, ESC or Q quit.
    """
    from pixelshift.live import MODES, run_live

    _check_choice(mode, MODES, "mode")
    config = _config(
        transition=transition,
        duration=duration,
        fps=30,
        easing=easing,
        delay=delay,
        jitter=jitter,
        wave=wave,
        wave_shape=wave_shape,
        wobble=wobble,
        swirl=swirl,
        burst=burst,
        seed=seed,
    )
    _, solve_pair = _solver_fn(solver=solver, lam=lam, color_space=color_space, seed=seed)
    src_img, tgt_img = _load([source, target], size)
    mapping = _solve_with_report(solve_pair, src_img, tgt_img, f"with '{solver}' (lam={lam})")

    typer.echo("Opening window: drop images on it, press Q or ESC to quit ...")
    try:
        stats = run_live(
            mapping,
            config,
            window_size=window,
            vsync=vsync,
            fullscreen=fullscreen,
            mode=mode,
            solve=solve_pair,
            target_image=tgt_img,
        )
    except GpuUnavailableError as exc:
        raise _fail(str(exc)) from None
    typer.echo(f"Closed after {stats.frames} frames ({stats.fps:.0f} fps average).")
