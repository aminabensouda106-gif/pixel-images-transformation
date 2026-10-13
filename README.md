# Pixel Image Transformation

> Rebuild one picture using **only the pixels of another**. Every pixel of a *source* image is moved to a new position so that, together, they form a *target* image. No color is invented, blended or repainted: the output is a pure **permutation** of the source pixels, and the transformation is shown as an animation of the pixels travelling across the canvas.

**Status:** 🚧 under active development. The core (solver, animation, CPU renderer, command-line tool, MP4/GIF export) works; the GPU renderer and live window are still planned. Progress follows the milestones in [`docs/PLAN.md`](docs/PLAN.md). Features marked *(planned)* do not exist yet.

<p align="center">
  <img src="examples/gallery/palette_to_smiley.gif" width="288" alt="A rainbow palette rearranging itself into a smiley face">
  &nbsp;&nbsp;
  <img src="examples/gallery/sunset_to_heart.gif" width="288" alt="A sunset rearranging itself into a heart">
</p>
<p align="center"><em>Left: a rainbow palette becomes a smiley. Right: a sunset becomes a heart. Same pixels, new arrangement.</em></p>

---

## Table of contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Technology stack](#technology-stack)
4. [Architecture](#architecture)
5. [Repository layout](#repository-layout)
6. [Getting started](#getting-started)
7. [Roadmap](#roadmap)
8. [Documentation](#documentation)
9. [Limitations](#limitations)
10. [Credits and inspiration](#credits-and-inspiration)
11. [Contributing and license](#contributing-and-license)

---

## What it does

Given two images, a **source** (the pixels we are allowed to use) and a **target** (the picture we want to see):

1. Both are resized to the same grid of `N × N` pixels, so both contain exactly `N²` pixels.
2. The program finds, for every target position, which source pixel should land there. This is a **bijection**: every source pixel is used exactly once.
3. The pixels are animated from their original positions to their assigned ones, and the result can be exported as an image, GIF or MP4.

The final picture looks like the target, yet if you sorted its pixels by color you would get exactly the same list as the source. Only the **color grading and the pixels of the first image** are used.

## How it works

The core of the project is an **assignment problem**, a discrete form of *optimal transport*. We look for the permutation `π` that minimizes

```
cost(π) = Σᵢ  ‖ Lab(source[π(i)]) − Lab(target[i]) ‖²  +  λ · ‖ pos(π(i)) − pos(i) ‖²
              └────────── color error ──────────┘        └──── travel distance ────┘
```

- the **color term** makes the result resemble the target (measured in the perceptual CIELAB space, not raw RGB),
- the **spatial term** `λ` keeps pixels close to where they started, so the animation looks like a flow rather than static noise. `λ = 0` gives the best likeness, and larger values give smoother motion.

Solving it exactly (Hungarian algorithm) costs `O(N⁶)` for an `N × N` image, which is too slow beyond small sizes. The project therefore uses a **three-stage pipeline**:

| Stage | Method | Role |
|---|---|---|
| 1. Initial guess | Recursive median-split matching in color+position space | Fast (`O(P log P)`, `P = N²`) and already close to good |
| 2. Refinement | Parallel pairwise swaps (JIT-compiled, multi-core) | Removes most of the remaining error |
| 3. Polish *(optional)* | Exact Hungarian solve on small tiles | Squeezes out the last local errors |

Then an **animation engine** interpolates each pixel along its path (easing, per-pixel delay, optional noise), and a **renderer** draws the frames, either on the CPU with NumPy or on the GPU with OpenGL point sprites.

Details and the mathematics are in [`docs/TECHNOLOGIES.md`](docs/TECHNOLOGIES.md).

## Technology stack

| Area | Choice | Why |
|---|---|---|
| Language | **Python 3.12** | Fast to prototype, huge scientific ecosystem |
| Numerics | **NumPy** | Vectorized array math on whole images |
| Hot loops | **Numba** (`@njit(parallel=True)`) | Compiles the swap solver to multi-core machine code |
| Exact solver | **SciPy** `linear_sum_assignment` | Reference/optimal solution for small grids and tile polishing |
| Image I/O | **Pillow** (+ OpenCV optional) | Loading, cropping, Lanczos resampling |
| Color science | Own CIELAB / OKLab implementation | Perceptual color distance |
| GPU rendering | **moderngl** + GLSL shaders (+ `pygame` or `glfw` window) | Smooth 60 fps animation of 10⁵ to 10⁶ points |
| Video / GIF export | **imageio** + FFmpeg (MP4), Pillow (GIF) | MP4 and GIF output |
| CLI | **Typer** | Clean command-line interface |
| Quality | **pytest**, **ruff**, **mypy**, GitHub Actions | Tests, lint, types, CI |
| Packaging | **uv** + `pyproject.toml` | Reproducible environments |

Every choice is justified in depth in [`docs/TECHNOLOGIES.md`](docs/TECHNOLOGIES.md).

## Architecture

```mermaid
flowchart LR
    A[Source image] --> P[Preprocess<br/>crop, resize, N×N]
    B[Target image] --> P
    P --> F[Feature builder<br/>Lab color + position]
    F --> S[Solver<br/>median-split → swaps → tile polish]
    S --> M[(Mapping π<br/>start → end positions)]
    M --> AN[Animation engine<br/>easing, delay, noise]
    AN --> R{Renderer}
    R --> CPU[CPU / NumPy]
    R --> GPU[GPU / OpenGL]
    CPU --> E[Exporter<br/>PNG, GIF, MP4]
    GPU --> W[Live window]
    GPU --> E
```

The design rule is **one-way data flow with a single hand-off object**: the solver produces a `Mapping`, and everything after it only reads that mapping. Solvers, renderers and exporters are interchangeable behind small interfaces. The full design is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Repository layout

```
.
├── README.md
├── pyproject.toml
├── docs/
│   ├── PLAN.md            project plan, requirements, milestones
│   ├── TECHNOLOGIES.md    deep dive for studying each technology
│   └── ARCHITECTURE.md    modules, interfaces, data flow
├── src/pixelshift/
│   ├── io.py              load / crop / resize / save
│   ├── color.py           sRGB ↔ CIELAB / OKLab
│   ├── features.py        feature vectors (color + position)
│   ├── solvers/
│   │   ├── base.py        Solver interface and registry
│   │   ├── sort1d.py      luminance-sort baseline
│   │   ├── mediansplit.py recursive median-split matching (Numba)
│   │   ├── swaps.py       parallel swap refinement (Numba)
│   │   ├── exact.py       exact Hungarian solver and chunked exact polish (SciPy)
│   │   └── combined.py    the full pipeline solver
│   ├── mapping.py         Mapping dataclass (the hand-off object)
│   ├── metrics.py         cost, color error, travel distance
│   ├── pipeline.py        images in, Mapping out
│   ├── animate.py         easing, departure delays, wobble
│   ├── render/            cpu.py (gpu.py is planned)
│   ├── export.py          GIF and MP4 writers
│   └── cli.py             `pixelshift` command (Typer)
├── tests/
├── examples/
│   ├── make_inputs.py     generates the example input images
│   ├── inputs/            example sources and targets
│   └── gallery/           example outputs shown above
└── .github/workflows/     CI
```

## Getting started

Requires Python 3.11 to 3.13. FFmpeg does not need to be installed: a copy is bundled.

```bash
git clone https://github.com/aminabensouda106-gif/pixel-images-transformation.git
cd pixel-images-transformation

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .

# Rebuild the pixels of a sunset into a heart (uses the bundled example images)
pixelshift run -s examples/inputs/sunset.png -t examples/inputs/heart.png -o result.mp4
```

The first run compiles the numeric kernels, so it takes a few extra seconds. Every later run reuses the compiled code.

**Common options** (`pixelshift run --help` lists them all):

| Option | Default | Meaning |
|---|---|---|
| `-s`, `--source` | required | Image whose pixels are used |
| `-t`, `--target` | required | Image to rebuild |
| `-o`, `--out` | `result.mp4` | `.mp4` or `.gif` for the animation, `.png` for the final image only |
| `--size` | 128 | Grid side `N`; both images become `N × N` pixels |
| `--lam` | 0.05 | Travel penalty. `0` gives the best likeness; larger values give shorter, smoother pixel movement |
| `--solver` | `pipeline` | `sort`, `exact` (up to 64×64), `mediansplit`, `swaps`, `pipeline` |
| `--duration`, `--fps` | 3.0 s, 30 | Animation length and frame rate |
| `--wobble` | 1.5 | Sideways shake in grid pixels (`0` for straight lines) |
| `--scale` | 4 | Output pixels per grid pixel |
| `--seed` | 0 | The output is deterministic for a given seed |
| `--png` | none | Also save the final rearranged image |

**Examples**

```bash
# The two animations at the top of this page
pixelshift run -s examples/inputs/palette.png -t examples/inputs/smiley.png \
  --size 96 --scale 3 --fps 20 -o examples/gallery/palette_to_smiley.gif
pixelshift run -s examples/inputs/sunset.png -t examples/inputs/heart.png \
  --size 96 --scale 3 --fps 20 -o examples/gallery/sunset_to_heart.gif

# Your own photos, larger grid, final image only
pixelshift run -s me.jpg -t friend.jpg --size 256 -o final.png
```

A live window (`pixelshift live`) is planned for milestone M5.

**Developing**

```bash
pip install -e ".[dev]"
pytest          # tests
ruff check .    # lint
mypy            # type checking
```

## Roadmap

| Milestone | Goal | Status |
|---|---|---|
| **M0** | Repo, packaging, CI, color conversion | ✅ done |
| **M1** | Baseline solver (luminance sort) and exact solver, with quality metrics | ✅ done |
| **M2** | Median-split + parallel swap solver, benchmarked against the exact one | ✅ done |
| **M3** | Animation engine and CPU renderer, GIF export | ✅ done |
| **M4** | CLI, MP4 export, documentation of results | ✅ done |
| **M5** | GPU renderer and live window | planned |
| **M6** | Stretch: web version, drag-and-drop, more transitions | planned |

### Measured performance

Measured on a 16-core laptop CPU, with two photos of faces at λ = 0.05 to 0.2:

| Grid | Pipeline solver | Exact solver | Quality of pipeline |
|---|---|---|---|
| 64 × 64 | about 0.5 s | 12 to 24 s | within 0.2% (λ = 0) and 2.5% (λ = 0.2) of the optimal cost |
| 128 × 128 | about 1.5 s | impractical | not measurable |
| 256 × 256 | about 3.5 to 5 s | impractical | not measurable |

Timings depend on your machine and core count.

Each milestone has acceptance criteria in [`docs/PLAN.md`](docs/PLAN.md).

## Documentation

- [`docs/PLAN.md`](docs/PLAN.md): requirements, scope, milestones, risks
- [`docs/TECHNOLOGIES.md`](docs/TECHNOLOGIES.md): the theory and tools to study
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): modules and data flow

## Limitations

- Source and target must have the same number of pixels, so both are cropped and resized to a common square grid. Fine detail in the target is limited by the **colors available in the source**: a source with no red cannot reproduce a red target.
- When the colors of the two images differ a lot, the printed color error (ΔE) is high, and the result is a stylized version of the target made from the source's palette. The two images in the gallery above work that way on purpose.
- Results are approximate: the fast solver does not guarantee the globally optimal permutation.
- GIF files are limited to 256 colors per frame, so they shimmer slightly in flight. Use MP4 for the best quality.
- Speed and memory depend on grid size. The exact solver is only practical on small grids.

## Credits and inspiration

The visual effect is inspired by pixel-rearrangement morph demos such as *Obamify*. The theory comes from optimal transport (Monge–Kantorovich) and the classical assignment problem. This project is an independent implementation.

## Contributing and license

Issues and pull requests are welcome. License: to be chosen before the first release (MIT is the usual default for a project like this).
