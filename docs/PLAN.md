# Project plan

## 1. Mission

Build a program that takes a **source image** and a **target image** and produces a new image that looks like the target but is made **only of the source's pixels** (a permutation of them), plus an **animation** where the pixels travel from their source positions to their final positions.

Reference behavior (from the recorded demo): a face photo dissolves into a cloud of moving pixels, which settle into a different face. The in-between frames are speckled because individual pixels are in flight. The final frame has the target's shape and the source's colors.

## 2. Requirements

### Functional

| ID | Requirement |
|---|---|
| F1 | Accept two images of any size and format (PNG, JPG, WebP) |
| F2 | Normalize both to the same `N × N` grid (center crop, then Lanczos resize) |
| F3 | Compute a bijective mapping source pixel → target position |
| F4 | Guarantee the output is an exact permutation of the source pixels (tested) |
| F5 | Expose a `λ` parameter trading likeness against travel distance |
| F6 | Render the transition as an animation with configurable duration and easing |
| F7 | Export the final image (PNG) and the animation (GIF, MP4) |
| F8 | Provide a CLI, and later a live window |
| F9 | Report quality metrics: mean ΔE, total displacement, solver time |

### Non-functional

| ID | Requirement |
|---|---|
| N1 | Solve a 128×128 grid (16 384 pixels) in under 10 s on a laptop CPU |
| N2 | Solve a 256×256 grid (65 536 pixels) in under 60 s |
| N3 | Deterministic output for a given seed |
| N4 | No network access needed at runtime |
| N5 | Tested, typed and linted; CI runs on every push |

### Out of scope (initially)

Video input, face detection or alignment, color transfer that alters pixel values, web hosting.

## 3. The core problem

With `P = N²` pixels, find permutation `π` minimizing

```
Σᵢ ‖Lab(S[π(i)]) − Lab(T[i])‖²  +  λ ‖pos(π(i)) − pos(i)‖²
```

This is the **linear assignment problem**. A `P × P` cost matrix is impossible at scale (`P = 16 384` means 268 M entries, about 1 GB in float32), so we avoid building it and use structured approximations. See [`TECHNOLOGIES.md`](TECHNOLOGIES.md) §3 to §6.

## 4. Strategy

```
baseline (sort)  →  good initial guess (median split)  →  refine (parallel swaps)  →  polish (exact on tiles)
```

Each stage is measured against the previous one, so every addition has to **prove** it improves quality or speed. The exact Hungarian solver on small grids is the ground truth for scoring the approximations.

## 5. Milestones

Each milestone ends with a **gate**: a concrete check that must pass before the next one starts.

### M0: Foundations
- `pyproject.toml`, `src/` layout, `ruff`, `mypy`, `pytest`, GitHub Actions
- `color.py`: sRGB ↔ linear ↔ XYZ ↔ CIELAB, plus OKLab
- `io.py`: load, center-crop, resize
- **Gate:** round-trip RGB → Lab → RGB error below 1/255; CI green.

### M1: Baselines and ground truth
- `solvers/sort1d.py`: sort both pixel sets by luminance and match by rank
- `solvers/exact.py`: Hungarian via SciPy (for grids up to about 64×64)
- Metrics module: mean ΔE, displacement, permutation-validity check
- **Gate:** both solvers return valid permutations; exact beats sort on mean ΔE for every test pair.

### M2: Fast solver
- `solvers/mediansplit.py`: recursive median-split matching on color + position features
- `solvers/swaps.py`: Numba-parallel disjoint-pair swap refinement with a shrinking neighborhood
- Optional tile polish with the exact solver
- **Gate:** on 64×64, within 5% of the exact solver's cost; 128×128 solved in under 10 s; speedup from parallelism measured and recorded.

### M3: Animation and CPU rendering
- `mapping.py`, `animate.py`: per-pixel path, easing (smoothstep), per-pixel delay by distance, optional curl-noise offset
- `render/cpu.py`: scatter pixels into a frame buffer, with splatting to avoid holes
- GIF export
- **Gate:** a GIF whose last frame is the permuted image; the pixel multiset of the last frame equals the source's (test).

### M4: CLI and MP4
- `cli.py` with `run` command; MP4 through imageio-ffmpeg
- Gallery of results in `examples/`, linked from the README
- **Gate:** the README quick-start works verbatim on a clean checkout.

### M5: GPU renderer and live window
- `render/gpu.py` with moderngl: one point per pixel, interpolation in the vertex shader
- `live` command with interactive controls (restart, swap images, speed)
- **Gate:** 128×128 at 60 fps; 512×512 (262 144 points) at 30 fps or better.

### M6: Stretch
- Drag-and-drop images, alternative transitions (wave, spiral, explosion), WebGL/WASM port.

## 6. Quality metrics

| Metric | Meaning |
|---|---|
| Mean ΔE (CIELAB) | Average perceptual color error between result and target; lower is better |
| Cost gap vs exact | `(cost − exact_cost) / exact_cost` on small grids |
| Mean displacement | Average pixel travel distance; relates to animation smoothness |
| Runtime and peak memory | Scalability |
| Permutation validity | Boolean, must always be true |

## 7. Risks

| Risk | Mitigation |
|---|---|
| Source lacks the colors needed (the result looks washed out) | Document it; optional histogram-aware warning; this is a limit of the idea, not a bug |
| Swap refinement stalls in a local minimum | Annealing schedule and restarts; tile polish |
| Visible holes during animation | Sub-pixel splatting on CPU; point sprites with size on GPU |
| OpenGL setup differs per machine | Keep the CPU renderer as the always-working fallback |
| Numba compile time on first run | Cache compiled functions (`cache=True`) |

## 8. Working agreement

- Build milestone by milestone, and stop at every gate to show results.
- Choose the simplest method that meets the gate. Add complexity only with measured evidence.
- Every algorithmic claim in the docs is backed by a benchmark or a test.
