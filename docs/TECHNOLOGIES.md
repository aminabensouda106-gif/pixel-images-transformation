# Technologies and theory: a study guide

This document explains **what each technology is, why it is used here, and what to study** to understand it in depth. Sections follow the pipeline order. Notation: the image has `P = N²` pixels, `S` is the source, `T` the target.

Contents: [1 Image basics](#1-digital-images-and-resampling) · [2 Color science](#2-color-science) · [3 Assignment and optimal transport](#3-the-core-problem-assignment-and-optimal-transport) · [4 Initial guess](#4-fast-initial-guess-recursive-median-split) · [5 Refinement](#5-refinement-parallel-swaps) · [6 Other solvers](#6-other-solvers-worth-knowing) · [7 Performance](#7-performance-engineering-numpy-and-numba) · [8 Animation](#8-animation-mathematics) · [9 Rendering](#9-rendering-cpu-and-gpu) · [10 Export](#10-video-and-gif-export) · [11 Tooling](#11-engineering-tooling) · [12 Study path](#12-suggested-study-path)

---

## 1. Digital images and resampling

**Concept.** An image is a 3-D array `H × W × 3` of 8-bit values (R, G, B). Two images can only be put in one-to-one correspondence if they have the same pixel count, so we **center-crop to a square** and **resample** to `N × N`.

**Resampling.** Lanczos resampling uses a windowed sinc kernel and preserves detail better than bilinear or bicubic when shrinking. Study: sampling theorem, aliasing, convolution kernels.

**Tools.** Pillow (`Image.resize(..., Image.LANCZOS)`), NumPy arrays (`np.asarray(img)`).

**Pitfalls.** Resampling creates new colors by blending. That is fine for *preparing the inputs*, because afterwards the source pixels are never altered. Keep **sRGB gamma** in mind (next section): pixel values are not linear in light.

## 2. Color science

The color term of the cost must reflect **how different two colors look**, and RGB distance does not.

### 2.1 sRGB and gamma
sRGB values are gamma-encoded. To do physics-like math you linearize: `c_lin = ((c + 0.055)/1.055)^2.4` for `c > 0.04045`, else `c/12.92`.

### 2.2 CIE XYZ and CIELAB
XYZ is a device-independent color space from human cone response. **CIELAB** (`L*`, `a*`, `b*`) is built from XYZ so that Euclidean distance approximates perceived difference:

```
ΔE76 = sqrt( (ΔL*)² + (Δa*)² + (Δb*)² )
```

Pipeline: sRGB → linear RGB → XYZ (3×3 matrix, D65 white) → Lab (cube-root nonlinearity).

### 2.3 OKLab
A more recent space (Björn Ottosson, 2020) with better hue linearity and a simpler formula. A strong alternative to CIELAB; the project keeps the color space **swappable** and compares both.

### 2.4 Better distances
CIEDE2000 is more accurate but slow and non-Euclidean, which breaks the geometric tricks used by the solvers (median splits, k-d trees assume Euclidean features). Plain Euclidean distance in Lab/OKLab is the pragmatic choice.

**Study:** human color vision, gamma, white points, why Euclidean RGB is a poor metric. **Tools:** NumPy matrix multiplication; optionally `colour-science` for cross-checking.

## 3. The core problem: assignment and optimal transport

### 3.1 Formulation
Two sets of `P` items. A cost `c(i, j)` for placing source pixel `j` at target slot `i`. Find a **bijection** `π` minimizing `Σᵢ c(i, π(i))`. This is the **linear assignment problem**. Equivalently, minimize `trace(C Xᵀ)` over permutation matrices `X`.

### 3.2 Link to optimal transport
Optimal transport (Monge 1781, Kantorovich 1942) moves one distribution onto another at minimal total cost. When both distributions are **uniform on `P` points**, the Kantorovich problem reduces exactly to the assignment problem (Birkhoff–von Neumann: extreme points of the doubly-stochastic polytope are permutation matrices).

With quadratic cost, **Brenier's theorem** says the optimal map is the gradient of a convex function. Two consequences used here:
- In **1-D** the optimal map is simply **sorting**: smallest to smallest, and so on. This is why sorting by luminance is a valid baseline and also the building block of sliced methods.
- Optimal maps do not cross: if two pixels are matched crosswise there is a swap that reduces cost. This is the justification for swap-based refinement (§5).

### 3.3 Our cost
```
c(i, j) = ‖Lab(S_j) − Lab(T_i)‖²  +  λ · ‖pos_j − pos_i‖² / N²
```
The spatial term is normalized by `N²` so `λ` has the same meaning at every resolution.

### 3.4 Exact algorithms
| Algorithm | Complexity | Notes |
|---|---|---|
| Hungarian (Kuhn–Munkres) | `O(P³)` | Textbook, pedagogical |
| Jonker–Volgenant (LAPJV) | `O(P³)` worst, fast in practice | What SciPy's `linear_sum_assignment` uses (a modified shortest-augmenting-path variant) |
| Auction (Bertsekas) | `O(P² log(P·C)/ε)`-ish | Parallel-friendly, GPU-suited |

At `P = 16 384` the dense matrix alone is about 1 GB (float32) and the solve takes many minutes, so exact solving is **limited to small grids and small tiles**. We use it as **ground truth** and as a **local polisher**.

**Study:** linear programming duality (the assignment dual gives intuition for *why* algorithms work), shortest augmenting paths, Birkhoff polytope. **Tools:** `scipy.optimize.linear_sum_assignment`.

## 4. Fast initial guess: recursive median split

### 4.1 Idea
Build a feature vector per pixel, `f = (L, a, b, w·x, w·y)`, where `w` encodes `λ`. Then match the two sets recursively:

1. Take the union of source and target feature vectors in the current group (equal sizes).
2. Pick the axis with the **largest variance**.
3. Sort both sets along that axis and split each at the **median** into two halves.
4. Match lower half with lower half and upper with upper, then recurse.
5. At size ≤ 8, solve exactly.

Because both halves are always equal in size, the result is **automatically a bijection**.

### 4.2 Why it works
It is a discrete, multi-dimensional generalization of the 1-D sorting optimum. Cutting along the direction of greatest spread keeps matched pixels close in feature space. It is a classical divide-and-conquer approach related to **k-d trees** and the **median-cut** color quantization algorithm.

### 4.3 Cost and quality
`O(P log P)` using partial selection (`np.argpartition`), about `O(P log² P)` with full sorts. It is not optimal because early splits can force poor matches near boundaries, which is exactly what refinement repairs.

**Study:** k-d trees, median-cut quantization, divide and conquer, quickselect.

## 5. Refinement: parallel swaps

### 5.1 The 2-swap move
Take two target slots `i` and `k` holding source pixels `a = π(i)` and `b = π(k)`. Swapping them changes total cost by

```
Δ = c(i,b) + c(k,a) − c(i,a) − c(k,b)
```

Accept the swap if `Δ < 0`. Each accepted swap strictly lowers the cost, so the process converges to a **2-opt local optimum**. Note that the cost of a pair is computed **on the fly from the features**, with no matrix.

### 5.2 Choosing candidates
Random pairs across the image rarely help. Better: pick `k` near `i` (within radius `r`) and **shrink `r`** over time (a simple annealing schedule). Optionally accept some uphill moves early (simulated annealing) to escape local minima.

### 5.3 Parallelism
Swaps conflict only if two threads touch the same slot. Each round: randomly permute the slots, pair them up **disjointly**, and process all pairs in parallel (`prange`). No locks, no data races, and the result is deterministic given the seed.

### 5.4 Cost
One round is `O(P)` work and scales near-linearly with cores. Hundreds of rounds are typical.

**Study:** local search, 2-opt (also used in the traveling salesman problem), simulated annealing, race conditions and data parallelism.

## 6. Other solvers worth knowing

| Method | Idea | Why mention it |
|---|---|---|
| **Sinkhorn / entropic OT** | Add an entropy term, then alternate row/column normalization of `exp(−C/ε)` | GPU friendly; yields a soft plan that must be rounded to a permutation; the matrix is dense unless structured |
| **Sliced Wasserstein / sliced assignment** | Project features on random 1-D directions, sort, and average the resulting maps | Uses only sorting; a cheap, parallel alternative |
| **Auction algorithm** | Pixels "bid" for slots with prices | Naturally parallel; strong GPU candidate (stretch goal) |
| **Min-cost flow on a sparse graph** | Restrict each pixel to `k` candidate slots | Exact on the restricted graph, scales better than dense |

The project implements §4 + §5 first, because they are simple, fast and good enough. The others are future experiments, benchmarked against the baseline.

## 7. Performance engineering: NumPy and Numba

### 7.1 NumPy
Vectorization pushes loops into C. Know: array broadcasting, memory layout (C-contiguous vs strided), `float32` vs `float64`, avoiding temporary arrays, `argsort` / `argpartition`.

### 7.2 Numba
A **JIT compiler** (built on LLVM) that turns numeric Python into machine code on first call.
- `@njit(parallel=True, cache=True, fastmath=True)`: compile in nopython mode, multi-thread `prange` loops, cache to disk.
- Restrictions: typed arrays and simple Python only; no arbitrary objects.
- Parallelism uses a thread pool, so the swap solver scales with core count.

**Study:** how a JIT works, CPU caches and memory bandwidth, SIMD, Amdahl's law. **Measure:** always benchmark (`time.perf_counter`, `pytest-benchmark`) and profile before optimizing.

## 8. Animation mathematics

Each pixel has start `p0` (its source position) and end `p1` (its assigned slot).

### 8.1 Interpolation
```
p(t) = p0 + (p1 − p0) · e(t')      t' = clamp((t − delayᵢ) / duration, 0, 1)
```
`e` is an **easing function**, usually *smoothstep* `e(u) = 3u² − 2u³` or smootherstep `6u⁵ − 15u⁴ + 10u³`, which starts and ends with zero velocity.

### 8.2 Per-pixel delay
`delayᵢ = k · ‖p1 − p0‖` (or based on the distance from a center) makes pixels leave in a **wave** instead of all at once, which is what makes the effect feel organic.

### 8.3 Organic motion
Add a perturbation `noise(p, t) · sin(π t')` that vanishes at both ends, so pixels start and finish exactly on their grid. **Curl noise** (the curl of a noise field) gives divergence-free, swirling motion that looks like fluid. Study: Perlin / simplex noise, curl of a vector field.

### 8.4 Physics alternative
Simulate springs: `a = k(p1 − p) − c·v`, integrated with semi-implicit Euler. More lifelike, but harder to control the duration. Study: ODE integration, damping.

## 9. Rendering: CPU and GPU

### 9.1 CPU path (reference)
For each frame, compute positions, round to integer pixels and write colors into a frame buffer. Because pixels move to non-integer positions, the naive scatter leaves **holes**. Fixes: splat each point on a 2×2 neighborhood, or draw with bilinear weights. Simple, portable, but slow for big grids.

### 9.2 GPU path
**OpenGL** renders each source pixel as a **point sprite**. The **vertex shader** does the interpolation of §8, so the CPU only uploads two buffers once:

```glsl
// vertex shader (sketch)
in vec2 p0; in vec2 p1; in vec3 color; in float delay;
uniform float t;
void main() {
    float u = clamp((t - delay) / duration, 0.0, 1.0);
    float e = u*u*(3.0 - 2.0*u);
    vec2 p = mix(p0, p1, e);
    gl_Position = vec4(p, 0.0, 1.0);
    gl_PointSize = pointSize;
    vColor = color;
}
```
`moderngl` is a thin, Pythonic OpenGL wrapper; the window comes from `pygame` or `glfw`. This is how a million points animate at 60 fps.

**Study:** the graphics pipeline (vertex → rasterization → fragment), GLSL, VBOs and VAOs, uniforms, point sprites, framebuffers (for offscreen rendering used by video export).

## 10. Video and GIF export

- **imageio** collects frames and talks to **FFmpeg** (bundled by `imageio-ffmpeg`).
- MP4 (H.264, `yuv420p` for broad compatibility) is small and smooth. GIF is limited to 256 colors with dithering and is large, but shareable everywhere.
- Offscreen GPU rendering reads frames back with `framebuffer.read()`.

**Study:** video codecs, chroma subsampling, GIF palette quantization.

## 11. Engineering tooling

| Tool | Role |
|---|---|
| `uv` | Fast dependency and virtualenv manager, lockfile |
| `pyproject.toml` | Single packaging and tool config file (PEP 621) |
| `pytest` | Tests, including property tests (the output is always a permutation) |
| `hypothesis` | Optional property-based testing |
| `ruff` | Lint and format |
| `mypy` | Static typing |
| GitHub Actions | CI on every push |
| Typer | CLI built from type hints |

**What to test:** color round-trips, permutation validity, solver cost never worse than the baseline, deterministic output with a fixed seed, last animation frame equals the permuted image.

## 12. Suggested study path

1. Color: gamma, XYZ, CIELAB. Reproduce the conversion in NumPy by hand.
2. The assignment problem: solve a tiny case by hand, then with SciPy; read about duality.
3. 1-D optimal transport = sorting. Implement the luminance baseline.
4. Median split and k-d trees.
5. Local search and 2-opt; why disjoint pairs avoid races.
6. NumPy performance, then Numba parallelism.
7. Easing and noise, then the OpenGL pipeline and GLSL.

**Good starting references:** *Computational Optimal Transport* (Peyré and Cuturi, free online); Ottosson, "A perceptual color space for image processing" (OKLab); Bertsekas on the auction algorithm; the Numba and moderngl documentation; *The Book of Shaders* for GLSL intuition.
