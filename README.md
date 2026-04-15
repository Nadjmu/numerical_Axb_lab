# Numerical Ax = b Lab

An interactive Streamlit application for studying numerical linear algebra algorithms, with emphasis on **stability analysis**, **conditioning**, and **numerical error characterisation** for the linear system **Ax = b**.

Supports both **CPU** (NumPy / SciPy / LAPACK) and **GPU** (CuPy / cuSolver) execution, switchable via a toggle in the sidebar. Matrices and vectors can be generated internally or **imported from external `.npy` files**.

Part of the research project: *High Performance Data Reduction and Numerical Error Analysis for Memory Constrained Computational Physics Simulations.*

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Installation & Running](#2-installation--running)
3. [GPU Setup](#3-gpu-setup)
4. [Connecting via SSH](#4-connecting-via-ssh)
5. [Importing Custom Matrices](#5-importing-custom-matrices)
6. [Libraries Used](#6-libraries-used)
7. [Experiment Design](#7-experiment-design)
8. [Matrix Types](#8-matrix-types)
9. [Sparsity Structures](#9-sparsity-structures)
10. [Solvers](#10-solvers)
11. [Analysis Metrics](#11-analysis-metrics)
12. [High-Precision Residual](#12-high-precision-residual)
13. [UI Layout](#13-ui-layout)

---

## 1. Project Structure

```
numerical_lab/
├── app.py                    # Main Streamlit entry point, experiment orchestration
├── requirements.txt          # Python dependencies
├── GPU_SETUP.md              # Detailed GPU installation guide
├── core/
│   ├── __init__.py
│   ├── device.py             # CPU/GPU abstraction layer
│   ├── problem_creation.py   # Matrix/vector generation, import, perturbation
│   ├── solvers.py            # All solver implementations + SOLVERS registry
│   └── analysis.py           # Stability metrics, high-precision residual computation
└── ui/
    ├── __init__.py
    ├── problem_ui.py         # Sidebar: device selector, import, matrix type, sweep/compare
    ├── solver_ui.py          # Sidebar: solver selection, norm, solver compare
    └── analysis_ui.py        # Heatmaps, dataframes, metric plots
```

### Key design principles

**Matrix TYPE** defines the algebraic structure of the entries (Toeplitz, Hilbert, etc.).  
**Matrix STRUCTURE** defines the sparsity pattern applied on top (tridiagonal, banded, etc.).  
These are orthogonal concepts enforced by the `COMPATIBILITY` dict in `problem_creation.py`.

**CPU / GPU** is an orthogonal axis controlled by `core/device.py`. The rest of the codebase uses `xp = get_array_module(use_gpu)` in place of `import numpy as np`, so the same algorithmic code runs on both devices.

**Import** is an orthogonal axis for A and b independently. Imported arrays bypass type/structure/size controls but still go through the perturbation pipeline.

---

## 2. Installation & Running

### CPU-only (no GPU required)

```bash
conda create -n nla_lab python=3.11
conda activate nla_lab
conda install -c conda-forge numpy scipy pandas matplotlib seaborn mpmath streamlit
streamlit run app.py
```

### CPU + GPU

```bash
conda create -n nla_lab python=3.11
conda activate nla_lab
conda install -c conda-forge numpy scipy pandas matplotlib seaborn mpmath streamlit cupy
streamlit run app.py
```

> **Why conda over pip?**  
> CuPy must be built against the exact CUDA version on the machine. Installing everything via `conda -c conda-forge` in a single command lets the solver pick mutually compatible versions automatically. Mixing `pip install` and `conda install` in the same environment can cause NumPy version conflicts.

---

## 3. GPU Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA toolkit installed (`nvidia-smi` shows the version)

### Check your CUDA version

```bash
nvidia-smi          # top-right corner: "CUDA Version: XX.X"
```

### Install CuPy

```bash
# Via conda (recommended — auto-detects CUDA version)
conda install -c conda-forge cupy

# Or via pip if you know your exact CUDA version
pip install cupy-cuda11x   # CUDA 11.x
pip install cupy-cuda12x   # CUDA 12.x
```

### Verify

```bash
python -c "import cupy; print(cupy.__version__)"
python -c "import cupy; cupy.show_config()"
```

### Using the GPU toggle

Once CuPy is installed a **⚙️ Device** selector appears at the top of the sidebar with a **CPU / GPU** radio button. Switching to GPU:

- transfers matrices to the GPU after generation (or after import)
- runs all solver operations (factorisation, iterative solve) on the GPU
- pulls results back to CPU for display and high-precision residual computation

If CuPy is not installed the GPU option is disabled with a clear install message — the app always works in CPU-only mode.

### GPU solver mapping

| CPU (NumPy / SciPy)                        | GPU (CuPy / cuPyx)                            |
|--------------------------------------------|-----------------------------------------------|
| `numpy.linalg.svd`                         | `cupy.linalg.svd`                             |
| `numpy.linalg.qr`                          | `cupy.linalg.qr`                              |
| `scipy.linalg.lu_factor` / `lu_solve`      | `cupyx.scipy.linalg.lu_factor` / `lu_solve`   |
| `scipy.linalg.cho_factor` / `cho_solve`    | `cupyx.scipy.linalg.cho_factor` / `cho_solve` |
| `scipy.linalg.solve_triangular`            | `cupyx.scipy.linalg.solve_triangular`         |
| `scipy.sparse.csc_matrix` + `splu`         | `cupyx.scipy.sparse.csc_matrix` + `splu`      |
| `scipy.sparse.linalg.gmres`                | `cupyx.scipy.sparse.linalg.gmres`             |
| `scipy.sparse.linalg.cg`                   | `cupyx.scipy.sparse.linalg.cg`                |

### Performance notes

- GPU speedup is most noticeable for **large dense matrices** (m ≥ 500) and iterative solvers (GMRES, CG)
- For **small matrices** (m < 100) the CPU is often faster due to GPU kernel launch overhead
- The high-precision residual (mpmath / float128) always runs on the CPU regardless of device choice — accuracy is identical on both devices

---

## 4. Connecting via SSH

When running on a remote GPU server, Streamlit's browser UI is accessed by forwarding a port over SSH.

### Start the app on the server

```bash
streamlit run app.py --server.port 8504
```

### Forward the port from your local machine

```bash
ssh -4 -L 8504:localhost:8504 username@server.address
```

The `-4` flag forces IPv4 and avoids the common `bind [::1]:XXXX: Cannot assign requested address` error.

### Open in your browser

```
http://localhost:8504
```

If port 8504 is already in use on your local machine, pick any free port (e.g. 8505) and use it consistently in both commands.

---

## 5. Importing Custom Matrices

A and b can be imported independently from external `.npy` files. This lets you analyse any matrix produced outside the app — from a simulation, a finite element assembly, a dataset, or a hand-crafted example.

### How to create a .npy file

```python
import numpy as np

# Any matrix you want to analyse
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=np.float64)

b = np.array([1.0, 2.0, 3.0])

np.save("A.npy", A)   # upload this in the sidebar
np.save("b.npy", b)   # upload this in the sidebar
```

### Using import in the sidebar

Two checkboxes appear at the top of the **Matrix A** and **Vector b** sections:

- **Import A from .npy file** — when checked, a file uploader replaces the type / size / structure / seed controls. Shape, dtype, and sparsity are all read from the file. The app shows a confirmation with shape, dtype, and non-zero count.
- **Import b from .npy file** — when checked, a file uploader replaces the dtype control. Length and dtype are read from the file.

Both are independent — you can import A but generate b randomly, import b but generate A from a type, or import both.

### What changes when importing

| Feature | Generated matrix | Imported matrix |
|---|---|---|
| Type / seed / size controls | Shown | Hidden |
| Structure selectbox | Shown | Hidden |
| Hermitian / PD checkboxes | Shown | Hidden |
| Perturbation | Uses selected structure mask | Inherits sparsity pattern of imported array |
| GPU transfer | After generation | After import, before solve |
| dtype cast | Via dtype selector | Via dtype selector (cast applied after load) |

### Perturbation on imported matrices

When an imported A is perturbed, the noise is masked to the **non-zero pattern of the original imported matrix**. This ensures the perturbation respects the structure of the matrix — a sparse FEM matrix stays sparse after perturbation — without requiring the user to manually specify a structure.

Implemented via `sparsity_mask(A)` in `core/problem_creation.py`, which returns a boolean array of non-zero entries passed as `custom_mask` to `apply_perturbation`.

### Requirements for imported arrays

- File format: `.npy` (saved with `numpy.save`)
- A must be 2-D and square
- b must be 1-D with length matching A's row count
- All values must be finite (no NaN or Inf)
- Any numeric dtype is accepted and cast to the selected dtype after loading

---

## 6. Libraries Used

| Library | Purpose | Docs |
|---|---|---|
| [NumPy](https://numpy.org/doc/stable/) | Matrix construction, dense linear algebra, float128 | [numpy.org](https://numpy.org/doc/stable/) |
| [SciPy](https://docs.scipy.org/doc/scipy/) | LAPACK wrappers, sparse LU, iterative solvers | [docs.scipy.org](https://docs.scipy.org/doc/scipy/) |
| [CuPy](https://cupy.dev/) | GPU array library — drop-in NumPy/SciPy replacement | [cupy.dev](https://cupy.dev/) |
| [cuPyx](https://docs.cupy.dev/en/stable/reference/scipy.html) | SciPy-compatible GPU routines (linalg, sparse) | [docs.cupy.dev](https://docs.cupy.dev/) |
| [Streamlit](https://docs.streamlit.io/) | Web UI framework | [docs.streamlit.io](https://docs.streamlit.io/) |
| [mpmath](https://mpmath.org/doc/current/) | Arbitrary-precision arithmetic for residual | [mpmath.org](https://mpmath.org/doc/current/) |
| [Pandas](https://pandas.pydata.org/docs/) | DataFrames for entry display | [pandas.pydata.org](https://pandas.pydata.org/docs/) |
| [Matplotlib](https://matplotlib.org/stable/) | Plot rendering backend | [matplotlib.org](https://matplotlib.org/stable/) |
| [Seaborn](https://seaborn.pydata.org/) | Heatmaps with diverging palettes | [seaborn.pydata.org](https://seaborn.pydata.org/) |

### Key functions worth knowing

- [`numpy.linalg.cond`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html) — condition number κ(A)
- [`numpy.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) — singular value decomposition
- [`numpy.linalg.qr`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html) — QR decomposition (Householder via LAPACK)
- [`scipy.linalg.lu_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html) / [`lu_solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html) — dense LU (LAPACK `dgetrf`/`dgetrs`)
- [`scipy.linalg.cho_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_factor.html) / [`cho_solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve.html) — Cholesky (LAPACK `dpotrf`/`dpotrs`)
- [`scipy.sparse.csc_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html) — Compressed Sparse Column format
- [`scipy.sparse.linalg.splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html) — SuperLU sparse factorisation with COLAMD reordering
- [`scipy.sparse.linalg.gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) — Generalised Minimum Residual
- [`scipy.sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html) — Conjugate Gradient
- [`scipy.linalg.solve_triangular`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html) — triangular back-substitution
- [`mpmath.matrix`](https://mpmath.org/doc/current/matrices.html) — arbitrary-precision matrix type
- [`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html) — annotated heatmaps
- [`seaborn.diverging_palette`](https://seaborn.pydata.org/generated/seaborn.diverging_palette.html) — diverging colormap (blue=negative, red=positive)
- [`cupy.linalg.svd`](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html) — GPU SVD
- [`cupy.linalg.qr`](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.qr.html) — GPU QR
- [`cupyx.scipy.linalg.lu_factor`](https://docs.cupy.dev/en/stable/reference/scipy_linalg.html) — GPU LU
- [`cupyx.scipy.sparse.linalg.gmres`](https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html) — GPU GMRES

---

## 7. Experiment Design

The app supports two axes of variation that produce multi-series, multi-instance experiments:

### Sweep axis (x-axis of plots)
Vary one parameter across instances within each series:
- **m** — matrix size (generated matrices only)
- **Perturbation order on A** — magnitude of noise added to A (10^k · ‖A‖)
- **Perturbation order on b** — magnitude of noise added to b (10^k · ‖b‖)

### Compare axis (legend of plots)
Vary one parameter across series:
- **Matrix type** — e.g. compare Hilbert vs Toeplitz vs Random Gaussian
- **Structure** — e.g. compare Dense vs Tridiagonal vs Banded
- **Solver** — e.g. compare LU vs GMRES vs QR Householder

### Data model

```
session_state.series_list = [
    {
        "label":     str,           # legend entry
        "instances": [              # one per sweep combo
            {
                "A":           ndarray,   # CPU numpy (converted from GPU after solve)
                "b":           ndarray,
                "delta_A":     ndarray | None,
                "delta_b":     ndarray | None,
                "result":      dict,      # from solver (x always CPU numpy)
                "metrics":     dict,      # from stability_analysis
                "use_gpu":     bool,      # whether this instance ran on GPU
                "imported_A":  bool,      # whether A came from a .npy file
                "imported_b":  bool,      # whether b came from a .npy file
                ...
            }
        ]
    }
]
```

Sections 1–2 show one selected instance; Sections 3–4 plot all instances across all series.

---

## 8. Matrix Types

All types are square (m × m). The **type** determines entry values; the **structure** then applies a sparsity mask on top (except Random SPD which is Dense only).

Matrix generation always runs on the **CPU** regardless of device choice — the algebraically complex constructors (Toeplitz, Block-Toeplitz, Circulant) are loop-heavy and the H→D transfer cost is negligible. The performance-critical operations (factorisation, iterative solve) run on the GPU.

When a matrix is **imported**, type and structure controls are not shown — the array is used as-is.

### Random Gaussian
```
A[i,j] ~ N(0, 1)  i.i.d.
```
Expected condition number κ(A) ~ O(m). Standard benchmark matrix with no algebraic structure. Seed controls the random generator.

**Reference:** [numpy.random.Generator.standard_normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html)

---

### Hilbert
```
A[i,j] = 1 / (i + j + 1)     i, j = 0, 1, ..., m-1
```
Symmetric positive definite. Condition number grows as κ(A) ~ (3.5)^m — the canonical example of a severely ill-conditioned matrix. No seed (deterministic).

**Reference:** [Wikipedia — Hilbert matrix](https://en.wikipedia.org/wiki/Hilbert_matrix)

---

### Toeplitz
```
A[i,j] = f(|i - j|)
f(k) = cos(k + φ) · exp(−k / 4)     φ = seed-dependent phase
```
Constant along each diagonal. Appears in convolution operators, time-series analysis, and 1-D finite difference stencils.

**Reference:** [Wikipedia — Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix)

---

### Block-Toeplitz
```
A[br, bc] = B[bc − br]
```
where each block `B[k]` is an independent random Gaussian matrix of size `bs × bs`. param = block size. Appears in 2-D PDE discretisations and multi-channel signal processing.

**Reference:** [Wikipedia — Block Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix#Block_Toeplitz_matrices)

---

### Circulant
```
A[i,j] = c[(j − i) mod m]
c[k] = N(0,1) · exp(−k / (m/4))
```
Each row is a cyclic shift of the first row. Appears in problems with periodic boundary conditions.

**Reference:** [Wikipedia — Circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix)

---

### Random SPD
```
A = Q Λ Qᵀ
Q  = random orthogonal matrix  (QR of Gaussian)
Λ  = diag(logspace(0, k, m))  →  κ(A) = 10^k  exactly
```
Symmetric positive definite with a prescribed condition number. Compatible with Dense only.

**Reference:** [numpy.linalg.qr](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html)

---

### Diagonally Dominant
```
A[i,j] ~ N(0,1)   for i ≠ j
A[i,i]  = Σ_{j≠i} |A[i,j]| + margin
```
Gershgorin's circle theorem guarantees non-singularity. Appears in finite difference discretisations of elliptic PDEs.

**Reference:** [Wikipedia — Diagonally dominant matrix](https://en.wikipedia.org/wiki/Diagonally_dominant_matrix)

---

### Compatibility table

| Matrix type | Dense | Sparse Tridiagonal | Sparse Block-Tridiagonal | Sparse Banded |
|---|:---:|:---:|:---:|:---:|
| Random Gaussian | ✅ | ✅ | ✅ | ✅ |
| Hilbert | ✅ | ✅ | ✅ | ✅ |
| Toeplitz | ✅ | ✅ | ✅ | ✅ |
| Block-Toeplitz | ✅ | ✅ | ✅ | ✅ |
| Circulant | ✅ | ✅ | ✅ | ✅ |
| Random SPD | ✅ | ❌ | ❌ | ❌ |
| Diagonally Dominant | ✅ | ✅ | ✅ | ✅ |
| **Custom (imported)** | — | — | — | — |

---

## 9. Sparsity Structures

Structures are **pure sparsity masks** applied after matrix generation via element-wise multiplication with a boolean mask. Not applicable to imported matrices — their sparsity pattern is inherited directly from the array.

### Dense
No entries are zeroed. All m² entries are retained.

### Sparse Tridiagonal
Only diagonals −1, 0, +1 are kept. 3m − 2 non-zeros.

### Sparse Block-Tridiagonal
Main block-diagonal and two neighbouring blocks retained. param = block size.

### Sparse Banded
`num_diags` diagonals centred on the main diagonal. Tridiagonal is a special case with `num_diags = 3`.

**Reference:** [Wikipedia — Band matrix](https://en.wikipedia.org/wiki/Band_matrix)

---

## 10. Solvers

All solvers share the same interface and accept a `use_gpu` keyword:

```python
def solve_*(A: np.ndarray, b: np.ndarray, use_gpu: bool = False) -> dict:
    # returns:
    # {
    #   "x":       np.ndarray   computed solution
    #   "method":  str          human-readable name  (includes [GPU] or [CPU])
    #   "success": bool
    #   "message": str          status / diagnostic
    #   "Q":       np.ndarray   Q factor (QR solvers only)
    # }
```

If the input array is already a CuPy array the solver automatically runs on the GPU regardless of the `use_gpu` flag. All solvers work identically on generated and imported matrices.

### Solver comparison summary

| Solver | Requires | Cost | Backward stable | GPU support |
|---|---|---|:---:|:---:|
| SVD (Reduced/Full) | Any shape | O(m²n) | ✅ | ✅ |
| QR Householder | Any shape | O(m²n) | ✅ | ✅ |
| QR Classical GS | Any shape | O(mk²) | ❌ | ✅ |
| QR Modified GS | Any shape | O(mk²) | ⚠️ | ✅ |
| LU | Square | O(m³/3) | ✅ | ✅ |
| LU (Sparse) | Square | O(m·bw²) | ✅ | ✅ |
| Cholesky | Square SPD | O(m³/6) | ✅ | ✅ |
| GMRES | Square | O(mk)/iter | ✅ | ✅ |
| CG | Square SPD | O(nnz)/iter | ✅ | ✅ |

---

## 11. Analysis Metrics

All metrics are computed in `core/analysis.py` by `stability_analysis(A, b, x, norm_type)`. Works with both NumPy and CuPy input arrays, and equally with generated and imported matrices.

### Condition number κ(A)

```
κ_p(A) = ‖A‖_p · ‖A⁺‖_p
```

On GPU, computed via `cupy.linalg.svd` (σ_max / σ_min) since `cupy.linalg.cond` is not available.

### Residual ‖r‖

```
r = b − A x̃
```

Always computed at **high precision on the CPU** (see Section 12), regardless of which device was used for the solve.

### Forward Error Bound (FEB)

```
FEB = κ(A) · ‖r‖_p / ( ‖A‖_p · ‖x̃‖_p )
```

### Backward Error (BE)

```
BE = ‖r‖_p / ( ‖A‖_p · ‖x̃‖_p + ‖b‖_p )
```

### Orthogonality error (QR solvers only)

```
orth_err = ‖QᵀQ − I‖_F
```

---

## 12. High-Precision Residual

The residual `r = b − Ax̃` is computed at the highest available precision to avoid catastrophic cancellation. This always runs on the **CPU**, even when the solve ran on the GPU — GPU arrays are transferred via `to_numpy()` first.

```
if mpmath available and m ≤ 200:
    compute using mpmath at 50 decimal places  (~166 bits)
elif float128 is genuinely wider than float64:
    compute using numpy.float128  (80-bit extended, x86 Linux only)
else:
    compute using float64  (fallback)
```

There is **no accuracy regression** on the GPU path or the import path — the high-precision residual is identical regardless of how the matrix was obtained or which device was used for the solve.

---

## 13. UI Layout

```
Sidebar                          Main area
───────────────────              ─────────────────────────────────────────────
⚙️ Device                        1. Problem creation (A, b)
  CPU / GPU toggle                  Device: 🔵 CPU / 🟢 GPU
                                    A: 🔧 generated / 📂 imported
Matrix A                            b: 🔧 generated / 📂 imported
  [ ] Import A from .npy            [A: shape dtype nnz density memory]
      └─ file uploader              [b: dtype memory               ]
  ── or, if not importing ──        A entries / heatmap  |  b entries / heatmap
  Type                              ΔA / ΔA heatmap      |  Δb (if perturbed)
  Seed
  Type param (if applicable)     2. x̃ via <solver>  [CPU] or [GPU]
  Size m  [Sweep checkbox]          x entries / heatmap
  Structure
  Hermitian / PD checkboxes      3. Problem-specific sensitivity metrics
  dtype                             κ(A) gauge or line plot | — | —

Perturbation                     4. Solution quality metrics
  Perturb A  [order + Sweep]        ‖r‖ gauge/line | FEB gauge/line | BE gauge/line
  Perturb b  [order + Sweep]
                                 5. Solver behaviour metrics  (coming soon)
Vector b
  [ ] Import b from .npy         6. Structural metrics  (coming soon)
      └─ file uploader
  ── or, if not importing ──     7. Summary  (coming soon)
  dtype
                                 8. Save results  (coming soon)
Compare
  None / Matrix type / Structure

Solver
  Method
  Compare solvers checkbox

Norm  (2 / inf)

[Run Experiment]
```
