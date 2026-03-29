# Numerical Ax = b Lab

An interactive Streamlit application for studying numerical linear algebra algorithms, with emphasis on **stability analysis**, **conditioning**, and **numerical error characterisation** for the linear system **Ax = b**.

Part of the research project: *High Performance Data Reduction and Numerical Error Analysis for Memory Constrained Computational Physics Simulations.*

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Installation & Running](#2-installation--running)
3. [Libraries Used](#3-libraries-used)
4. [Experiment Design](#4-experiment-design)
5. [Matrix Types](#5-matrix-types)
6. [Sparsity Structures](#6-sparsity-structures)
7. [Solvers](#7-solvers)
8. [Analysis Metrics](#8-analysis-metrics)
9. [High-Precision Residual](#9-high-precision-residual)
10. [UI Layout](#10-ui-layout)

---

## 1. Project Structure

```
numerical_lab/
├── app.py                    # Main Streamlit entry point, experiment orchestration
├── requirements.txt          # Python dependencies
├── core/
│   ├── problem_creation.py   # Matrix/vector generation, perturbation, compatibility table
│   ├── solvers.py            # All solver implementations + SOLVERS registry
│   └── analysis.py          # Stability metrics, high-precision residual computation
└── ui/
    ├── problem_ui.py         # Sidebar: matrix type, structure, sweep/compare axes
    ├── solver_ui.py          # Sidebar: solver selection, norm, solver compare
    └── analysis_ui.py        # Heatmaps, dataframes, metric plots
```

### Key design principle

**Matrix TYPE** defines the algebraic structure of the entries (Toeplitz, Hilbert, etc.).  
**Matrix STRUCTURE** defines the sparsity pattern applied on top (tridiagonal, banded, etc.).  
These are orthogonal concepts enforced by the `COMPATIBILITY` dict in `problem_creation.py`.

---

## 2. Installation & Running

```bash
conda create -n nla_lab python=3.11
conda activate nla_lab
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt`:
```
numpy>=1.24.0
scipy>=1.10.0
streamlit>=1.28.0
pandas>=2.0.0
matplotlib>=3.7.0
mpmath>=1.3.0
seaborn
```

---

## 3. Libraries Used

| Library | Purpose | Docs |
|---|---|---|
| [NumPy](https://numpy.org/doc/stable/) | Matrix construction, dense linear algebra, float128 | [numpy.org](https://numpy.org/doc/stable/) |
| [SciPy](https://docs.scipy.org/doc/scipy/) | LAPACK wrappers, sparse LU, iterative solvers | [docs.scipy.org](https://docs.scipy.org/doc/scipy/) |
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

---

## 4. Experiment Design

The app supports two axes of variation that produce multi-series, multi-instance experiments:

### Sweep axis (x-axis of plots)
Vary one parameter across instances within each series:
- **m** — matrix size
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
                "A":          ndarray,
                "b":          ndarray,
                "delta_A":    ndarray | None,
                "delta_b":    ndarray | None,
                "result":     dict,   # from solver
                "metrics":    dict,   # from stability_analysis
                ...
            }
        ]
    }
]
```

Sections 1–2 show one selected instance; Sections 3–4 plot all instances across all series.

---

## 5. Matrix Types

All types are square (m × m). The **type** determines entry values; the **structure** then applies a sparsity mask on top (except Random SPD which is Dense only).

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
Constant along each diagonal — the defining property of a Toeplitz matrix. Built directly from the first row and column, not by post-processing. Appears in convolution operators, time-series analysis, and 1-D finite difference stencils.

**Reference:** [Wikipedia — Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix)

---

### Block-Toeplitz
```
A[br, bc] = B[bc − br]
```
where each block `B[k]` is an **independent** random Gaussian matrix of size `bs × bs`, and `br`, `bc` are block-row and block-column indices. The block size `bs` is set via the `type_param` slider.

This is the mathematically correct direct construction — each offset `d = bc − br` gets its own independently drawn block, then the matrix is tiled. Appears in 2-D PDE discretisations (where each block corresponds to a 1-D slice) and multi-channel signal processing.

**Reference:** [Wikipedia — Block Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix#Block_Toeplitz_matrices)

---

### Circulant
```
A[i,j] = c[(j − i) mod m]
c[k] = N(0,1) · exp(−k / (m/4))     (decaying Gaussian envelope)
```
Each row is a cyclic shift of the first row. Eigenvalues are the DFT of the first row, so circulant systems can be solved in O(m log m) via FFT. Appears in problems with periodic boundary conditions.

**Reference:** [Wikipedia — Circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix) | [numpy.fft](https://numpy.org/doc/stable/reference/routines.fft.html)

---

### Random SPD
```
A = Q Λ Qᵀ
Q  = random orthogonal matrix  (QR of Gaussian)
Λ  = diag(logspace(0, k, m))  →  κ(A) = 10^k  exactly
```
Symmetric positive definite with a prescribed condition number κ(A) = 10^k, where k is set by the `log₁₀(κ_target)` slider (1–16). Compatible with Dense only — sparse zeroing destroys positive definiteness.

**Reference:** [numpy.linalg.qr](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html) | [numpy.logspace](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html)

---

### Diagonally Dominant
```
A[i,j] ~ N(0,1)   for i ≠ j
A[i,i]  = Σ_{j≠i} |A[i,j]| + margin
```
The diagonal entry in each row exceeds the sum of absolute off-diagonal entries by a fixed `margin` (set via slider, default 1). **Gershgorin's circle theorem** guarantees non-singularity regardless of sparsity pattern applied. Appears in finite difference discretisations of elliptic PDEs.

**Reference:** [Wikipedia — Diagonally dominant matrix](https://en.wikipedia.org/wiki/Diagonally_dominant_matrix) | [Gershgorin circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)

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

---

## 6. Sparsity Structures

Structures are **pure sparsity masks** — they zero out entries outside the pattern, leaving the remaining entries from the chosen matrix type unchanged. Applied after matrix generation via element-wise multiplication with a boolean mask.

### Dense
No entries are zeroed. All m² entries are retained. Default structure.

---

### Sparse Tridiagonal
```
A[i,j] = 0   if  |i − j| > 1
```
Only the main diagonal and the two neighbouring diagonals (−1 and +1) are kept. For an m × m matrix: 3m − 2 non-zeros. Arises naturally in 1-D finite difference schemes (e.g. the 1-D Laplacian with second-order central differences).

---

### Sparse Block-Tridiagonal
```
A[br, bc] = 0   if  |br − bc| > 1
```
The matrix is partitioned into blocks of size `bs × bs`. Only the main block-diagonal and its two neighbours are retained. Non-zeros: approximately (3·(m/bs) − 2) · bs² = 3m·bs − 2·bs². Arises in 2-D finite difference schemes where each block-row corresponds to a row of grid points.

Block size slider range: 1 (near-tridiagonal) to m (one dense block).

---

### Sparse Banded
```
A[i,j] = 0   if  |i − j| > (num_diags − 1) / 2
```
Keeps `num_diags` diagonals centred on the main diagonal. Tridiagonal is a special case with `num_diags = 3`; pentadiagonal uses `num_diags = 5`. The bandwidth parameter controls the number of diagonals retained (odd values only, stepped by 2 in the UI).

**Reference:** [Wikipedia — Band matrix](https://en.wikipedia.org/wiki/Band_matrix)

---

## 7. Solvers

All solvers share the same interface:
```python
def solve_*(A: np.ndarray, b: np.ndarray) -> dict:
    # returns:
    # {
    #   "x":       np.ndarray   computed solution
    #   "method":  str          human-readable name
    #   "success": bool
    #   "message": str          status / diagnostic
    #   "Q":       np.ndarray   Q factor (QR solvers only)
    # }
```

---

### SVD (Reduced)

Computes the **thin SVD** A = UΣVᵀ where U is m×k, Σ is k×k, Vᵀ is k×n, k = min(m,n). The solution is the **pseudoinverse**:

```
x = V Σ⁺ Uᵀ b
```

Singular values below `ε · max(m,n) · σ_max` are treated as zero (numerical rank truncation), giving the **minimum-norm least-squares solution**.

- **Stability:** Optimal — backward error ≈ ε_mach always
- **Cost:** O(m²n) — most expensive solver
- **Works for:** Any shape, rank-deficient matrices

**NumPy docs:** [`numpy.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)

---

### SVD (Full)

Same as SVD (Reduced) but computes the full unitary factor U (m×m) and full Vᵀ (n×n). Numerically identical result for x. Useful for inspecting the full singular value spectrum.

**NumPy docs:** [`numpy.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) with `full_matrices=True`

---

### QR Householder (Reduced)

Computes the **thin QR decomposition** A = QR via LAPACK Householder reflections, then solves by back-substitution:

```
A = QR   →   Ax = b   →   QRx = b   →   Rx = Qᵀb
```

Q is m×k, R is k×k (k = min(m,n)). Backward stable: orthogonality error ‖QᵀQ − I‖_F ≈ ε_mach always, because Householder reflections are orthogonal transformations applied in finite precision with no accumulation of rounding errors.

- **Stability:** Backward stable
- **Cost:** O(m²n − n³/3) for overdetermined systems
- **Works for:** Any shape

**NumPy docs:** [`numpy.linalg.qr`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html) with `mode='reduced'` | **SciPy docs:** [`scipy.linalg.solve_triangular`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html)

---

### QR Householder (Full)

Same algorithm as Reduced but Q is square (m×m). Numerically identical result for x. Useful when the full orthogonal factor is needed.

**NumPy docs:** [`numpy.linalg.qr`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html) with `mode='complete'`

---

### QR Classical Gram-Schmidt (CGS)

Manually computes the thin QR decomposition by orthogonalising each column against all previously computed q-vectors using the **original column a_j**:

```
for j = 0, ..., k-1:
    v = a_j
    for i = 0, ..., j-1:
        r[i,j] = qᵢᵀ · a_j        ← projection uses original a_j
        v      = v − r[i,j] · qᵢ
    r[j,j] = ‖v‖
    q_j    = v / r[j,j]
```

Numerically **unstable** for ill-conditioned A: all projections use the original `a_j` before any correction, so rounding errors from earlier projections are not cancelled. Orthogonality error ‖QᵀQ − I‖_F can reach O(1) for ill-conditioned problems. Included to demonstrate instability.

- **Stability:** Numerically unstable
- **Cost:** O(mk²)
- **Works for:** Any shape

**Reference:** [Trefethen & Bau, *Numerical Linear Algebra*, Lecture 7](https://people.maths.ox.ac.uk/trefethen/NLA.html)

---

### QR Modified Gram-Schmidt (MGS)

Manually computes thin QR but orthogonalises against **the current (partially corrected) vector** at each step:

```
for i = 0, ..., k-1:
    r[i,i] = ‖q_i‖;   q_i = q_i / r[i,i]
    for j = i+1, ..., k-1:
        r[i,j]  = qᵢᵀ · q_j      ← uses current q_j, not original
        q_j     = q_j − r[i,j] · qᵢ
```

Algebraically equivalent to CGS but numerically much better: rounding errors from earlier steps are progressively corrected. Orthogonality error ‖QᵀQ − I‖_F ≈ ε_mach · κ(A).

- **Stability:** Substantially better than CGS; worse than Householder for very ill-conditioned A
- **Cost:** O(mk²) — same as CGS
- **Works for:** Any shape

**Reference:** [Björck, *Numerics of Gram-Schmidt Orthogonalization*](https://doi.org/10.1016/0024-3795(94)90493-6)

---

### LU (partial pivoting)

Factors A as **PA = LU** where P is a permutation matrix, L is unit lower triangular, and U is upper triangular. Solves via forward substitution (Ly = Pb) then back substitution (Ux = y). Uses LAPACK's `dgetrf` and `dgetrs`.

Partial pivoting (row interchanges only) ensures |L[i,j]| ≤ 1, which controls element growth for most practical matrices.

- **Stability:** Backward stable for most matrices
- **Cost:** O(m³ / 3)
- **Requires:** Square matrix

**SciPy docs:** [`scipy.linalg.lu_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html) | [`scipy.linalg.lu_solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html)

---

### LU (Sparse / SuperLU)

Same LU factorisation but the matrix is first converted to **CSC (Compressed Sparse Column)** format, which stores only the non-zero entries and their row indices. SuperLU then applies **COLAMD (Column Approximate Minimum Degree)** reordering to permute columns before factorisation, minimising fill-in in L and U.

```
A  (dense m×m, stores m² floats)
↓  sp.csc_matrix(A)
A_csc  (sparse, stores only nnz floats + index arrays)
↓  splu(A_csc)   with COLAMD reordering
L, U  (sparse triangular factors)
↓  lu.solve(b)
x
```

For matrices with a sparse structure (tridiagonal, banded, block-tridiagonal) this is substantially faster than dense LU at large m. For dense matrices the conversion overhead makes it slightly slower.

- **Stability:** Same as dense LU (partial pivoting preserved)
- **Cost:** O(m · bandwidth²) for banded matrices; O(m³) worst case
- **Requires:** Square matrix

**SciPy docs:** [`scipy.sparse.csc_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html) | [`scipy.sparse.linalg.splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html) | [COLAMD paper](https://dl.acm.org/doi/10.1145/1024074.1024075)

---

### Cholesky

For symmetric positive definite A, factors as **A = LLᵀ** and solves via forward/backward substitution. Uses LAPACK's `dpotrf` and `dpotrs`. About twice as fast as LU because it exploits symmetry and only processes the lower triangle.

- **Stability:** Backward stable — even better than LU for SPD matrices (no pivoting needed)
- **Cost:** O(m³ / 6) — roughly 2× faster than LU
- **Requires:** Square, symmetric, positive definite matrix

**SciPy docs:** [`scipy.linalg.cho_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_factor.html) | [`scipy.linalg.cho_solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve.html)

---

### GMRES

**Generalised Minimum Residual** — an iterative Krylov subspace method that at iteration k finds the x_k minimising ‖b − Ax‖₂ over the Krylov subspace:

```
𝒦_k(A, b) = span{ b, Ab, A²b, ..., A^(k-1)b }

x_k = argmin_{x ∈ 𝒦_k} ‖b − Ax‖₂
```

Does not require symmetry or positive definiteness. Convergence depends on the spectral properties of A. Restarts after a fixed number of iterations to limit memory growth.

- **Stability:** Backward stable (minimises residual at each step)
- **Cost per iteration:** O(mk) — total cost depends on convergence
- **Requires:** Square matrix (non-singular)

**SciPy docs:** [`scipy.sparse.linalg.gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) | [Wikipedia — GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)

---

### CG (Conjugate Gradient)

**Conjugate Gradient** — an iterative Krylov method optimal for SPD matrices. At iteration k finds the x_k minimising the **A-norm of the error** over 𝒦_k(A, b):

```
‖e_k‖_A = √( (x_true − x_k)ᵀ A (x_true − x_k) )

Convergence bound:
‖e_k‖_A / ‖e_0‖_A  ≤  2 · ( (√κ − 1) / (√κ + 1) )^k
```

For well-conditioned matrices (small κ) converges in very few iterations. For ill-conditioned matrices (large κ) many iterations are required — this is the fundamental limitation of CG.

- **Stability:** Backward stable in exact arithmetic; finite-precision CG can lose conjugacy for ill-conditioned A
- **Cost per iteration:** O(m²) for dense A; O(nnz) for sparse A
- **Requires:** Square, symmetric, positive definite matrix

**SciPy docs:** [`scipy.sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html) | [Wikipedia — Conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

---

### Solver comparison summary

| Solver | Requires | Cost | Backward stable | Best for |
|---|---|---|:---:|---|
| SVD (Reduced/Full) | Any shape | O(m²n) | ✅ | Rank-deficient, ill-conditioned |
| QR Householder | Any shape | O(m²n) | ✅ | Overdetermined, general |
| QR Classical GS | Any shape | O(mk²) | ❌ | Demonstrating instability |
| QR Modified GS | Any shape | O(mk²) | ⚠️ | Better than CGS, worse than HH |
| LU | Square | O(m³/3) | ✅ | General square systems |
| LU (Sparse) | Square | O(m·bw²) | ✅ | Large sparse systems |
| Cholesky | Square SPD | O(m³/6) | ✅ | SPD systems (fastest direct) |
| GMRES | Square | O(mk)/iter | ✅ | Large non-symmetric systems |
| CG | Square SPD | O(nnz)/iter | ✅ | Large SPD systems |

---

## 8. Analysis Metrics

All metrics are computed in `core/analysis.py` by `stability_analysis(A, b, x, norm_type)`.

### Condition number κ(A)

```
κ_p(A) = ‖A‖_p · ‖A⁺‖_p
```

where A⁺ is the pseudoinverse and p is the chosen norm (2 or ∞).

- **‖A‖₂ = σ_max(A)** — largest singular value
- **‖A‖∞ = max_i Σ_j |A[i,j]|** — maximum absolute row sum

Interpretation: a relative perturbation of size ε in A or b can cause a relative error up to κ(A)·ε in x. Equivalently, log₁₀(κ(A)) digits of accuracy are lost.

**Reference:** [`numpy.linalg.cond`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html)

---

### Residual ‖r‖

```
r = b − A x̃
‖r‖_p  =  p-norm of r
```

Measures how well the computed solution x̃ satisfies the linear system. A small residual is **necessary but not sufficient** for accuracy — an ill-conditioned matrix can produce a small residual for a solution far from the true x.

Computed at **high precision** (see Section 9).

---

### Forward Error Bound (FEB)

```
FEB = κ(A) · ‖r‖_p / ( ‖A‖_p · ‖x̃‖_p )
```

An upper bound on the **relative forward error** ‖x_true − x̃‖ / ‖x_true‖. Derived from the standard perturbation bound for linear systems:

```
‖x_true − x̃‖     ‖r‖
──────────────  ≤  κ(A) · ──────────
  ‖x_true‖               ‖A‖ · ‖x̃‖
```

Note: this bound can be very pessimistic — κ(A) amplifies even a tiny residual into a large apparent error bound.

**Reference:** Trefethen & Bau, *Numerical Linear Algebra*, Theorem 12.1

---

### Backward Error (BE)

```
BE = ‖r‖_p / ( ‖A‖_p · ‖x̃‖_p + ‖b‖_p )
```

The **normwise backward error** — the size of the smallest relative perturbation [ΔA, Δb] such that x̃ is the exact solution of the perturbed system (A + ΔA)x̃ = b + Δb:

```
η(x̃) =  min { max(‖ΔA‖/‖A‖, ‖Δb‖/‖b‖) :  (A + ΔA)x̃ = b + Δb }

      =     ‖r‖
         ──────────────────
         ‖A‖ · ‖x̃‖ + ‖b‖
```

A solver is called **backward stable** if BE ≈ ε_mach. LU, QR Householder, Cholesky, GMRES, and CG are all backward stable.

**Reference:** Higham, *Accuracy and Stability of Numerical Algorithms*, Definition 7.1

---

### Orthogonality error (QR solvers only)

```
orth_err = ‖QᵀQ − I‖_F
```

Measures how far the computed Q factor is from being exactly orthogonal.

| Solver | Expected orth_err |
|---|---|
| QR Householder | O(ε_mach) |
| QR Modified GS | O(ε_mach · κ(A)) |
| QR Classical GS | O(1) for ill-conditioned A |

**Reference:** [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) with `ord='fro'`

---

## 9. High-Precision Residual

The residual `r = b − Ax̃` is the most numerically sensitive quantity in the analysis. When x̃ is a good solution, `Ax̃ ≈ b` and **catastrophic cancellation** occurs in the subtraction — computing r in float64 can give a result orders of magnitude wrong.

### Precision strategy (in `core/analysis.py`)

```
if mpmath available and m ≤ 200:
    compute r and ‖r‖ using mpmath at 50 decimal places  (~166 bits, ε ~ 10⁻⁵⁰)
elif float128 is genuinely wider than float64:
    compute r and ‖r‖ using numpy.float128  (80-bit extended, ε ≈ 1.1 × 10⁻¹⁹)
else:
    compute r and ‖r‖ using float64  (ε ≈ 2.2 × 10⁻¹⁶, fallback)
```

**Critical detail:** ‖r‖ is computed **inside** the high-precision context before casting r back to float64. Computing r in high precision but then calling `numpy.linalg.norm(r.astype(float64))` would re-introduce float64 rounding errors in the norm, defeating the purpose. Since FEB and BE both use ‖r‖ directly, they inherit the full precision of this computation.

### Benchmark (m = 500, Random Gaussian)

| Method | Time | Machine epsilon |
|---|---|---|
| float64 | 0.1 ms | 2.2 × 10⁻¹⁶ |
| float128 | 4.5 ms | 1.1 × 10⁻¹⁹ |
| mpmath 50 dps | 1500 ms | ~10⁻⁵⁰ |

### Concrete example: Hilbert 10×10 with LU

| Method | ‖r‖ | backward error |
|---|---|---|
| float64 | 2.32 × 10⁻⁴ | 2.68 × 10⁻¹⁴ (wrong) |
| mpmath 50 dps | 1.01 × 10⁻⁴ | 1.17 × 10⁻¹⁷ ≈ ε_mach ✅ |

LU is backward stable — the mpmath result correctly shows this. The float64 result falsely suggests instability due to cancellation in the residual computation.

**References:**
- [`mpmath.matrix`](https://mpmath.org/doc/current/matrices.html)
- [`numpy.float128`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float128)
- Higham, *Accuracy and Stability of Numerical Algorithms*, Chapter 3

---

## 10. UI Layout

```
Sidebar                          Main area
───────────────────              ─────────────────────────────────────────────
Matrix A                         1. Problem creation (A, b)
  Type                             [A: shape dtype nnz density memory]
  Seed                             [b: dtype memory               ]
  Type param (if applicable)       A entries / heatmap  |  b entries / heatmap
  Size m  [Sweep checkbox]         ΔA / ΔA heatmap      |  Δb (if perturbed)
  Structure
  Hermitian / PD checkboxes      2. x̃ via <solver>
  dtype                            x entries / heatmap

Perturbation                     3. Problem-specific sensitivity metrics
  Perturb A  [order + Sweep]       κ(A) gauge or line plot | — | —
  Perturb b  [order + Sweep]
                                 4. Solution quality metrics
Vector b                           ‖r‖ gauge/line | FEB gauge/line | BE gauge/line
  dtype
                                 5. Solver behaviour metrics  (coming soon)
Compare
  None / Matrix type / Structure 6. Structural metrics  (coming soon)

Solver                           7. Summary  (coming soon)
  Method
  Compare solvers checkbox       8. Save results  (coming soon)

Norm  (2 / inf)

[Run Experiment]
```

### Gauge plots (single instance)
Horizontal gradient bar (green=good, red=bad) with a vertical marker at the current value and an automated interpretation caption (success/info/warning).

### Line plots (multiple instances or series)
One coloured line per series. Up to 10 series via the `_SERIES_COLORS` palette. Points annotated with their log₁₀ value when ≤ 10 instances per series. Reference lines mark ε_mach and κ = 1.

### Heatmaps
- Diverging colourmap (blue=negative, red=positive, white=zero) via [`seaborn.diverging_palette(220, 20)`](https://seaborn.pydata.org/generated/seaborn.diverging_palette.html)
- Cell annotations for matrices ≤ 20×20
- Spy plot fallback for matrices > 150×150
- Vectors shown as single-column heatmaps aligned with the matrix display
