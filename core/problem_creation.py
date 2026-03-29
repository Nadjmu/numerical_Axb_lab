"""
core/problem_creation.py
========================
Generates the coefficient matrix A and right-hand side vector b for Ax = b.

Public API
----------
create_matrix(...)          -> np.ndarray
create_rhs(...)             -> np.ndarray
apply_perturbation(...)     -> np.ndarray
matrix_info(...)            -> dict
compatible_structures(type) -> list[str]

Design principle
----------------
Matrix TYPE defines the entry values and their algebraic structure.
Matrix STRUCTURE defines the sparsity pattern (which entries are zero).

These are orthogonal concepts — a Toeplitz matrix can be dense or tridiagonal;
a Random Gaussian matrix can be dense or banded.  The exception is Random SPD,
whose positive definiteness is destroyed by any sparsity pattern, so it is
restricted to Dense only.

Matrix types
------------
Random Gaussian      Entries i.i.d. N(0,1).  κ ~ O(n).  General benchmark.
Hilbert              H[i,j] = 1/(i+j+1).  κ ~ (3.5)^n.  Ill-conditioned SPD.
Toeplitz             Constant along each diagonal.  Built directly from a first
                     row/column drawn from a decaying cosine sequence.
                     Appears in convolution, time-series, 1-D FDM stencils.
Block-Toeplitz       Built directly from independent random blocks B_0,…,B_{p-1}
                     for the first block-row and B_{-1},…,B_{-(q-1)} for the
                     first block-column, then tiled: T_{ij} = B_{j-i}.
                     param = block size.  Appears in 2-D PDE discretisations
                     and multi-channel signal processing.
Circulant            Each row is a cyclic shift of the first.  Eigenvalues = DFT
                     of first row.  Appears in periodic BC problems.
Random SPD           A = QΛQᵀ with prescribed κ = 10^type_param.  Dense only.
Diagonally Dominant  aᵢᵢ = Σ|aᵢⱼ| + margin.  Nonsingular by Gershgorin.
                     Appears in FDM of elliptic PDEs.

Structures (sparsity patterns only)
------------------------------------
Dense                All entries retained.
Sparse Tridiagonal   Diagonals -1, 0, +1 only.
Sparse Block-Tridiagonal  Main block-diagonal + two neighbouring blocks.
Sparse Banded        param diagonals centred on main diagonal.

Compatibility
-------------
Random Gaussian      : all structures
Hilbert              : all structures
Toeplitz             : all structures
Block-Toeplitz       : all structures
Circulant            : all structures
Random SPD           : Dense only  (sparse zeroing destroys SPD)
Diagonally Dominant  : all structures
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility table
# ──────────────────────────────────────────────────────────────────────────────

_SPARSE_STRUCTURES = [
    "Dense",
    "Sparse Tridiagonal",
    "Sparse Block-Tridiagonal",
    "Sparse Banded",
]

COMPATIBILITY: dict[str, list[str]] = {
    "Random Gaussian":     _SPARSE_STRUCTURES,
    "Hilbert":             _SPARSE_STRUCTURES,
    "Toeplitz":            _SPARSE_STRUCTURES,
    "Block-Toeplitz":      _SPARSE_STRUCTURES,
    "Circulant":           _SPARSE_STRUCTURES,
    "Random SPD":          ["Dense"],
    "Diagonally Dominant": _SPARSE_STRUCTURES,
}


def compatible_structures(matrix_type: str) -> list[str]:
    """Return the list of valid sparsity structures for the given matrix type."""
    return COMPATIBILITY.get(matrix_type, _SPARSE_STRUCTURES)


# ──────────────────────────────────────────────────────────────────────────────
# Base matrix generators
# ──────────────────────────────────────────────────────────────────────────────

def _random_gaussian(m: int, n: int, seed: int = 42, **_) -> np.ndarray:
    """Entries i.i.d. N(0,1).  Expected κ ~ O(n)."""
    return np.random.default_rng(seed).standard_normal((m, n))


def _hilbert(m: int, n: int, **_) -> np.ndarray:
    """H[i,j] = 1/(i+j+1).  κ ~ (3.5)^min(m,n)."""
    i = np.arange(m, dtype=np.float64).reshape(-1, 1)
    j = np.arange(n, dtype=np.float64).reshape(1, -1)
    return 1.0 / (i + j + 1.0)


def _toeplitz(m: int, n: int, seed: int = 42, **_) -> np.ndarray:
    """
    Scalar Toeplitz matrix: T[i,j] = f(j - i).

    The generating function f is built from a decaying cosine sequence with a
    seed-dependent phase shift:
        f[k] = cos(k + phase) * exp(-|k| / 4)

    The result is a true Toeplitz matrix (constant along every diagonal)
    built directly from its first row and first column — no post-processing.
    """
    rng   = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)

    # f[k] for k = 0, 1, ..., max(m,n)-1
    max_k = max(m, n)
    k     = np.arange(max_k, dtype=np.float64)
    f     = np.cos(k + phase) * np.exp(-k / 4.0)

    # Build T[i,j] = f[|j-i|] using the sign convention f[-k] = f[k]
    # (i.e. symmetric Toeplitz).  For a non-symmetric version pass a separate
    # first column; here we keep it symmetric for clarity.
    A = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            A[i, j] = f[abs(j - i)]
    return A


def _block_toeplitz(m: int, n: int, seed: int = 42,
                    type_param: int = 4, **_) -> np.ndarray:
    """
    Block-Toeplitz matrix built directly from independent random blocks.

    A Block-Toeplitz matrix is defined by its first block-row and first
    block-column:
        T[br, bc] = B[bc - br]

    where B[k] for k >= 0 comes from the first block-row, and B[k] for k < 0
    comes from the first block-column.  Each block B[k] is an independent
    random Gaussian block of size bs × bs.

    This is the mathematically correct construction — no projection, no tiling
    of a structured matrix, no ambiguity.

    type_param = block size (bs).  Defaults to 4.
    """
    bs     = max(1, int(type_param))
    rng    = np.random.default_rng(seed)
    num_rb = (m + bs - 1) // bs
    num_cb = (n + bs - 1) // bs

    # Generate independent random blocks for each distinct offset d = bc - br
    # d ranges from -(num_rb-1) to +(num_cb-1)
    blocks: dict[int, np.ndarray] = {}
    for d in range(-(num_rb - 1), num_cb):
        blocks[d] = rng.standard_normal((bs, bs))

    # Tile
    A = np.zeros((m, n), dtype=np.float64)
    for br in range(num_rb):
        for bc in range(num_cb):
            d   = bc - br
            B   = blocks[d]
            r0, r1 = br * bs, min((br + 1) * bs, m)
            c0, c1 = bc * bs, min((bc + 1) * bs, n)
            h, w = r1 - r0, c1 - c0
            A[r0:r1, c0:c1] = B[:h, :w]
    return A


def _circulant(m: int, n: int, seed: int = 42, **_) -> np.ndarray:
    """
    Circulant matrix: each row is a cyclic shift of the first row.

    The first row has a Gaussian envelope so the spectrum decays naturally.
    Always square (k = min(m, n)).
    """
    k   = min(m, n)
    rng = np.random.default_rng(seed)
    c   = rng.standard_normal(k) * np.exp(-np.arange(k, dtype=float) / (k / 4.0))
    A   = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        A[i] = np.roll(c, i)
    out = np.zeros((m, n), dtype=np.float64)
    out[:k, :k] = A
    return out


def _random_spd(m: int, n: int, seed: int = 42,
                type_param: int = 6, **_) -> np.ndarray:
    """
    Random SPD matrix with prescribed κ(A) = 10^type_param.

        A = Q Λ Qᵀ,  Q random orthogonal,  Λ = logspace(0, type_param, k)
    """
    k    = min(m, n)
    rng  = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    lam  = np.logspace(0, type_param, k)
    A    = (Q * lam) @ Q.T
    out  = np.zeros((m, n), dtype=np.float64)
    out[:k, :k] = A
    return out


def _diagonally_dominant(m: int, n: int, seed: int = 42,
                          type_param: int = 1, **_) -> np.ndarray:
    """
    Strictly diagonally dominant random matrix.

        aᵢᵢ = Σⱼ≠ᵢ |aᵢⱼ| + margin,   margin = type_param ≥ 1

    Gershgorin guarantees non-singularity.  Sparse structures only
    remove off-diagonal entries, which strengthens diagonal dominance.
    """
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((m, n))
    k   = min(m, n)
    for i in range(k):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        A[i, i] = row_sum + float(type_param)
    return A


_BASE_GENERATORS: dict[str, callable] = {
    "Random Gaussian":     _random_gaussian,
    "Hilbert":             _hilbert,
    "Toeplitz":            _toeplitz,
    "Block-Toeplitz":      _block_toeplitz,
    "Circulant":           _circulant,
    "Random SPD":          _random_spd,
    "Diagonally Dominant": _diagonally_dominant,
}

# Human-readable notes shown in the UI
MATRIX_TYPE_NOTES: dict[str, str] = {
    "Random Gaussian": (
        "Entries i.i.d. N(0,1).  κ(A) ~ O(n).  Standard benchmark."
    ),
    "Hilbert": (
        "H[i,j] = 1/(i+j+1).  κ(A) ~ (3.5)ⁿ.  "
        "Canonical severely ill-conditioned matrix."
    ),
    "Toeplitz": (
        "Constant along each diagonal: T[i,j] = f(|i−j|) with a decaying cosine f.  "
        "Appears in convolution operators, time-series analysis, 1-D FDM stencils."
    ),
    "Block-Toeplitz": (
        "T[br,bc] = B[bc−br] where each B[k] is an independent random block.  "
        "param = block size.  Appears in 2-D PDE discretisations and "
        "multi-channel signal processing."
    ),
    "Circulant": (
        "Each row is a cyclic shift of the first row.  "
        "Eigenvalues = DFT of first row.  Appears in periodic BC problems."
    ),
    "Random SPD": (
        "A = QΛQᵀ with κ(A) = 10^k exactly (set k with the slider).  "
        "Dense only — sparse zeroing destroys positive definiteness."
    ),
    "Diagonally Dominant": (
        "aᵢᵢ = Σ|aᵢⱼ| + margin.  Nonsingular by Gershgorin.  "
        "Appears in FDM of elliptic PDEs.  Compatible with all structures."
    ),
}

MATRIX_TYPE_PARAM_NOTES: dict[str, str] = {
    "Block-Toeplitz":      "Block size — size of each random block.",
    "Random SPD":          "log₁₀(κ_target) — controls the condition number directly.",
    "Diagonally Dominant": "Diagonal margin — larger = better conditioned.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Sparsity structure masks
# ──────────────────────────────────────────────────────────────────────────────

def _tridiagonal_mask(m: int, n: int) -> np.ndarray:
    mask = np.zeros((m, n), dtype=bool)
    for d in (-1, 0, 1):
        rows = np.arange(max(0, -d), min(m, n - d))
        if rows.size:
            mask[rows, rows + d] = True
    return mask


def _block_tridiagonal_mask(m: int, n: int, block_size: int) -> np.ndarray:
    bs = max(1, block_size)
    mask = np.zeros((m, n), dtype=bool)
    num_rb = (m + bs - 1) // bs
    num_cb = (n + bs - 1) // bs
    for br in range(num_rb):
        for bc in range(num_cb):
            if abs(br - bc) <= 1:
                r0, r1 = br * bs, min((br + 1) * bs, m)
                c0, c1 = bc * bs, min((bc + 1) * bs, n)
                mask[r0:r1, c0:c1] = True
    return mask


def _banded_mask(m: int, n: int, num_diags: int) -> np.ndarray:
    half = num_diags // 2
    mask = np.zeros((m, n), dtype=bool)
    for d in range(-half, half + 1):
        rows = np.arange(max(0, -d), min(m, n - d))
        if rows.size:
            mask[rows, rows + d] = True
    return mask


def _apply_structure(A: np.ndarray, structure: str,
                     struct_param: int) -> np.ndarray:
    """Apply a sparsity pattern to A by zeroing entries outside the pattern."""
    if structure == "Dense":
        return A.copy()
    m, n = A.shape
    if structure == "Sparse Tridiagonal":
        return A * _tridiagonal_mask(m, n)
    if structure == "Sparse Block-Tridiagonal":
        return A * _block_tridiagonal_mask(m, n, struct_param)
    if structure == "Sparse Banded":
        return A * _banded_mask(m, n, struct_param)
    raise ValueError(f"Unknown structure: {structure!r}")


# Human-readable structure descriptions
STRUCTURE_NOTES: dict[str, str] = {
    "Dense":                    "",
    "Sparse Tridiagonal":       "Non-zeros on diagonals −1, 0, +1 only.",
    "Sparse Block-Tridiagonal": "Main block-diagonal + two neighbouring blocks.  param = block size.",
    "Sparse Banded":            "param diagonals centred on the main diagonal.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Matrix modifications
# ──────────────────────────────────────────────────────────────────────────────

def _symmetrize(A: np.ndarray) -> np.ndarray:
    return (A + A.T) / 2.0


def _make_spd(A: np.ndarray) -> np.ndarray:
    n     = A.shape[1]
    B     = A.T @ A
    alpha = n * np.finfo(np.float64).eps * np.linalg.norm(B, 1) + 1e-10
    B    += alpha * np.eye(n)
    return B


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def create_matrix(
    matrix_type: str,
    m: int,
    n: int,
    structure: str,
    struct_param: int,
    make_hermitian: bool,
    make_positive_definite: bool,
    dtype: np.dtype,
    seed: int = 42,
    type_param: int = 4,
) -> np.ndarray:
    """
    Construct the coefficient matrix A.

    Parameters
    ----------
    matrix_type           : one of the keys in _BASE_GENERATORS
    m, n                  : rows and columns
    structure             : sparsity pattern ("Dense" | "Sparse Tridiagonal" |
                            "Sparse Block-Tridiagonal" | "Sparse Banded")
    struct_param          : block size (Block-Tridiagonal) or num diagonals (Banded)
    make_hermitian        : symmetrise via (A + Aᵀ) / 2
    make_positive_definite: transform A → Aᵀ A + α I
    dtype                 : np.float32 or np.float64
    seed                  : RNG seed
    type_param            : type-specific parameter:
                            Block-Toeplitz      → block size
                            Random SPD          → log₁₀(κ_target)
                            Diagonally Dominant → diagonal margin

    Raises ValueError for incompatible (matrix_type, structure) combinations.
    """
    generator = _BASE_GENERATORS.get(matrix_type)
    if generator is None:
        raise ValueError(f"Unknown matrix type: {matrix_type!r}")

    allowed = compatible_structures(matrix_type)
    if structure not in allowed:
        raise ValueError(
            f"Structure {structure!r} is not compatible with {matrix_type!r}.  "
            f"Allowed: {allowed}"
        )

    A = generator(m, n, seed=seed, type_param=type_param)
    A = _apply_structure(A, structure, struct_param)

    if make_hermitian:
        A = _symmetrize(A)
    if make_positive_definite:
        A = _make_spd(A)

    return A.astype(dtype)


def create_rhs(m: int, dtype: np.dtype, seed: int = 7) -> np.ndarray:
    """Random Gaussian right-hand side vector b of length m."""
    return np.random.default_rng(seed).standard_normal(m).astype(dtype)


def apply_perturbation(
    arr: np.ndarray,
    order: int,
    seed: int = 13,
    structure: str = "Dense",
    struct_param: int = 1,
    make_hermitian: bool = False,
    make_positive_definite: bool = False,
) -> np.ndarray:
    """
    Add a structure-aware random perturbation with magnitude 10^order × ‖arr‖.
    """
    rng   = np.random.default_rng(seed)
    noise = rng.standard_normal(arr.shape)

    if arr.ndim == 2:
        noise = _apply_structure(noise, structure, struct_param)
        if make_hermitian:
            noise = _symmetrize(noise)
        if make_positive_definite:
            noise = _symmetrize(noise)

    noise_norm = np.linalg.norm(noise)
    arr_norm   = np.linalg.norm(arr)
    if noise_norm > 0 and arr_norm > 0:
        noise = noise * (arr_norm * (10.0 ** order) / noise_norm)
    else:
        noise = noise * (10.0 ** order)

    return (arr + noise).astype(arr.dtype)


def matrix_info(A: np.ndarray) -> dict:
    """Basic descriptive statistics for a dense matrix A."""
    m, n  = A.shape
    nnz   = int(np.count_nonzero(A))
    total = m * n
    return {
        "shape":        (m, n),
        "dtype":        str(A.dtype),
        "nnz":          nnz,
        "density":      nnz / total if total > 0 else 0.0,
        "memory_bytes": A.nbytes,
    }
