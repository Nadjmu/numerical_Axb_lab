"""
core/problem_creation.py
========================
Generates the coefficient matrix A and right-hand side vector b for Ax = b.

GPU support
-----------
All public functions accept an optional ``use_gpu`` boolean (default False).
When True, arrays are allocated and returned as CuPy arrays on the current
CUDA device.  The internal ``xp`` variable holds either ``numpy`` or ``cupy``
depending on the device choice.

Base generators run on the CPU using NumPy, then the result is transferred to
the GPU via ``xp.asarray()``.  This is intentional: the matrix generation
logic is algebraically complex (Toeplitz, Block-Toeplitz, Circulant, etc.) and
the transfer cost is negligible for the matrix sizes used in this lab.  The
performance-critical operations (factorisation, iterative solve) run on the GPU.

Public API
----------
create_matrix(...)          -> np.ndarray  or  cp.ndarray
create_rhs(...)             -> np.ndarray  or  cp.ndarray
apply_perturbation(...)     -> np.ndarray  or  cp.ndarray
matrix_info(...)            -> dict  (always CPU scalars)
compatible_structures(type) -> list[str]
"""

from __future__ import annotations

import numpy as np

from core.device import get_array_module, to_numpy


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
    return COMPATIBILITY.get(matrix_type, _SPARSE_STRUCTURES)


# ──────────────────────────────────────────────────────────────────────────────
# Base matrix generators  (always produce numpy arrays — cheap CPU logic)
# ──────────────────────────────────────────────────────────────────────────────

def _random_gaussian(m: int, n: int, seed: int = 42, **_) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((m, n))


def _hilbert(m: int, n: int, **_) -> np.ndarray:
    i = np.arange(m, dtype=np.float64).reshape(-1, 1)
    j = np.arange(n, dtype=np.float64).reshape(1, -1)
    return 1.0 / (i + j + 1.0)


def _toeplitz(m: int, n: int, seed: int = 42, **_) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    max_k = max(m, n)
    k     = np.arange(max_k, dtype=np.float64)
    f     = np.cos(k + phase) * np.exp(-k / 4.0)
    A     = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            A[i, j] = f[abs(j - i)]
    return A


def _block_toeplitz(m: int, n: int, seed: int = 42,
                    type_param: int = 4, **_) -> np.ndarray:
    bs     = max(1, int(type_param))
    rng    = np.random.default_rng(seed)
    num_rb = (m + bs - 1) // bs
    num_cb = (n + bs - 1) // bs
    blocks: dict[int, np.ndarray] = {}
    for d in range(-(num_rb - 1), num_cb):
        blocks[d] = rng.standard_normal((bs, bs))
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
# Sparsity structure masks  (NumPy — cheap, always CPU)
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


def _apply_structure_numpy(A: np.ndarray, structure: str,
                            struct_param: int) -> np.ndarray:
    """Apply a sparsity pattern.  Always operates on NumPy arrays."""
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
# Matrix modifications  (NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def _symmetrize_numpy(A: np.ndarray) -> np.ndarray:
    return (A + A.T) / 2.0


def _make_spd_numpy(A: np.ndarray) -> np.ndarray:
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
    use_gpu: bool = False,
):
    """
    Construct the coefficient matrix A.

    When ``use_gpu=True`` the returned array is a CuPy array on the GPU.
    Generation always happens on the CPU (NumPy) and is then transferred.

    Parameters
    ----------
    use_gpu : bool
        Transfer the final matrix to the GPU after generation.
    (all other parameters identical to CPU version)
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

    # Generate on CPU
    A = generator(m, n, seed=seed, type_param=type_param)
    A = _apply_structure_numpy(A, structure, struct_param)

    if make_hermitian:
        A = _symmetrize_numpy(A)
    if make_positive_definite:
        A = _make_spd_numpy(A)

    A = A.astype(dtype)

    # Transfer to GPU if requested
    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(A)
    return A


def create_rhs(m: int, dtype: np.dtype, seed: int = 7,
               use_gpu: bool = False):
    """Random Gaussian right-hand side vector b of length m."""
    b = np.random.default_rng(seed).standard_normal(m).astype(dtype)
    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(b)
    return b


def apply_perturbation(
    arr,
    order: int,
    seed: int = 13,
    structure: str = "Dense",
    struct_param: int = 1,
    make_hermitian: bool = False,
    make_positive_definite: bool = False,
    use_gpu: bool = False,
    custom_mask: np.ndarray | None = None,
):
    """
    Add a structure-aware random perturbation with magnitude 10^order × ‖arr‖.

    Works whether ``arr`` is a NumPy or CuPy array.  The perturbation is
    generated on the CPU and transferred if needed.

    Parameters
    ----------
    custom_mask : np.ndarray | None
        Boolean mask of the same shape as arr.  When provided (for imported
        matrices), noise is zeroed outside the mask so the perturbation
        inherits the sparsity pattern of the original array.  Overrides
        the ``structure`` / ``struct_param`` arguments.
    """
    arr_cpu = to_numpy(arr)
    rng     = np.random.default_rng(seed)
    noise   = rng.standard_normal(arr_cpu.shape)

    if arr_cpu.ndim == 2:
        if custom_mask is not None:
            # Imported matrix: inherit sparsity pattern from the array itself
            noise = noise * custom_mask.astype(float)
        else:
            noise = _apply_structure_numpy(noise, structure, struct_param)
        if make_hermitian:
            noise = _symmetrize_numpy(noise)
        if make_positive_definite:
            noise = _symmetrize_numpy(noise)

    noise_norm = np.linalg.norm(noise)
    arr_norm   = np.linalg.norm(arr_cpu)
    if noise_norm > 0 and arr_norm > 0:
        noise = noise * (arr_norm * (10.0 ** order) / noise_norm)
    else:
        noise = noise * (10.0 ** order)

    result_cpu = (arr_cpu + noise).astype(arr_cpu.dtype)

    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(result_cpu)
    return result_cpu


def load_npy(file_obj, expected_ndim: int, use_gpu: bool = False):
    """
    Load a .npy file uploaded via Streamlit and return a numpy or cupy array.

    Parameters
    ----------
    file_obj    : file-like object from st.file_uploader
    expected_ndim : 1 for vectors, 2 for matrices
    use_gpu     : transfer to GPU after loading

    Returns
    -------
    arr : np.ndarray or cp.ndarray

    Raises
    ------
    ValueError  if the array has the wrong number of dimensions or
                contains non-finite values.
    """
    import io
    arr = np.load(io.BytesIO(file_obj.read()))

    if arr.ndim != expected_ndim:
        raise ValueError(
            f"Expected a {expected_ndim}-D array but got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Imported array contains NaN or Inf values.")

    # Ensure float dtype
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)

    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(arr)
    return arr


def sparsity_mask(A) -> np.ndarray:
    """
    Return a boolean mask of the non-zero entries of A (CPU numpy).
    Used to inherit the sparsity pattern of an imported matrix for perturbation.
    """
    A_cpu = to_numpy(A)
    return A_cpu != 0.0


def matrix_info(A) -> dict:
    """
    Basic descriptive statistics for a dense matrix A.

    Works for both NumPy and CuPy arrays.  All returned values are
    plain Python scalars so they are safe to display in Streamlit.
    """
    A_cpu = to_numpy(A)
    m, n  = A_cpu.shape
    nnz   = int(np.count_nonzero(A_cpu))
    total = m * n
    return {
        "shape":        (m, n),
        "dtype":        str(A_cpu.dtype),
        "nnz":          nnz,
        "density":      nnz / total if total > 0 else 0.0,
        "memory_bytes": A_cpu.nbytes,
    }
