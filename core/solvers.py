"""
core/solvers.py
===============
Solver implementations for the linear system  Ax = b.

GPU support
-----------
Every public ``solve_*`` function accepts an optional ``use_gpu`` keyword
argument (default False).  When True, the solver uses CuPy / cuPyx routines
instead of NumPy / SciPy equivalents.

When an input array is already a CuPy array the solver will detect this
automatically via ``device.is_gpu_array`` and run on the GPU regardless of
the explicit ``use_gpu`` flag — this avoids accidental CPU round-trips.

GPU solver mapping
------------------
| CPU solver                | GPU solver                           |
|---------------------------|--------------------------------------|
| numpy.linalg.svd          | cupy.linalg.svd                      |
| numpy.linalg.qr           | cupy.linalg.qr                       |
| scipy.linalg.lu_factor/   | cupyx.scipy.linalg.lu_factor/        |
|   lu_solve                |   lu_solve                           |
| scipy.linalg.cho_factor/  | cupyx.scipy.linalg.cho_factor/       |
|   cho_solve               |   cho_solve                          |
| scipy.linalg.             | cupyx.scipy.linalg.                  |
|   solve_triangular        |   solve_triangular                   |
| scipy.sparse.csc_matrix + | cupyx.scipy.sparse.csc_matrix +      |
|   sparse.linalg.splu      |   sparse.linalg.splu                 |
| scipy.sparse.linalg.gmres | cupyx.scipy.sparse.linalg.gmres      |
| scipy.sparse.linalg.cg    | cupyx.scipy.sparse.linalg.cg         |

Notes
-----
- ``numpy.linalg.cond`` is not available in CuPy; condition number is
  computed via ``cupy.linalg.svd`` directly (σ_max / σ_min).
- The classical and modified Gram-Schmidt implementations are hand-written
  loops; they run on whatever array type (numpy/cupy) is passed in.
- All returned ``x`` values are arrays on the same device as the input.
  The caller (app.py / analysis.py) is responsible for calling
  ``device.to_numpy`` before plotting or display.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.device import (
    get_array_module, get_linalg, get_sparse, get_sparse_linalg,
    is_gpu_array, to_numpy,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to pick CPU vs GPU sub-modules
# ──────────────────────────────────────────────────────────────────────────────

def _use_gpu_from_array(A, use_gpu: bool) -> bool:
    """Return True if we should run on GPU, considering both flag and array type."""
    return use_gpu or is_gpu_array(A)


def _xp(use_gpu: bool):
    return get_array_module(use_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# Precondition helpers  (device-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def _check_compatible(A, b) -> None:
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but b has length {b.shape[0]}."
        )


def _check_square(A, name: str) -> None:
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"'{name}' requires a square matrix "
            f"(got {A.shape[0]} × {A.shape[1]})."
        )


def _check_symmetric(A, name: str, tol: float = 1e-8) -> None:
    _check_square(A, name)
    xp = get_array_module(_use_gpu_from_array(A, False))
    # Use numpy for the allclose test (CuPy has np.allclose equivalent)
    A_cpu = to_numpy(A)
    if not np.allclose(A_cpu, A_cpu.T, atol=tol, rtol=tol):
        raise ValueError(
            f"'{name}' requires a symmetric matrix. "
            "Enable the 'Hermitian' option in the problem creation panel."
        )


def _check_positive_definite(A, name: str) -> None:
    _check_symmetric(A, name)
    try:
        np.linalg.cholesky(to_numpy(A))
    except np.linalg.LinAlgError:
        raise ValueError(
            f"'{name}' requires a positive definite matrix. "
            "Enable the 'Positive Definite' option in the problem creation panel."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility: triangular solve
# ──────────────────────────────────────────────────────────────────────────────

def _triangular_lstsq(R, rhs, use_gpu: bool = False):
    """Solve R x = rhs where R is upper triangular (or trapezoidal)."""
    if R.shape[0] == R.shape[1]:
        sla_mod = get_linalg(use_gpu)
        return sla_mod.solve_triangular(R, rhs)
    # Fallback lstsq — must happen on CPU for both backends
    R_cpu, rhs_cpu = to_numpy(R), to_numpy(rhs)
    x_cpu, *_ = np.linalg.lstsq(R_cpu, rhs_cpu, rcond=None)
    if use_gpu:
        xp = get_array_module(True)
        return xp.asarray(x_cpu)
    return x_cpu


# ──────────────────────────────────────────────────────────────────────────────
# SVD solvers
# ──────────────────────────────────────────────────────────────────────────────

def _svd_solve(A, b, full_matrices: bool, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    gpu  = _use_gpu_from_array(A, use_gpu)
    xp   = get_array_module(gpu)
    label = "Full" if full_matrices else "Reduced"

    A_f = A.astype(float)
    b_f = b.astype(float)

    U, s, Vt = xp.linalg.svd(A_f, full_matrices=full_matrices)

    eps   = float(np.finfo(float).eps)
    s_cpu = to_numpy(s)
    tol   = eps * max(A_f.shape) * float(s_cpu[0])
    s_inv_cpu = np.where(s_cpu > tol, 1.0 / s_cpu, 0.0)
    rank  = int(np.sum(s_cpu > tol))
    k     = len(s_cpu)

    s_inv = xp.asarray(s_inv_cpu)
    Utb   = U[:, :k].T @ b_f
    x     = Vt[:k].T @ (s_inv * Utb)

    return {
        "x":       x,
        "method":  f"SVD ({label})",
        "success": True,
        "message": (
            f"Pseudoinverse via {label} SVD {'[GPU]' if gpu else '[CPU]'}.  "
            f"Numerical rank = {rank} / {k}  (threshold = {tol:.2e})."
        ),
    }


def solve_svd_reduced(A, b, use_gpu: bool = False):
    return _svd_solve(A, b, False, use_gpu)

def solve_svd_full(A, b, use_gpu: bool = False):
    return _svd_solve(A, b, True, use_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# Householder QR
# ──────────────────────────────────────────────────────────────────────────────

def _householder_qr_solve(A, b, full_matrices: bool,
                           use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    gpu   = _use_gpu_from_array(A, use_gpu)
    xp    = get_array_module(gpu)
    m, n  = A.shape
    label = "Full" if full_matrices else "Reduced"
    mode  = "complete" if full_matrices else "reduced"

    A_f = A.astype(float)
    b_f = b.astype(float)

    Q, R = xp.linalg.qr(A_f, mode=mode)
    k    = min(m, n)
    Qtb  = Q[:, :k].T @ b_f
    x    = _triangular_lstsq(R[:k, :], Qtb, gpu)

    return {
        "x":       x,
        "Q":       Q[:, :k],
        "R":       R[:k, :],
        "method":  f"QR Householder ({label})",
        "success": True,
        "message": (
            f"{'CuPy' if gpu else 'LAPACK'} Householder QR ({label}) "
            f"{'[GPU]' if gpu else '[CPU]'}.  "
            f"System: {'overdetermined/square' if m >= n else 'underdetermined'}."
        ),
    }


def solve_qr_householder_reduced(A, b, use_gpu: bool = False):
    return _householder_qr_solve(A, b, False, use_gpu)

def solve_qr_householder_full(A, b, use_gpu: bool = False):
    return _householder_qr_solve(A, b, True, use_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# Classical Gram-Schmidt  (device-agnostic loop)
# ──────────────────────────────────────────────────────────────────────────────

def _classical_gram_schmidt(A):
    """Works for both numpy and cupy arrays."""
    xp   = get_array_module(is_gpu_array(A))
    m, n = A.shape
    k    = min(m, n)
    Q    = xp.zeros((m, k), dtype=float)
    R    = xp.zeros((k, k), dtype=float)

    for j in range(k):
        a_j = A[:, j].astype(float)
        v   = a_j.copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ a_j
            v       -= R[i, j] * Q[:, i]
        R[j, j] = xp.linalg.norm(v)
        if float(R[j, j]) > 1e-14:
            Q[:, j] = v / R[j, j]
    return Q, R


def solve_cgs(A, b, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    gpu  = _use_gpu_from_array(A, use_gpu)
    xp   = get_array_module(gpu)
    A_f  = xp.asarray(A.astype(float))
    Q, R = _classical_gram_schmidt(A_f)
    Qtb  = Q.T @ b.astype(float)
    x    = _triangular_lstsq(R, Qtb, gpu)
    return {
        "x":       x,
        "Q":       Q,
        "R":       R,
        "method":  "QR (Classical Gram-Schmidt)",
        "success": True,
        "message": (
            f"Classical GS {'[GPU]' if gpu else '[CPU]'}: projections from the "
            "original column — can lose orthogonality for ill-conditioned matrices."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Modified Gram-Schmidt  (device-agnostic loop)
# ──────────────────────────────────────────────────────────────────────────────

def _modified_gram_schmidt(A):
    """Works for both numpy and cupy arrays."""
    xp   = get_array_module(is_gpu_array(A))
    m, n = A.shape
    k    = min(m, n)
    Q    = xp.asarray(A[:, :k].astype(float).copy() if not is_gpu_array(A)
                      else A[:, :k].astype(float).copy())
    R    = xp.zeros((k, k), dtype=float)

    for i in range(k):
        R[i, i] = xp.linalg.norm(Q[:, i])
        if float(R[i, i]) > 1e-14:
            Q[:, i] /= R[i, i]
        for j in range(i + 1, k):
            R[i, j]  = Q[:, i] @ Q[:, j]
            Q[:, j] -= R[i, j] * Q[:, i]
    return Q, R


def solve_mgs(A, b, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    gpu  = _use_gpu_from_array(A, use_gpu)
    xp   = get_array_module(gpu)
    A_f  = xp.asarray(A.astype(float))
    Q, R = _modified_gram_schmidt(A_f)
    Qtb  = Q.T @ b.astype(float)
    x    = _triangular_lstsq(R, Qtb, gpu)
    return {
        "x":       x,
        "Q":       Q,
        "R":       R,
        "method":  "QR (Modified Gram-Schmidt)",
        "success": True,
        "message": (
            f"Modified GS {'[GPU]' if gpu else '[CPU]'}: orthogonalises against "
            "current vector — substantially better than Classical GS."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LU with partial pivoting
# ──────────────────────────────────────────────────────────────────────────────

def solve_lu(A, b, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    _check_square(A, "LU")
    gpu    = _use_gpu_from_array(A, use_gpu)
    sla    = get_linalg(gpu)
    A_f    = A.astype(float)
    b_f    = b.astype(float)
    lu, piv = sla.lu_factor(A_f)
    x       = sla.lu_solve((lu, piv), b_f)
    return {
        "x":       x,
        "method":  "LU (partial pivoting)",
        "success": True,
        "message": (
            f"PA = LU with partial pivoting {'[GPU]' if gpu else '[CPU/LAPACK]'}.  "
            "Requires a square matrix."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sparse LU  (SuperLU / cuSolver)
# ──────────────────────────────────────────────────────────────────────────────

def solve_lu_sparse(A, b, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    _check_square(A, "LU (Sparse)")
    gpu  = _use_gpu_from_array(A, use_gpu)

    A_f   = to_numpy(A).astype(float)
    b_f_cpu = to_numpy(b).astype(float)
    nnz   = int(np.count_nonzero(A_f))
    total = A_f.shape[0] ** 2

    if gpu:
        # CuPy sparse path
        try:
            import cupyx.scipy.sparse as cpsp
            import cupyx.scipy.sparse.linalg as cpspla
            import cupy as cp
            A_csc = cpsp.csc_matrix(cp.asarray(A_f))
            b_gpu = cp.asarray(b_f_cpu)
            lu    = cpspla.splu(A_csc)
            x     = lu.solve(b_gpu)
            backend = "cuSolver/SuperLU [GPU]"
        except Exception as e:
            # Fallback: dense cupy solve if sparse path fails
            import cupy as cp
            x = cp.linalg.solve(cp.asarray(A_f), cp.asarray(b_f_cpu))
            backend = f"cupy.linalg.solve [GPU] (sparse fallback, reason: {e})"
    else:
        A_csc = sp.csc_matrix(A_f)
        lu    = spla.splu(A_csc)
        x     = lu.solve(b_f_cpu)
        backend = "SuperLU/COLAMD [CPU]"

    density = nnz / total if total > 0 else 1.0
    return {
        "x":       x if gpu else x,
        "method":  "LU (Sparse / SuperLU)",
        "success": True,
        "message": (
            f"{backend}.  "
            f"Matrix density = {density:.1%}  ({nnz:,} non-zeros of {total:,})."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cholesky
# ──────────────────────────────────────────────────────────────────────────────

def solve_cholesky(A, b, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    _check_positive_definite(A, "Cholesky")
    gpu    = _use_gpu_from_array(A, use_gpu)
    sla    = get_linalg(gpu)
    A_f    = A.astype(float)
    b_f    = b.astype(float)
    c, low = sla.cho_factor(A_f)
    x      = sla.cho_solve((c, low), b_f)
    return {
        "x":       x,
        "method":  "Cholesky",
        "success": True,
        "message": (
            f"Cholesky A = LLᵀ {'[GPU]' if gpu else '[CPU/LAPACK]'}.  "
            "Requires symmetric positive definite matrix."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# GMRES
# ──────────────────────────────────────────────────────────────────────────────

def solve_gmres(A, b, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    _check_square(A, "GMRES")
    gpu    = _use_gpu_from_array(A, use_gpu)
    spla_m = get_sparse_linalg(gpu)
    xp     = get_array_module(gpu)

    A_f = A.astype(float)
    b_f = b.astype(float)

    # cupyx gmres signature matches scipy
    x, info = spla_m.gmres(A_f, b_f, rtol=1e-10, atol=1e-14)
    success  = info == 0
    msg = (
        f"{'CuPy' if gpu else 'SciPy'} GMRES {'[GPU]' if gpu else '[CPU]'} — "
        + ("Converged." if success else f"Did not converge (info={info}).")
    )
    return {"x": x, "method": "GMRES", "success": success, "message": msg}


# ──────────────────────────────────────────────────────────────────────────────
# Conjugate Gradient
# ──────────────────────────────────────────────────────────────────────────────

def solve_cg(A, b, use_gpu: bool = False) -> dict:
    _check_compatible(A, b)
    _check_positive_definite(A, "CG")
    gpu    = _use_gpu_from_array(A, use_gpu)
    spla_m = get_sparse_linalg(gpu)

    A_f = A.astype(float)
    b_f = b.astype(float)

    x, info = spla_m.cg(A_f, b_f, rtol=1e-10, atol=1e-14)
    success  = info == 0
    msg = (
        f"{'CuPy' if gpu else 'SciPy'} CG {'[GPU]' if gpu else '[CPU]'} — "
        + ("Converged." if success else f"Did not converge (info={info}).")
    )
    return {"x": x, "method": "CG (Conjugate Gradient)",
            "success": success, "message": msg}


# ──────────────────────────────────────────────────────────────────────────────
# Solver registry
# ──────────────────────────────────────────────────────────────────────────────

SOLVERS: dict = {
    "SVD (Reduced)":                   solve_svd_reduced,
    "SVD (Full)":                      solve_svd_full,
    "QR Householder (Reduced)":        solve_qr_householder_reduced,
    "QR Householder (Full)":           solve_qr_householder_full,
    "QR Classical Gram-Schmidt":       solve_cgs,
    "QR Modified Gram-Schmidt":        solve_mgs,
    "LU":                              solve_lu,
    "LU (Sparse)":                     solve_lu_sparse,
    "Cholesky":                        solve_cholesky,
    "GMRES":                           solve_gmres,
    "CG (Conjugate Gradient)":         solve_cg,
}
