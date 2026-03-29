"""
core/solvers.py
===============
Solver implementations for the linear system  Ax = b.

Solvers
-------
SVD      : Reduced, Full
QR       : Householder Reduced, Householder Full,
           Classical Gram-Schmidt, Modified Gram-Schmidt
Direct   : LU (partial pivoting), Cholesky
Iterative: GMRES, CG (Conjugate Gradient)

Each public  solve_*  function has the signature::

    (A: np.ndarray, b: np.ndarray) -> dict

The returned dict always contains:

    x          : np.ndarray  – computed solution
    method     : str         – human-readable method name
    success    : bool        – whether the solver completed
    message    : str         – status / diagnostic text

QR-based solvers additionally return:

    Q          : np.ndarray  – the Q factor (used for orthogonality analysis)

The public registry ``SOLVERS`` maps display names to solver functions.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ──────────────────────────────────────────────────────────────────────────────
# Precondition helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_compatible(A: np.ndarray, b: np.ndarray) -> None:
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but b has length {b.shape[0]}."
        )


def _check_square(A: np.ndarray, name: str) -> None:
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"'{name}' requires a square matrix "
            f"(got {A.shape[0]} × {A.shape[1]})."
        )


def _check_symmetric(A: np.ndarray, name: str, tol: float = 1e-8) -> None:
    _check_square(A, name)
    if not np.allclose(A, A.T, atol=tol, rtol=tol):
        raise ValueError(
            f"'{name}' requires a symmetric matrix. "
            "Enable the 'Hermitian' option in the problem creation panel."
        )


def _check_positive_definite(A: np.ndarray, name: str) -> None:
    """Check SPD by attempting a Cholesky factorisation."""
    _check_symmetric(A, name)
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"'{name}' requires a positive definite matrix. "
            "Enable the 'Positive Definite' option in the problem creation panel."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility: least-squares triangular solve
# ──────────────────────────────────────────────────────────────────────────────

def _triangular_lstsq(R: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solve  R x = rhs  where R is upper triangular (or upper trapezoidal).

    For square R this calls the numerically preferred ``solve_triangular``.
    For non-square R (underdetermined) it falls back to ``lstsq``.
    """
    if R.shape[0] == R.shape[1]:
        return la.solve_triangular(R, rhs)
    x, *_ = np.linalg.lstsq(R, rhs, rcond=None)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# SVD solvers
# ──────────────────────────────────────────────────────────────────────────────

def _svd_solve(A: np.ndarray, b: np.ndarray, full_matrices: bool) -> dict:
    """
    Solve  Ax ≈ b  via the pseudoinverse  x = V Σ⁺ Uᵀ b.

    Singular values below  ε · max(m, n) · σ_max  are treated as zero,
    giving the minimum-norm least-squares solution.
    """
    _check_compatible(A, b)
    label = "Full" if full_matrices else "Reduced"

    A_f = A.astype(float)
    b_f = b.astype(float)

    U, s, Vt = np.linalg.svd(A_f, full_matrices=full_matrices)

    # Threshold for numerical rank determination
    tol   = np.finfo(float).eps * max(A_f.shape) * s[0]
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    rank  = int(np.sum(s > tol))
    k     = len(s)

    # x = V diag(s_inv) Uᵀ b
    Utb = U[:, :k].T @ b_f      # shape (k,)
    x   = Vt[:k].T @ (s_inv * Utb)  # shape (n,)

    return {
        "x":       x,
        "method":  f"SVD ({label})",
        "success": True,
        "message": (
            f"Pseudoinverse via {label} SVD.  "
            f"Numerical rank = {rank} / {k}  (threshold = {tol:.2e})."
        ),
    }


def solve_svd_reduced(A, b): return _svd_solve(A, b, full_matrices=False)
def solve_svd_full(A, b):    return _svd_solve(A, b, full_matrices=True)


# ──────────────────────────────────────────────────────────────────────────────
# Householder QR  (via LAPACK  dgeqrf / dorgqr)
# ──────────────────────────────────────────────────────────────────────────────

def _householder_qr_solve(A: np.ndarray, b: np.ndarray, full_matrices: bool) -> dict:
    """
    Solve  Ax ≈ b  via Householder QR then back-substitution.

    For  m >= n: overdetermined / square — computes the least-squares solution
                 by solving the triangular system  R x = Qᵀ b.
    For  m <  n: underdetermined — falls back to lstsq on the trapezoidal R.
    """
    _check_compatible(A, b)
    m, n  = A.shape
    label = "Full" if full_matrices else "Reduced"
    mode  = "complete" if full_matrices else "reduced"

    A_f = A.astype(float)
    b_f = b.astype(float)

    Q, R = np.linalg.qr(A_f, mode=mode)
    k    = min(m, n)

    # Project rhs onto the column space of A
    Qtb = Q[:, :k].T @ b_f   # shape (k,)
    x   = _triangular_lstsq(R[:k, :], Qtb)

    return {
        "x":       x,
        "Q":       Q[:, :k],
        "R":       R[:k, :],
        "method":  f"QR Householder ({label})",
        "success": True,
        "message": (
            f"LAPACK Householder QR ({label}).  "
            f"System: {'overdetermined/square' if m >= n else 'underdetermined'}."
        ),
    }


def solve_qr_householder_reduced(A, b): return _householder_qr_solve(A, b, False)
def solve_qr_householder_full(A, b):    return _householder_qr_solve(A, b, True)


# ──────────────────────────────────────────────────────────────────────────────
# Classical Gram-Schmidt  (CGS)
# ──────────────────────────────────────────────────────────────────────────────

def _classical_gram_schmidt(A: np.ndarray):
    """
    Compute the thin QR decomposition  A = Q R  via Classical Gram-Schmidt.

    Returns Q (m × k) and R (k × k), where  k = min(m, n).

    Numerical behaviour: CGS can lose orthogonality rapidly when A is
    ill-conditioned because projections are computed against the *original*
    column  a_j  before any correction.
    """
    m, n  = A.shape
    k     = min(m, n)
    Q     = np.zeros((m, k), dtype=float)
    R     = np.zeros((k, k), dtype=float)

    for j in range(k):
        a_j = A[:, j].astype(float)   # original column — used for all projections
        v   = a_j.copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ a_j  # projection uses original a_j
            v       -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-14:
            Q[:, j] = v / R[j, j]
        # If R[j,j] ≈ 0 the column is nearly linearly dependent; Q[:,j] stays 0.

    return Q, R


def solve_cgs(A: np.ndarray, b: np.ndarray) -> dict:
    """Classical Gram-Schmidt QR solver."""
    _check_compatible(A, b)
    Q, R = _classical_gram_schmidt(A)
    Qtb  = Q.T @ b.astype(float)
    x    = _triangular_lstsq(R, Qtb)
    return {
        "x":       x,
        "Q":       Q,
        "R":       R,
        "method":  "QR (Classical Gram-Schmidt)",
        "success": True,
        "message": (
            "Classical GS: computes projections from the original column, "
            "which can cause catastrophic loss of orthogonality for "
            "ill-conditioned matrices."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Modified Gram-Schmidt  (MGS)
# ──────────────────────────────────────────────────────────────────────────────

def _modified_gram_schmidt(A: np.ndarray):
    """
    Compute the thin QR decomposition  A = Q R  via Modified Gram-Schmidt.

    Returns Q (m × k) and R (k × k), where  k = min(m, n).

    Numerical behaviour: MGS re-orthogonalises against each q_i using the
    *current* (partially corrected) vector, which significantly improves
    orthogonality retention compared with CGS — at the same O(mn²) cost.
    """
    m, n  = A.shape
    k     = min(m, n)
    Q     = A[:, :k].astype(float).copy()   # working copy; overwritten in place
    R     = np.zeros((k, k), dtype=float)

    for i in range(k):
        R[i, i] = np.linalg.norm(Q[:, i])
        if R[i, i] > 1e-14:
            Q[:, i] /= R[i, i]
        for j in range(i + 1, k):
            R[i, j]  = Q[:, i] @ Q[:, j]   # projection uses the *current* Q[:,j]
            Q[:, j] -= R[i, j] * Q[:, i]

    return Q, R


def solve_mgs(A: np.ndarray, b: np.ndarray) -> dict:
    """Modified Gram-Schmidt QR solver."""
    _check_compatible(A, b)
    Q, R = _modified_gram_schmidt(A)
    Qtb  = Q.T @ b.astype(float)
    x    = _triangular_lstsq(R, Qtb)
    return {
        "x":       x,
        "Q":       Q,
        "R":       R,
        "method":  "QR (Modified Gram-Schmidt)",
        "success": True,
        "message": (
            "Modified GS: orthogonalises against the current vector at each step. "
            "Substantially better orthogonality retention than Classical GS."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LU with partial pivoting
# ──────────────────────────────────────────────────────────────────────────────

def solve_lu(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Solve  Ax = b  via LU factorisation with partial pivoting  (PA = LU).

    Requires a square matrix.  Uses LAPACK's  dgetrf / dgetrs  via SciPy.
    """
    _check_compatible(A, b)
    _check_square(A, "LU")
    lu, piv = la.lu_factor(A.astype(float))
    x       = la.lu_solve((lu, piv), b.astype(float))
    return {
        "x":       x,
        "method":  "LU (partial pivoting)",
        "success": True,
        "message": "LAPACK PA = LU with partial pivoting.  Requires a square matrix.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sparse LU  (SuperLU via scipy.sparse)
# ──────────────────────────────────────────────────────────────────────────────

def solve_lu_sparse(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Solve  Ax = b  via sparse LU factorisation (SuperLU).

    The dense matrix A is converted to CSC (Compressed Sparse Column) format
    internally.  SuperLU applies column reordering (COLAMD) to minimise
    fill-in in the L and U factors, then performs the factorisation.

    For matrices with a sparse structure (tridiagonal, banded, block-tridiagonal)
    this exploits the sparsity and is substantially faster than dense LU for
    large m.  For dense matrices the conversion overhead makes it slightly
    slower than regular LU.

    Requires a square matrix.
    """
    _check_compatible(A, b)
    _check_square(A, "LU (Sparse)")

    A_f   = A.astype(float)
    b_f   = b.astype(float)
    nnz   = int(np.count_nonzero(A_f))
    total = A_f.shape[0] ** 2

    # Convert dense → CSC sparse format (only non-zeros stored)
    A_csc = sp.csc_matrix(A_f)

    # SuperLU factorisation with COLAMD column reordering
    lu    = spla.splu(A_csc)
    x     = lu.solve(b_f)

    density = nnz / total if total > 0 else 1.0
    return {
        "x":       x,
        "method":  "LU (Sparse / SuperLU)",
        "success": True,
        "message": (
            f"SuperLU with COLAMD reordering.  "
            f"Matrix density = {density:.1%}  ({nnz:,} non-zeros of {total:,}).  "
            f"Sparse format exploits zeros for faster factorisation."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cholesky
# ──────────────────────────────────────────────────────────────────────────────

def solve_cholesky(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Solve  Ax = b  via Cholesky factorisation  (A = L Lᵀ).

    Requires A to be square, symmetric, and positive definite.
    Uses LAPACK's  dpotrf / dpotrs  via SciPy.
    """
    _check_compatible(A, b)
    _check_positive_definite(A, "Cholesky")
    c, low = la.cho_factor(A.astype(float))
    x      = la.cho_solve((c, low), b.astype(float))
    return {
        "x":       x,
        "method":  "Cholesky",
        "success": True,
        "message": "LAPACK Cholesky A = LLᵀ.  Requires a symmetric positive definite matrix.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# GMRES
# ──────────────────────────────────────────────────────────────────────────────

def solve_gmres(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Solve  Ax = b  via the Generalised Minimum Residual method.

    GMRES minimises the residual over Krylov subspaces and works for any
    non-singular square matrix.  No symmetry is required.
    """
    _check_compatible(A, b)
    _check_square(A, "GMRES")
    A_f = A.astype(float)
    b_f = b.astype(float)
    x, info = spla.gmres(A_f, b_f, rtol=1e-10, atol=1e-14)
    success  = info == 0
    msg      = "Converged." if success else f"Did not converge (SciPy info code = {info})."
    return {
        "x":       x,
        "method":  "GMRES",
        "success": success,
        "message": msg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conjugate Gradient  (CG)
# ──────────────────────────────────────────────────────────────────────────────

def solve_cg(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Solve  Ax = b  via the Conjugate Gradient method.

    CG is optimal for symmetric positive definite matrices: it minimises
    the A-norm of the error over Krylov subspaces.  Convergence rate
    depends on  sqrt(κ(A)).
    """
    _check_compatible(A, b)
    _check_positive_definite(A, "CG")
    A_f = A.astype(float)
    b_f = b.astype(float)
    x, info = spla.cg(A_f, b_f, rtol=1e-10, atol=1e-14)
    success  = info == 0
    msg      = "Converged." if success else f"Did not converge (SciPy info code = {info})."
    return {
        "x":       x,
        "method":  "CG (Conjugate Gradient)",
        "success": success,
        "message": msg,
    }


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
