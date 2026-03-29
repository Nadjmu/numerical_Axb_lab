"""
core/analysis.py
================
Conditioning and numerical stability analysis for  Ax = b.

Public API
----------
condition_number(A, norm_type)              -> float
stability_analysis(A, b, x, norm_type)     -> dict
orthogonality_error(Q)                     -> float   (for QR-based solvers)

Residual precision
------------------
The residual  r = b − A x_comp  is the foundation of all three stability
metrics.  It is computed in the highest precision available on the platform:

  1. mpmath at `MP_DPS` decimal places — used when m ≤ MP_SIZE_THRESHOLD.
     On this system gives the same result as float128 but is platform-
     independent and arbitrarily accurate.

  2. numpy float128 (80-bit extended precision on x86 Linux) — used for
     larger matrices where mpmath would be too slow.  Falls back to float64
     automatically on platforms where float128 == float64 (e.g. macOS ARM).

  3. float64 — final fallback if both above are unavailable or fail.

The returned residual_vec is always a float64 array so it is compatible with
the rest of the codebase.  The precision gain only affects its accuracy.

Metrics returned by stability_analysis
---------------------------------------
kappa          – condition number  κ_p(A)
residual_vec   – r = b − A x_comp  (high-precision)
residual_norm  – ‖r‖_p
norm_A         – ‖A‖_p
norm_x         – ‖x_comp‖_p
norm_b         – ‖b‖_p
forward_bound  – κ(A) ‖r‖ / (‖A‖ ‖x‖)
backward_error – ‖r‖ / (‖A‖ ‖x‖ + ‖b‖)
residual_prec  – string describing which precision method was used
"""

from __future__ import annotations

import numpy as np

# ── mpmath configuration ──────────────────────────────────────────────────────
# Decimal places for mpmath residual.  50 dps ≈ 166 bits ≈ 10× float64.
MP_DPS = 50

# Use mpmath for matrices up to this size; float128 above (speed trade-off).
# At m=200 mpmath takes ~250 ms; float128 takes ~2 ms.
MP_SIZE_THRESHOLD = 200

try:
    import mpmath as _mpmath
    _MPMATH_AVAILABLE = True
except ImportError:
    _MPMATH_AVAILABLE = False

# Check whether float128 is genuinely wider than float64 on this platform.
_F128_EPS = float(np.finfo(np.float128).eps)
_F64_EPS  = float(np.finfo(np.float64).eps)
_FLOAT128_WIDER = _F128_EPS < _F64_EPS   # False on macOS ARM, True on x86 Linux


# ──────────────────────────────────────────────────────────────────────────────
# High-precision residual
# ──────────────────────────────────────────────────────────────────────────────

def _residual_mpmath(A: np.ndarray, b: np.ndarray,
                     x: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Compute r = b − A x and ‖r‖₂ using mpmath at MP_DPS decimal places.

    The norm is computed in high precision before casting r back to float64,
    so norm_r is not contaminated by float64 rounding of the already-small
    residual entries.

    Returns (r as float64, norm_r as float, precision label).
    """
    _mpmath.mp.dps = MP_DPS
    A_mp = _mpmath.matrix(A.tolist())
    b_mp = _mpmath.matrix(b.tolist())
    x_mp = _mpmath.matrix(x.tolist())
    r_mp = b_mp - A_mp * x_mp
    # Compute ‖r‖₂ in mpmath before losing precision
    norm_r = float(_mpmath.sqrt(sum(r_mp[i]**2 for i in range(len(b)))))
    r      = np.array([float(r_mp[i]) for i in range(len(b))], dtype=np.float64)
    return r, norm_r, f"mpmath ({MP_DPS} dps)"


def _residual_float128(A: np.ndarray, b: np.ndarray,
                       x: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Compute r = b − A x and ‖r‖₂ using numpy float128 (80-bit on x86 Linux).

    The norm is computed in float128 before casting r back to float64.

    Returns (r as float64, norm_r as float, precision label).
    """
    r_128  = (b.astype(np.float128)
              - A.astype(np.float128) @ x.astype(np.float128))
    norm_r = float(np.linalg.norm(r_128))   # norm in float128
    r      = r_128.astype(np.float64)
    return r, norm_r, "float128 (80-bit extended)"


def _residual_float64(A: np.ndarray, b: np.ndarray,
                      x: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Standard float64 residual and norm — fallback."""
    r      = b.astype(np.float64) - A.astype(np.float64) @ x.astype(np.float64)
    norm_r = float(np.linalg.norm(r))
    return r, norm_r, "float64 (standard)"


def compute_residual(A: np.ndarray, b: np.ndarray,
                     x: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Compute r = b − A x and ‖r‖₂ in the highest precision available.

    Both r and its norm are computed at high precision — the norm is extracted
    before casting r back to float64, so norm_r is not affected by the
    precision loss of the cast.

    Strategy
    --------
    m ≤ MP_SIZE_THRESHOLD and mpmath installed → mpmath at MP_DPS digits
    m >  MP_SIZE_THRESHOLD and float128 wider  → numpy float128
    otherwise                                  → float64

    Returns
    -------
    r      : np.ndarray (float64) — the residual vector
    norm_r : float                — ‖r‖₂ computed at high precision
    label  : str                  — description of precision used
    """
    m = A.shape[0]

    if _MPMATH_AVAILABLE and m <= MP_SIZE_THRESHOLD:
        try:
            return _residual_mpmath(A, b, x)
        except Exception:
            pass   # fall through on any mpmath failure

    if _FLOAT128_WIDER:
        try:
            return _residual_float128(A, b, x)
        except Exception:
            pass

    return _residual_float64(A, b, x)


# ──────────────────────────────────────────────────────────────────────────────
# Norm helpers
# ──────────────────────────────────────────────────────────────────────────────

def _norm_order(norm_type: str) -> int | float:
    return 2 if norm_type == "2" else np.inf


def _vec_norm(v: np.ndarray, norm_type: str) -> float:
    return float(np.linalg.norm(v, ord=_norm_order(norm_type)))


def _mat_norm(A: np.ndarray, norm_type: str) -> float:
    """
    ‖A‖₂ = σ_max (expensive for large A).
    ‖A‖∞ = max row-sum (O(mn), always fast).
    """
    return float(np.linalg.norm(A, ord=_norm_order(norm_type)))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def condition_number(A: np.ndarray, norm_type: str) -> float:
    """
    Compute κ_p(A) = ‖A‖_p · ‖A⁺‖_p.

    Returns np.inf for singular or near-singular matrices.
    """
    order = _norm_order(norm_type)
    try:
        return float(np.linalg.cond(A.astype(float), p=order))
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")


def stability_analysis(
    A:         np.ndarray,
    b:         np.ndarray,
    x:         np.ndarray,
    norm_type: str,
) -> dict:
    """
    Compute the three key numerical stability metrics.

    The residual r = b − A x is computed in the highest precision available
    (mpmath for small matrices, float128 for large ones) so that FEB and BE
    are not contaminated by floating-point cancellation in the residual itself.

    Parameters
    ----------
    A, b, x   : coefficient matrix, rhs, computed solution
    norm_type : '2' or 'inf'

    Returns
    -------
    dict with keys: kappa, residual_vec, residual_norm, norm_A, norm_x,
                    norm_b, forward_bound, backward_error, norm_type,
                    residual_prec
    """
    A_f = A.astype(np.float64)
    b_f = b.astype(np.float64)
    x_f = x.astype(np.float64)

    # ── High-precision residual and its norm ─────────────────────────────────
    # norm_r is computed in the same high precision as r — before any cast
    # back to float64 — so FEB and BE are not contaminated by float64 rounding.
    # For the inf-norm we fall back to float64 (mpmath/float128 gave ‖r‖₂).
    r, norm_r_2, residual_prec = compute_residual(A_f, b_f, x_f)

    if norm_type == "2":
        norm_r = norm_r_2           # use high-precision ‖r‖₂ directly
    else:
        norm_r = _vec_norm(r, norm_type)   # ‖r‖∞ in float64 (no cancellation issue)

    # ── Remaining norms (float64 — no catastrophic cancellation here) ────────
    norm_A = _mat_norm(A_f,  norm_type)
    norm_x = _vec_norm(x_f,  norm_type)
    norm_b = _vec_norm(b_f,  norm_type)
    kappa  = condition_number(A_f, norm_type)

    # ── Derived metrics ───────────────────────────────────────────────────────
    denom_fwd     = norm_A * norm_x
    forward_bound = (kappa * norm_r / denom_fwd
                     if denom_fwd > 0 else float("inf"))

    denom_bwd      = norm_A * norm_x + norm_b
    backward_error = (norm_r / denom_bwd
                      if denom_bwd > 0 else float("inf"))

    return {
        "kappa":          kappa,
        "residual_vec":   r,
        "residual_norm":  norm_r,
        "norm_A":         norm_A,
        "norm_x":         norm_x,
        "norm_b":         norm_b,
        "forward_bound":  forward_bound,
        "backward_error": backward_error,
        "norm_type":      norm_type,
        "residual_prec":  residual_prec,
    }


def orthogonality_error(Q: np.ndarray) -> float:
    """
    ‖QᵀQ − I‖_F — measures loss of orthogonality in the Q factor.

    Householder QR  → O(ε_mach)
    Modified GS     → O(ε_mach · κ(A))
    Classical GS    → can be O(1) for ill-conditioned A
    """
    k   = Q.shape[1]
    err = np.linalg.norm(Q.T @ Q - np.eye(k), ord="fro")
    return float(err)