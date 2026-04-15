"""
core/analysis.py
================
Conditioning and numerical stability analysis for  Ax = b.

GPU support
-----------
When input arrays live on the GPU (CuPy), the main computation path uses
CuPy operations.  mpmath and float128 always run on the CPU — they are
only used for the high-precision residual on small matrices where the
CPU round-trip cost is negligible.

High-precision residual strategy (GPU path)
--------------------------------------------
GPU arrays are transferred to CPU (to_numpy) before the mpmath / float128
path so arbitrary precision can still be used for small matrices.  The
residual vector is then transferred back to the same device as the input.

This means the GPU path also benefits from high-precision residuals for
small matrices — there is no accuracy regression compared to the CPU path.

Public API
----------
condition_number(A, norm_type)              -> float
stability_analysis(A, b, x, norm_type)     -> dict
orthogonality_error(Q)                     -> float
"""

from __future__ import annotations

import numpy as np

from core.device import get_array_module, is_gpu_array, to_numpy

# ── mpmath configuration ──────────────────────────────────────────────────────
MP_DPS            = 50
MP_SIZE_THRESHOLD = 200

try:
    import mpmath as _mpmath
    _MPMATH_AVAILABLE = True
except ImportError:
    _MPMATH_AVAILABLE = False

_F128_EPS       = float(np.finfo(np.float128).eps)
_F64_EPS        = float(np.finfo(np.float64).eps)
_FLOAT128_WIDER = _F128_EPS < _F64_EPS


# ──────────────────────────────────────────────────────────────────────────────
# High-precision residual  (always on CPU numpy/mpmath)
# ──────────────────────────────────────────────────────────────────────────────

def _residual_mpmath(A_cpu, b_cpu, x_cpu):
    _mpmath.mp.dps = MP_DPS
    A_mp = _mpmath.matrix(A_cpu.tolist())
    b_mp = _mpmath.matrix(b_cpu.tolist())
    x_mp = _mpmath.matrix(x_cpu.tolist())
    r_mp = b_mp - A_mp * x_mp
    norm_r = float(_mpmath.sqrt(sum(r_mp[i]**2 for i in range(len(b_cpu)))))
    r = np.array([float(r_mp[i]) for i in range(len(b_cpu))], dtype=np.float64)
    return r, norm_r, f"mpmath ({MP_DPS} dps)"


def _residual_float128(A_cpu, b_cpu, x_cpu):
    r_128  = (b_cpu.astype(np.float128)
              - A_cpu.astype(np.float128) @ x_cpu.astype(np.float128))
    norm_r = float(np.linalg.norm(r_128))
    r      = r_128.astype(np.float64)
    return r, norm_r, "float128 (80-bit extended)"


def _residual_float64(A_cpu, b_cpu, x_cpu):
    r      = b_cpu.astype(np.float64) - A_cpu.astype(np.float64) @ x_cpu.astype(np.float64)
    norm_r = float(np.linalg.norm(r))
    return r, norm_r, "float64 (standard)"


def compute_residual(A, b, x):
    """
    Compute r = b − A x and ‖r‖₂ in the highest precision available.

    GPU arrays are automatically pulled to CPU for the high-precision path.
    The returned residual vector is always a CPU NumPy float64 array
    (callers that need a GPU array can do xp.asarray(r) themselves).
    """
    A_cpu = to_numpy(A).astype(np.float64)
    b_cpu = to_numpy(b).astype(np.float64)
    x_cpu = to_numpy(x).astype(np.float64)
    m     = A_cpu.shape[0]

    if _MPMATH_AVAILABLE and m <= MP_SIZE_THRESHOLD:
        try:
            return _residual_mpmath(A_cpu, b_cpu, x_cpu)
        except Exception:
            pass

    if _FLOAT128_WIDER:
        try:
            return _residual_float128(A_cpu, b_cpu, x_cpu)
        except Exception:
            pass

    return _residual_float64(A_cpu, b_cpu, x_cpu)


# ──────────────────────────────────────────────────────────────────────────────
# Norm helpers  (device-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def _norm_order(norm_type: str):
    return 2 if norm_type == "2" else np.inf


def _vec_norm(v, norm_type: str) -> float:
    """Works for numpy and cupy vectors."""
    xp = get_array_module(is_gpu_array(v))
    return float(xp.linalg.norm(v, ord=_norm_order(norm_type)))


def _mat_norm(A, norm_type: str) -> float:
    """
    ‖A‖₂ = σ_max  or  ‖A‖∞ = max row-sum.

    CuPy supports np.inf norm directly.  For the 2-norm CuPy does not
    support ord=2 for matrices, so we compute σ_max via SVD.
    """
    if norm_type == "2":
        if is_gpu_array(A):
            import cupy as cp
            s = cp.linalg.svd(A.astype(float), compute_uv=False)
            return float(s[0])
        return float(np.linalg.norm(A.astype(float), ord=2))
    else:
        xp = get_array_module(is_gpu_array(A))
        return float(xp.linalg.norm(A.astype(float), ord=np.inf))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def condition_number(A, norm_type: str) -> float:
    """
    Compute κ_p(A) = ‖A‖_p · ‖A⁺‖_p.

    For CuPy arrays, uses singular values directly (cupy.linalg.svd)
    since cupy.linalg.cond is not available.
    """
    try:
        if is_gpu_array(A):
            import cupy as cp
            A_f = A.astype(float)
            if norm_type == "2":
                s = cp.linalg.svd(A_f, compute_uv=False)
                s_cpu = to_numpy(s)
                if s_cpu[-1] == 0:
                    return float("inf")
                return float(s_cpu[0] / s_cpu[-1])
            else:
                # ‖A‖∞ · ‖A⁻¹‖∞  — compute via numpy on CPU copy
                A_cpu = to_numpy(A_f)
                return float(np.linalg.cond(A_cpu, p=np.inf))
        else:
            order = _norm_order(norm_type)
            return float(np.linalg.cond(A.astype(float), p=order))
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")


def stability_analysis(A, b, x, norm_type: str) -> dict:
    """
    Compute the three key numerical stability metrics.

    Works with both NumPy (CPU) and CuPy (GPU) arrays.
    The residual is always computed in high precision on the CPU.
    """
    # High-precision residual — always CPU numpy
    r_cpu, norm_r_2, residual_prec = compute_residual(A, b, x)

    if norm_type == "2":
        norm_r = norm_r_2
    else:
        norm_r = float(np.linalg.norm(r_cpu, ord=np.inf))

    # Remaining norms — can use GPU operations
    norm_A = _mat_norm(A,            norm_type)
    norm_x = _vec_norm(x,            norm_type)
    norm_b = _vec_norm(b,            norm_type)
    kappa  = condition_number(A,     norm_type)

    denom_fwd     = norm_A * norm_x
    forward_bound = (kappa * norm_r / denom_fwd
                     if denom_fwd > 0 else float("inf"))

    denom_bwd      = norm_A * norm_x + norm_b
    backward_error = (norm_r / denom_bwd
                      if denom_bwd > 0 else float("inf"))

    # Always return residual as a CPU numpy array for display
    return {
        "kappa":          kappa,
        "residual_vec":   r_cpu,
        "residual_norm":  norm_r,
        "norm_A":         norm_A,
        "norm_x":         norm_x,
        "norm_b":         norm_b,
        "forward_bound":  forward_bound,
        "backward_error": backward_error,
        "norm_type":      norm_type,
        "residual_prec":  residual_prec,
    }


def orthogonality_error(Q) -> float:
    """
    ‖QᵀQ − I‖_F — works for numpy and cupy Q.
    """
    if is_gpu_array(Q):
        import cupy as cp
        k   = Q.shape[1]
        err = cp.linalg.norm(Q.T @ Q - cp.eye(k), ord="fro")
        return float(err)
    k   = Q.shape[1]
    err = np.linalg.norm(Q.T @ Q - np.eye(k), ord="fro")
    return float(err)
