"""
ui/analysis_ui.py
=================
Streamlit display functions for experiment results.

Functions
---------
render_array(label, arr)          – scrollable dataframe of raw entries
render_heatmap(label, arr)        – seaborn heatmap for 2-D matrices
render_vector_heatmap(label, arr) – seaborn heatmap for 1-D vectors
render_matrix_info(A, b)          – dimensions, dtype, density, memory
render_sparsity_pattern(A)        – heatmap / spy depending on size
render_solver_result(result)      – solution status and x info
render_stability(metrics)         – condition number + 3 key metrics
render_orthogonality(result)      – Q-factor orthogonality (QR solvers)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st

from core.problem_creation import matrix_info
from core.analysis import orthogonality_error


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(val: float, decimals: int = 6) -> str:
    """Format a float for display; handles inf and zero gracefully."""
    if val is None:
        return "N/A"
    if np.isinf(val):
        return "∞"
    if np.isnan(val):
        return "NaN"
    if val == 0.0:
        return "0"
    return f"{val:.{decimals}e}"


def _bytes_to_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


# ──────────────────────────────────────────────────────────────────────────────
# Seaborn style defaults
# ──────────────────────────────────────────────────────────────────────────────

# Maximum dimension for annotated heatmaps (annotations become unreadable beyond this)
_ANNOT_MAX   = 20
# Maximum dimension for any heatmap (spy plot used beyond this)
_HEATMAP_MAX = 150


def _apply_heatmap_style() -> None:
    """Apply a consistent seaborn / matplotlib style for all heatmaps."""
    sns.set_theme(style="white", font_scale=0.9)


def _diverging_cmap():
    """Diverging colormap centred at zero: blue=negative, red=positive."""
    return sns.diverging_palette(220, 20, as_cmap=True)


def _annotate_fmt(data: np.ndarray) -> str:
    """Choose annotation format based on data magnitude."""
    absmax = np.nanmax(np.abs(data))
    if absmax == 0:
        return ".2f"
    if absmax < 0.01 or absmax >= 1e4:
        return ".2e"
    return ".3f"


def _heatmap_norm_and_cmap(data: np.ndarray):
    """
    Choose norm + colormap based on the sign structure of data.

    - Mixed-sign  → SymLogNorm(vmin=-absmax, vmax=absmax) + diverging cmap
                    (zero = white; red = positive, blue = negative)
    - All positive → SymLogNorm(vmin=0, vmax=absmax)       + 'plasma'
    - All negative → SymLogNorm(vmin=-absmax, vmax=0)      + 'plasma_r'
    - All zero     → returns (None, 'plasma', 'value')

    Using the full colormap range for single-sign data means every order of
    magnitude gets equal visual weight, eliminating the "all one shade" problem.

    Returns (norm_or_None, cmap, cbar_label).
    """
    tiny   = np.finfo(float).tiny * 10
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None, "plasma", "value"

    absmax = float(np.max(np.abs(finite)))
    if absmax == 0:
        return None, "plasma", "value"

    linthresh  = max(absmax * 1e-3, tiny)
    has_pos    = bool(np.any(finite > 0))
    has_neg    = bool(np.any(finite < 0))

    if has_pos and has_neg:
        norm  = mcolors.SymLogNorm(linthresh=linthresh, linscale=0.5,
                                   vmin=-absmax, vmax=absmax, base=10)
        return norm, _diverging_cmap(), "value (symlog)"

    if has_pos:
        norm  = mcolors.SymLogNorm(linthresh=linthresh, linscale=0.5,
                                   vmin=0, vmax=absmax, base=10)
        return norm, "plasma", "value (log)"

    # all non-positive
    norm  = mcolors.SymLogNorm(linthresh=linthresh, linscale=0.5,
                               vmin=-absmax, vmax=0, base=10)
    return norm, "plasma_r", "−value (log)"


def _symlog_norm(data: np.ndarray):
    """Legacy helper used by PDF builder — returns (norm, absmax)."""
    absmax = float(np.nanmax(np.abs(data)))
    if absmax == 0:
        return None, 1.0
    linthresh = max(absmax * 1e-3, np.finfo(float).tiny * 10)
    norm = mcolors.SymLogNorm(
        linthresh=linthresh, linscale=0.5,
        vmin=-absmax, vmax=absmax, base=10,
    )
    return norm, absmax


# ──────────────────────────────────────────────────────────────────────────────
# 2-D matrix heatmap
# ──────────────────────────────────────────────────────────────────────────────

def render_heatmap(label: str, arr: np.ndarray) -> None:
    """
    Render a seaborn heatmap for a 2-D numpy array.

    For complex arrays, Re and Im are shown as two side-by-side panels.
    For large matrices (> _HEATMAP_MAX) falls back to a spy dot plot.
    """
    _apply_heatmap_style()
    is_cx  = np.iscomplexobj(arr)
    parts  = [("Re", arr.real), ("Im", arr.imag)] if is_cx else [("", arr)]
    m, n   = arr.shape

    if m > _HEATMAP_MAX or n > _HEATMAP_MAX:
        # imshow is instant for large matrices and supports full colorbar + norm
        ncols = 2 if is_cx else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        for ax, (part_label, data) in zip(axes, parts):
            norm, cmap, cbar_label = _heatmap_norm_and_cmap(data)
            im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest", origin="upper")
            fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
            ax.set_title(f"{label}  [{part_label}]" if is_cx else label,
                         fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("column", fontsize=8)
            ax.set_ylabel("row",    fontsize=8)
        st.caption(f"Matrix is {m} × {n} — colour = value magnitude (log scale).")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    annot     = (m <= _ANNOT_MAX and n <= _ANNOT_MAX)
    cell_size = max(0.4, min(0.9, 8.0 / max(m, n)))
    ncols = 2 if is_cx else 1
    fig_w = min(n * cell_size * ncols + 1.5 * ncols, 14.0)
    fig_h = min(m * cell_size + 1.2, 8.0)

    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    if ncols == 1:
        axes = [axes]

    for ax, (part_label, data) in zip(axes, parts):
        norm, cmap, cbar_label = _heatmap_norm_and_cmap(data)
        fmt     = _annotate_fmt(data) if annot else ""
        norm_kw = {"norm": norm} if norm is not None else {"vmin": 0, "vmax": 1}
        sns.heatmap(
            data,
            ax          = ax,
            cmap        = cmap,
            annot       = annot,
            fmt         = fmt,
            annot_kws   = {"size": max(6, min(10, int(80 / max(m, n))))},
            linewidths  = 0.4 if m <= 40 else 0.0,
            linecolor   = "#cccccc",
            square      = True,
            cbar_kws    = {"shrink": 0.75, "label": cbar_label},
            xticklabels = n <= 30,
            yticklabels = m <= 30,
            **norm_kw,
        )
        ax.set_title(f"{label}  [{part_label}]" if is_cx else label,
                     fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("column", fontsize=8)
        ax.set_ylabel("row",    fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        if n > 10:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 1-D vector heatmap
# ──────────────────────────────────────────────────────────────────────────────

def render_vector_heatmap(label: str, arr: np.ndarray) -> None:
    """
    Render a seaborn heatmap for a 1-D numpy array displayed as a column.

    The vector is shown as a single-column heatmap so it visually aligns
    with the matrix heatmap beside it (same colour scale logic).
    Annotations are shown for short vectors (≤ _ANNOT_MAX entries).
    """
    _apply_heatmap_style()
    m = arr.shape[0]

    if m > _HEATMAP_MAX:
        fig, ax  = plt.subplots(figsize=(1.2, 5))
        data_2d  = arr[:_HEATMAP_MAX].reshape(-1, 1)
        norm, cmap, cbar_label = _heatmap_norm_and_cmap(data_2d)
        norm_kw  = {"norm": norm} if norm is not None else {"vmin": 0, "vmax": 1}
        sns.heatmap(
            data_2d, ax=ax, cmap=cmap,
            annot=False, linewidths=0.0, square=False,
            cbar_kws={"shrink": 0.6, "label": cbar_label},
            xticklabels=False, yticklabels=False,
            **norm_kw,
        )
        ax.set_title(label, fontsize=9, fontweight="bold", pad=6)
        st.caption(f"Showing first {_HEATMAP_MAX} of {m} entries.")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    annot                    = m <= _ANNOT_MAX
    fmt                      = _annotate_fmt(arr) if annot else ""
    norm, cmap, cbar_label   = _heatmap_norm_and_cmap(arr.reshape(-1, 1))
    norm_kw                  = {"norm": norm} if norm is not None else {"vmin": 0, "vmax": 1}

    cell_h = max(0.35, min(0.7, 6.0 / m))
    fig_h  = min(m * cell_h + 1.0, 9.0)

    fig, ax = plt.subplots(figsize=(1.6, fig_h))

    sns.heatmap(
        arr.reshape(-1, 1),
        ax          = ax,
        cmap        = cmap,
        annot       = annot,
        fmt         = fmt,
        annot_kws   = {"size": max(6, min(9, int(60 / m)))},
        linewidths  = 0.4 if m <= 40 else 0.0,
        linecolor   = "#cccccc",
        square      = False,
        cbar_kws    = {"shrink": 0.6, "label": cbar_label},
        xticklabels = False,
        yticklabels = True if m <= 30 else False,
        **norm_kw,
    )

    ax.set_title(label, fontsize=9, fontweight="bold", pad=6)
    ax.set_ylabel("index", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Raw array display (scrollable dataframe)
# ──────────────────────────────────────────────────────────────────────────────

_DISPLAY_MAX = 200   # max rows/cols shown in the dataframe widget

def render_array(label: str, arr: np.ndarray) -> None:
    """
    Display entries of a 1-D or 2-D numpy array as a scrollable dataframe.

    Truncated to _DISPLAY_MAX rows/cols with a caption for larger arrays.
    """
    st.markdown(f"**{label}**")

    if arr.ndim == 1:
        truncated = arr.shape[0] > _DISPLAY_MAX
        data = arr[:_DISPLAY_MAX] if truncated else arr
        df = pd.DataFrame(data, columns=["value"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if truncated:
            st.caption(f"Showing first {_DISPLAY_MAX} of {arr.shape[0]} entries.")

    elif arr.ndim == 2:
        m, n = arr.shape
        data = arr[:_DISPLAY_MAX, :_DISPLAY_MAX]
        if np.iscomplexobj(data):
            st.caption("Re(A)  —  real part:")
            st.dataframe(pd.DataFrame(data.real), use_container_width=True, hide_index=True)
            st.caption("Im(A)  —  imaginary part:")
            st.dataframe(pd.DataFrame(data.imag), use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        if m > _DISPLAY_MAX or n > _DISPLAY_MAX:
            st.caption(
                f"Showing [{min(m, _DISPLAY_MAX)} × {min(n, _DISPLAY_MAX)}] "
                f"of [{m} × {n}].  Full matrix stored in memory."
            )
    else:
        st.warning(f"Cannot display array with ndim={arr.ndim}.")


# ──────────────────────────────────────────────────────────────────────────────
# Matrix information block
# ──────────────────────────────────────────────────────────────────────────────

def render_matrix_info(A: np.ndarray, b: np.ndarray) -> None:
    """Display shape, dtype, sparsity and memory info for A and b."""
    info = matrix_info(A)
    m, n = info["shape"]

    st.subheader("Matrix A")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Shape",     f"{m} × {n}")
    col2.metric("dtype",     info["dtype"])
    col3.metric("Non-zeros", f"{info['nnz']:,}")
    col4.metric("Density",   f"{info['density']:.2%}")
    col5.metric("Memory",    _bytes_to_human(info["memory_bytes"]))

    b_mem = _bytes_to_human(b.nbytes)
    st.caption(
        f"Vector b — length {b.shape[0]},  dtype {b.dtype},  memory {b_mem}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sparsity pattern — convenience wrapper kept for backward compatibility
# ──────────────────────────────────────────────────────────────────────────────

def render_sparsity_pattern(A: np.ndarray) -> None:
    """Render the heatmap / spy plot for matrix A."""
    render_heatmap("Sparsity / value pattern", A)


# ──────────────────────────────────────────────────────────────────────────────
# Solver result block
# ──────────────────────────────────────────────────────────────────────────────

def render_solver_result(result: dict) -> None:
    """Display solver status, method name and basic solution statistics."""
    st.subheader(f"Solution via {result['method']}")

    if result["success"]:
        st.success(result["message"])
    else:
        st.warning(result["message"])

    x = result["x"]
    c1, c2, c3 = st.columns(3)
    c1.metric("x length", str(x.shape[0]))
    c2.metric("‖x‖₂",    _fmt(float(np.linalg.norm(x))))
    c3.metric("‖x‖∞",    _fmt(float(np.linalg.norm(x, np.inf))))

    render_array("Computed solution x", x)


# ──────────────────────────────────────────────────────────────────────────────
# Conditioning block
# ──────────────────────────────────────────────────────────────────────────────

def render_conditioning(metrics: dict) -> None:
    """Display the condition number with a plain interpretation."""
    norm_label = "‖·‖₂" if metrics["norm_type"] == "2" else "‖·‖∞"
    kappa = metrics["kappa"]

    st.subheader("Conditioning")

    c1, c2 = st.columns(2)
    c1.metric(
        f"κ(A)  ({norm_label})",
        _fmt(kappa),
        help="κ(A) = ‖A‖ · ‖A⁺‖.  Measures amplification of data errors.",
    )

    if not (np.isinf(kappa) or np.isnan(kappa) or kappa <= 0):
        digits_lost      = np.log10(kappa)
        digits_available = -np.log10(np.finfo(float).eps)
        c2.metric(
            "Digits of accuracy lost",
            f"≈ {digits_lost:.1f}",
            help=(
                f"float64 gives ≈{digits_available:.0f} significant digits.  "
                f"Conditioning costs ≈log₁₀(κ) = {digits_lost:.1f} of them."
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Stability metrics block
# ──────────────────────────────────────────────────────────────────────────────

def render_stability(metrics: dict) -> None:
    """
    Display the three key numerical stability metrics in a table.

    Metrics
    -------
    1. Residual ‖r‖              r = b − A x_comp
    2. Forward error bound       κ(A) ‖r‖ / (‖A‖ ‖x‖)
    3. Backward error            ‖r‖ / (‖A‖ ‖x‖ + ‖b‖)
    """
    norm_label = "‖·‖₂" if metrics["norm_type"] == "2" else "‖·‖∞"

    render_conditioning(metrics)

    st.subheader("Numerical Stability Metrics")

    rows = [
        {
            "Metric":  f"1. Residual  ‖r‖  ({norm_label})",
            "Value":   _fmt(metrics["residual_norm"]),
            "Formula": "r = b − A x",
            "Interpretation": (
                "How well x satisfies the linear system.  "
                "A small residual is necessary but not sufficient for accuracy."
            ),
        },
        {
            "Metric":  f"2. Forward error bound  ({norm_label})",
            "Value":   _fmt(metrics["forward_bound"]),
            "Formula": "κ(A) · ‖r‖ / (‖A‖ · ‖x‖)",
            "Interpretation": (
                "Upper bound on the relative error ‖x_true − x‖ / ‖x‖.  "
                "Controlled by both the conditioning and the residual."
            ),
        },
        {
            "Metric":  f"3. Backward error  ({norm_label})",
            "Value":   _fmt(metrics["backward_error"]),
            "Formula": "‖r‖ / (‖A‖ · ‖x‖ + ‖b‖)",
            "Interpretation": (
                "Smallest relative perturbation to [A, b] making x an exact solution.  "
                "A backward-stable solver gives this ≈ ε_mach."
            ),
        },
    ]

    df = pd.DataFrame(rows)
    st.table(df[["Metric", "Value", "Formula"]])

    with st.expander("Metric interpretations"):
        for row in rows:
            st.markdown(f"**{row['Metric']}** — {row['Interpretation']}")

    st.subheader("Intermediate norms")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"‖A‖  ({norm_label})", _fmt(metrics["norm_A"]))
    c2.metric(f"‖x‖  ({norm_label})", _fmt(metrics["norm_x"]))
    c3.metric(f"‖b‖  ({norm_label})", _fmt(metrics["norm_b"]))

    eps = np.finfo(float).eps
    prec = metrics.get("residual_prec", "float64")
    st.caption(
        f"Reference: ε_mach (float64) = {eps:.2e}.  "
        "A backward error near ε_mach indicates a backward-stable solve.  "
        f"Residual computed in: **{prec}**."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Orthogonality block (QR-based solvers only)
# ──────────────────────────────────────────────────────────────────────────────

def render_orthogonality(result: dict) -> None:
    """
    Display the orthogonality error ‖QᵀQ − I‖_F for QR-based solvers.

    - Householder QR  → O(ε_mach)
    - Modified GS     → O(ε_mach · κ(A))
    - Classical GS    → can be O(1) for ill-conditioned problems
    """
    if "Q" not in result:
        return

    Q   = result["Q"]
    err = orthogonality_error(Q)
    eps = np.finfo(float).eps

    st.subheader("Q-factor orthogonality")

    c1, c2 = st.columns(2)
    c1.metric(
        "‖QᵀQ − I‖_F",
        _fmt(err),
        help="Measures how far Q is from being exactly orthogonal.",
    )

    ratio = err / eps if eps > 0 else float("inf")
    c2.metric(
        "Ratio to ε_mach",
        f"{ratio:.2e}",
        help="Values ≫ 1 indicate significant orthogonality loss.",
    )

    if ratio < 100:
        st.success("Near-perfect orthogonality (ratio < 100 × ε_mach).")
    elif ratio < 1e6:
        st.info("Moderate orthogonality loss — typical for Modified GS on ill-conditioned problems.")
    else:
        st.warning(
            "Severe orthogonality loss — consistent with Classical GS on an ill-conditioned matrix."
        )
