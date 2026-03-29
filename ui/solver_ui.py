"""
ui/solver_ui.py
===============
Streamlit sidebar widgets for solver and norm selection.

When compare axis = "solver" is selected here, a multiselect lets the user
pick several solvers.  Each becomes a separate series (legend) in the plots.

Returns
-------
render_solver_ui() -> dict with keys:
    solver_name : str          — the primary / fixed solver
    norm_type   : '2' | 'inf'
    compare: {
        axis         : "solver" | None
        solver_values: list[str]   — solvers to compare (includes solver_name)
    }
"""

from __future__ import annotations

import streamlit as st
from core.solvers import SOLVERS


SOLVER_NOTES: dict[str, str] = {
    "SVD (Reduced)": (
        "Thin SVD A = UΣVᵀ (U: m×k, Σ: k×k, Vᵀ: k×n, k = min(m,n)). "
        "Gives minimum-norm least-squares solution. Best stability, highest cost."
    ),
    "SVD (Full)": (
        "Full SVD A = UΣVᵀ (U: m×m, Vᵀ: n×n). "
        "Identical numerical result to Reduced SVD for the solution x."
    ),
    "QR Householder (Reduced)": (
        "Economy QR via LAPACK Householder reflections. "
        "Backward stable: orthogonality error ≈ ε_mach."
    ),
    "QR Householder (Full)": (
        "Full QR via LAPACK Householder reflections. "
        "Same stability as Reduced; Q is square (m×m)."
    ),
    "QR Classical Gram-Schmidt": (
        "Manual CGS QR. All projections use the original column a_j. "
        "Numerically unstable for ill-conditioned A: orthogonality can be lost entirely."
    ),
    "QR Modified Gram-Schmidt": (
        "Manual MGS QR. Projections use the current (updated) vector. "
        "Much better than CGS; orthogonality error ≈ ε_mach · κ(A)."
    ),
    "LU": (
        "PA = LU with partial pivoting via LAPACK. "
        "Requires square A. Backward stable for most matrices."
    ),
    "Cholesky": (
        "A = LLᵀ via LAPACK. Requires square, symmetric, positive definite A. "
        "Twice as fast as LU; numerically stable for SPD systems."
    ),
    "GMRES": (
        "Generalised Minimum Residual. Works for any non-singular square A. "
        "Minimises ‖r‖ over growing Krylov subspaces. No symmetry required."
    ),
    "CG (Conjugate Gradient)": (
        "Conjugate Gradient. Requires square SPD A. "
        "Minimises the A-norm of the error. Convergence rate ~ sqrt(κ(A))."
    ),
}

NORM_NOTES = {
    "2":   "‖v‖₂ = √(Σ vᵢ²),  ‖A‖₂ = σ_max(A)  — expensive for large A.",
    "inf": "‖v‖∞ = max|vᵢ|,  ‖A‖∞ = max row-sum  — always O(mn).",
}

ALL_SOLVERS = list(SOLVERS.keys())


def render_solver_ui() -> dict:
    """
    Render solver and norm selection widgets.

    Returns
    -------
    dict with keys: solver_name, norm_type, compare
    """
    st.markdown("---")
    st.header("Solver")

    solver_name = st.selectbox("Method", ALL_SOLVERS)

    with st.expander("Solver info", expanded=False):
        st.caption(SOLVER_NOTES.get(solver_name, ""))

    # ── Solver compare ────────────────────────────────────────────────────────
    compare_solvers = st.checkbox(
        "Compare solvers",
        key="compare_solvers_checkbox",
        help="Select multiple solvers to compare as separate series in plots.",
    )

    solver_values     = [solver_name]
    solver_compare_axis = None

    if compare_solvers:
        solver_values = st.multiselect(
            "Solvers to compare",
            options = ALL_SOLVERS,
            default = [solver_name],
            key     = "compare_solver_values",
        )
        if not solver_values:
            st.warning("Select at least one solver.")
            solver_values = [solver_name]
        # Ensure the primary solver is always first
        if solver_name in solver_values:
            solver_values = [solver_name] + [s for s in solver_values
                                             if s != solver_name]
        solver_compare_axis = "solver"
        st.caption("✓ " + ",  ".join(solver_values))

    # ── Norm ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Norm")

    norm_type = st.radio("Norm for analysis", ["2", "inf"], horizontal=True)
    st.caption(NORM_NOTES[norm_type])

    return {
        "solver_name":  solver_name,
        "norm_type":    norm_type,
        "compare": {
            "axis":          solver_compare_axis,
            "solver_values": solver_values,
        },
    }
