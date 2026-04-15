"""
app.py
======
Numerical Linear Algebra Lab — main Streamlit entry point.

GPU support is threaded through via prob_params["use_gpu"].
All matrices, vectors and solutions live on the chosen device;
``device.to_numpy`` is called before any display / plot operation.
"""

from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st

from core.problem_creation import (
    create_matrix, create_rhs, apply_perturbation, matrix_info,
    compatible_structures, sparsity_mask,
)
from core.solvers import SOLVERS
from core.analysis import stability_analysis
from core.device import to_numpy, gpu_available, gpu_info
from ui.problem_ui import render_problem_ui, structure_label
from ui.solver_ui import render_solver_ui
from ui.analysis_ui import (
    render_array,
    render_heatmap,
    render_vector_heatmap,
    render_stability,
    render_orthogonality,
    _fmt,
    _bytes_to_human,
)

_SERIES_COLORS = [
    "#2980b9", "#e74c3c", "#27ae60", "#8e44ad",
    "#e67e22", "#16a085", "#c0392b", "#2c3e50",
    "#f39c12", "#1abc9c",
]


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NLA Lab", layout="wide")

st.title("Numerical Ax = b Lab")
st.caption(
    "Explore the conditioning and numerical stability of **Ax = b** solvers.  "
    "Configure the problem and solver in the sidebar, then click **Run Experiment**."
)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    prob_params   = render_problem_ui()   # now includes use_gpu
    solver_params = render_solver_ui()

    st.markdown("---")

    p_compare = prob_params["compare"]
    s_compare = solver_params["compare"]

    if p_compare["axis"] and s_compare["axis"]:
        st.warning(
            "Both a matrix/structure and a solver compare axis are active.  "
            "Please pick only one.  Using the matrix/structure axis."
        )
        s_compare = {"axis": None, "solver_values": [solver_params["solver_name"]]}

    if p_compare["axis"]:
        compare = p_compare
        compare["solver_values"] = [solver_params["solver_name"]]
    elif s_compare["axis"]:
        compare = {
            "axis":               "solver",
            "matrix_type_values": [prob_params["matrix_type"]],
            "structure_values":   [{"name": prob_params["structure"],
                                    "param": prob_params["struct_param"]}],
            "solver_values":      s_compare["solver_values"],
        }
    else:
        compare = {
            "axis":               None,
            "matrix_type_values": [prob_params["matrix_type"]],
            "structure_values":   [{"name": prob_params["structure"],
                                    "param": prob_params["struct_param"]}],
            "solver_values":      [solver_params["solver_name"]],
        }

    sw     = prob_params["sweep"]
    n_inst = (len(sw["m_values"])
              * len(sw["order_A_values"])
              * len(sw["order_b_values"]))

    if compare["axis"] == "matrix_type":
        n_series = len(compare["matrix_type_values"])
    elif compare["axis"] == "structure":
        n_series = len(compare["structure_values"])
    elif compare["axis"] == "solver":
        n_series = len(compare["solver_values"])
    else:
        n_series = 1

    n_total = n_series * n_inst
    if n_total > 1:
        st.caption(
            f"Experiment: **{n_series} series** × **{n_inst} instances** "
            f"= **{n_total} runs**."
        )

    # Show active device in sidebar footer
    use_gpu = prob_params.get("use_gpu", False)
    if use_gpu:
        info = gpu_info()
        dev  = info["devices"][0] if info["devices"] else "GPU"
        st.caption(f"🟢 Running on GPU: {dev}")
    else:
        st.caption("🔵 Running on CPU")

    run = st.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

if "series_list" not in st.session_state:
    st.session_state.series_list = None


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one instance
# ──────────────────────────────────────────────────────────────────────────────

def _run_instance(
    p: dict, solver_name: str,
    matrix_type_i: str,
    structure_i: str, struct_param_i: int,
    m_i: int, order_A_i: int, order_b_i: int,
) -> dict:
    use_gpu = p.get("use_gpu", False)
    try:
        # ── Build A ───────────────────────────────────────────────────────────
        if p.get("import_A") and p.get("imported_A_array") is not None:
            # Use imported array — cast dtype and transfer to GPU if needed
            A_cpu = p["imported_A_array"].astype(p["dtype_A"])
            if use_gpu:
                xp = __import__("cupy")
                A  = xp.asarray(A_cpu)
            else:
                A  = A_cpu
            # Sparsity mask for perturbation (inherited from imported matrix)
            A_mask = sparsity_mask(A_cpu)
        else:
            A      = create_matrix(
                matrix_type            = matrix_type_i,
                m                      = m_i,
                n                      = m_i,
                structure              = structure_i,
                struct_param           = struct_param_i,
                make_hermitian         = p["make_hermitian"],
                make_positive_definite = p["make_pd"],
                dtype                  = p["dtype_A"],
                seed                   = p["seed"],
                type_param             = p.get("type_param", 6),
                use_gpu                = use_gpu,
            )
            A_mask = None   # use structure-based perturbation

        # ── Build b ───────────────────────────────────────────────────────────
        if p.get("import_b") and p.get("imported_b_array") is not None:
            b_cpu = p["imported_b_array"].astype(p["dtype_b"])
            if use_gpu:
                xp = __import__("cupy")
                b  = xp.asarray(b_cpu)
            else:
                b  = b_cpu
        else:
            b = create_rhs(A.shape[0], p["dtype_b"], use_gpu=use_gpu)

        A_original = A.copy()
        b_original = b.copy()

        if p["perturb_A"]:
            A = apply_perturbation(
                A,
                order                  = order_A_i,
                structure              = structure_i,
                struct_param           = struct_param_i,
                make_hermitian         = p["make_hermitian"],
                make_positive_definite = p["make_pd"],
                use_gpu                = use_gpu,
                custom_mask            = A_mask,
            )
        if p["perturb_b"]:
            b = apply_perturbation(b, order=order_b_i, use_gpu=use_gpu)

        delta_A = (A - A_original) if p["perturb_A"] else None
        delta_b = (b - b_original) if p["perturb_b"] else None

        solver_fn = SOLVERS[solver_name]
        result    = solver_fn(A, b, use_gpu=use_gpu)
        metrics   = stability_analysis(A, b, result["x"],
                                       p.get("norm_type", "2"))

        def _cpu(arr):
            return to_numpy(arr) if arr is not None else None

        return {
            "error":        None,
            "matrix_type":  matrix_type_i,
            "structure":    structure_i,
            "struct_param": struct_param_i,
            "solver_name":  solver_name,
            "m":            m_i,
            "order_A":      order_A_i,
            "order_b":      order_b_i,
            "use_gpu":      use_gpu,
            "imported_A":   p.get("import_A", False),
            "imported_b":   p.get("import_b", False),
            "A":            _cpu(A),
            "b":            _cpu(b),
            "A_original":   _cpu(A_original),
            "b_original":   _cpu(b_original),
            "delta_A":      _cpu(delta_A),
            "delta_b":      _cpu(delta_b),
            "result":       {
                **result,
                "x": _cpu(result["x"]),
                **({} if "Q" not in result else {"Q": _cpu(result["Q"])}),
            },
            "metrics":      metrics,
        }

    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc), "m": m_i, "order_A": order_A_i,
                "order_b": order_b_i, "solver_name": solver_name}
    except MemoryError:
        return {"error": "Out of memory — reduce size or switch to a sparser structure.",
                "m": m_i, "order_A": order_A_i, "order_b": order_b_i,
                "solver_name": solver_name}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}", "m": m_i,
                "order_A": order_A_i, "order_b": order_b_i,
                "solver_name": solver_name}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build series list
# ──────────────────────────────────────────────────────────────────────────────

def _build_series_specs(p: dict, s: dict, cmp: dict) -> list[dict]:
    base_mt  = p["matrix_type"]
    base_st  = p["structure"]
    base_sp  = p["struct_param"]
    base_sol = s["solver_name"]
    axis     = cmp["axis"]

    if axis == "matrix_type":
        specs = []
        for mt in cmp["matrix_type_values"]:
            st_name = base_st if base_st in compatible_structures(mt) else "Dense"
            sp      = base_sp if st_name == base_st else 1
            specs.append({
                "label":        mt,
                "matrix_type":  mt,
                "structure":    st_name,
                "struct_param": sp,
                "solver_name":  base_sol,
            })
        return specs

    if axis == "structure":
        return [
            {
                "label":        structure_label(sv["name"], sv["param"]),
                "matrix_type":  base_mt,
                "structure":    sv["name"],
                "struct_param": sv["param"],
                "solver_name":  base_sol,
            }
            for sv in cmp["structure_values"]
        ]

    if axis == "solver":
        return [
            {
                "label":        sol,
                "matrix_type":  base_mt,
                "structure":    base_st,
                "struct_param": base_sp,
                "solver_name":  sol,
            }
            for sol in cmp["solver_values"]
        ]

    return [
        {
            "label":        _single_series_label(p, s),
            "matrix_type":  base_mt,
            "structure":    base_st,
            "struct_param": base_sp,
            "solver_name":  base_sol,
        }
    ]


def _single_series_label(p: dict, s: dict) -> str:
    device = "GPU" if p.get("use_gpu") else "CPU"
    parts  = [p["matrix_type"],
              structure_label(p["structure"], p["struct_param"]),
              s["solver_name"],
              device]
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment execution
# ──────────────────────────────────────────────────────────────────────────────

if run:
    p   = prob_params
    s   = solver_params
    sw  = p["sweep"]
    cmp = compare

    p["norm_type"] = s["norm_type"]

    series_specs = _build_series_specs(p, s, cmp)
    combos       = list(itertools.product(
        sw["m_values"],
        sw["order_A_values"],
        sw["order_b_values"],
    ))

    total_runs  = len(series_specs) * len(combos)
    bar         = st.progress(0, text="Running…")
    run_idx     = 0
    series_list = []

    for spec in series_specs:
        instances = []
        for (m_i, oA_i, ob_i) in combos:
            run_idx += 1
            bar.progress(
                run_idx / total_runs,
                text=(f"[{run_idx}/{total_runs}]  "
                      f"Series: {spec['label']}  |  "
                      f"m={m_i}, ord_A={oA_i}, ord_b={ob_i}  "
                      f"{'[GPU]' if p.get('use_gpu') else '[CPU]'}"),
            )
            inst = _run_instance(
                p,
                solver_name    = spec["solver_name"],
                matrix_type_i  = spec["matrix_type"],
                structure_i    = spec["structure"],
                struct_param_i = spec["struct_param"],
                m_i            = m_i,
                order_A_i      = oA_i,
                order_b_i      = ob_i,
            )
            inst["prob_params"]   = p
            inst["solver_params"] = s
            instances.append(inst)

        series_list.append({
            "label":     spec["label"],
            "instances": instances,
            "spec":      spec,
        })

    bar.empty()
    st.session_state.series_list = series_list
    st.session_state.compare     = cmp


# ──────────────────────────────────────────────────────────────────────────────
# Sweep / instance helpers  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_axis_label(sweep_param: str | None) -> str:
    return {
        "m":               "Matrix size  m",
        "perturb_A_order": "Perturbation order  k  (‖ΔA‖ ≈ 10^k · ‖A‖)",
        "perturb_b_order": "Perturbation order  k  (‖Δb‖ ≈ 10^k · ‖b‖)",
    }.get(sweep_param or "", "Instance")


def _sweep_x_values(instances: list, sweep_param: str | None) -> list:
    if sweep_param == "m":
        return [inst["m"] for inst in instances]
    if sweep_param == "perturb_A_order":
        return [inst["order_A"] for inst in instances]
    if sweep_param == "perturb_b_order":
        return [inst["order_b"] for inst in instances]
    return list(range(1, len(instances) + 1))


def _get_sweep_param(series_list: list) -> str | None:
    try:
        return series_list[0]["instances"][0]["prob_params"]["sweep"]["param"]
    except (IndexError, KeyError):
        return None


def _format_x_ticks(ax, x_vals: list, sweep_param: str | None) -> None:
    if sweep_param in ("perturb_A_order", "perturb_b_order"):
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"10^{v}" for v in x_vals], fontsize=7)


def _instance_label(inst: dict, n_total: int, idx: int,
                    sweep_param: str | None) -> str:
    if n_total == 1:
        return f"m={inst['m']}"
    if sweep_param == "m":
        return f"m = {inst['m']}"
    if sweep_param == "perturb_A_order":
        return f"ord_A = 10^{inst['order_A']}"
    if sweep_param == "perturb_b_order":
        return f"ord_b = 10^{inst['order_b']}"
    return f"#{idx + 1}"


# ──────────────────────────────────────────────────────────────────────────────
# Render sections 1 and 2  (all arrays are already CPU numpy here)
# ──────────────────────────────────────────────────────────────────────────────

def _render_instance(inst: dict) -> None:
    if inst.get("error"):
        st.error(inst["error"])
        return

    A       = inst["A"]
    b       = inst["b"]
    result  = inst["result"]
    metrics = inst["metrics"]

    device_badge = "🟢 GPU" if inst.get("use_gpu") else "🔵 CPU"
    A_badge      = "📂 imported" if inst.get("imported_A") else "🔧 generated"
    b_badge      = "📂 imported" if inst.get("imported_b") else "🔧 generated"

    st.header("1. Problem creation (A, b)")
    st.caption(f"Device: {device_badge}  ·  A: {A_badge}  ·  b: {b_badge}")

    info = matrix_info(A)
    m, n = info["shape"]

    col_A_hdr, col_b_hdr = st.columns([5, 1])
    with col_A_hdr:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Shape",     f"{m} × {n}")
        c2.metric("dtype",     info["dtype"])
        c3.metric("Non-zeros", f"{info['nnz']:,}")
        c4.metric("Density",   f"{info['density']:.2%}")
        c5.metric("Memory",    _bytes_to_human(info["memory_bytes"]))
    with col_b_hdr:
        st.metric("dtype",  str(b.dtype))
        st.metric("Memory", _bytes_to_human(b.nbytes))

    col_A, col_b = st.columns([5, 1])
    with col_A:
        tab_tbl, tab_heat = st.tabs(["Entries", "Heatmap"])
        with tab_tbl:
            render_array("A", A)
        with tab_heat:
            render_heatmap("A", A)
    with col_b:
        tab_tbl_b, tab_heat_b = st.tabs(["Entries", "Heatmap"])
        with tab_tbl_b:
            render_array("b", b)
        with tab_heat_b:
            render_vector_heatmap("b", b)

    if inst["delta_A"] is not None or inst["delta_b"] is not None:
        st.markdown("**Perturbation**")
        col_dA, col_db = st.columns([5, 1])
        with col_dA:
            if inst["delta_A"] is not None:
                tab_tbl_dA, tab_heat_dA = st.tabs(["ΔA entries", "ΔA heatmap"])
                with tab_tbl_dA:
                    render_array("ΔA", inst["delta_A"])
                with tab_heat_dA:
                    render_heatmap("ΔA", inst["delta_A"])
            else:
                st.caption("No perturbation applied to A.")
        with col_db:
            if inst["delta_b"] is not None:
                tab_tbl_db, tab_heat_db = st.tabs(["Δb entries", "Δb heatmap"])
                with tab_tbl_db:
                    render_array("Δb", inst["delta_b"])
                with tab_heat_db:
                    render_vector_heatmap("Δb", inst["delta_b"])
            else:
                st.caption("No perturbation applied to b.")

    st.divider()

    st.header(f"2. x̃ via {result['method']}")

    if result["success"]:
        st.success(result["message"])
    else:
        st.warning(result["message"])

    x = result["x"]
    col_x, col_empty = st.columns([1, 5])
    with col_x:
        st.metric("dtype",  str(x.dtype))
        st.metric("Memory", _bytes_to_human(x.nbytes))
        tab_tbl_x, tab_heat_x = st.tabs(["Entries", "Heatmap"])
        with tab_tbl_x:
            render_array("x̃", x)
        with tab_heat_x:
            render_vector_heatmap("x̃", x)

    if "Q" in result:
        with st.expander("Q-factor orthogonality (QR solvers)", expanded=False):
            render_orthogonality(result)

    st.divider()

    with st.expander("Stability analysis (detailed)", expanded=False):
        render_stability(metrics)


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — κ(A) plot  (identical logic, arrays already CPU)
# ──────────────────────────────────────────────────────────────────────────────

def _render_kappa_plot(series_list: list, sweep_param: str | None) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    eps        = np.finfo(float).eps
    kappa_max  = 1.0 / eps
    digits_tot = -np.log10(eps)

    all_good = [(s, [i for i in s["instances"] if not i.get("error")])
                for s in series_list]
    all_good = [(s, insts) for s, insts in all_good if insts]
    if not all_good:
        st.warning("No valid instances to plot.")
        return

    single_inst = sum(len(insts) for _, insts in all_good) == 1

    if single_inst and len(all_good) == 1:
        kappa = all_good[0][1][0]["metrics"]["kappa"]
        log_k = np.log10(kappa) if kappa > 0 and not np.isinf(kappa) else digits_tot
        digits_lost = min(log_k, digits_tot)

        fig, ax = plt.subplots(figsize=(5, 2.2))
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", extent=[0, digits_tot, -0.4, 0.4],
                  cmap="RdYlGn_r", alpha=0.25, zorder=0)
        ax.axvline(digits_lost, color="#c0392b", lw=2.5, zorder=3)
        ax.text(digits_lost + 0.15, 0.25,
                f"log₁₀(κ) = {log_k:.1f}\n({digits_lost:.1f} digits lost)",
                fontsize=8, color="#c0392b", va="center")
        ax.axvline(0, color="#27ae60", lw=1.2, ls="--", zorder=2)
        ax.text(0.15, -0.28, "ε_mach", fontsize=7, color="#27ae60")
        ax.set_xlim(-0.5, digits_tot + 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Significant digits lost  = log₁₀(κ(A))", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"κ(A) = {_fmt(kappa)}", fontsize=9, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if np.isinf(kappa):
            st.error("κ(A) = ∞ — matrix is numerically singular.")
        elif kappa > kappa_max / 10:
            st.warning(f"κ(A) ≈ {_fmt(kappa)} — effectively singular.")
        elif digits_lost > digits_tot * 0.75:
            st.warning(f"κ(A) ≈ {_fmt(kappa)} — severely ill-conditioned.")
        elif digits_lost > 4:
            st.info(f"κ(A) ≈ {_fmt(kappa)} — moderately ill-conditioned.")
        else:
            st.success(f"κ(A) ≈ {_fmt(kappa)} — well-conditioned.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhspan(np.log10(kappa_max) - 2, digits_tot + 1,
               color="#e74c3c", alpha=0.07, zorder=0)
    ax.axhline(np.log10(kappa_max), color="#e74c3c", lw=1.0, ls="--",
               alpha=0.6, label=f"1/ε_mach")
    ax.axhline(0, color="#27ae60", lw=1.0, ls=":", alpha=0.7, label="κ = 1")

    x_label = _sweep_axis_label(sweep_param)
    all_x   = []

    for si, (series, insts) in enumerate(all_good):
        color      = _SERIES_COLORS[si % len(_SERIES_COLORS)]
        kappas     = [i["metrics"]["kappa"] for i in insts]
        log_kappas = [
            np.log10(k) if k > 0 and not np.isinf(k) else digits_tot
            for k in kappas
        ]
        x_vals = _sweep_x_values(insts, sweep_param)
        all_x.extend(x_vals)
        ax.plot(x_vals, log_kappas, color=color, lw=2.0, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.8,
                zorder=4, label=series["label"])
        if len(insts) <= 10:
            for xv, lk in zip(x_vals, log_kappas):
                ax.annotate(f"{lk:.1f}", xy=(xv, lk), xytext=(0, 6),
                            textcoords="offset points",
                            fontsize=7, ha="center", color=color)

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("log₁₀(κ(A))", fontsize=9)
    ax.set_title("Condition number κ(A)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=8)
    _format_x_ticks(ax, sorted(set(all_x)), sweep_param)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"ε_mach = {eps:.2e},  1/ε_mach ≈ {kappa_max:.2e}.")


def _render_section3(series_list: list) -> None:
    st.header("3. Problem specific sensitivity metrics")
    sweep_param = _get_sweep_param(series_list)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**κ(A)**")
        _render_kappa_plot(series_list, sweep_param)
    with col2:
        st.markdown("**—**")
        st.caption("Coming soon.")
    with col3:
        st.markdown("**—**")
        st.caption("Coming soon.")


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Solution quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def _make_metric_plot(
    series_list: list, sweep_param: str | None,
    metric_key: str, title: str, ylabel: str,
    color_single: str, single_label: str,
) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    eps     = np.finfo(float).eps
    log_eps = np.log10(eps)

    all_good = [(s, [i for i in s["instances"] if not i.get("error")])
                for s in series_list]
    all_good = [(s, insts) for s, insts in all_good if insts]
    if not all_good:
        st.warning("No valid instances to plot.")
        return

    def _safe_log(v: float) -> float:
        if v is None or np.isnan(v): return log_eps
        if np.isinf(v) or v <= 0:   return 0.0
        return np.log10(v)

    single_inst = sum(len(insts) for _, insts in all_good) == 1

    if single_inst and len(all_good) == 1:
        val     = all_good[0][1][0]["metrics"][metric_key]
        log_val = _safe_log(val)
        lo, hi  = log_eps - 1, 2.0

        fig, ax = plt.subplots(figsize=(5, 2.0))
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", extent=[lo, hi, -0.4, 0.4],
                  cmap="RdYlGn", alpha=0.22, zorder=0)
        ax.axvline(log_val,  color=color_single, lw=2.5, zorder=3)
        ax.axvline(log_eps,  color="#7f8c8d",    lw=1.2, ls="--", zorder=2)
        ax.axvline(0.0,      color="#27ae60",    lw=1.0, ls=":",  zorder=2)
        ax.text(log_val + 0.1, 0.28, f"log₁₀ = {log_val:.1f}",
                fontsize=8, color=color_single, va="center")
        ax.text(log_eps + 0.1, -0.30, "ε_mach", fontsize=7, color="#7f8c8d")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(f"log₁₀({single_label})", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"{title}  =  {_fmt(val)}", fontsize=9, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if log_val <= log_eps + 1:
            st.success(f"{single_label} ≈ ε_mach — excellent.")
        elif log_val <= log_eps + 4:
            st.info(f"{single_label} ≈ {_fmt(val)} — good.")
        elif log_val <= -4:
            st.info(f"{single_label} ≈ {_fmt(val)} — acceptable.")
        else:
            st.warning(f"{single_label} ≈ {_fmt(val)} — large, check conditioning.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhline(log_eps, color="#7f8c8d", lw=1.0, ls="--", alpha=0.7,
               label=f"ε_mach")
    ax.axhspan(log_eps - 2, log_eps + 1,
               color="#27ae60", alpha=0.07, label="Near ε_mach")

    x_label = _sweep_axis_label(sweep_param)
    all_x   = []

    for si, (series, insts) in enumerate(all_good):
        color    = _SERIES_COLORS[si % len(_SERIES_COLORS)]
        vals     = [i["metrics"][metric_key] for i in insts]
        log_vals = [_safe_log(v) for v in vals]
        x_vals   = _sweep_x_values(insts, sweep_param)
        all_x.extend(x_vals)
        ax.plot(x_vals, log_vals, color=color, lw=2.0, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.8,
                zorder=4, label=series["label"])
        if len(insts) <= 10:
            for xv, lv in zip(x_vals, log_vals):
                ax.annotate(f"{lv:.1f}", xy=(xv, lv), xytext=(0, 6),
                            textcoords="offset points",
                            fontsize=7, ha="center", color=color)

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(ylabel,   fontsize=9)
    ax.set_title(title,     fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=8)
    _format_x_ticks(ax, sorted(set(all_x)), sweep_param)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"ε_mach = {eps:.2e}.")


def _render_section4(series_list: list) -> None:
    st.header("4. Solution quality metrics")
    sweep_param = _get_sweep_param(series_list)
    try:
        norm_type = series_list[0]["instances"][0]["solver_params"]["norm_type"]
    except (IndexError, KeyError):
        norm_type = "2"
    norm_label = "‖·‖₂" if norm_type == "2" else "‖·‖∞"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Residual  ‖r‖  ({norm_label})**")
        _make_metric_plot(series_list, sweep_param,
                          metric_key="residual_norm",
                          title=f"Residual  ‖r‖  ({norm_label})",
                          ylabel="log₁₀(‖r‖)",
                          color_single="#2980b9", single_label="‖r‖")
    with col2:
        st.markdown(f"**Forward error bound  ({norm_label})**")
        _make_metric_plot(series_list, sweep_param,
                          metric_key="forward_bound",
                          title=f"Forward error bound  ({norm_label})",
                          ylabel="log₁₀(FEB)",
                          color_single="#e67e22", single_label="FEB")
    with col3:
        st.markdown(f"**Backward error  ({norm_label})**")
        _make_metric_plot(series_list, sweep_param,
                          metric_key="backward_error",
                          title=f"Backward error  ({norm_label})",
                          ylabel="log₁₀(BE)",
                          color_single="#8e44ad", single_label="BE")


# ──────────────────────────────────────────────────────────────────────────────
# Main display
# ──────────────────────────────────────────────────────────────────────────────

series_list = st.session_state.series_list

if series_list is None:
    st.info(
        "Configure the problem and solver in the sidebar, "
        "then click **Run Experiment**."
    )
else:
    sweep_param = _get_sweep_param(series_list)

    n_series = len(series_list)
    if n_series == 1:
        active_series = series_list[0]
    else:
        series_labels = [s["label"] for s in series_list]
        si = st.selectbox(
            f"Series  ({n_series} total — compare axis)",
            options=range(n_series),
            format_func=lambda i: series_labels[i],
            index=0,
        )
        active_series = series_list[si]

    instances = active_series["instances"]
    n_inst    = len(instances)

    if n_inst == 1:
        active_inst = instances[0]
    else:
        inst_labels = [
            _instance_label(inst, n_inst, i, sweep_param)
            for i, inst in enumerate(instances)
        ]
        ii = st.selectbox(
            f"Instance  ({n_inst} per series — sweep axis)",
            options=range(n_inst),
            format_func=lambda i: inst_labels[i],
            index=0,
        )
        active_inst = instances[ii]

    _render_instance(active_inst)

    _render_section3(series_list)
    st.divider()

    _render_section4(series_list)
    st.divider()

    st.header("5. Solver behaviour metrics")
    st.caption("Coming soon.")
    st.divider()

    st.header("6. Structural metrics")
    st.caption("Coming soon.")
    st.divider()

    st.header("7. Summary")
    st.caption("Coming soon.")
    st.divider()

    st.header("8. Save results")
    st.caption("Coming soon.")
