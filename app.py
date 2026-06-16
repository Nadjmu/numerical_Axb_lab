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

if "crop_count_lab1" not in st.session_state:
    st.session_state.crop_count_lab1 = 1


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
            # Use imported array — preserve original dtype, transfer to GPU if needed
            A_cpu = p["imported_A_array"]
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
            b_cpu = p["imported_b_array"]
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
    st.session_state.pop("_lab1_pdf_bytes", None)


# ──────────────────────────────────────────────────────────────────────────────
# Sweep / instance helpers  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_axis_label(sweep_param: str | None) -> str:
    return {
        "m":               "Matrix size  m",
        "perturb_A_order": "Perturbation order  k  (||dA|| ~ 10^k * ||A||)",
        "perturb_b_order": "Perturbation order  k  (||db|| ~ 10^k * ||b||)",
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

    info = matrix_info(A)
    m, n = info["shape"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Problem creation (A, b)")
        st.caption(f"Device: {device_badge}  ·  A: {A_badge}  ·  b: {b_badge}")

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

        crop_bound = min(A.shape[0], A.shape[1])
        _n_crops = st.session_state.crop_count_lab1
        for _ci in range(_n_crops):
            with st.expander(
                f"Zoom: crop #{_ci + 1}  A[n_low : n_high, n_low : n_high]",
                expanded=(_ci == 0),
            ):
                crop_c1, crop_c2 = st.columns(2)
                with crop_c1:
                    crop_low = int(st.number_input(
                        "n_low  (inclusive)", min_value=0,
                        max_value=crop_bound - 1, value=0, step=1,
                        key=f"crop_low_{_ci}"))
                with crop_c2:
                    crop_high = int(st.number_input(
                        "n_high  (exclusive)", min_value=1,
                        max_value=crop_bound, value=min(crop_bound, 10), step=1,
                        key=f"crop_high_{_ci}"))
                if crop_low >= crop_high:
                    st.warning("n_low must be strictly less than n_high.")
                else:
                    crop_sub = A[crop_low:crop_high, crop_low:crop_high]
                    crop_sz  = crop_high - crop_low
                    st.caption(
                        f"A[{crop_low}:{crop_high}, {crop_low}:{crop_high}]  —  "
                        f"{crop_sz} × {crop_sz}"
                    )
                    crop_tab_e, crop_tab_h = st.tabs(["Entries", "Heatmap"])
                    with crop_tab_e:
                        render_array(f"A[{crop_low}:{crop_high}]", crop_sub)
                    with crop_tab_h:
                        render_heatmap(f"A[{crop_low}:{crop_high}]", crop_sub)
        if st.button("+ Add another crop", key="add_crop_lab1"):
            st.session_state.crop_count_lab1 += 1
            st.rerun()

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

    with col2:
        st.subheader(f"2. x̃ via {result['method']}")

        if result["success"]:
            st.success(result["message"])
        else:
            st.warning(result["message"])

        x = result["x"]
        col_x, col_stab = st.columns([1, 5])
        with col_x:
            st.metric("dtype",  str(x.dtype))
            st.metric("Memory", _bytes_to_human(x.nbytes))
            tab_tbl_x, tab_heat_x = st.tabs(["Entries", "Heatmap"])
            with tab_tbl_x:
                render_array("x̃", x)
            with tab_heat_x:
                render_vector_heatmap("x̃", x)
        with col_stab:
            with st.expander("Stability analysis (detailed)", expanded=False):
                render_stability(metrics)
            if "Q" in result:
                with st.expander("Q-factor orthogonality (QR solvers)", expanded=False):
                    render_orthogonality(result)

    st.divider()


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
                f"log10(kappa) = {log_k:.1f}\n({digits_lost:.1f} digits lost)",
                fontsize=8, color="#c0392b", va="center")
        ax.axvline(0, color="#27ae60", lw=1.2, ls="--", zorder=2)
        ax.text(0.15, -0.28, "eps_mach", fontsize=7, color="#27ae60")
        ax.set_xlim(-0.5, digits_tot + 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Significant digits lost = log10(kappa(A))", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"kappa(A) = {_fmt(kappa)}", fontsize=9, fontweight="bold")
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
               alpha=0.6, label="1/eps_mach")
    ax.axhline(0, color="#27ae60", lw=1.0, ls=":", alpha=0.7, label="kappa = 1")

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
    ax.set_ylabel("log10(kappa(A))", fontsize=9)
    ax.set_title("Condition number kappa(A)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=8)
    _format_x_ticks(ax, sorted(set(all_x)), sweep_param)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"ε_mach = {eps:.2e},  1/ε_mach ≈ {kappa_max:.2e}.")


def _render_section3(series_list: list) -> None:
    st.subheader("3. Problem specific sensitivity metrics")
    sweep_param = _get_sweep_param(series_list)
    st.markdown("**κ(A)**")
    _render_kappa_plot(series_list, sweep_param)
    st.markdown("**—**")
    st.caption("Coming soon.")
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
        ax.text(log_val + 0.1, 0.28, f"log10 = {log_val:.1f}",
                fontsize=8, color=color_single, va="center")
        ax.text(log_eps + 0.1, -0.30, "eps_mach", fontsize=7, color="#7f8c8d")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(f"log10({single_label})", fontsize=8)
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
               label="eps_mach")
    ax.axhspan(log_eps - 2, log_eps + 1,
               color="#27ae60", alpha=0.07, label="Near eps_mach")

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
    st.subheader("4. Solution quality metrics")
    sweep_param = _get_sweep_param(series_list)
    try:
        norm_type = series_list[0]["instances"][0]["solver_params"]["norm_type"]
    except (IndexError, KeyError):
        norm_type = "2"
    norm_label      = "||.||_2" if norm_type == "2" else "||.||_inf"
    norm_label_html = "‖·‖₂"   if norm_type == "2" else "‖·‖∞"

    st.markdown(f"**Residual  ‖r‖  ({norm_label_html})**")
    _make_metric_plot(series_list, sweep_param,
                      metric_key="residual_norm",
                      title=f"Residual ||r|| ({norm_label})",
                      ylabel="log10(||r||)",
                      color_single="#2980b9", single_label="||r||")
    st.markdown(f"**Forward error bound  ({norm_label_html})**")
    _make_metric_plot(series_list, sweep_param,
                      metric_key="forward_bound",
                      title=f"Forward error bound ({norm_label})",
                      ylabel="log10(FEB)",
                      color_single="#e67e22", single_label="FEB")
    st.markdown(f"**Backward error  ({norm_label_html})**")
    _make_metric_plot(series_list, sweep_param,
                      metric_key="backward_error",
                      title=f"Backward error ({norm_label})",
                      ylabel="log10(BE)",
                      color_single="#8e44ad", single_label="BE")


# ──────────────────────────────────────────────────────────────────────────────
# Sections 7 & 8 — summary and PDF export
# ──────────────────────────────────────────────────────────────────────────────

def _build_summary_text(active_inst: dict, series_list: list) -> str:
    p       = active_inst.get("prob_params", {})
    s       = active_inst.get("solver_params", {})
    res     = active_inst["result"]
    metrics = active_inst["metrics"]
    A       = active_inst["A"]
    b       = active_inst["b"]
    x       = res["x"]

    device_str = "GPU" if active_inst.get("use_gpu") else "CPU"
    src_A  = "imported (.npy)" if active_inst.get("imported_A") else active_inst.get("matrix_type", "N/A")
    src_b  = "imported (.npy)" if active_inst.get("imported_b") else "generated (random)"

    lines = [
        "=== Numerical Ax = b Lab — Experiment Summary ===",
        "",
        "-- PROBLEM --",
        f"  Source (A)    : {src_A}",
        f"  Source (b)    : {src_b}",
        f"  Structure     : {active_inst.get('structure', 'N/A')} (param={active_inst.get('struct_param', 'N/A')})",
        f"  Size          : {active_inst['m']} x {active_inst['m']}",
        f"  dtype A       : {A.dtype}",
        f"  dtype b       : {b.dtype}",
        f"  Seed          : {p.get('seed', 'N/A')}",
        f"  Hermitian     : {p.get('make_hermitian', False)}",
        f"  Positive def. : {p.get('make_pd', False)}",
        f"  Perturbed A   : {p.get('perturb_A', False)}"
        + (f" (order 10^{active_inst.get('order_A')})" if p.get("perturb_A") else ""),
        f"  Perturbed b   : {p.get('perturb_b', False)}"
        + (f" (order 10^{active_inst.get('order_b')})" if p.get("perturb_b") else ""),
        f"  Device        : {device_str}",
        "",
        "-- SOLVER --",
        f"  Method        : {res.get('method', 'N/A')}",
        f"  Norm type     : {p.get('norm_type', '2')} ({'||.||_2 Euclidean' if p.get('norm_type','2') == '2' else '||.||_inf max'}) ",
        f"  Success       : {res.get('success', 'N/A')}",
        f"  Message       : {res.get('message', '')}",
        "",
        "-- NUMERICAL ANALYSIS --",
        f"  kappa(A)           : {metrics['kappa']:.6e}",
        f"  ||A||              : {metrics['norm_A']:.6e}",
        f"  ||x~||             : {metrics['norm_x']:.6e}",
        f"  ||b||              : {metrics['norm_b']:.6e}",
        f"  ||r|| (residual)   : {metrics['residual_norm']:.6e}",
        f"  Forward err. bound : {metrics['forward_bound']:.6e}",
        f"  Backward error     : {metrics['backward_error']:.6e}",
        f"  Residual precision : {metrics['residual_prec']}",
    ]

    if "Q" in res and res.get("Q") is not None:
        Q    = res["Q"]
        orth = float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), ord="fro"))
        lines.append(f"  Q orth. error      : {orth:.6e}")

    lines += ["", "-- SOLUTION x~ (first 10 entries) --"]
    for i in range(min(10, len(x))):
        xi = x[i]
        if np.iscomplexobj(x):
            lines.append(f"  x~[{i:3d}]: Re={xi.real:+.6e}  Im={xi.imag:+.6e}")
        else:
            lines.append(f"  x~[{i:3d}]: {float(xi):+.6e}")
    if len(x) > 10:
        lines.append(f"  ... ({len(x) - 10} more entries)")

    lines += [
        "",
        "-- EXPERIMENT SCALE --",
        f"  Series            : {len(series_list)}",
        f"  Instances/series  : {len(series_list[0]['instances']) if series_list else 0}",
        "",
        "=== END SUMMARY ===",
    ]
    return "\n".join(lines)


def _build_pdf_bytes(active_inst: dict, series_list: list) -> bytes:
    import io
    from matplotlib.backends.backend_pdf import PdfPages

    summary = _build_summary_text(active_inst, series_list)
    A       = active_inst["A"]
    metrics = active_inst["metrics"]
    sweep_p = _get_sweep_param(series_list)
    eps     = np.finfo(float).eps

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # Page 1 — summary text
        fig = plt.figure(figsize=(8.5, 11))
        ax  = fig.add_axes([0.05, 0.03, 0.90, 0.94])
        ax.axis("off")
        ax.text(0, 1, summary, fontsize=6.5, family="monospace",
                va="top", ha="left", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2 — A heatmap (Re + Im side-by-side for complex matrices)
        is_cx  = np.iscomplexobj(A)
        parts  = [("Re(A)", A.real), ("Im(A)", A.imag)] if is_cx else [("A", A)]
        ncols  = len(parts)
        pal    = sns.diverging_palette(220, 10, as_cmap=True)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
        if ncols == 1:
            axes = [axes]
        for ax, (part_lbl, data) in zip(axes, parts):
            if min(A.shape) > 150:
                ax.spy(data, markersize=1, color="#2980b9")
                ax.set_title(f"{part_lbl} — sparsity pattern", fontsize=10, fontweight="bold")
            else:
                absmax_d  = float(np.nanmax(np.abs(data))) or 1.0
                linthresh = max(absmax_d * 1e-3, np.finfo(float).tiny * 10)
                import matplotlib.colors as mcolors
                norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=0.5,
                                          vmin=-absmax_d, vmax=absmax_d, base=10)
                sns.heatmap(data, ax=ax, cmap=pal, norm=norm,
                            annot=min(A.shape) <= 20, fmt=".2g",
                            linewidths=0.3 if min(A.shape) <= 20 else 0,
                            cbar_kws={"label": "value (log scale)"})
                ax.set_title(part_lbl, fontsize=10, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3 — kappa(A) gauge
        kappa      = metrics["kappa"]
        digits_tot = -np.log10(eps)
        log_k      = (np.log10(kappa) if kappa > 0 and not np.isinf(kappa) else digits_tot)
        dl         = min(log_k, digits_tot)
        fig, ax = plt.subplots(figsize=(8, 2.6))
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", extent=[0, digits_tot, -0.4, 0.4],
                  cmap="RdYlGn_r", alpha=0.30, zorder=0)
        ax.axvline(dl, color="#c0392b", lw=2.5, zorder=3)
        ax.text(dl + 0.2, 0.22, f"log10(kappa) = {log_k:.2f}\n({dl:.1f} digits lost)",
                fontsize=9, color="#c0392b")
        ax.axvline(0, color="#27ae60", lw=1.2, ls="--", zorder=2)
        ax.set_xlabel("Significant digits lost  =  log10(kappa(A))", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(-0.5, digits_tot + 1)
        ax.set_title(f"Condition number  kappa(A) = {_fmt(kappa)}", fontsize=10, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4 — residual / FEB / backward error gauges
        lo_g = np.log10(eps) - 1
        hi_g = 2.0
        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        for ax_i, (key, lbl, color) in enumerate([
            ("residual_norm",  "||r||  (residual)",      "#2980b9"),
            ("forward_bound",  "Forward error bound",    "#e67e22"),
            ("backward_error", "Backward error",         "#8e44ad"),
        ]):
            val = metrics[key]
            lv  = (np.log10(val) if val is not None and val > 0 and not np.isinf(val)
                   else np.log10(eps))
            grad = np.linspace(0, 1, 256).reshape(1, -1)
            axes[ax_i].imshow(grad, aspect="auto", extent=[lo_g, hi_g, -0.4, 0.4],
                              cmap="RdYlGn", alpha=0.25)
            axes[ax_i].axvline(lv, color=color, lw=2.5)
            axes[ax_i].axvline(np.log10(eps), color="#7f8c8d", lw=1.2, ls="--")
            axes[ax_i].set_xlabel("log10", fontsize=8)
            axes[ax_i].set_yticks([])
            axes[ax_i].set_title(f"{lbl}\n= {_fmt(val)}", fontsize=9, fontweight="bold")
            axes[ax_i].set_xlim(lo_g, hi_g)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5 — sweep plots (only if multi-instance)
        all_good = [(s, [i for i in s["instances"] if not i.get("error")])
                    for s in series_list]
        all_good = [(s, insts) for s, insts in all_good if insts]
        if sum(len(insts) for _, insts in all_good) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax_i, (key, lbl, ylabel) in enumerate([
                ("kappa",          "kappa(A)",      "log10(kappa(A))"),
                ("residual_norm",  "||r||",         "log10(||r||)"),
                ("backward_error", "Backward error","log10(BE)"),
            ]):
                for si, (ser, insts) in enumerate(all_good):
                    color = _SERIES_COLORS[si % len(_SERIES_COLORS)]
                    raw   = [i["metrics"][key] for i in insts]
                    logged = [np.log10(v) if v is not None and v > 0 and not np.isinf(v)
                              else np.log10(eps) for v in raw]
                    x_vals = _sweep_x_values(insts, sweep_p)
                    axes[ax_i].plot(x_vals, logged, color=color, lw=1.8,
                                    marker="o", markersize=4, label=ser["label"])
                axes[ax_i].axhline(np.log10(eps), color="#7f8c8d", lw=1, ls="--", alpha=0.6)
                axes[ax_i].set_xlabel(_sweep_axis_label(sweep_p), fontsize=8)
                axes[ax_i].set_ylabel(ylabel, fontsize=8)
                axes[ax_i].set_title(lbl, fontsize=9, fontweight="bold")
                axes[ax_i].tick_params(labelsize=7)
                if len(series_list) > 1:
                    axes[ax_i].legend(fontsize=6)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buf.seek(0)
    return buf.read()


def _render_section7(active_inst: dict, series_list: list) -> None:
    st.subheader("7. Summary")
    if active_inst.get("error"):
        st.error(active_inst["error"])
        return
    summary_text = _build_summary_text(active_inst, series_list)
    st.text_area(
        "Copy and paste into an LLM to verify results:",
        value=summary_text,
        height=440,
    )


def _render_section8(active_inst: dict, series_list: list) -> None:
    st.subheader("8. Save results")
    if active_inst.get("error"):
        st.error(active_inst["error"])
        return
    if st.button("Generate PDF", key="gen_pdf_lab1"):
        with st.spinner("Building PDF…"):
            pdf_bytes = _build_pdf_bytes(active_inst, series_list)
            st.session_state["_lab1_pdf_bytes"] = pdf_bytes
    pdf_bytes = st.session_state.get("_lab1_pdf_bytes")
    if pdf_bytes is not None:
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="nla_lab_results.pdf",
            mime="application/pdf",
        )
        st.caption("Includes: summary, A heatmap, condition number, residual, FEB, and backward error.")


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

    col3, col4 = st.columns(2)
    with col3:
        _render_section3(series_list)
    with col4:
        _render_section4(series_list)
    st.divider()

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("5. Solver behaviour metrics")
        st.caption("Coming soon.")
    with col6:
        st.subheader("6. Structural metrics")
        st.caption("Coming soon.")
    st.divider()

    _render_section7(active_inst, series_list)
    st.divider()

    _render_section8(active_inst, series_list)
