"""One ``render_*`` function per tab. Heavy suites run behind ``st.form`` submit buttons,
are cached with ``st.cache_data`` (key = JSON of all configs) and persisted in session_state."""
from __future__ import annotations

import json
import platform
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import ca3model as cm
from ca3model import plotting as pl
from ca3model.experiments import (_kind_spec, default_load_conditions, default_load_values,
                                  run_trial_conditions)
from ca3model.stats import paired_sweep_table, summarise

from .sidebar import Settings
from .text import ASSUMPTIONS, BIO_TABLE, DISCLAIMER, GLOSSARY, SHOW_NOT_SHOW

LEVELS = [round(x, 1) for x in np.arange(0.0, 0.91, 0.1)]
ATTRACTOR_HELP = "target / other_stored / mirror / spurious (among surviving neurons)"


# ------------------------------------------------------------------ helpers
def _cfg_json(s: Settings, **params) -> str:
    return json.dumps({"net": s.net.to_dict(), "ret": s.ret.to_dict(), "exp": s.exp.to_dict(),
                       "mode": s.damage.synaptic_mode, "params": params}, sort_keys=True)


def _unpack(cfg_json: str):
    c = json.loads(cfg_json)
    return (cm.NetworkConfig(**c["net"]), cm.ExperimentConfig(**c["exp"]),
            cm.RetrievalConfig(**c["ret"]), c["mode"], c["params"])


def _sub(progress, i, n):
    return (lambda f: progress((i + f) / n)) if progress else None


@st.cache_data(show_spinner=False)
def _run_suite(name: str, cfg_json: str, _progress=None):
    """Cached heavy computation; ``_progress`` is excluded from the cache key."""
    net, exp, ret, mode, p = _unpack(cfg_json)
    if name == "sweeps":
        lv = p["levels"]
        return {"A": cm.exp_healthy_vs_weakened(net, exp, ret, lv, mode, _sub(_progress, 0, 3)),
                "B": cm.exp_healthy_vs_neuron_loss(net, exp, ret, lv, mode, _sub(_progress, 1, 3)),
                "C": cm.exp_weakening_vs_loss(net, exp, ret, lv, mode, _sub(_progress, 2, 3))}
    if name == "combined":
        return cm.exp_combined(net, exp, ret, p["syn"], p["neu"], mode, _progress)
    if name == "learning":
        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        e = cm.exp_new_learning(net, exp, ret, p["levels"], p["kind"], mode, _sub(_progress, 0, 3))
        seq = cm.sequential_learning_curve(net, exp, ret, _kind_spec(p["kind"], p["seq_level"], mode),
                                           _sub(_progress, 1, 3))
        parts = []
        for k, eta in enumerate(etas):
            d = cm.exp_new_learning(net, replace(exp, plasticity_efficacy=eta), ret,
                                    [p["seq_level"]], p["kind"], mode)
            d = d[d.condition == "damage_then_learn"].assign(eta=eta)
            parts.append(d)
            if _progress:
                _progress((2 + (k + 1) / len(etas)) / 3)
        return {"E": e, "seq": seq, "eta": pd.concat(parts, ignore_index=True)}
    if name == "load":
        conds = default_load_conditions(p["syn"], p["neu"], mode)
        return cm.exp_memory_load(net, exp, ret, p["loads"], conds, _progress)
    raise KeyError(name)


def _heavy(name: str, s: Settings, **params) -> None:
    """Run (or fetch from cache) a heavy suite with a progress bar; persist in session_state."""
    cfg = _cfg_json(s, **params)
    bar = st.progress(0.0, text="Running…")
    res = _run_suite(name, cfg, _progress=lambda f: bar.progress(min(float(f), 1.0), text="Running…"))
    bar.empty()
    st.session_state[f"res_{name}"] = {"result": res, "cfg": cfg}


def _stored(name: str, s: Settings):
    """Persisted result or None; warns if sidebar settings changed since it was computed."""
    r = st.session_state.get(f"res_{name}")
    if r is None:
        st.info("Choose settings above and press **Run** (results persist while you switch tabs).")
        return None
    cur = json.loads(r["cfg"])
    now = json.loads(_cfg_json(s))
    if any(cur[k] != now[k] for k in ("net", "ret", "exp", "mode")):
        st.warning("Showing results computed with earlier sidebar settings. Press Run to update.")
    return r["result"]


def _chart(fig, container=None) -> None:
    """Render a figure with a deterministic unique key (counter reset at each script run)."""
    st.session_state["_chart_n"] = st.session_state.get("_chart_n", 0) + 1
    (container or st).plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state['_chart_n']}")


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(4)


def _stack(net: cm.HopfieldNetwork) -> np.ndarray:
    return np.stack(net.patterns)


# --------------------------------------------------------------- 1. demo
def render_demo(s: Settings) -> None:
    N, P = s.net.n_neurons, s.exp.n_patterns
    rngs = cm.trial_rngs(s.exp.base_seed, 0)
    pats = (cm.random_patterns(P, N, rngs.patterns) if s.pattern_type == "random"
            else cm.smooth_patterns(P, N, rngs.patterns))
    net = cm.HopfieldNetwork.from_patterns(pats)
    dnet = cm.apply_damage(net, s.damage, cm.draw_damage_randomness(N, rngs.damage))
    idx = st.slider("Stored memory index", 0, P - 1, 0, help="Which stored pattern to cue.") if P > 1 else 0
    cue = cm.corrupt_cue(pats[idx], s.ret.cue_noise, s.ret.cue_mode, cm.trial_rngs(s.exp.base_seed, 1000 + idx).cues)
    rh = cm.retrieve(net, cue, s.ret.max_sweeps, rngs.fresh_dynamics())
    rd = cm.retrieve(dnet, cue, s.ret.max_sweeps, rngs.fresh_dynamics(), record_states=True)
    mh = cm.evaluate_retrieval(rh, pats[idx], net.alive, _stack(net), s.ret.success_threshold)
    md = cm.evaluate_retrieval(rd, pats[idx], dnet.alive, _stack(dnet), s.ret.success_threshold)

    hist = rd.state_history or [rd.final_state]
    k = (st.slider("Damaged network: sweep", 0, len(hist) - 1, len(hist) - 1,
                   help="Step through the recorded states (0 = the cue) of the damaged network.")
         if len(hist) > 1 else 0)
    _chart(pl.fig_pattern_panel(pats[idx], cue, rh.final_state, hist[k],
                                titles=("Original", "Corrupted cue", "Healthy reconstruction",
                                        f"Damaged network, sweep {k}")))
    for col, title, m in zip(st.columns(2), ("Healthy network", "Damaged network"), (mh, md)):
        with col:
            st.subheader(title)
            c = st.columns(5)
            c[0].metric("Accuracy (surv.)", f"{m['accuracy_surv']:.3f}", help="1 − Hamming/n_alive")
            c[1].metric("Hamming (all)", m["hamming_all"], help="Errors over all N; lost neurons count as errors.")
            c[2].metric("Sweeps", m["sweeps"], help="Sweeps until the first zero-flip sweep.")
            c[3].metric("Success", "yes" if m["success"] else "no",
                        help="Converged AND survivor accuracy ≥ threshold (primary measure).")
            c[4].metric("Attractor", m["attractor_type"], help=ATTRACTOR_HELP)
            st.caption(f"accuracy over all N: {m['accuracy_all']:.3f} · flips: {m['n_flips']} · "
                       f"converged: {m['converged']}")
    a, b = st.columns(2)
    a.caption("Healthy energy trace")
    _chart(pl.fig_energy_trace(rh.energy_trace), a)
    b.caption("Damaged energy trace (never increases under asynchronous updates)")
    _chart(pl.fig_energy_trace(rd.energy_trace), b)


# --------------------------------------------------------------- 2. network
def render_network(s: Settings) -> None:
    N, P = s.net.n_neurons, s.exp.n_patterns
    rngs = cm.trial_rngs(s.exp.base_seed, 0)
    pats = cm.random_patterns(P, N, rngs.patterns)
    net = cm.HopfieldNetwork.from_patterns(pats)
    dnet = cm.apply_damage(net, s.damage, cm.draw_damage_randomness(N, rngs.damage))
    Wh, Wd = net.effective_weights(), dnet.effective_weights()
    _chart(pl.fig_weight_compare(Wh, Wd, dnet.alive))
    st.caption("Grey lines mark lost neurons (their rows/columns are zero).")
    cs = dnet.connectivity_stats()
    c = st.columns(5)
    c[0].metric("Synaptic efficacy lost", f"{100 * cs['frac_synapses_lost']:.1f}%",
                help="1 − total efficacy / healthy N(N−1); attenuation counts partially.")
    c[1].metric("Synapses exactly zero", f"{100 * cs['frac_synapses_zero']:.1f}%",
                help="Deleted synapses plus all synapses of lost neurons.")
    c[2].metric("Neurons alive", f"{cs['n_alive']} / {N}")
    c[3].metric("Mean effective degree", f"{cs['mean_degree']:.1f}", help="Nonzero links per surviving neuron.")
    c[4].metric("Mean efficacy (alive pairs)", f"{cs['mean_efficacy_alive_pairs']:.3f}")
    _chart(pl.fig_weight_hist(Wh, Wd))


# --------------------------------------------------------------- 3. sweeps
def render_sweeps(s: Settings) -> None:
    with st.form("form_sweeps"):
        levels = st.multiselect("Severity / loss levels (model parameters)", LEVELS, default=LEVELS,
                                help="Used for synaptic severity, neuron-loss fraction and the matched comparison.")
        matched = st.checkbox("Show matched synapse-loss comparison", value=True,
                              help="Weakening at s = 1−(1−f)², the synapse fraction removed by losing a fraction f of neurons.")
        go = st.form_submit_button("Run damage sweeps (A+B+C)")
    if go:
        if not levels:
            st.error("Select at least one level.")
        else:
            _heavy("sweeps", s, levels=sorted(levels))
    res = _stored("sweeps", s)
    if res is None:
        return
    A, B, C = res["A"], res["B"], res["C"]
    parts = [A[A.condition == "healthy"].assign(level=0.0),
             A[A.condition == "synaptic"].assign(level=A.synaptic_severity),
             B[B.condition == "neuron"].assign(level=B.neuron_loss)]
    if matched:
        m = C[C.condition == "matched_synaptic"]
        parts.append(m.assign(level=1 - np.sqrt(1 - m.synaptic_severity)))
    long = pd.concat(parts, ignore_index=True)
    metrics = [("success", "retrieval success rate"), ("accuracy_surv", "accuracy (survivors)"),
               ("hamming_surv", "Hamming distance (survivors)"), ("sweeps", "convergence time (sweeps)")]
    summ = summarise(long, ["condition", "level"], [m for m, _ in metrics])
    xl = "nominal level: synaptic severity s or neuron-loss fraction f (model parameter)"
    cols = st.columns(2)
    for i, (m, lab) in enumerate(metrics):
        with cols[i % 2]:
            _chart(pl.fig_metric_vs_severity(summ, "level", m, "condition", lab, xl))
    if matched:
        st.caption("'matched_synaptic' is plotted at the neuron-loss level f that implies the same synapse fraction.")
    st.subheader("Paired differences vs healthy (exploratory)")
    st.caption("Mean difference in success rate, 95% t-interval over trials, Wilcoxon signed-rank p (exploratory).")
    ta = paired_sweep_table(A, "synaptic", "healthy", "synaptic_severity", "success").rename(
        columns={"synaptic_severity": "level"}).assign(condition="synaptic")
    tb = paired_sweep_table(B, "neuron", "healthy", "neuron_loss", "success").rename(
        columns={"neuron_loss": "level"}).assign(condition="neuron")
    st.dataframe(_fmt(pd.concat([ta, tb])[["condition", "level", "mean_diff", "lo", "hi", "wilcoxon_p", "n_trials"]]),
                 use_container_width=True, hide_index=True)


# --------------------------------------------------------------- 4. combined
def render_combined(s: Settings) -> None:
    with st.form("form_combined"):
        syn = st.multiselect("Synaptic severities", LEVELS, default=[0.0, 0.2, 0.4, 0.6, 0.8])
        neu = st.multiselect("Neuron-loss fractions", LEVELS, default=[0.0, 0.2, 0.4, 0.6, 0.8])
        go = st.form_submit_button("Run combined-damage grid (D)")
    if go:
        if not syn or not neu:
            st.error("Select at least one level on each axis.")
        else:
            _heavy("combined", s, syn=sorted(syn), neu=sorted(neu))
    df = _stored("combined", s)
    if df is None:
        return
    comb = summarise(df[df.condition == "combined"], ["synaptic_severity", "neuron_loss"], ["success"])
    _chart(pl.fig_heatmap_grid(comb, "synaptic_severity", "neuron_loss", "success_mean",
                               "success rate", 0, 1))
    sv = sorted(df.synaptic_severity.unique())
    nv = sorted(df.neuron_loss.unique())
    c1, c2 = st.columns(2)
    s_sel = c1.selectbox("Synaptic severity", sv, index=min(2, len(sv) - 1))
    n_sel = c2.selectbox("Neuron loss", nv, index=min(2, len(nv) - 1))
    pick = pd.concat([
        df[df.condition == "healthy"],
        df[(df.condition == "synaptic_only") & (df.synaptic_severity == s_sel)],
        df[(df.condition == "neuron_only") & (df.neuron_loss == n_sel)],
        df[(df.condition == "combined") & (df.synaptic_severity == s_sel) & (df.neuron_loss == n_sel)]])
    pick = pick.assign(condition=pick.condition.replace({"synaptic_only": "synaptic", "neuron_only": "neuron"}))
    _chart(pl.fig_condition_bars(summarise(pick, ["condition"], ["success"]), "success", ylabel="retrieval success rate"))
    st.subheader("Interaction (additivity) table")
    it = cm.interaction_table(df, "success")
    st.dataframe(_fmt(it), use_container_width=True, hide_index=True)
    st.caption("deficit = healthy success − condition success (paired per trial). interaction = combined deficit − "
               "(synaptic deficit + neuron deficit). Positive: combined damage is worse than the sum of its parts; "
               "negative: better than additive (partly overlapping damage). Descriptive only; deficits are not clipped.")


# --------------------------------------------------------------- 5. learning
def render_learning(s: Settings) -> None:
    with st.form("form_learning"):
        kind = st.radio("Damage type", ["synaptic", "neuron", "combined"], horizontal=True,
                        help="Which damage is applied before/after learning the new patterns.")
        levels = st.multiselect("Damage levels", LEVELS, default=[0.0, 0.2, 0.4, 0.6, 0.8])
        seq_level = st.slider("Level for sequential-learning and plasticity analyses", 0.0, 0.9, 0.4, 0.1)
        go = st.form_submit_button("Run new-learning experiments (E)")
    st.caption(f"Plasticity efficacy for damage-then-learn comes from the sidebar (currently "
               f"{s.exp.plasticity_efficacy:.2f}); {s.exp.n_new_patterns} new patterns are learned.")
    if go:
        if not levels:
            st.error("Select at least one level.")
        else:
            _heavy("learning", s, kind=kind, levels=sorted(levels), seq_level=float(seq_level))
    res = _stored("learning", s)
    if res is None:
        return
    E, seq, eta = res["E"], res["seq"], res["eta"]
    st.subheader("Before vs after damage: new and old patterns")
    _chart(pl.fig_new_learning(summarise(E, ["condition", "pattern_group", "level"], ["success"])))
    st.info("In this abstraction effective weights are W_raw ⊙ efficacy ⊙ alive, so with plasticity efficacy = 1 "
            "'damage then learn' and 'learn then damage' coincide. They separate only when plasticity "
            "efficacy < 1 (impaired encoding) — see the sidebar slider and the plot below.")
    a, b = st.columns(2)
    with a:
        st.subheader("Sequential learning")
        imm = summarise(seq[seq.stage == "immediate"], ["condition", "step"], ["success"])
        _chart(pl.fig_metric_vs_severity(imm, "step", "success", "condition", "success right after learning",
                                         "new pattern number learned"))
        fin = summarise(seq[seq.stage == "final"], ["condition"], ["success"])
        _chart(pl.fig_condition_bars(fin, "success", ylabel="new-pattern success after all learning"))
    with b:
        st.subheader("Plasticity-efficacy effect")
        pe = summarise(eta[eta.pattern_group == "new"], ["condition", "eta"], ["success"])
        _chart(pl.fig_metric_vs_severity(pe, "eta", "success", "condition", "new-pattern success",
                                         "plasticity efficacy η (model parameter)"))


# --------------------------------------------------------------- 6. load
def render_load(s: Settings) -> None:
    n = s.net.n_neurons
    with st.form("form_load"):
        c1, c2 = st.columns(2)
        syn = c1.slider("Synaptic severity for this sweep", 0.0, 0.9, 0.3, 0.05)
        neu = c2.slider("Neuron loss for this sweep", 0.0, 0.9, 0.2, 0.05)
        go = st.form_submit_button("Run memory-load sweep (F)")
    st.caption(f"Loads: {default_load_values(n)} patterns (up to ≈0.25 N). At most 10 patterns are tested per load.")
    if go:
        _heavy("load", s, syn=float(syn), neu=float(neu), loads=default_load_values(n))
    df = _stored("load", s)
    if df is None:
        return
    _chart(pl.fig_memory_load(summarise(df, ["condition", "n_patterns"], ["success"]), n))
    st.caption("The dashed line marks the theoretical α_c ≈ 0.138 for large random-pattern networks; agreement is "
               "only approximate at these small sizes.")
    _chart(pl.fig_metric_vs_severity(summarise(df, ["condition", "load_alpha"], ["accuracy_surv"]), "load_alpha",
                                     "accuracy_surv", "condition", "accuracy (survivors)", "load α = P/N"))


# --------------------------------------------------------------- 7. summary
@st.cache_data(show_spinner=False)
def _summary_rows(cfg_json: str) -> pd.DataFrame:
    net, exp, ret, mode, p = _unpack(cfg_json)
    conds = {"healthy@0": cm.DamageSpec(),
             "synaptic@x": cm.DamageSpec(p["syn"], mode, 0.0),
             "neuron@x": cm.DamageSpec(0.0, mode, p["neu"]),
             "combined@x": cm.DamageSpec(p["syn"], mode, p["neu"])}
    rows = []
    for t in range(exp.n_trials):
        rows += run_trial_conditions(net, exp, ret, conds, t, "S")
    df = pd.DataFrame(rows)
    df.attrs["meta"] = {"network": net.to_dict(), "experiment": exp.to_dict(), "retrieval": ret.to_dict(),
                        "damage": p, "mode": mode}
    return df


def render_summary(s: Settings) -> None:
    cfg = _cfg_json(s, syn=s.damage.synaptic_severity, neu=s.damage.neuron_loss)
    df = _summary_rows(cfg)
    st.caption("All conditions at the current sidebar damage settings (paired over trials). "
               "**success** uses surviving neurons; **success_all** counts lost neurons as errors.")
    summ = summarise(df, ["condition"], ["success", "success_all", "accuracy_surv", "accuracy_all", "sweeps"])
    order = ["healthy", "synaptic", "neuron", "combined"]
    summ = summ.set_index("condition").loc[order].reset_index()
    a, b = st.columns(2)
    with a:
        _chart(pl.fig_condition_bars(summ, "success", ylabel="success (survivors)"))
    with b:
        _chart(pl.fig_condition_bars(summ, "success_all", ylabel="success (all N)"))
    st.dataframe(_fmt(summ), use_container_width=True, hide_index=True)
    meta = json.dumps(df.attrs.get("meta", {}), sort_keys=True)
    csv = f"# config: {meta}\n" + df.to_csv(index=False)
    st.download_button("Download trial-level CSV (config JSON in header comment)", csv,
                       file_name="ca3_summary.csv", mime="text/csv")
    with st.expander("Reproducibility", expanded=True):
        import plotly, scipy
        try:
            import numba
            nb = numba.__version__
        except Exception:
            nb = "not installed (NumPy fallback)"
        st.code(json.dumps({"base_seed": s.exp.base_seed, "N": s.net.n_neurons, "trials": s.exp.n_trials,
                            "python": platform.python_version(), "numpy": np.__version__, "numba": nb,
                            "scipy": scipy.__version__, "pandas": pd.__version__, "plotly": plotly.__version__,
                            "streamlit": st.__version__, "backend": "numba" if cm.HAVE_NUMBA else "numpy"},
                           indent=2), language="json")
        st.code(cfg, language="json")


# --------------------------------------------------------------- 8. about
def render_about() -> None:
    st.info(DISCLAIMER)
    st.subheader("Biology vs. computational abstraction")
    st.markdown(BIO_TABLE)
    st.subheader("Glossary")
    for k, v in GLOSSARY.items():
        st.markdown(f"**{k}** — {v}")
    st.subheader("Model assumptions")
    st.markdown(ASSUMPTIONS)
    st.subheader("What the results do and do not show")
    st.markdown(SHOW_NOT_SHOW)
