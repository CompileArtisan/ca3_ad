"""RECALL4 NEUROLAB simulator view: one brain, an Alzheimer's severity slider linked to synaptic
connectivity, input noise, cue strength and two drug toggles. The model and its values are the CA3
Hopfield reference model of the two Monte Carlo reports (see ``ca3model.neurolab``). The animation
replays the network's real asynchronous update order, so every change on screen happened in the
simulation."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ca3model.neurolab import (BASELINE_CUE, BASELINE_NOISE, CUE_SIZE, CUE_STRENGTH_RANGE,
                               DEFAULT_SEVERITY, DRUG_STRENGTHS, GRID, MAX_SWEEPS,
                               MIN_CONNECTIVITY, N_NEURONS, N_PAIRS, OUTCOMES, SUCCESS_OVERLAP,
                               TRIAL_SEED_BASE, Condition, connectivity_for_severity, cue_neurons,
                               hebbian_weights, neuron_layout, run_trial, run_trials,
                               severity_for_connectivity, summarise)

from .settings import Settings, SimSettings

TEMPLATE = (Path(__file__).with_name("brain_view.html")).read_text(encoding="utf-8")
HEALTHY_RGB = np.array([45, 212, 191])
SICK_RGB = np.array([251, 113, 133])
NOISE_MAX = 50


def _accent(severity: float) -> str:
    rgb = HEALTHY_RGB + (SICK_RGB - HEALTHY_RGB) * min(1.0, severity / 0.6)
    return ",".join(str(int(round(v))) for v in rgb)


def _verdict(m: dict) -> str:
    mm = f"m = {m['m']:.2f}, {m['pct_correct']:.0f}% of neurons correct"
    return {"correct": f"recalled memory #1 ({mm})",
            "different memory": "settled into stored memory #"
                                f"{2 + int(np.argmax(m['overlaps'][1:]))} instead ({mm})",
            "mirror": f"settled into the mirror image of memory #1 ({mm})",
            "mixture": f"settled into a mixture / spurious state ({mm})",
            "not converged": f"did not settle within {MAX_SWEEPS} sweeps ({mm})"}[m["outcome"]]


# ---------------------------------------------------------------- drugs
def _memantine_step(sim: SimSettings) -> int:
    """Points of input noise memantine removes (Experiment 2: 10% -> 8 / 6 / 4%)."""
    return int(round(100 * (BASELINE_NOISE - DRUG_STRENGTHS[sim.strength][1])))


def _condition(conn_pct: int, noise_pct: int, cue_pct: int, done: bool, mem: bool,
               sim: SimSettings) -> Condition:
    """Experiment 2 (default): donepezil = cue amplitude A = s / 0.20; memantine already moved the
    noise slider. Experiment 1: donepezil = persistent cue gain g; memantine = activity gate."""
    cond = Condition(connectivity=conn_pct / 100, noise=noise_pct / 100,
                     cue_fraction=cue_pct / 100, n_stored=sim.n_stored,
                     random_fill=sim.random_fill)
    if sim.drug_model == "exp1":
        return replace(cond, cue_gain=sim.cue_gain if done else 0.0,
                       activity_threshold=sim.activity_threshold if mem else 0.0)
    if done:
        cond = replace(cond, cue_amplitude=DRUG_STRENGTHS[sim.strength][0] / BASELINE_CUE)
    return cond


# ---------------------------------------------------------------- simulation (cached)
@st.cache_data(show_spinner=False, max_entries=64)
def _monte_carlo(cond: Condition, n_trials: int) -> dict:
    return summarise(run_trials(cond, n_trials))


@st.cache_data(show_spinner=False, max_entries=64)
def _animation(cond: Condition, trial: int, sev: float, title: str, mem_note: str) -> str:
    return _payload(run_trial(cond, trial), sev, title, mem_note)


def _weight_matrix(W: np.ndarray, scale: float) -> list:
    """25x25 weights scaled to [-1, 1] by the strongest healthy synapse, 2 decimals."""
    return np.round(np.clip(W / scale, -1, 1), 2).tolist()


def _round(a, nd=3):
    return [None if np.isnan(v) else round(float(v), nd) for v in a]


def _payload(run, sev: float, title: str, mem_note: str) -> str:
    cond, m = run.cond, run.metrics
    n = run.patterns.shape[1]
    xy, edges = neuron_layout(n)
    W_healthy = hebbian_weights(run.patterns)
    w_max = float(np.abs(W_healthy).max())
    wsign = np.sign(W_healthy[edges[:, 0], edges[:, 1]]).astype(int)
    eff = run.mask[edges[:, 0], edges[:, 1]]
    k = m["pairs_kept"]
    chips = [f"synaptic connectivity {cond.connectivity:.0%} ({k}/{N_PAIRS} pairs)",
             f"{m['n_corrupted']} input neuron{'s' * (m['n_corrupted'] != 1)} corrupted",
             f"{m['sweeps']} sweep{'s' * (m['sweeps'] != 1)}"]
    if cond.cue_amplitude != 1:
        chips.append(f"cue amplitude ×{cond.cue_amplitude:.2f}")
    if cond.cue_gain:
        chips.append(f"cue gain g = {cond.cue_gain:g}")
    if cond.activity_threshold:
        chips.append(f"gate θ = {cond.activity_threshold:g}")
    data = {
        "n": n, "grid": list(GRID),
        "xy": [[round(float(a), 4), round(float(c), 4)] for a, c in xy],
        "edges": edges.tolist(), "wsign": wsign.tolist(),
        "target": run.patterns[0].astype(int).tolist(),
        "others": run.patterns[1:].astype(int).tolist(),
        "thr": SUCCESS_OVERLAP,
        "brains": [{
            "title": title, "accentRGB": _accent(sev), "success": bool(m["success"]),
            "sweeps": int(m["sweeps"]), "states": [s.astype(int).tolist() for s in run.states],
            "pos": [_round(p) for p in run.update_pos], "alive": [1] * n,
            "eff": [round(float(e), 2) for e in eff], "chips": chips, "verdict": _verdict(m),
            "atrophy": 1.0 - 0.22 * sev,
        }],
        "cueNote": _cue_note(run, mem_note),
        "weights": {"healthy": _weight_matrix(W_healthy, w_max),
                    "brain": _weight_matrix(run.weights, w_max),
                    "brainLabel": "This brain",
                    "brainSub": f"{k} of {N_PAIRS} synaptic pairs kept ({cond.connectivity:.0%})"},
    }
    return TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def _cue_note(run, mem_note: str) -> str:
    cond = run.cond
    k = cond.n_cued
    base = ", ".join(str(int(i)) for i in cue_neurons()[:min(k, CUE_SIZE)])
    note = f"{k} of {N_NEURONS} neurons cued ({cond.cue_fraction:.0%} cue strength): {base}"
    if k > CUE_SIZE:
        note += f" plus {k - CUE_SIZE} more"
    rest = "start at random ±1" if cond.random_fill else "start silent"
    note += (f", with memory #1's values; the other {N_NEURONS - k} {rest}. "
             f"Input noise {cond.noise:.0%}: {run.metrics['n_corrupted']} neuron"
             f"{'s' * (run.metrics['n_corrupted'] != 1)} corrupted in this trial")
    if cond.cue_amplitude != 1:
        note += (f". Donepezil: cued neurons start at amplitude {cond.cue_amplitude:.2f} "
                 f"(effective cue strength {cond.effective_cue:.0%})")
    if mem_note:
        note += f". {mem_note}"
    return note + "."


# ---------------------------------------------------------------- controls
def _on_memantine() -> None:
    """Experiment 2: memantine moves the input-noise slider itself (down 6 points at high
    strength, 10% -> 4%); off puts it back."""
    ss = st.session_state
    if ss.get("sim_drug_model", "").startswith("Experiment 1"):
        return
    step = ss.get("_nl_mem_step", 6)
    if ss["nl_mem"]:
        ss["nl_noise_before_mem"] = ss["nl_noise"]
        ss["nl_noise"] = max(0, ss["nl_noise"] - step)
        ss["nl_noise_with_mem"] = ss["nl_noise"]
    else:
        before = ss.pop("nl_noise_before_mem", None)
        after = ss.pop("nl_noise_with_mem", None)
        if before is not None and ss["nl_noise"] == after:
            ss["nl_noise"] = before
        else:
            ss["nl_noise"] = min(NOISE_MAX, ss["nl_noise"] + step)


def _on_severity() -> None:
    st.session_state["nl_conn"] = connectivity_for_severity(st.session_state["nl_sev"])


def _on_connectivity() -> None:
    st.session_state["nl_sev"] = severity_for_connectivity(st.session_state["nl_conn"])


def _controls(sim: SimSettings) -> tuple[int, int, int, int, bool, bool]:
    ss = st.session_state
    ss.setdefault("nl_noise", int(round(100 * BASELINE_NOISE)))
    ss.setdefault("nl_sev", DEFAULT_SEVERITY)
    ss.setdefault("nl_conn", connectivity_for_severity(ss["nl_sev"]))
    ss["_nl_mem_step"] = _memantine_step(sim)
    exp1 = sim.drug_model == "exp1"
    cue_s, noise_s = DRUG_STRENGTHS[sim.strength]
    lo, hi, step = CUE_STRENGTH_RANGE
    with st.container(border=True, key="nl_controls"):
        c1, c0, c2, c3, c4 = st.columns([1.1, 1.1, 1, 1, 1.05], gap="medium")
        sev = c1.slider("Alzheimer's severity", 0, 100, step=1, format="%d%%", key="nl_sev",
                        on_change=_on_severity,
                        help="0% is a healthy brain. Higher severity removes synaptic connections: "
                             f"connectivity falls linearly from 100% to {MIN_CONNECTIVITY}% (the "
                             "reports' range). It is the same brain getting worse: a higher setting "
                             "removes a superset of the pairs removed below it.")
        conn = c0.slider("Synaptic connectivity (wᵢⱼ)", MIN_CONNECTIVITY, 100, step=1,
                         format="%d%%", key="nl_conn", on_change=_on_connectivity,
                         help=f"Share of the {N_PAIRS} neuron pairs kept: exactly round(c × "
                              f"{N_PAIRS}), chosen at random. A removed pair deletes both wᵢⱼ and "
                              "wⱼᵢ; surviving weights keep full strength. Linked to severity: moving "
                              "one moves the other.")
        noise = c2.slider("Input noise", 0, NOISE_MAX, step=1, format="%d%%", key="nl_noise",
                          help="Each neuron is corrupted with this probability: a cued neuron has "
                               "its sign flipped, a silent one becomes active with a random sign. "
                               "Memantine lowers it.")
        if ss.get("nl_mem") and "nl_noise_before_mem" in ss and not exp1:
            c2.caption(f"Memantine: {ss['nl_noise_before_mem']}% → {ss['nl_noise']}%")
        cue = c3.slider("Cue strength", lo, hi, int(round(100 * BASELINE_CUE)), step,
                        format="%d%%", key="nl_cue",
                        help=f"Share of the {N_NEURONS} neurons given memory #1's value in the cue. "
                             f"20% is the reports' cue: neurons 2, 4, 6, 7, 11. More reveals extra "
                             "neurons (the reports' comparison run); the rest start silent.")
        with c4:
            st.markdown("<div class='nl-label'>Medical intervention</div>", unsafe_allow_html=True)
            done = st.toggle("Donepezil", key="nl_done",
                             help=(f"Modelled as a persistent cue input, h += g·cue, g = "
                                   f"{sim.cue_gain:g} (Experiment 1)." if exp1 else
                                   f"Modelled as a stronger cue: the cued neurons start at amplitude "
                                   f"{cue_s / BASELINE_CUE:.2f} (effective cue strength 20% → "
                                   f"{cue_s:.0%}, Experiment 2). No extra neurons are revealed and "
                                   "the weights are untouched."))
            mem = st.toggle("Memantine", key="nl_mem", on_change=_on_memantine,
                            help=(f"Modelled as an activity gate: a neuron fires only when |h| > θ, "
                                  f"θ = {sim.activity_threshold:g} (Experiment 1)." if exp1 else
                                  f"Modelled as less input noise: {_memantine_step(sim)} points lower "
                                  f"(10% → {noise_s:.0%}, Experiment 2). Turning it on moves the "
                                  "Input noise slider down; turning it off moves it back."))
    return sev, conn, noise, cue, done, mem


# ---------------------------------------------------------------- reliability
def _row(label: str, r: dict) -> dict:
    lo, hi = r["ci"]
    o = r["outcomes"]
    return {"Condition": label, "Success %": 100 * r["success"],
            "95% CI (Wilson)": f"{100 * lo:.1f}–{100 * hi:.1f}",
            "Mean overlap (SD)": f"{r['mean_overlap']:.2f} ({r['sd_overlap']:.2f})",
            "% neurons correct": round(r["pct_correct"], 1),
            "Mean sweeps": round(r["mean_sweeps"], 2),
            **{k.capitalize(): o[k] for k in OUTCOMES}}


def _reliability(rel: dict, healthy: dict, sev: float, rows: list) -> None:
    n = rel["n"]
    st.markdown(f"<div class='nl-label'>Is this run typical? Same settings, {n:,} matched trials"
                "</div>", unsafe_allow_html=True)
    k, h = rel["successes"], healthy["successes"]
    ref = "" if sev == 0 else f"<small>A healthy brain recalls {h:,}/{n:,} of the same trials</small>"
    st.markdown(f"<div class='nl-rel' style='--acc:rgb({_accent(sev)})'><span>This brain</span>"
                f"<b>{k:,}/{n:,} recalled</b><div class='nl-track'><i style='width:{100 * k / n:.1f}%'>"
                f"</i></div>{ref}</div>", unsafe_allow_html=True)
    with st.expander("Monte Carlo details"):
        st.dataframe(pd.DataFrame([_row(lbl, r) for lbl, r in rows]), hide_index=True,
                     width="stretch",
                     column_config={"Success %": st.column_config.ProgressColumn(
                         "Success %", min_value=0, max_value=100, format="%.1f%%")})
        st.caption(f"Trial t uses seed {TRIAL_SEED_BASE:,} + t (t = 0…{n - 1:,}); every row uses the "
                   "same trials. Success = settled with overlap m ≥ 0.8 with memory #1.")


def render_neurolab(s: Settings) -> None:
    sim = s.sim
    sev_pct, conn, noise, cue, done, mem = _controls(sim)
    sev = sev_pct / 100
    cond = _condition(conn, noise, cue, done, mem, sim)
    noise_untreated = st.session_state.get("nl_noise_before_mem", noise) if mem else noise
    untreated = _condition(conn, noise_untreated, cue, False, False, sim)
    healthy = _condition(100, noise, cue, False, False, sim)
    title = "Healthy brain" if sev_pct == 0 else f"Alzheimer's-inspired brain · {sev_pct}% severity"
    title += "".join(f" · {d}" for d, on in (("donepezil", done), ("memantine", mem)) if on)
    mem_note = ""
    if mem and sim.drug_model == "exp2" and "nl_noise_before_mem" in st.session_state:
        mem_note = (f"Memantine has lowered input noise from {st.session_state['nl_noise_before_mem']}%"
                    f" to {noise}%")
    with st.spinner("Running the brain…"):
        html = _animation(cond, sim.trial, sev, title, mem_note)
        rel = _monte_carlo(cond, sim.n_trials)
        ref = _monte_carlo(healthy, sim.n_trials)
        rows = [("This brain", rel)]
        if done or mem:
            rows.append(("Same brain, untreated", _monte_carlo(untreated, sim.n_trials)))
        if conn < 100:
            rows.append(("Healthy brain (100% connectivity)", ref))
    st.iframe(html, height="content")
    _reliability(rel, ref, sev, rows)
    with st.expander("How to read this"):
        st.markdown(
            "- The brain is the CA3 Hopfield reference model of the two Monte Carlo reports: 25 model "
            "neurons with states ±1 and three stored memories (random patterns, seed 20261118), "
            "learned with the Hebbian rule. The target is **memory #1**; #2 and #3 are shown under it.\n"
            "- The cue gives part of memory #1: at 20% cue strength, neurons 2, 4, 6, 7, 11. The rest "
            "start silent (0). These 5 values also match memory #2 on 4 of 5 neurons, so even a healthy "
            "brain often settles into memory #2.\n"
            f"- **Alzheimer's severity** removes synaptic connections: connectivity falls from 100% to "
            f"{MIN_CONNECTIVITY}% and keeps exactly round(c × {N_PAIRS}) of the {N_PAIRS} neuron "
            "pairs. **Input noise** corrupts each neuron with that probability.\n"
            "- Neurons update one at a time in a fresh random order each sweep: s = sign(h), and h = 0 "
            f"keeps the state. The run stops after a full sweep with no change, or after {MAX_SWEEPS} "
            f"sweeps. **Recalled** = settled with overlap m ≥ {SUCCESS_OVERLAP} with memory #1.\n"
            "- **Donepezil** makes the cue stronger: cued neurons start at amplitude 1.75 (effective "
            "cue strength 35%). **Memantine** lowers input noise by 6 points (10% → 4%); you will see "
            "the Input noise slider move. Strength and Experiment 1's version are in **Advanced**.\n"
            "- In the disc each dot is a neuron: **amber = +1**, **violet = −1**, hollow = 0 (silent). "
            "Neurons update in the exact order the simulation used; a ring marks each change. Lines are "
            "a sample of synapses: blue = consistent with the current state, red = in conflict, "
            "dotted = pair removed.\n"
            "- **Synaptic weights wᵢⱼ**: all 25×25 connection strengths, scaled so +1 is the strongest "
            "healthy synapse. A removed pair is 0.\n"
            "- The bar under the brain repeats the same settings over many matched trials. Each trial "
            "draws its own connections, noise and update order.")
        st.caption("Drug toggles are illustrative abstractions, not pharmacology, and have no "
                   "clinical meaning.")
