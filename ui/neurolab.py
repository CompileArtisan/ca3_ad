#!/usr/bin/env python3

"""RECALL4 NEUROLAB simulator view: one brain, an Alzheimer's severity slider, input noise, cue
strength and two drug toggles. The animation replays the network's real asynchronous update
order (see ``ca3model.neurolab``), so every flip on screen happened in the simulation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import streamlit as st

from ca3model.neurolab import (DONEPEZIL_CUE_GAIN, DONEPEZIL_RECOVERY, GRID,
                               MEMANTINE_NOISE_REDUCTION, NEURON_LOSS_AT_MAX, SIMULATOR_MODE,
                               SYNAPSE_LOSS_AT_MAX, connectivity_for_severity, covariance_weights,
                               cue_counts, effective_cue_strength, memory_pattern,
                               monte_carlo_recall, neuron_layout, run_stimulator,
                               severity_for_connectivity)

from .settings import Settings

TEMPLATE = (Path(__file__).with_name("brain_view.html")).read_text(encoding="utf-8")
# Part of the simulation cache key: any edit to this file or the template invalidates cached pages
# (st.cache_data only hashes the cached function's own code, so text changes elsewhere, such as a
# relabelled note, would otherwise keep being served from the cache after a hot redeploy).
CODE_KEY = hashlib.sha256(Path(__file__).read_bytes() + TEMPLATE.encode()).hexdigest()[:16]
HEALTHY_RGB = np.array([45, 212, 191])
SICK_RGB = np.array([251, 113, 133])
NOISE_MAX = 50


def _accent(severity: float) -> str:
    rgb = HEALTHY_RGB + (SICK_RGB - HEALTHY_RGB) * min(1.0, severity / 0.6)
    return ",".join(str(int(round(v))) for v in rgb)


def _verdict(m: dict) -> str:
    if m["kind"] == "recalled":
        return "completed the X"
    if m["kind"] == "silent":
        return "went silent: the cue could not drive recall"
    if m["kind"] == "unsettled":
        return "did not settle"
    stray = f", {m['stray']} stray neuron{'s' * (m['stray'] != 1)} firing" if m["stray"] else ""
    return f"partial recall: {m['hit']} of {m['n_on']} X neurons firing{stray}"


@st.cache_data(show_spinner=False, max_entries=64)
def _simulate(args: tuple, mem: bool, noise_off: float, code_key: str = CODE_KEY) -> tuple[str, dict, dict]:
    """The animated run, the Monte Carlo recall at these settings, the four drug scenarios at the
    same brain settings (noise_off = input noise without memantine), and a healthy reference."""
    sev, noise, strength, done, mode, sweeps, thr, seed, conn = args
    mc = monte_carlo_recall(*args)
    noise_on = _half(round(noise_off * 100)) / 100
    scen = {}
    for key, d, nz in (("none", False, noise_off), ("done", True, noise_off),
                       ("mem", False, noise_on), ("both", True, noise_on)):
        scen[key] = mc if (d == done and abs(nz - noise) < 1e-9) else monte_carlo_recall(
            sev, nz, strength, d, mode, sweeps, thr, seed, conn)
        scen[key] = {**scen[key], "noise": nz}
    healthy = monte_carlo_recall(0.0, noise, strength, False, mode, sweeps, thr, seed)
    # Animate the most typical trial, not trial 1: a single lucky trial (e.g. one of memantine's
    # ~15% full recalls) would otherwise show 96% next to a 69% average.
    rep_args = args[:7] + (seed + mc["rep"],) + args[8:]
    return _payload(run_stimulator(*rep_args), args, mc, mem), scen, healthy


def _weight_matrix(W: np.ndarray, scale: float) -> list:
    """25x25 weights scaled to [-1, 1] by the strongest healthy synapse, 2 decimals."""
    return np.round(np.clip(W / scale, -1, 1), 2).tolist()


def _round(a, nd=3):
    return [None if np.isnan(v) else round(float(v), nd) for v in a]


def _payload(r: dict, args: tuple, mc: dict, mem: bool) -> str:
    sev, noise, strength, done, *_ = args
    b = r["brain"]
    brain, m = b.brain, b.metrics
    n = brain.n
    xy, edges = neuron_layout(n)
    wsign = np.sign(covariance_weights(r["target"])[edges[:, 0], edges[:, 1]]).astype(int)
    eff = brain.efficacy[edges[:, 0], edges[:, 1]]
    W_healthy = covariance_weights(memory_pattern())
    w_max = float(np.abs(W_healthy).max())
    title = "Healthy brain" if sev == 0 else f"Alzheimer's-inspired brain · {sev:.0%} severity"
    drugs = [d for d, on in (("donepezil", done), ("memantine", mem)) if on]
    if drugs:
        title += " · " + " + ".join(drugs)
    n_alive = int(brain.alive.sum())
    chips = [f"synaptic strength {100 * brain.synaptic_strength():.0f}%",
             f"neurons {n_alive}/{n}",
             f"{m['sweeps']} sweep{'s' * (m['sweeps'] != 1)}"]
    data = {
        "n": n, "grid": list(GRID),
        "xy": [[round(float(a), 4), round(float(c), 4)] for a, c in xy],
        "edges": edges.tolist(), "wsign": wsign.tolist(), "target": r["target"].astype(int).tolist(),
        "thr": args[6],
        "brains": [{
            "title": title, "accentRGB": _accent(sev), "success": bool(m["success"]),
            "sweeps": int(m["sweeps"]), "states": [s.astype(int).tolist() for s in b.states],
            "pos": [_round(p) for p in b.update_pos], "alive": brain.alive.astype(int).tolist(),
            "eff": [round(float(e), 2) for e in eff], "chips": chips, "verdict": _verdict(m),
            "atrophy": 1.0 - 0.22 * sev,
        }],
        "cueNote": _cue_note(strength, noise, done),
        "mc": {"recall": mc["recall"], "lo": mc["lo"], "hi": mc["hi"], "recalled": mc["recalled"],
               "n": mc["n"], "rep": mc["rep"]},
        "weights": {"healthy": _weight_matrix(W_healthy, w_max),
                    "brain": _weight_matrix(b.weights, w_max),
                    "brainLabel": "This brain" + (" + donepezil" if done else ""),
                    "brainSub": f"{100 * brain.synaptic_strength():.0f}% synaptic strength left"},
    }
    return TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def _cue_note(strength: float, noise: float, donepezil: bool = False) -> str:
    eff = effective_cue_strength(strength, donepezil)
    lit, n_on, stray, n_off = cue_counts(eff, noise)
    note = f"{lit} of the X's {n_on} neurons lit ({strength:.0%} clue strength"
    note += f", ×{DONEPEZIL_CUE_GAIN:g} with donepezil)" if donepezil else ")"
    if stray:
        note += f", plus {stray} of the other {n_off} firing by mistake ({noise:.0%} input noise)"
    if st.session_state.get("nl_mem"):
        note += ". Memantine has halved the input noise"
    return note + "."


def _half(v: int) -> int:
    return int(np.floor(v * (1 - MEMANTINE_NOISE_REDUCTION) + 0.5))


def _on_memantine() -> None:
    """Memantine moves the input-noise slider itself: on halves it, off puts it back."""
    ss = st.session_state
    if ss["nl_mem"]:
        ss["nl_noise_before_mem"] = ss["nl_noise"]
        ss["nl_noise"] = _half(ss["nl_noise"])
        ss["nl_noise_with_mem"] = ss["nl_noise"]
    else:
        before = ss.pop("nl_noise_before_mem", None)
        after = ss.pop("nl_noise_with_mem", None)
        if before is not None and ss["nl_noise"] == after:
            ss["nl_noise"] = before
        else:
            ss["nl_noise"] = min(NOISE_MAX, int(round(ss["nl_noise"] / (1 - MEMANTINE_NOISE_REDUCTION))))


def _on_severity() -> None:
    st.session_state["nl_conn"] = connectivity_for_severity(st.session_state["nl_sev"])


def _on_connectivity() -> None:
    st.session_state["nl_sev"] = severity_for_connectivity(st.session_state["nl_conn"])


def _controls() -> tuple[float, float, float, float, bool, bool]:
    ss = st.session_state
    ss.setdefault("nl_noise", 40)
    ss.setdefault("nl_sev", 57)
    ss.setdefault("nl_conn", connectivity_for_severity(ss["nl_sev"]))
    lo_conn = connectivity_for_severity(100)
    with st.container(border=True, key="nl_controls"):
        c1, c0, c2, c3, c4 = st.columns([1.1, 1.1, 1, 1, 1.05], gap="medium")
        sev = c1.slider("Alzheimer's severity", 0, 100, step=1, format="%d%%", key="nl_sev",
                        on_change=_on_severity,
                        help="0% is a healthy brain. Higher severity lowers synaptic connectivity "
                             f"(down to {1 - SYNAPSE_LOSS_AT_MAX:.0%} at 100%) and removes a few "
                             f"neurons (up to {NEURON_LOSS_AT_MAX:.0%}). It is the same brain getting "
                             "worse: damage at a higher setting includes all the damage below it.") / 100
        conn = c0.slider("Synaptic connectivity (wᵢⱼ)", lo_conn, 100, step=1, format="%d%%",
                         key="nl_conn", on_change=_on_connectivity,
                         help="Share of healthy synaptic strength kept (every synapse is weakened "
                              "by a random amount). Linked to severity: moving one moves the "
                              "other.") / 100
        noise = c2.slider("Input noise", 0, NOISE_MAX, step=1, format="%d%%", key="nl_noise",
                          help="Share of the neurons outside the X that fire by mistake in the cue. "
                               "Memantine halves it.") / 100
        if ss.get("nl_mem") and "nl_noise_before_mem" in ss:
            c2.caption(f"Memantine: {ss['nl_noise_before_mem']}% → {ss['nl_noise']}%")
        strength = c3.slider("Clue strength", 10, 100, 40, 10, format="%d%%", key="nl_cue",
                             help="Share of the X's 9 neurons lit in the cue; the rest of the "
                                  "memory starts silent.") / 100
        with c4:
            st.markdown("<div class='nl-label'>Medical intervention</div>", unsafe_allow_html=True)
            done = st.toggle("Donepezil", key="nl_done",
                             help=f"AChE inhibitor, modelled as restoring {DONEPEZIL_RECOVERY:.0%} "
                                  "of the lost synaptic strength on surviving neurons and "
                                  f"multiplying clue strength by {DONEPEZIL_CUE_GAIN:g} (stronger "
                                  "cholinergic cue signal).")
            mem = st.toggle("Memantine", key="nl_mem", on_change=_on_memantine,
                            help=f"NMDA antagonist, modelled as cutting input noise by "
                                 f"{MEMANTINE_NOISE_REDUCTION:.0%}. Fewer stray neurons fire in the "
                                 "cue, so recall rises modestly. Turning it on moves the Input "
                                 "noise slider down; turning it off moves it back.")
    return sev, conn, noise, strength, done, mem


def _noise_without_memantine(noise_pct: int, mem: bool) -> int:
    """Input noise (in %) the brain would get with memantine off, for the scenario comparison."""
    if not mem:
        return noise_pct
    ss = st.session_state
    if "nl_noise_before_mem" in ss and ss.get("nl_noise_with_mem") == noise_pct:
        return int(ss["nl_noise_before_mem"])
    return min(NOISE_MAX, int(round(noise_pct / (1 - MEMANTINE_NOISE_REDUCTION))))


SCENARIOS = (("none", "No medication"), ("done", "Donepezil"), ("mem", "Memantine"),
             ("both", "Donepezil + memantine"))


def _scenarios(scen: dict, healthy: dict, active: str, sev: float) -> None:
    n = scen["none"]["n"]
    st.markdown(f"<div class='nl-label'>Monte Carlo recall · {n:,} simulated trials per scenario, "
                "each with a fresh brain at this severity, a fresh cue and a fresh update order"
                "</div>", unsafe_allow_html=True)
    base = scen["none"]["recall"]
    tiles = []
    for key, name in SCENARIOS:
        r = scen[key]
        delta = "" if key == "none" else f"<em>{100 * (r['recall'] - base):+.1f} pts</em>"
        tiles.append(
            f"<div class='nl-sc{' on' if key == active else ''}'><span>{name}</span>"
            f"<b>{100 * r['recall']:.1f}%</b>{delta}"
            f"<div class='nl-track'><i style='width:{100 * r['recall']:.1f}%'></i></div>"
            f"<small>input noise {100 * r['noise']:.0f}% · 95% CI {100 * r['lo']:.1f}–"
            f"{100 * r['hi']:.1f}% · {r['recalled']}/{r['n']} fully recalled</small></div>")
    st.markdown(f"<div class='nl-scs' style='--acc:rgb({_accent(sev)})'>{''.join(tiles)}</div>",
                unsafe_allow_html=True)
    ref = "" if sev == 0 else (f"A healthy brain at the current cue and noise: "
                               f"{100 * healthy['recall']:.1f}%. ")
    st.caption(ref + "Recall = mean share of the 25 neurons matching the stored X once the network "
               "settles (a silent network still matches the 16 neurons outside the X, so it scores "
               "64%). Simulated outputs of an illustrative model, not clinical trial results.")


def render_neurolab(s: Settings) -> None:
    sev, conn, noise, strength, done, mem = _controls()
    args = (sev, noise, strength, done, SIMULATOR_MODE, s.ret.max_sweeps,
            s.ret.success_threshold, s.exp.base_seed, conn)
    noise_off = _noise_without_memantine(round(noise * 100), mem) / 100
    with st.spinner("Running the brain…"):
        html, scen, healthy = _simulate(args, bool(mem), noise_off, CODE_KEY)
    st.iframe(html, height="content")
    active = {(False, False): "none", (True, False): "done", (False, True): "mem",
              (True, True): "both"}[(bool(done), bool(mem))]
    _scenarios(scen, healthy, active, sev)
    with st.expander("How to read this"):
        st.markdown(
            "- The brain has stored one memory: an **X** on a 5×5 grid of 25 model neurons "
            "(hippocampal CA3). The cue lights part of the X, plus a few wrong neurons if there is "
            "input noise, and the network tries to complete the rest.\n"
            "- A neuron fires only when the input from the neurons it is connected to beats a "
            "firing threshold. **Alzheimer's severity** removes synaptic strength (and, further "
            "along, a neuron or two), so each lit neuron delivers less input. A healthy brain "
            "completes the X from a 40% cue; a damaged one needs a stronger cue, and when it is "
            "severe enough no cue is enough.\n"
            "- **Donepezil** gives back part of the lost synaptic strength and strengthens the cue. "
            "**Memantine** halves the input noise (you will see the Input noise slider move), so "
            "fewer stray neurons fight the memory and recall improves a little. Together they do "
            "best.\n"
            "- **Recall** is a Monte Carlo estimate: the same settings are simulated on 1,000 "
            "random brains, cues and update orders, and the share of neurons that end up matching "
            "the X is averaged. The animation replays the most typical of those 1,000 trials "
            "(the one whose recall is closest to the average), so it shows what usually happens.\n"
            "- In the disc each dot is a neuron: **amber = firing**, **violet = silent**, hollow = "
            "not in the cue yet. Neurons update one at a time in the exact order the simulation "
            "used; a ring marks each change. Lines are a sample of synapses (the model is fully "
            "connected): blue = consistent with the memory, red = in conflict, dotted = lost.\n"
            "- The **Memory retrieved** grid is the same neurons laid out as the 5×5 image. Recall counts "
            f"as successful when the network settles and at least {s.ret.success_threshold:.0%} of "
            "all 25 neurons match the X (a lost neuron cannot fire, so it counts as silent).\n"
            "- **Synaptic weights wᵢⱼ**: the heatmaps show all 25×25 connection strengths, scaled so "
            "+1 is the strongest healthy synapse. Yellow = excitatory (the two neurons are both in "
            "the X, or both outside it), purple = inhibitory (one in, one out), 0 = no effect. "
            "Damage pulls every value toward 0; lost neurons leave blank rows and columns.\n"
            "- Sweeps, success threshold and the random seed are in **Advanced**.")
        st.caption("Drug toggles are illustrative abstractions, not pharmacology, and have no "
                   "clinical meaning.")
