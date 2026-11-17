#!/usr/bin/env python3

"""RECALL4 NEUROLAB simulator view: one brain, an Alzheimer's severity slider, input noise, cue
strength and two drug toggles. The animation replays the network's real asynchronous update
order (see ``ca3model.neurolab``), so every flip on screen happened in the simulation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from ca3model.neurolab import (DONEPEZIL_RECOVERY, GRID, MEMANTINE_NOISE_REDUCTION,
                               NEURON_LOSS_AT_MAX, SIMULATOR_MODE, SYNAPSE_LOSS_AT_MAX,
                               covariance_weights,
                               cue_counts, neuron_layout, recall_reliability, run_stimulator)

from .settings import Settings

TEMPLATE = (Path(__file__).with_name("brain_view.html")).read_text(encoding="utf-8")
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
def _simulate(args: tuple) -> tuple[str, dict, dict]:
    sev, noise, strength, _, *rest = args
    return (_payload(run_stimulator(*args), args), recall_reliability(*args),
            recall_reliability(0.0, noise, strength, False, *rest))


def _round(a, nd=3):
    return [None if np.isnan(v) else round(float(v), nd) for v in a]


def _payload(r: dict, args: tuple) -> str:
    sev, noise, strength, done, *_ = args
    b = r["brain"]
    brain, m = b.brain, b.metrics
    n = brain.n
    xy, edges = neuron_layout(n)
    wsign = np.sign(covariance_weights(r["target"])[edges[:, 0], edges[:, 1]]).astype(int)
    eff = brain.efficacy[edges[:, 0], edges[:, 1]]
    title = "Healthy brain" if sev == 0 else f"Alzheimer's-inspired brain · {sev:.0%} severity"
    if done:
        title += " · donepezil"
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
        "cueNote": _cue_note(strength, noise),
    }
    return TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def _cue_note(strength: float, noise: float) -> str:
    lit, n_on, stray, n_off = cue_counts(strength, noise)
    note = f"{lit} of the X's {n_on} neurons lit ({strength:.0%} cue strength)"
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


def _controls() -> tuple[float, float, float, bool, bool]:
    ss = st.session_state
    ss.setdefault("nl_noise", 10)
    with st.container(border=True, key="nl_controls"):
        c1, c2, c3, c4 = st.columns([1.15, 1, 1, 1.05], gap="large")
        sev = c1.slider("Alzheimer's severity", 0, 100, 65, 5, format="%d%%", key="nl_sev",
                        help="0% is a healthy brain. Higher severity removes synaptic strength "
                             f"(up to {SYNAPSE_LOSS_AT_MAX:.0%} at 100%) and a few neurons (up to "
                             f"{NEURON_LOSS_AT_MAX:.0%}). It is the same brain getting worse: "
                             "damage at a higher setting includes all the damage below it.") / 100
        noise = c2.slider("Input noise", 0, NOISE_MAX, step=1, format="%d%%", key="nl_noise",
                          help="Share of the neurons outside the X that fire by mistake in the cue. "
                               "Memantine halves it.") / 100
        if ss.get("nl_mem") and "nl_noise_before_mem" in ss:
            c2.caption(f"Memantine: {ss['nl_noise_before_mem']}% → {ss['nl_noise']}%")
        strength = c3.slider("Cue strength", 10, 100, 40, 10, format="%d%%", key="nl_cue",
                             help="Share of the X's 9 neurons lit in the cue; the rest of the "
                                  "memory starts silent.") / 100
        with c4:
            st.markdown("<div class='nl-label'>Medical intervention</div>", unsafe_allow_html=True)
            done = st.toggle("Donepezil", key="nl_done",
                             help=f"AChE inhibitor, modelled as restoring {DONEPEZIL_RECOVERY:.0%} "
                                  "of the lost synaptic strength on surviving neurons.")
            mem = st.toggle("Memantine", key="nl_mem", on_change=_on_memantine,
                            help=f"NMDA antagonist, modelled as cutting input noise by "
                                 f"{MEMANTINE_NOISE_REDUCTION:.0%}. Turning it on moves the "
                                 "Input noise slider down; turning it off moves it back.")
    return sev, noise, strength, done, mem


def _reliability(rel: dict, healthy: dict, sev: float) -> None:
    n = rel["n"]
    st.markdown(f"<div class='nl-label'>Is this run typical? Same brain, {n} fresh cues at these "
                "settings</div>", unsafe_allow_html=True)
    k, h = rel["recalled"], healthy["recalled"]
    ref = "" if sev == 0 else (f"<small>A healthy brain recalls {h}/{n} of the same cues</small>")
    st.markdown(f"<div class='nl-rel' style='--acc:rgb({_accent(sev)})'><span>This brain</span>"
                f"<b>{k}/{n} recalled</b><div class='nl-track'><i style='width:{100 * k / n:.1f}%'>"
                f"</i></div>{ref}</div>", unsafe_allow_html=True)


def render_neurolab(s: Settings) -> None:
    sev, noise, strength, done, _ = _controls()
    args = (sev, noise, strength, done, SIMULATOR_MODE, s.ret.max_sweeps,
            s.ret.success_threshold, s.exp.base_seed)
    with st.spinner("Running the brain…"):
        html, rel, healthy = _simulate(args)
    st.iframe(html, height="content")
    _reliability(rel, healthy, sev)
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
            "- **Donepezil** gives back part of the lost synaptic strength. **Memantine** halves the "
            "input noise; you will see the Input noise slider move.\n"
            "- In the disc each dot is a neuron: **amber = firing**, **violet = silent**, hollow = "
            "not in the cue yet. Neurons update one at a time in the exact order the simulation "
            "used; a ring marks each change. Lines are a sample of synapses (the model is fully "
            "connected): blue = consistent with the memory, red = in conflict, dotted = lost.\n"
            "- The grid under the brain is the same neurons laid out as the 5×5 image. Recall counts "
            f"as successful when the network settles and at least {s.ret.success_threshold:.0%} of "
            "all 25 neurons match the X (a lost neuron cannot fire, so it counts as silent).\n"
            "- Sweeps, success threshold and the random seed are in **Advanced**.")
        st.caption("Drug toggles are illustrative abstractions, not pharmacology, and have no "
                   "clinical meaning.")
