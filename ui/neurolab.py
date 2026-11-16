#!/usr/bin/env python3

"""Neurolab simulator view: five controls (connectivity, noise, cue size, two drug toggles) and an
animated healthy-vs-damaged brain. The animation replays the network's real asynchronous update
order (see ``ca3model.neurolab``), so every flip on screen happened in the simulation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from ca3model.neurolab import (DONEPEZIL_RECOVERY, MEMANTINE_NOISE_REDUCTION, neuron_layout,
                               recall_reliability, run_stimulator)
from ca3model.patterns import grid_shape

from .settings import Settings

TEMPLATE = (Path(__file__).with_name("brain_view.html")).read_text(encoding="utf-8")
ACCENT = {"healthy": "45,212,191", "damaged": "251,113,133"}
ATTRACTOR_WORDS = {"target": "settled on the stored memory",
                   "other_stored": "settled on a different stored memory",
                   "mirror": "settled on the memory's mirror image",
                   "spurious": "settled on a blend of memories (spurious state)"}


@st.cache_data(show_spinner=False, max_entries=64)
def _simulate(args: tuple) -> tuple[str, dict]:
    r = run_stimulator(*args)
    return _payload(r, args), recall_reliability(*args)


def _round(a, nd=3):
    return [None if np.isnan(v) else round(float(v), nd) for v in a]


def _payload(r: dict, args: tuple) -> str:
    n, _, conn, noise, given, done, mem, loss, mode, _, thr, _ = args
    xy, edges = neuron_layout(n)
    wsign = np.sign(r["healthy"].net.W_raw[edges[:, 0], edges[:, 1]]).astype(int)
    brains = []
    for key in ("healthy", "damaged"):
        b = r[key]
        m, net = b.metrics, b.net
        eff = net.efficacy[edges[:, 0], edges[:, 1]]
        cs = net.connectivity_stats()
        title = "Healthy brain" if key == "healthy" else "Alzheimer's-inspired brain"
        if key == "damaged" and (done or mem):
            title += " · treated"
        chips = [f"{m['sweeps']} sweep{'s' * (m['sweeps'] != 1)}",
                 f"neurons {cs['n_alive']}/{n}",
                 f"synaptic strength {100 * (1 - cs['frac_synapses_lost']):.0f}%"]
        verdict = (f"{ATTRACTOR_WORDS[m['attractor_type']]}"
                   + ("" if m["converged"] else " · did not settle"))
        brains.append({
            "title": title, "accentRGB": ACCENT[key], "success": bool(m["success"]),
            "sweeps": int(m["sweeps"]), "states": [s.astype(int).tolist() for s in b.states],
            "pos": [_round(p) for p in b.update_pos], "alive": net.alive.astype(int).tolist(),
            "eff": [round(float(e), 2) for e in eff], "chips": chips, "verdict": verdict,
            "atrophy": 1.0 - 0.25 * float(np.mean(~net.alive)),
        })
    shown = int(round(given * n))
    wrong = int(round(noise * shown))
    note = f"{shown} of {n} neurons shown ({given:.0%}), {wrong} of them wrong ({noise:.0%})."
    if mem:
        w2 = int(round(noise * (1 - MEMANTINE_NOISE_REDUCTION) * shown))
        note += f" With memantine the treated brain gets {w2} wrong."
    data = {"n": n, "grid": list(grid_shape(n)), "xy": [[round(float(a), 4), round(float(c), 4)]
                                                          for a, c in xy],
            "edges": edges.tolist(), "wsign": wsign.tolist(), "target": r["target"].astype(int).tolist(),
            "thr": thr, "brains": brains, "cueNote": note}
    return TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def _controls() -> tuple[float, float, float, bool, bool]:
    with st.container(border=True, key="nl_controls"):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.05], gap="large")
        conn = c1.slider("Synaptic connectivity", 0, 100, 10, 1, format="%d%%",
                         help="How much synaptic strength the Alzheimer's-inspired brain keeps "
                              "(100% = healthy). The healthy brain always keeps 100%.") / 100
        noise = c2.slider("Input noise", 0, 50, 10, 1, format="%d%%",
                          help="Share of the shown cue neurons that are wrong.") / 100
        given = c3.slider("Cue given", 5, 100, 30, 5, format="%d%%",
                          help="Share of the memory shown to the network; the rest starts unknown.") / 100
        with c4:
            st.markdown("<div class='nl-label'>Medical intervention</div>", unsafe_allow_html=True)
            done = st.toggle("Donepezil", help=f"AChE inhibitor, modelled as restoring "
                             f"{DONEPEZIL_RECOVERY:.0%} of the lost synaptic strength on surviving "
                             "neurons. Treated brain only.")
            mem = st.toggle("Memantine", help=f"NMDA antagonist, modelled as removing "
                            f"{MEMANTINE_NOISE_REDUCTION:.0%} of the input noise. Treated brain only.")
    return conn, noise, given, done, mem


def _reliability(rel: dict, treated: bool) -> None:
    n = rel["n"]
    st.markdown("<div class='nl-label'>Is this run typical? Same two brains, "
                f"{n} fresh cues across every stored memory</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    for col, key, name in ((a, "healthy", "Healthy brain"),
                           (b, "damaged", "Alzheimer's-inspired" + (" · treated" if treated else ""))):
        k = rel[key]
        with col:
            st.markdown(f"<div class='nl-rel nl-{key}'><span>{name}</span><b>{k}/{n} recalled</b>"
                        f"<div class='nl-track'><i style='width:{100 * k / n:.1f}%'></i></div></div>",
                        unsafe_allow_html=True)


def render_neurolab(s: Settings) -> None:
    conn, noise, given, done, mem = _controls()
    args = (s.net.n_neurons, s.exp.n_patterns, conn, noise, given, done, mem,
            s.damage.neuron_loss, s.damage.synaptic_mode, s.ret.max_sweeps,
            s.ret.success_threshold, s.exp.base_seed)
    with st.spinner("Running both brains…"):
        html, rel = _simulate(args)
    st.iframe(html, height="content")
    _reliability(rel, done or mem)
    with st.expander("How to read this"):
        st.markdown(
            "- Both brains store the same memories and get the same incomplete cue. The healthy "
            "brain keeps every synapse; the Alzheimer's-inspired brain keeps only the *synaptic "
            "connectivity* you set.\n"
            "- The disc is a model of the hippocampal CA3 network. Each dot is a model neuron, "
            "**amber = firing**, **violet = silent**, hollow = not given in the cue. Neurons update "
            "one at a time in the exact order the simulation used; a ring marks each flip.\n"
            "- Lines are a sample of synapses between neighbours on screen (the model itself is "
            "fully connected). Blue = the two neurons agree with what the synapse learned, "
            "red = they conflict, dotted = synapse lost. As a memory is recalled the network "
            "turns blue and the hippocampus glows.\n"
            "- The grid under each brain is the same neurons laid out as an image; compare it with "
            "the stored memory on the left. Hover any neuron or grid cell to inspect it.\n"
            "- Recall counts as successful when the network stops changing and at least "
            f"{s.ret.success_threshold:.0%} of surviving neurons match the stored memory.\n"
            "- Network size, number of memories, neuron loss, damage mode and the success "
            "threshold are in **Advanced**.")
        st.caption("Drug toggles are illustrative abstractions, not pharmacology, and have no "
                   "clinical meaning.")
