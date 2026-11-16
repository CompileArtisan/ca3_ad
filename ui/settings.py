"""Model settings. The simulator shows only its own five controls; everything else lives in
the Advanced tab and is returned as config objects (no model logic here)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import streamlit as st

from ca3model import DamageSpec, ExperimentConfig, NetworkConfig, RetrievalConfig

from .text import HELP

SQUARES = [100, 144, 196, 256, 324, 400]


@dataclass(frozen=True)
class Settings:
    net: NetworkConfig
    ret: RetrievalConfig
    damage: DamageSpec
    exp: ExperimentConfig
    pattern_type: str

    def key(self) -> str:
        import json
        return json.dumps({"net": self.net.to_dict(), "ret": self.ret.to_dict(),
                           "damage": self.damage.to_dict(), "exp": self.exp.to_dict(),
                           "ptype": self.pattern_type}, sort_keys=True)


def _group(title: str, note: str = ""):
    box = st.container(border=True)
    box.markdown(f"**{title}**")
    if note:
        box.caption(note)
    return box


def render_settings() -> Settings:
    if "base_seed" not in st.session_state:
        st.session_state["base_seed"] = 12345

    c1, c2, c3 = st.columns(3)
    with c1, _group("Network", "Also used by the simulator."):
        n = st.select_slider("Network size N", options=SQUARES, value=256, help=HELP["n"])
        p = st.slider("Stored memories P", 1, 60, 8, help=HELP["p"])
        st.caption(f"load α = P/N = **{p / n:.3f}**  (theory: α_c ≈ 0.138 for random patterns)")
        sweeps = st.slider("Max sweeps", 5, 100, 30, help=HELP["sweeps"])
        thr = st.slider("Success threshold (survivor accuracy)", 0.80, 1.00, 0.95, 0.01,
                        help=HELP["thr"])
    with c2, _group("Damage", "Neuron loss and weakening mode also apply to the simulator."):
        neu = st.slider("Neuronal loss fraction", 0.0, 0.9, 0.0, 0.01, help=HELP["neu"])
        mode = st.radio("Weakening mode", ["deletion", "attenuation"], horizontal=True,
                        help=HELP["syn_mode"])
        syn = st.slider("Synaptic weakening severity", 0.0, 1.0, 0.0, 0.01,
                        help=HELP["syn"] + " Analysis tabs only; the simulator uses its "
                                          "connectivity slider.")
        eta = st.slider("Plasticity efficacy (optional encoding impairment)", 0.0, 1.0, 1.0, 0.05,
                        help=HELP["eta"])
        n_new = st.slider("New patterns (learning experiments)", 1, 20, 5, help=HELP["new"])
    with c3, _group("Analysis cues & reproducibility", "Cue settings here apply to the analysis "
                    "tabs below; the simulator uses its own cue sliders."):
        noise = st.slider("Cue noise", 0.0, 0.5, 0.20, 0.01, help=HELP["noise"])
        cue_mode = st.radio("Cue mode", ["flip", "occlude"], horizontal=True, help=HELP["mode_cue"])
        trials = st.slider("Trials", 5, 200, 30, help=HELP["trials"])
        a, b = st.columns([2, 1], vertical_alignment="bottom")
        seed = a.number_input("Base seed", step=1, key="base_seed", help=HELP["seed"])
        if b.button("Re-roll", help="Pick a new random base seed.", width="stretch"):
            st.session_state["base_seed"] = random.randint(0, 2**31 - 1)
            st.rerun()
        ptype = st.radio("Pattern type", ["random", "smooth"], horizontal=True, help=HELP["ptype"])
        if ptype == "smooth":
            st.warning("Smooth patterns: visual demo only; statistics use random patterns.")

    return Settings(
        net=NetworkConfig(int(n)),
        ret=RetrievalConfig(cue_noise=noise, cue_mode=cue_mode, max_sweeps=int(sweeps),
                            success_threshold=float(thr)),
        damage=DamageSpec(syn, mode, neu),
        exp=ExperimentConfig(n_patterns=int(p), n_trials=int(trials), base_seed=int(seed),
                             n_new_patterns=int(n_new), plasticity_efficacy=float(eta)),
        pattern_type=ptype,
    )
