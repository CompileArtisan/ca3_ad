"""Global parameter sidebar. Returns config objects (no model logic here)."""
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


def render_sidebar() -> Settings:
    sb = st.sidebar
    sb.header("Model parameters")
    if "base_seed" not in st.session_state:
        st.session_state["base_seed"] = 12345

    n = sb.select_slider("Network size N", options=SQUARES, value=256, help=HELP["n"])
    p = sb.slider("Stored memories P", 1, 60, 8, help=HELP["p"])
    sb.caption(f"load α = P/N = **{p / n:.3f}**  (theory: α_c ≈ 0.138 for random patterns)")

    sb.subheader("Retrieval")
    noise = sb.slider("Cue noise", 0.0, 0.5, 0.20, 0.01, help=HELP["noise"])
    cue_mode = sb.radio("Cue mode", ["flip", "occlude"], horizontal=True, help=HELP["mode_cue"])
    sweeps = sb.slider("Max sweeps", 5, 100, 30, help=HELP["sweeps"])
    thr = sb.slider("Success threshold (survivor accuracy)", 0.80, 1.00, 0.95, 0.01, help=HELP["thr"])

    sb.subheader("Damage (model parameters)")
    syn = sb.slider("Synaptic weakening severity", 0.0, 1.0, 0.0, 0.01, help=HELP["syn"])
    mode = sb.radio("Weakening mode", ["deletion", "attenuation"], horizontal=True, help=HELP["syn_mode"])
    neu = sb.slider("Neuronal loss fraction", 0.0, 0.9, 0.0, 0.01, help=HELP["neu"])
    eta = sb.slider("Plasticity efficacy (optional encoding impairment)", 0.0, 1.0, 1.0, 0.05,
                    help=HELP["eta"])
    n_new = sb.slider("New patterns (learning experiments)", 1, 20, 5, help=HELP["new"])

    sb.subheader("Reproducibility")
    trials = sb.slider("Trials", 5, 200, 30, help=HELP["trials"])
    c1, c2 = sb.columns([2, 1])
    seed = c1.number_input("Base seed", step=1, key="base_seed", help=HELP["seed"])
    c2.write("")
    if c2.button("Re-roll", help="Pick a new random base seed."):
        st.session_state["base_seed"] = random.randint(0, 2**31 - 1)
        st.rerun()
    ptype = sb.radio("Pattern type", ["random", "smooth"], horizontal=True, help=HELP["ptype"])
    if ptype == "smooth":
        sb.warning("Smooth patterns: visual demo only; statistics use random patterns.")

    return Settings(
        net=NetworkConfig(int(n)),
        ret=RetrievalConfig(cue_noise=noise, cue_mode=cue_mode, max_sweeps=int(sweeps),
                            success_threshold=float(thr)),
        damage=DamageSpec(syn, mode, neu),
        exp=ExperimentConfig(n_patterns=int(p), n_trials=int(trials), base_seed=int(seed),
                             n_new_patterns=int(n_new), plasticity_efficacy=float(eta)),
        pattern_type=ptype,
    )
