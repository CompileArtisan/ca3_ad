"""Model settings. The simulator shows only its own controls; everything else lives in
the Advanced tab and is returned as config objects (no model logic here)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import streamlit as st

from ca3model import DamageSpec, ExperimentConfig, NetworkConfig, RetrievalConfig
from ca3model.neurolab import (EXP1_ACTIVITY_THRESHOLD, EXP1_CUE_GAIN, EXP1_GAIN_LEVELS,
                               EXP1_THRESHOLD_LEVELS, N_TRIALS, TRIAL_SEED_BASE)

from .text import HELP

SQUARES = [100, 144, 196, 256, 324, 400]


@dataclass(frozen=True)
class SimSettings:
    """Simulator options. Defaults are the primary settings of the two reports."""

    drug_model: str = "exp2"          # "exp2": cue amplitude / input noise; "exp1": gain g / gate theta
    cue_gain: float = EXP1_CUE_GAIN
    activity_threshold: float = EXP1_ACTIVITY_THRESHOLD
    n_stored: int = 3
    random_fill: bool = False
    trial: int = 0
    n_trials: int = N_TRIALS
    strength: str = "High"            # Experiment 2 strength: cue 25/30/35%, noise 8/6/4%


@dataclass(frozen=True)
class Settings:
    net: NetworkConfig
    ret: RetrievalConfig
    damage: DamageSpec
    exp: ExperimentConfig
    pattern_type: str
    sim: SimSettings = SimSettings()

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
    with c1, _group("Network", "Analysis tabs only. The simulator uses the reports' fixed "
                    "values: 25 neurons, 3 memories, at most 100 sweeps, success at m ≥ 0.8."):
        n = st.select_slider("Network size N", options=SQUARES, value=256, help=HELP["n"])
        p = st.slider("Stored memories P", 1, 60, 8, help=HELP["p"])
        st.caption(f"load α = P/N = **{p / n:.3f}**  (theory: α_c ≈ 0.138 for random patterns)")
        sweeps = st.slider("Max sweeps", 5, 100, 30, help=HELP["sweeps"])
        thr = st.slider("Success threshold (survivor accuracy)", 0.80, 1.00, 0.95, 0.01,
                        help=HELP["thr"])
    with c2, _group("Damage", "Analysis tabs only; the simulator's damage comes from its "
                    "connectivity slider."):
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

    sim = _simulator_settings()
    return Settings(
        net=NetworkConfig(int(n)),
        ret=RetrievalConfig(cue_noise=noise, cue_mode=cue_mode, max_sweeps=int(sweeps),
                            success_threshold=float(thr)),
        damage=DamageSpec(syn, mode, neu),
        exp=ExperimentConfig(n_patterns=int(p), n_trials=int(trials), base_seed=int(seed),
                             n_new_patterns=int(n_new), plasticity_efficacy=float(eta)),
        pattern_type=ptype,
        sim=sim,
    )


def _simulator_settings() -> SimSettings:
    with _group("Simulator", "Defaults are the primary settings of the two Monte Carlo reports. "
                "The alternatives are the reports' own sensitivity runs and Experiment 1's drug model."):
        a, b, c = st.columns(3)
        model = a.radio("Drug model", ["Experiment 2: cue amplitude / input noise",
                                       "Experiment 1: cue gain g / activity gate θ"],
                        key="sim_drug_model",
                        help="Experiment 2 (default): donepezil-inspired = the 5 cue neurons start at "
                             "amplitude 1.25 / 1.50 / 1.75 (effective cue 25 / 30 / 35%); "
                             "memantine-inspired = input noise 2 / 4 / 6 points lower (10% → 8 / 6 / 4%). "
                             "Experiment 1: donepezil-inspired = persistent cue input h += g·cue; "
                             "memantine-inspired = a neuron fires only when |h| > θ.")
        exp1 = model.startswith("Experiment 1")
        strength = a.select_slider("Intervention strength (Experiment 2)",
                                   options=["Low", "Moderate", "High"], value="High",
                                   disabled=exp1, key="sim_strength",
                                   help="Donepezil: effective cue strength 25 / 30 / 35% (amplitude "
                                        "1.25 / 1.50 / 1.75). Memantine: input noise 2 / 4 / 6 points "
                                        "lower (10% → 8 / 6 / 4%).")
        gain = a.select_slider("Cue gain g (Experiment 1)", options=list(EXP1_GAIN_LEVELS),
                               value=EXP1_CUE_GAIN, disabled=not exp1, key="sim_gain")
        theta = a.select_slider("Activity threshold θ (Experiment 1)",
                                options=list(EXP1_THRESHOLD_LEVELS), value=EXP1_ACTIVITY_THRESHOLD,
                                disabled=not exp1, key="sim_theta")
        n_stored = b.radio("Stored memories", [3, 1], horizontal=True, key="sim_p",
                           format_func=lambda p: "3 (primary)" if p == 3 else "1 (sensitivity)")
        fill = b.radio("Uncued neurons start", ["silent (0)", "random ±1"], horizontal=True,
                       key="sim_fill", help="Silent is the primary model; random ±1 is Experiment "
                                            "A's random-fill sensitivity run.")
        trial = c.number_input("Trial shown", 0, N_TRIALS - 1, 0, step=1, key="sim_trial",
                               help=f"The animated run uses trial t, seed {TRIAL_SEED_BASE:,} + t: its "
                                    "connectivity mask, noise draws and update order.")
        n_trials = c.select_slider("Monte Carlo trials", options=[100, 500, N_TRIALS],
                                   value=N_TRIALS, key="sim_trials",
                                   help="Experiment 2 used 2,000 matched trials per condition; "
                                        "Experiment 1 used 100.")
    return SimSettings("exp1" if exp1 else "exp2", float(gain), float(theta), int(n_stored),
                       fill.startswith("random"), int(trial), int(n_trials), strength)
