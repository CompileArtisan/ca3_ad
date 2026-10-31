import numpy as np
import pandas as pd
from ca3model import (DamageSpec, ExperimentConfig, NetworkConfig, RetrievalConfig, exp_combined,
                      exp_healthy_vs_neuron_loss, exp_healthy_vs_weakened, exp_memory_load,
                      exp_new_learning, exp_weakening_vs_loss, interaction_table,
                      sequential_learning_curve, trial_rngs)
from ca3model.experiments import TIDY_COLUMNS, run_trial_conditions

NET = NetworkConfig(100)
EXP = ExperimentConfig(n_patterns=6, n_trials=3, base_seed=7, n_new_patterns=3)
RET = RetrievalConfig(cue_noise=0.2, max_sweeps=15)
M = ["accuracy_surv", "hamming_all", "sweeps", "n_flips", "success", "final_energy"]


def test_determinism_and_columns():
    a = exp_healthy_vs_weakened(NET, EXP, RET, [0.0, 0.3, 0.6])
    b = exp_healthy_vs_weakened(NET, EXP, RET, [0.0, 0.3, 0.6])
    pd.testing.assert_frame_equal(a, b)
    assert set(TIDY_COLUMNS) <= set(a.columns)
    assert a.attrs["meta"]["suite"] == "A" and "network" in a.attrs["meta"]


def test_severity_zero_equals_healthy_exactly():
    df = exp_healthy_vs_weakened(NET, EXP, RET, [0.0, 0.5], mode="attenuation")
    h = df[df.condition == "healthy"].sort_values(["trial", "pattern_idx"])[M].reset_index(drop=True)
    z = df[(df.condition == "synaptic") & (df.synaptic_severity == 0)] \
        .sort_values(["trial", "pattern_idx"])[M].reset_index(drop=True)
    pd.testing.assert_frame_equal(h, z)


def test_paired_cues_identical_across_conditions():
    rows = run_trial_conditions(NET, EXP, RET, {"healthy@0": DamageSpec(),
                                                "synaptic@0.5": DamageSpec(0.5)}, 0)
    rngs = trial_rngs(EXP.base_seed, 0)
    from ca3model import corrupt_cue
    c1, c2 = rngs.fresh_cues(), rngs.fresh_cues()
    pats = np.random.default_rng(0).integers(0, 2, (3, 100)) * 2 - 1
    for p in pats:
        assert np.array_equal(corrupt_cue(p, .2, "flip", c1), corrupt_cue(p, .2, "flip", c2))
    assert {r["condition"] for r in rows} == {"healthy", "synaptic"}


def test_suites_B_C_D_shapes():
    b = exp_healthy_vs_neuron_loss(NET, EXP, RET, [0.0, 0.3])
    assert set(b.condition) == {"healthy", "neuron"}
    c = exp_weakening_vs_loss(NET, EXP, RET, [0.0, 0.2])
    assert set(c.condition) == {"healthy", "synaptic", "neuron", "matched_synaptic"}
    nl = c[(c.condition == "neuron") & (c.neuron_loss == 0.2)]["matched_synapse_loss_frac"]
    assert np.allclose(nl, 1 - 0.8 ** 2)
    ms = c[(c.condition == "matched_synaptic") & (c.synaptic_severity > 0)]
    assert np.allclose(ms.synaptic_severity, 1 - 0.8 ** 2)
    d = exp_combined(NET, EXP, RET, [0.0, 0.3], [0.0, 0.2])
    assert set(d.condition) == {"healthy", "synaptic_only", "neuron_only", "combined"}
    it = interaction_table(d)
    assert len(it) == 4 and {"deficit_additive", "interaction"} <= set(it.columns)
    zero = it[(it.synaptic_severity == 0) & (it.neuron_loss == 0)].iloc[0]
    assert zero["interaction"] == 0 and zero["deficit_combined"] == 0


def test_new_learning_invariants():
    e = exp_new_learning(NET, EXP, RET, [0.0, 0.4], kind="synaptic")
    assert set(e.condition) == {"healthy_learn", "damage_then_learn", "learn_then_damage"}
    assert set(e.pattern_group) == {"old", "new"}
    hl = e[e.condition == "healthy_learn"].sort_values(["trial", "pattern_idx"])[M].reset_index(drop=True)
    ld0 = e[(e.condition == "learn_then_damage") & (e.level == 0)] \
        .sort_values(["trial", "pattern_idx"])[M].reset_index(drop=True)
    pd.testing.assert_frame_equal(hl, ld0)
    # with eta = 1 the two damage orders give the same W_eff -> identical rows
    dl = e[(e.condition == "damage_then_learn") & (e.level == 0.4)] \
        .sort_values(["trial", "pattern_idx"])[M].reset_index(drop=True)
    ld = e[(e.condition == "learn_then_damage") & (e.level == 0.4)] \
        .sort_values(["trial", "pattern_idx"])[M].reset_index(drop=True)
    pd.testing.assert_frame_equal(dl, ld)


def test_sequential_and_memory_load():
    s = sequential_learning_curve(NET, EXP, RET, DamageSpec(0.4))
    assert set(s.stage) == {"immediate", "final"} and set(s.condition) == {"healthy", "damaged"}
    f = exp_memory_load(NET, EXP, RET, load_values=[1, 5, 10])
    assert np.allclose(f.load_alpha, f.n_patterns / 100)
    assert set(f.condition) == {"healthy", "synaptic", "neuron", "combined"}


def test_progress_callback():
    seen = []
    exp_healthy_vs_weakened(NET, EXP, RET, [0.1], progress=seen.append)
    assert seen[-1] == 1.0 and len(seen) == EXP.n_trials
