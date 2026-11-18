"""The simulator model must use the values and settings of the two Monte Carlo reports."""
import numpy as np
import pytest

from ca3model.neurolab import (CUE_STRENGTHS, MAX_SWEEPS, N_NEURONS, N_PAIRS, REDUCED_NOISE,
                               SUCCESS_OVERLAP, Condition, apply_drugs, connectivity_masks,
                               cue_neurons, hebbian_weights, initial_input, neuron_layout,
                               pairs_kept, run_trial, run_trials, stored_patterns, summarise,
                               trial_draws, wilson)


def _success(cond, n=2000):
    return 100 * summarise(run_trials(cond, n))["success"]


def test_network_memories_and_cue_match_the_reports():
    P = stored_patterns()
    assert P.shape == (3, 25) and set(np.unique(P)) == {-1, 1}
    assert list(cue_neurons()) == [2, 4, 6, 7, 11]
    c = cue_neurons()
    # "The 5-neuron cue shares 4 of its 5 values with stored memory #2."
    assert int((P[1, c] == P[0, c]).sum()) == 4
    assert np.array_equal(stored_patterns(1)[0], P[0])          # one-memory run: same seed, P = 1
    W = hebbian_weights(P)
    assert np.allclose(W, P.T.astype(float) @ P / 25 - np.eye(25) * 3 / 25)
    assert np.all(np.diag(W) == 0) and np.allclose(W, W.T)
    assert N_PAIRS == 300 and MAX_SWEEPS == 100 and SUCCESS_OVERLAP == 0.8


def test_connectivity_keeps_exactly_round_c_300_pairs_nested():
    d = trial_draws(0, 20)
    prev = None
    for c in (1.0, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2):
        M = connectivity_masks(c, d)
        assert pairs_kept(c) == round(c * 300)
        assert np.all(np.triu(M, 1).sum(axis=(1, 2)) == pairs_kept(c))
        assert np.allclose(M, M.transpose(0, 2, 1)) and np.all(np.einsum("tii->ti", M) == 0)
        if prev is not None:
            assert np.all(M <= prev)                             # lower c removes a superset
        prev = M


def test_cue_and_noise_rules():
    d = trial_draws(0, 2000)
    xi = stored_patterns()[0]
    x = initial_input(Condition(noise=0.0), d)
    assert np.all(x[:, cue_neurons()] == xi[cue_neurons()])
    assert np.count_nonzero(x, axis=1).max() == 5               # the other 20 start silent
    # Experiment 2: average corrupted neurons per trial at 10 / 8 / 6 / 4% noise
    counts = [round(float((d.u < p).sum(1).mean()), 2) for p in REDUCED_NOISE]
    assert counts == [2.53, 2.05, 1.54, 1.03]
    x = initial_input(Condition(noise=0.1, cue_amplitude=1.75), d)
    hit = d.u < 0.1
    cue = np.zeros(25, bool)
    cue[cue_neurons()] = True
    assert np.allclose(np.abs(x[:, cue]), 1.75)                  # amplitude kept when flipped
    assert np.all(np.abs(x[hit & ~cue[None]]) == 1)              # spurious activity at +-1
    assert np.all(x[~hit & ~cue[None]] == 0)


def test_reveal_comparison_lights_5_6_8_9_neurons():
    d = trial_draws(0, 50)
    for s, k in zip(CUE_STRENGTHS, (5, 6, 8, 9)):
        x = initial_input(Condition(noise=0.0, cue_fraction=s), d)
        assert np.all(np.count_nonzero(x, axis=1) == k) and set(np.unique(np.abs(x))) <= {0, 1}


@pytest.mark.parametrize("conn,group,pdf", [
    (1.0, "A", 51.6), (1.0, "B", 60.7), (1.0, "C", 63.9), (1.0, "D", 72.0),
    (0.6, "A", 45.0), (0.4, "A", 36.2), (0.3, "A", 25.8), (0.3, "D", 33.0)])
def test_experiment_2_table_reproduced_within_monte_carlo_error(conn, group, pdf):
    kw = {"A": {}, "B": {"cue_amplitude": 1.75}, "C": {"noise": 0.04},
          "D": {"cue_amplitude": 1.75, "noise": 0.04}}[group]
    assert _success(Condition(conn, **kw)) == pytest.approx(pdf, abs=2.5)


def test_all_trials_converge_and_sweeps_match_experiment_a():
    r = summarise(run_trials(Condition(), 2000))
    assert r["outcomes"]["not converged"] == 0
    assert r["mean_sweeps"] == pytest.approx(2.15, abs=0.1)     # Experiment A, 100%
    assert r["outcomes"]["different memory"] > r["outcomes"]["mixture"]  # failures mostly memory #2


def test_one_memory_sensitivity_tolerates_connection_loss():
    assert _success(Condition(1.0, n_stored=1), 500) > 95
    assert _success(Condition(0.2, noise=0.0, n_stored=1), 500) == 100


def test_drug_toggles():
    base = Condition(0.4, noise=0.10)
    for name, (cue, noise) in {"Low": (0.25, 0.08), "Moderate": (0.30, 0.06),
                               "High": (0.35, 0.04)}.items():
        c = apply_drugs(base, True, True, "exp2", name)
        assert c.cue_amplitude == pytest.approx(cue / 0.2) and c.noise == pytest.approx(noise)
        assert c.effective_cue == pytest.approx(cue)
        assert c.cue_gain == 0 and c.activity_threshold == 0
    c = apply_drugs(base, True, True, "exp1")
    assert (c.cue_gain, c.activity_threshold, c.cue_amplitude, c.noise) == (0.1, 0.04, 1.0, 0.1)
    assert apply_drugs(base, False, False, "exp2") == base


def test_animation_order_reproduces_the_simulation():
    for cond in (Condition(), Condition(0.4, noise=0.3, cue_amplitude=1.75, cue_fraction=0.4),
                 Condition(0.3, cue_gain=0.1, activity_threshold=0.04)):
        for trial in (0, 5):
            run = run_trial(cond, trial)
            res = run_trials(cond, 1, first_trial=trial)
            assert np.array_equal(np.sign(res["final"][0]), run.states[-1])
            assert run.metrics["sweeps"] == res["sweeps"][0] == len(run.states) - 1
            for k in range(run.metrics["sweeps"]):
                assert np.array_equal(np.argsort(run.update_pos[k]), run.orders[k])


def test_severity_dial_spans_the_reports_connectivity_range():
    from ca3model.neurolab import connectivity_for_severity, severity_for_connectivity
    assert [connectivity_for_severity(v) for v in (0, 25, 50, 75, 100)] == [100, 80, 60, 40, 20]
    for sev in range(0, 101):
        assert abs(severity_for_connectivity(connectivity_for_severity(sev)) - sev) <= 1


def test_reveal_more_neurons_comparison_run():
    # Experiment 2 comparison: revealing 9 correct neurons raised intact retrieval to 77.0%
    assert _success(Condition(cue_fraction=0.35)) == pytest.approx(77.0, abs=2.5)


def test_cue_below_20pct_uses_a_subset_of_the_5_cue_neurons():
    x = initial_input(Condition(noise=0.0, cue_fraction=0.08), trial_draws(0, 3))
    assert np.all(np.flatnonzero(x[0]) == cue_neurons()[:2])


def test_wilson_and_layout():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    xy, edges = neuron_layout(N_NEURONS)
    assert xy.shape == (N_NEURONS, 2) and np.all(np.hypot(*xy.T) <= 1.0)
    assert np.all(edges[:, 0] < edges[:, 1]) and edges.max() < N_NEURONS
