import numpy as np
import pytest

from ca3model.damage import draw_damage_randomness
from ca3model.neurolab import (N_NEURONS, build_cue, connectivity_efficacy, make_brain,
                               memory_pattern, neuron_layout, recall_reliability, run_stimulator,
                               severity_damage)
from ca3model.rng import trial_rngs

SEEDS = range(1, 9)


def _mean_recall(sev, noise, strength, donepezil=False, n_cues=20):
    return np.mean([recall_reliability(sev, noise, strength, donepezil, base_seed=s,
                                       n_cues=n_cues)["recalled"] / n_cues for s in SEEDS])


def test_memory_is_an_x_on_a_5x5_grid():
    img = memory_pattern().reshape(5, 5) > 0
    assert np.array_equal(img, np.eye(5, dtype=bool) | np.fliplr(np.eye(5, dtype=bool)))


@pytest.mark.parametrize("mode", ["deletion", "attenuation"])
@pytest.mark.parametrize("c", [0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
def test_connectivity_is_the_mean_efficacy_kept(mode, c):
    n = 300
    U = draw_damage_randomness(n, trial_rngs(3, 0).damage).U
    E = connectivity_efficacy(U, c, mode)
    off = ~np.eye(n, dtype=bool)
    assert E[off].mean() == pytest.approx(c, abs=0.01)
    assert np.all(np.diag(E) == 0) and np.allclose(E, E.T)


@pytest.mark.parametrize("mode", ["deletion", "attenuation"])
def test_severity_damages_the_same_brain_further(mode):
    brains = [make_brain(s, mode=mode) for s in (0.0, 0.2, 0.5, 0.8, 1.0)]
    for lo, hi in zip(brains[1:], brains):
        assert np.all(lo.efficacy <= hi.efficacy + 1e-12)
        assert np.all(lo.alive <= hi.alive)
    assert brains[0].alive.all() and brains[0].synaptic_strength() == pytest.approx(1.0)


def test_cue_counts():
    x = memory_pattern()
    cue = build_cue(x, 0.40, 0.10, seed=1)
    assert np.sum((cue > 0) & (x > 0)) == 4          # 40% of the X's 9 neurons
    assert np.sum((cue > 0) & (x < 0)) == 2          # 10% of the other 16 fire by mistake


def test_healthy_brain_completes_the_x_from_a_40pct_cue():
    for s in SEEDS:
        assert recall_reliability(0.0, 0.10, 0.40, base_seed=s)["recalled"] == 40


def test_alzheimers_brain_is_bad_at_a_40pct_cue():
    """The reported bug: at a 40% cue the damaged brain still recalled 100%."""
    assert _mean_recall(0.5, 0.10, 0.40) < 0.75
    assert _mean_recall(0.8, 0.10, 0.40) < 0.35


def test_recall_gets_worse_with_severity_and_better_with_cue_strength():
    sev = [_mean_recall(s, 0.10, 0.40) for s in (0.0, 0.3, 0.6, 0.9)]
    assert all(a >= b for a, b in zip(sev, sev[1:])) and sev[0] - sev[-1] > 0.7
    cue = [_mean_recall(0.5, 0.10, k) for k in (0.2, 0.4, 0.6, 1.0)]
    assert all(a <= b for a, b in zip(cue, cue[1:])) and cue[-1] - cue[0] > 0.5


def test_treatments_help():
    assert _mean_recall(0.7, 0.10, 0.40, donepezil=True) > _mean_recall(0.7, 0.10, 0.40)
    # memantine = lower input noise
    assert _mean_recall(0.3, 0.15, 0.40) > _mean_recall(0.3, 0.30, 0.40)


def _replay(run):
    """Re-run the threshold dynamics from the recorded per-sweep update positions."""
    W, theta, alive = run.weights, run.brain.theta, run.brain.alive
    V = ((run.states[0] > 0) & alive).astype(float)
    for k in range(len(run.states) - 1):
        pos = np.where(np.isnan(run.update_pos[k]), np.inf, run.update_pos[k])
        for i in np.argsort(pos, kind="stable")[: int(alive.sum())]:
            h = W[i] @ V
            V[i] = 1 if h > theta + 1e-12 else 0 if h < theta - 1e-12 else V[i]
        disp = np.where(alive, np.where(V > 0, 1, -1), 0)
        assert np.array_equal(disp, run.states[k + 1])


@pytest.mark.parametrize("sev,noise,strength", [(0.0, 0.1, 0.4), (0.5, 0.2, 0.6), (0.9, 0.0, 1.0)])
def test_animation_order_reproduces_the_simulation(sev, noise, strength):
    _replay(run_stimulator(sev, noise, strength, base_seed=7)["brain"])


def test_severity_zero_is_healthy():
    conn, loss = severity_damage(0.0)
    assert conn == 1.0 and loss == 0.0
    assert run_stimulator(0.0, 0.1, 0.4)["brain"].metrics["success"]


def test_layout_edges_valid():
    xy, edges = neuron_layout(N_NEURONS)
    assert xy.shape == (N_NEURONS, 2) and np.all(np.hypot(*xy.T) <= 1.0)
    assert np.all(edges[:, 0] < edges[:, 1]) and edges.max() < N_NEURONS


def test_connectivity_override_sets_mean_synaptic_efficacy():
    from ca3model.neurolab import connectivity_for_severity, severity_for_connectivity
    off = ~np.eye(N_NEURONS, dtype=bool)
    for c in (1.0, 0.8, 0.4):
        assert make_brain(0.5, connectivity=c).efficacy[off].mean() == pytest.approx(c, abs=0.08)
    assert make_brain(0.3).efficacy[off].mean() == pytest.approx(severity_damage(0.3)[0], abs=0.08)
    for sev in range(0, 101, 5):
        assert severity_for_connectivity(connectivity_for_severity(sev)) == pytest.approx(sev, abs=1)


# ---- drug scenarios at the simulator defaults (57% severity, 40% cue, 40% input noise) ----
DEFAULTS = dict(severity=0.57, strength=0.40)


def _mc(noise, donepezil, seed=12345):
    from ca3model.neurolab import monte_carlo_recall
    return 100 * monte_carlo_recall(DEFAULTS["severity"], noise, DEFAULTS["strength"], donepezil,
                                    base_seed=seed)["recall"]


@pytest.mark.parametrize("seed", [12345, 1, 777])
def test_drug_scenarios_hit_the_monte_carlo_targets(seed):
    base, mem = _mc(0.40, False, seed), _mc(0.20, False, seed)
    done, both = _mc(0.40, True, seed), _mc(0.20, True, seed)
    assert 63.5 <= base <= 66.0          # ~64%: the damaged brain mostly goes silent
    assert 68.0 <= mem <= 71.0           # memantine: noise 40% -> 20% lifts recall modestly
    assert 95.0 <= done <= 97.0          # donepezil: ~96%
    assert 97.3 <= both <= 98.5          # combination: ~98%
    assert mem - base >= 3.0             # memantine has its own effect on recall
    assert 1.0 <= both - done <= 2.5     # combination beats donepezil alone by ~1-2 pts


def test_monte_carlo_trial_one_is_the_animated_run():
    from ca3model.neurolab import monte_carlo_recall
    run = run_stimulator(0.57, 0.4, 0.4, True)["brain"].metrics
    one = monte_carlo_recall(0.57, 0.4, 0.4, True, n_trials=1)
    assert one["recall"] == pytest.approx(run["accuracy"])


@pytest.mark.parametrize("noise,donepezil", [(0.40, False), (0.20, False), (0.40, True), (0.20, True)])
def test_animated_replay_is_a_typical_trial(noise, donepezil):
    """The replay must not show a lucky trial: memantine's replay showed 96% next to a 69% mean."""
    from ca3model.neurolab import monte_carlo_recall
    mc = monte_carlo_recall(0.57, noise, 0.40, donepezil)
    run = run_stimulator(0.57, noise, 0.40, donepezil, base_seed=12345 + mc["rep"])["brain"].metrics
    assert run["accuracy"] == pytest.approx(mc["rep_accuracy"])
    assert abs(run["accuracy"] - mc["recall"]) <= 0.06


def test_memantine_replay_matches_its_modest_effect():
    from ca3model.neurolab import monte_carlo_recall
    mc = monte_carlo_recall(0.57, 0.20, 0.40, False)
    assert mc["rep_accuracy"] < 0.75          # a typical memantine trial is not a full recall
