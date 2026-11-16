import numpy as np
import pytest

from ca3model.damage import draw_damage_randomness
from ca3model.neurolab import (build_cue, connectivity_efficacy, neuron_layout, recall_reliability,
                               run_stimulator)
from ca3model.rng import trial_rngs


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
def test_connectivity_damage_is_nested(mode):
    U = draw_damage_randomness(120, trial_rngs(4, 0).damage).U
    levels = [connectivity_efficacy(U, c, mode) for c in (0.05, 0.2, 0.45, 0.5, 0.7, 1.0)]
    for lo, hi in zip(levels, levels[1:]):
        assert np.all(lo <= hi + 1e-12)


def test_cue_counts():
    pat = np.ones(256, dtype=np.int8)
    cue = build_cue(pat, 0.30, 0.10, seed=1)
    shown = int(round(0.30 * 256))
    assert np.count_nonzero(cue) == shown
    assert np.sum(cue == -1) == int(round(0.10 * shown))


def _replay(run):
    """Re-run the async dynamics from the recorded per-sweep update positions."""
    W, s = run.weights, run.states[0].astype(float).copy()
    tol = 1e-9 * np.abs(W).max()
    n_alive = int(run.net.alive.sum())
    for k in range(len(run.states) - 1):
        pos = np.where(np.isnan(run.update_pos[k]), np.inf, run.update_pos[k])
        for i in np.argsort(pos, kind="stable")[:n_alive]:
            h = W[i] @ s
            s[i] = 1 if h > tol else -1 if h < -tol else s[i]
        assert np.array_equal(s.astype(np.int8), run.states[k + 1])


@pytest.mark.parametrize("conn,loss", [(0.1, 0.0), (0.3, 0.2), (1.0, 0.0)])
def test_animation_order_reproduces_the_simulation(conn, loss):
    r = run_stimulator(144, 6, conn, 0.1, 0.3, neuron_loss=loss, base_seed=7)
    _replay(r["healthy"])
    _replay(r["damaged"])


def test_healthy_brain_unaffected_by_controls_meant_for_the_damaged_one():
    a = run_stimulator(144, 6, 0.1, 0.2, 0.3, base_seed=5)
    b = run_stimulator(144, 6, 0.9, 0.2, 0.3, donepezil=True, memantine=True, base_seed=5)
    assert np.array_equal(a["healthy"].states[-1], b["healthy"].states[-1])


def test_treatments_help_on_average():
    base = recall_reliability(256, 8, 0.10, 0.10, 0.30, n_cues=24)
    done = recall_reliability(256, 8, 0.10, 0.10, 0.30, donepezil=True, n_cues=24)
    noisy = recall_reliability(256, 8, 0.12, 0.30, 0.30, n_cues=24)
    mem = recall_reliability(256, 8, 0.12, 0.30, 0.30, memantine=True, n_cues=24)
    assert base["healthy"] == base["n"]
    assert done["damaged"] > base["damaged"]
    assert mem["damaged"] >= noisy["damaged"]


def test_layout_edges_valid():
    xy, edges = neuron_layout(256)
    assert xy.shape == (256, 2) and np.all(np.hypot(*xy.T) <= 1.0)
    assert np.all(edges[:, 0] < edges[:, 1]) and edges.max() < 256
