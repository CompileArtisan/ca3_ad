import numpy as np
from ca3model import HopfieldNetwork, random_patterns, retrieve, corrupt_cue


def pats(P=5, N=100, seed=0):
    return random_patterns(P, N, np.random.default_rng(seed))


def test_symmetric_zero_diag_and_formula():
    p = pats()
    net = HopfieldNetwork.from_patterns(p)
    X = p.astype(float)
    ref = X.T @ X / 100
    np.fill_diagonal(ref, 0)
    assert np.allclose(net.W_raw, ref) and np.array_equal(net.W_raw, net.W_raw.T)
    assert np.all(np.diag(net.W_raw) == 0)


def test_incremental_equals_batch():
    p = pats()
    inc = HopfieldNetwork(100)
    for row in p:
        inc.store(row[None, :])
    assert np.allclose(inc.W_raw, HopfieldNetwork.from_patterns(p).W_raw)


def test_plasticity_zero_leaves_W_unchanged():
    net = HopfieldNetwork.from_patterns(pats())
    before = net.W_raw.copy()
    net.store(pats(2, seed=9), plasticity_efficacy=0.0)
    assert np.array_equal(net.W_raw, before)


def test_store_does_not_touch_damage_state():
    net = HopfieldNetwork.from_patterns(pats())
    net.alive[:3] = False
    net.efficacy[0, 1] = net.efficacy[1, 0] = 0.0
    e, a = net.efficacy.copy(), net.alive.copy()
    net.store(pats(2, seed=3))
    assert np.array_equal(net.efficacy, e) and np.array_equal(net.alive, a)
    assert np.all(net.effective_weights()[:3] == 0) and np.all(net.effective_weights()[:, :3] == 0)


def test_low_load_fixed_points_and_retrieval():
    rng = np.random.default_rng(0)
    p = random_patterns(3, 256, rng)
    net = HopfieldNetwork.from_patterns(p)
    for pat in p:
        r = retrieve(net, pat, 30, rng)
        assert np.array_equal(r.final_state, pat) and r.sweeps == 1 and r.converged
        r2 = retrieve(net, corrupt_cue(pat, 0.1, "flip", rng), 30, rng)
        assert np.array_equal(r2.final_state, pat)


def test_connectivity_stats_healthy():
    s = HopfieldNetwork.from_patterns(pats()).connectivity_stats()
    assert abs(s["frac_synapses_lost"]) < 1e-12 and s["n_alive"] == 100 and s["mean_degree"] == 99
