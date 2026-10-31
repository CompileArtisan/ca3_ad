import numpy as np
import pytest
from ca3model import (DamageSpec, HopfieldNetwork, apply_damage, corrupt_cue,
                      draw_damage_randomness, random_patterns, retrieve)
from ca3model.retrieval import HAVE_NUMBA


def setup(P=8, N=144, seed=0):
    rng = np.random.default_rng(seed)
    p = random_patterns(P, N, rng)
    return p, HopfieldNetwork.from_patterns(p), rng


def test_uniform_scaling_invariance():
    p, net, rng = setup()
    cue = corrupt_cue(p[0], 0.25, "flip", rng)
    net5 = net.copy()
    net5.W_raw = net5.W_raw * 5
    r1 = retrieve(net, cue, 30, np.random.default_rng(5))
    r2 = retrieve(net5, cue, 30, np.random.default_rng(5))
    assert np.array_equal(r1.final_state, r2.final_state)
    assert (r1.sweeps, r1.n_flips) == (r2.sweeps, r2.n_flips)


def test_energy_non_increasing_healthy_and_damaged():
    p, net, rng = setup(P=12)
    d = draw_damage_randomness(net.n, np.random.default_rng(3))
    for spec in (DamageSpec(), DamageSpec(0.4, "deletion", 0.2), DamageSpec(0.6, "attenuation", 0.1)):
        dn = apply_damage(net, spec, d)
        for i in range(4):
            cue = corrupt_cue(p[i], 0.3, "flip", rng)
            tr = retrieve(dn, cue, 30, rng).energy_trace
            assert np.all(np.diff(tr) <= 1e-9)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_numba_matches_numpy_fallback():
    p, net, rng = setup(P=14)
    d = draw_damage_randomness(net.n, np.random.default_rng(3))
    for spec in (DamageSpec(), DamageSpec(0.5, "attenuation", 0.2)):
        dn = apply_damage(net, spec, d)
        for i in range(6):
            cue = corrupt_cue(p[i], 0.3, "flip", rng)
            a = retrieve(dn, cue, 30, np.random.default_rng(i), backend="numba")
            b = retrieve(dn, cue, 30, np.random.default_rng(i), backend="numpy")
            assert np.array_equal(a.final_state, b.final_state)
            assert (a.sweeps, a.n_flips, a.converged) == (b.sweeps, b.n_flips, b.converged)
            assert np.allclose(a.energy_trace, b.energy_trace)


def test_dead_neurons_stay_zero_and_never_flip():
    p, net, rng = setup()
    d = draw_damage_randomness(net.n, np.random.default_rng(2))
    dn = apply_damage(net, DamageSpec(neuron_loss=0.3), d)
    r = retrieve(dn, corrupt_cue(p[0], 0.2, "flip", rng), 30, rng, record_states=True)
    assert np.all(r.final_state[~dn.alive] == 0)
    assert all(np.all(s[~dn.alive] == 0) for s in r.state_history)


def test_convergence_reporting_and_rng_stream_independent_of_damage():
    p, net, rng = setup(P=3)
    r = retrieve(net, p[0], 30, rng)
    assert r.sweeps == 1 and r.n_flips == 0 and r.converged
    d = draw_damage_randomness(net.n, np.random.default_rng(2))
    g1, g2 = np.random.default_rng(9), np.random.default_rng(9)
    retrieve(net, p[0], 10, g1)
    retrieve(apply_damage(net, DamageSpec(neuron_loss=0.5), d), p[0], 10, g2)
    assert g1.random() == g2.random()


def test_all_neurons_dead_edge_case():
    p, net, rng = setup()
    net.alive[:] = False
    r = retrieve(net, p[0], 10, rng)
    assert np.all(r.final_state == 0) and r.converged
