import numpy as np
from ca3model import DamageSpec, HopfieldNetwork, apply_damage, build_damage, draw_damage_randomness, random_patterns
from ca3model.damage import n_lost_neurons

N = 200


def draw(seed=0):
    return draw_damage_randomness(N, np.random.default_rng(seed))


def test_deletion_fraction_matches_severity():
    d = draw()
    for s in (0.1, 0.5, 0.8):
        E, _ = build_damage(d, DamageSpec(s, "deletion"))
        off = ~np.eye(N, dtype=bool)
        frac = 1 - E[off].mean()
        sd = np.sqrt(s * (1 - s) / (N * (N - 1) / 2))
        assert abs(frac - s) < 5 * sd


def test_efficacy_symmetric_zero_diag_both_modes():
    for mode in ("deletion", "attenuation"):
        E, _ = build_damage(draw(), DamageSpec(0.4, mode))
        assert np.array_equal(E, E.T) and np.all(np.diag(E) == 0)
    E, _ = build_damage(draw(), DamageSpec(0.4, "attenuation"))
    assert E[~np.eye(N, dtype=bool)].mean() > 0.75 and E.max() <= 1


def test_nested_synaptic_and_neuron():
    d = draw()
    for mode in ("deletion", "attenuation"):
        E1, _ = build_damage(d, DamageSpec(0.2, mode))
        E2, _ = build_damage(d, DamageSpec(0.6, mode))
        assert np.all(E2 <= E1 + 1e-15)
    _, a1 = build_damage(d, DamageSpec(neuron_loss=0.1))
    _, a2 = build_damage(d, DamageSpec(neuron_loss=0.4))
    assert np.all(~a1 <= ~a2)  # dead(f1) subset of dead(f2)


def test_neuron_loss_count_exact():
    d = draw()
    for f in (0.0, 0.1, 0.333, 0.9):
        _, a = build_damage(d, DamageSpec(neuron_loss=f))
        assert (~a).sum() == round(f * N) == n_lost_neurons(N, f)


def test_combined_order_independent_and_severity_zero_identity():
    d = draw()
    net = HopfieldNetwork.from_patterns(random_patterns(5, N, np.random.default_rng(1)))
    syn, neu = DamageSpec(0.3), DamageSpec(0.0, neuron_loss=0.2)
    both = apply_damage(net, DamageSpec(0.3, neuron_loss=0.2), d)
    ab = apply_damage(apply_damage(net, syn, d), neu, d)
    ba = apply_damage(apply_damage(net, neu, d), syn, d)
    assert np.array_equal(ab.effective_weights(), ba.effective_weights())
    assert np.array_equal(ab.effective_weights(), both.effective_weights())
    assert np.array_equal(apply_damage(net, DamageSpec(), d).effective_weights(),
                          net.effective_weights())


def test_apply_damage_returns_copy():
    net = HopfieldNetwork.from_patterns(random_patterns(3, N, np.random.default_rng(1)))
    apply_damage(net, DamageSpec(0.5, neuron_loss=0.5), draw())
    assert net.alive.all() and net.efficacy.sum() == N * (N - 1)
