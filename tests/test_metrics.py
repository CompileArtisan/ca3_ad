import numpy as np
from ca3model import HopfieldNetwork, RetrievalResult, evaluate_retrieval, hamming, overlap, random_patterns


def fake(state):
    return RetrievalResult(state.astype(np.int8), 2, 5, True, np.array([0.0, -1.0]), None)


def test_accuracy_hamming_overlap_identity():
    rng = np.random.default_rng(0)
    t = random_patterns(1, 100, rng)[0]
    s = t.copy()
    s[:10] = -s[:10]
    alive = np.ones(100, bool)
    m = evaluate_retrieval(fake(s), t, alive, t[None, :], 0.95)
    assert m["hamming_all"] == 10 and np.isclose(m["accuracy_all"], 0.9)
    assert np.isclose(m["accuracy_all"], (1 + overlap(s, t)) / 2)
    assert np.isclose(m["accuracy_surv"], (1 + overlap(s, t, alive)) / 2)
    assert not m["success"]


def test_all_leq_surv_with_dead_and_equal_without():
    t = random_patterns(1, 100, np.random.default_rng(1))[0]
    alive = np.ones(100, bool)
    alive[:20] = False
    s = np.where(alive, t, 0)
    m = evaluate_retrieval(fake(s), t, alive, t[None, :], 0.95)
    assert m["accuracy_surv"] == 1.0 and np.isclose(m["accuracy_all"], 0.8)
    assert m["accuracy_all"] <= m["accuracy_surv"] and m["success"] and not m["success_all"]
    m2 = evaluate_retrieval(fake(t), t, np.ones(100, bool), t[None, :], 0.95)
    assert m2["accuracy_all"] == m2["accuracy_surv"]


def test_attractor_classes():
    p = random_patterns(4, 200, np.random.default_rng(2))
    alive = np.ones(200, bool)
    ev = lambda s, t: evaluate_retrieval(fake(s), t, alive, p, 0.95)["attractor_type"]
    assert ev(p[0], p[0]) == "target"
    assert ev(-p[0], p[0]) == "mirror"
    assert ev(p[1], p[0]) == "other_stored"
    mixed = np.sign(p[0] + p[1] + p[2]).astype(np.int8)
    mixed[mixed == 0] = 1
    assert ev(mixed, p[0]) == "spurious"


def test_hamming_mask():
    s, t = np.array([1, -1, 0, 0]), np.array([1, 1, 1, 1])
    assert hamming(s, t) == 3 and hamming(s, t, np.array([1, 1, 0, 0], bool)) == 1
