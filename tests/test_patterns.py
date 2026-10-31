import numpy as np
from ca3model import corrupt_cue, random_patterns, smooth_patterns, to_image, trial_rngs


def test_random_patterns_dtype_values_shape():
    p = random_patterns(5, 100, np.random.default_rng(0))
    assert p.dtype == np.int8 and p.shape == (5, 100)
    assert set(np.unique(p)) <= {-1, 1}


def test_seed_determinism():
    a = random_patterns(4, 100, np.random.default_rng(1))
    b = random_patterns(4, 100, np.random.default_rng(1))
    c = random_patterns(4, 100, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_trial_rngs_reproducible_and_distinct():
    a, b, c = trial_rngs(7, 0), trial_rngs(7, 0), trial_rngs(7, 1)
    assert a.patterns.random() == b.patterns.random() != c.patterns.random()
    assert a.fresh_cues().random() == b.fresh_cues().random()


def test_flip_exact_count_and_occlude_bounds():
    rng = np.random.default_rng(0)
    p = random_patterns(1, 200, rng)[0]
    assert np.sum(corrupt_cue(p, 0.2, "flip", rng) != p) == 40
    assert np.sum(corrupt_cue(p, 0.2, "occlude", rng) != p) <= 40


def test_to_image_and_smooth():
    assert to_image(np.ones(256)).shape == (16, 16)
    img = to_image(np.ones(110))
    assert img.shape[0] * img.shape[1] >= 110 and np.isnan(img).sum() == img.size - 110
    s = smooth_patterns(3, 256, np.random.default_rng(0))
    assert s.dtype == np.int8 and set(np.unique(s)) <= {-1, 1}
