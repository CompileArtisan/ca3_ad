import numpy as np
import pandas as pd
from scipy import stats as sps
from ca3model import mean_ci, paired_difference, summarise


def test_ci_known_data():
    x = np.array([1.0, 2, 3, 4, 5])
    m, lo, hi = mean_ci(x)
    h = sps.t.ppf(0.975, 4) * x.std(ddof=1) / np.sqrt(5)
    assert m == 3 and np.isclose(lo, 3 - h) and np.isclose(hi, 3 + h)


def test_small_n():
    assert np.isnan(mean_ci([])[0])
    m, lo, hi = mean_ci([2.0])
    assert m == 2 and np.isnan(lo) and np.isnan(hi)
    assert mean_ci([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)


def test_summarise_averages_patterns_within_trial_first():
    df = pd.DataFrame({"condition": "a", "trial": [0, 0, 0, 1], "pattern_idx": [0, 1, 2, 0],
                       "v": [1.0, 1.0, 1.0, 0.0]})
    out = summarise(df, ["condition"], ["v"])
    assert np.isclose(out.loc[0, "v_mean"], 0.5)  # NOT 0.75 (pooled over patterns)
    assert out.loc[0, "n_trials"] == 2


def test_paired_difference():
    rows = [{"condition": c, "trial": t, "v": v + (1 if c == "a" else 0) + 0.1 * t}
            for t in range(6) for c, v in (("a", 0.0), ("b", 0.0))]
    r = paired_difference(pd.DataFrame(rows), "a", "b", "v")
    assert np.isclose(r["mean_diff"], 1.0) and r["n_trials"] == 6 and r["wilcoxon_p"] < 0.05
    same = pd.DataFrame([{"condition": c, "trial": t, "v": 1.0} for t in range(4) for c in "ab"])
    assert paired_difference(same, "a", "b", "v")["wilcoxon_p"] == 1.0
