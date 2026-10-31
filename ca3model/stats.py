"""Statistics (spec 4.9, 6.7). Unit of replication = trial."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def mean_ci(x, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided t-interval: mean +- t_{(1+c)/2, n-1} * sd/sqrt(n).

    NaNs are dropped. n == 0 -> (nan, nan, nan); n == 1 -> (x, nan, nan).
    Zero variance gives lo == hi == mean.
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(a.mean())
    if n == 1:
        return m, float("nan"), float("nan")
    half = float(sps.t.ppf((1 + confidence) / 2, n - 1) * a.std(ddof=1) / np.sqrt(n))
    return m, m - half, m + half


def summarise(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Average over test patterns within (group_cols + trial), THEN mean/CI across trials.

    Output columns: group_cols, n_trials, and for each v: ``v_mean, v_lo, v_hi``.
    """
    d = df.copy()
    for v in value_cols:
        d[v] = d[v].astype(float)
    keys = list(group_cols)
    per_trial = d.groupby(keys + ["trial"], dropna=False, sort=False)[value_cols].mean().reset_index()
    rows = []
    grouped = per_trial.groupby(keys, dropna=False, sort=False) if keys else [((), per_trial)]
    for key, g in grouped:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row["n_trials"] = int(g["trial"].nunique())
        for v in value_cols:
            m, lo, hi = mean_ci(g[v].to_numpy())
            row[f"{v}_mean"], row[f"{v}_lo"], row[f"{v}_hi"] = m, lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    if keys and len(out):
        out = out.sort_values(keys, kind="stable").reset_index(drop=True)
    return out


def paired_difference(df: pd.DataFrame, cond_a: str, cond_b: str, value_col: str,
                      condition_col: str = "condition") -> dict:
    """Paired difference a - b over trials (patterns averaged within trial first).

    Returns mean_diff, 95% t-CI (lo, hi), Wilcoxon signed-rank p (exploratory;
    p = 1.0 if all differences are exactly zero) and n_trials. The caller should
    filter ``df`` so each condition appears once per trial.
    """
    d = df[df[condition_col].isin([cond_a, cond_b])].copy()
    d[value_col] = d[value_col].astype(float)
    pt = d.groupby([condition_col, "trial"])[value_col].mean().unstack(0)
    if cond_a not in pt or cond_b not in pt:
        return {"mean_diff": np.nan, "lo": np.nan, "hi": np.nan, "wilcoxon_p": np.nan, "n_trials": 0}
    diff = (pt[cond_a] - pt[cond_b]).dropna().to_numpy()
    m, lo, hi = mean_ci(diff)
    if diff.size == 0:
        p = float("nan")
    elif np.all(diff == 0):
        p = 1.0
    else:
        try:
            p = float(sps.wilcoxon(diff).pvalue)
        except ValueError:
            p = float("nan")
    return {"mean_diff": m, "lo": lo, "hi": hi, "wilcoxon_p": p, "n_trials": int(diff.size)}


def paired_sweep_table(df: pd.DataFrame, treat: str, base: str, sweep_col: str,
                       value_col: str) -> pd.DataFrame:
    """Paired difference (treat - base) at each level of ``sweep_col`` (exploratory).

    ``base`` rows are paired with ``treat`` rows of the same trial at every level.
    """
    rows = []
    base_rows = df[df["condition"] == base]
    for lvl, g in df[df["condition"] == treat].groupby(sweep_col, sort=True):
        r = paired_difference(pd.concat([g, base_rows]), treat, base, value_col)
        r[sweep_col] = lvl
        rows.append(r)
    out = pd.DataFrame(rows)
    return out[[sweep_col] + [c for c in out.columns if c != sweep_col]] if len(out) else out
