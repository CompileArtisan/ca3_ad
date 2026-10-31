"""Trial runner and experiment suites A-F (spec 6.8).

Design: unit of replication = trial. Within a trial all conditions share patterns,
cues, update orders and damage randomness (paired). Cue/dynamics generators are
re-created from the same child seed for every condition.

Condition keys use ``"name@level"``; the text before ``@`` becomes the ``condition``
column, the damage columns carry the numeric levels.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import DamageSpec, ExperimentConfig, NetworkConfig, RetrievalConfig
from .damage import DamageDraw, apply_damage, draw_damage_randomness
from .metrics import evaluate_retrieval
from .network import HopfieldNetwork
from .patterns import corrupt_cue, random_patterns
from .retrieval import retrieve
from .rng import TrialRNGs, trial_rngs
from .stats import mean_ci

Progress = Callable[[float], None] | None

TIDY_COLUMNS = [
    "experiment", "condition", "trial", "seed", "pattern_idx", "pattern_group",
    "n_neurons", "n_patterns", "cue_noise", "synaptic_mode", "synaptic_severity",
    "neuron_loss", "plasticity_efficacy", "accuracy_all", "accuracy_surv",
    "hamming_all", "hamming_surv", "sweeps", "n_flips", "converged", "success",
    "success_all", "attractor_type", "final_energy",
]
_DROP = {"overlap_surv"}

# A test pattern is (pattern_idx, pattern_group, pattern vector)
TestPattern = tuple[int, str, np.ndarray]


def _evaluate_net(net: HopfieldNetwork, tests: list[TestPattern], ret_cfg: RetrievalConfig,
                  rngs: TrialRNGs, label: str, spec: DamageSpec,
                  extra: dict | None = None) -> list[dict]:
    """Retrieve every test pattern from a corrupted cue on ``net`` (already damaged)."""
    W = net.effective_weights()
    stored = (np.stack(net.patterns) if net.patterns
              else np.empty((0, net.n), dtype=np.int8))
    cue_rng, dyn_rng = rngs.fresh_cues(), rngs.fresh_dynamics()
    rows = []
    for idx, group, pat in tests:
        cue = corrupt_cue(pat, ret_cfg.cue_noise, ret_cfg.cue_mode, cue_rng)
        res = retrieve(net, cue, ret_cfg.max_sweeps, dyn_rng, W_eff=W)
        m = evaluate_retrieval(res, pat, net.alive, stored, ret_cfg.success_threshold)
        row = {
            "condition": label.split("@")[0], "trial": rngs.trial, "seed": rngs.base_seed,
            "pattern_idx": idx, "pattern_group": group, "n_neurons": net.n,
            "cue_noise": ret_cfg.cue_noise, "synaptic_mode": spec.synaptic_mode,
            "synaptic_severity": spec.synaptic_severity, "neuron_loss": spec.neuron_loss,
        }
        row.update({k: v for k, v in m.items() if k not in _DROP})
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def run_condition_trial(net_healthy: HopfieldNetwork, damage_spec: DamageSpec, draw: DamageDraw,
                        patterns_to_test: list[TestPattern], ret_cfg: RetrievalConfig,
                        rngs: TrialRNGs, label: str, extra: dict | None = None,
                        reported_spec: DamageSpec | None = None) -> list[dict]:
    """Damage a copy of ``net_healthy`` with ``damage_spec`` and test the patterns.

    ``reported_spec`` overrides the damage columns when the network was already
    damaged before (damage-then-learn). ``extra`` is merged into every row.
    """
    net = apply_damage(net_healthy, damage_spec, draw)
    return _evaluate_net(net, patterns_to_test, ret_cfg, rngs, label,
                         reported_spec or damage_spec, extra)


def run_trial_conditions(net_cfg: NetworkConfig, exp_cfg: ExperimentConfig, ret_cfg: RetrievalConfig,
                         conditions: dict[str, DamageSpec], trial: int,
                         experiment: str = "", extra: dict | None = None) -> list[dict]:
    """One trial: store ``n_patterns`` random patterns, then test them under each condition."""
    rngs = trial_rngs(exp_cfg.base_seed, trial)
    n = net_cfg.n_neurons
    pats = random_patterns(exp_cfg.n_patterns, n, rngs.patterns)
    net = HopfieldNetwork.from_patterns(pats)
    draw = draw_damage_randomness(n, rngs.damage)
    tests = [(i, "all", pats[i]) for i in range(len(pats))]
    base = {"experiment": experiment, "n_patterns": exp_cfg.n_patterns, "plasticity_efficacy": 1.0}
    base.update(extra or {})
    rows: list[dict] = []
    for key, spec in conditions.items():
        rows += run_condition_trial(net, spec, draw, tests, ret_cfg, rngs, key, base)
    return rows


def _finish(rows: list[dict], suite: str, meta: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    ordered = [c for c in TIDY_COLUMNS if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in TIDY_COLUMNS]]
    df.attrs["meta"] = {"suite": suite, **meta}
    return df


def _meta(net_cfg, exp_cfg, ret_cfg, **params) -> dict:
    return {"network": net_cfg.to_dict(), "experiment": exp_cfg.to_dict(),
            "retrieval": ret_cfg.to_dict(), "params": params}


def _run(suite: str, net_cfg, exp_cfg, ret_cfg, make_conditions, progress: Progress, meta: dict):
    rows: list[dict] = []
    for t in range(exp_cfg.n_trials):
        rows += run_trial_conditions(net_cfg, exp_cfg, ret_cfg, make_conditions(), t, suite)
        if progress:
            progress((t + 1) / exp_cfg.n_trials)
    return _finish(rows, suite, meta)


# ----------------------------------------------------------------- A, B, C, D
def exp_healthy_vs_weakened(net_cfg, exp_cfg, ret_cfg, severities, mode="deletion",
                            progress: Progress = None) -> pd.DataFrame:
    """A: sweep synaptic severity s (neuron_loss = 0); 'healthy' baseline included."""
    def conds():
        c = {"healthy@0": DamageSpec()}
        c.update({f"synaptic@{s}": DamageSpec(float(s), mode, 0.0) for s in severities})
        return c
    return _run("A", net_cfg, exp_cfg, ret_cfg, conds, progress,
                _meta(net_cfg, exp_cfg, ret_cfg, severities=list(map(float, severities)), mode=mode))


def exp_healthy_vs_neuron_loss(net_cfg, exp_cfg, ret_cfg, fractions, mode="deletion",
                               progress: Progress = None) -> pd.DataFrame:
    """B: sweep neuron-loss fraction f (synaptic severity = 0)."""
    def conds():
        c = {"healthy@0": DamageSpec()}
        c.update({f"neuron@{f}": DamageSpec(0.0, mode, float(f)) for f in fractions})
        return c
    return _run("B", net_cfg, exp_cfg, ret_cfg, conds, progress,
                _meta(net_cfg, exp_cfg, ret_cfg, fractions=list(map(float, fractions))))


def matched_synapse_loss(f: float) -> float:
    """Neuron loss f silences 1-(1-f)^2 of all synapses (both endpoints must survive)."""
    return 1.0 - (1.0 - f) ** 2


def exp_weakening_vs_loss(net_cfg, exp_cfg, ret_cfg, levels, mode="deletion",
                          progress: Progress = None) -> pd.DataFrame:
    """C: weakening vs neuron loss on one nominal axis + 'matched_synaptic'
    (weakening at s = 1-(1-f)^2). Adds ``matched_synapse_loss_frac`` = expected
    fraction of total synaptic efficacy removed (neuron: 1-(1-f)^2; synaptic:
    s for deletion, s/2 for attenuation; matched_synaptic: as synaptic)."""
    def conds():
        c = {"healthy@0": DamageSpec()}
        for l in levels:
            l = float(l)
            c[f"synaptic@{l}"] = DamageSpec(l, mode, 0.0)
            c[f"neuron@{l}"] = DamageSpec(0.0, mode, l)
            c[f"matched_synaptic@{l}"] = DamageSpec(matched_synapse_loss(l), mode, 0.0)
        return c
    df = _run("C", net_cfg, exp_cfg, ret_cfg, conds, progress,
              _meta(net_cfg, exp_cfg, ret_cfg, levels=list(map(float, levels)), mode=mode))
    k = 1.0 if mode == "deletion" else 0.5
    df["matched_synapse_loss_frac"] = np.where(
        df["condition"] == "neuron", 1.0 - (1.0 - df["neuron_loss"]) ** 2,
        k * df["synaptic_severity"])
    return df


def exp_combined(net_cfg, exp_cfg, ret_cfg, syn_levels, neuron_levels, mode="deletion",
                 progress: Progress = None) -> pd.DataFrame:
    """D: full grid. Conditions healthy / synaptic_only / neuron_only / combined.
    Use :func:`interaction_table` on the result for the additivity analysis."""
    def conds():
        c = {"healthy@0": DamageSpec()}
        for s in syn_levels:
            c[f"synaptic_only@{s}"] = DamageSpec(float(s), mode, 0.0)
        for f in neuron_levels:
            c[f"neuron_only@{f}"] = DamageSpec(0.0, mode, float(f))
        for s in syn_levels:
            for f in neuron_levels:
                c[f"combined@{s}|{f}"] = DamageSpec(float(s), mode, float(f))
        return c
    return _run("D", net_cfg, exp_cfg, ret_cfg, conds, progress,
                _meta(net_cfg, exp_cfg, ret_cfg, syn_levels=list(map(float, syn_levels)),
                      neuron_levels=list(map(float, neuron_levels)), mode=mode))


def interaction_table(df: pd.DataFrame, value_col: str = "success") -> pd.DataFrame:
    """Additivity check for suite D, per trial then summarised across trials.

    deficit_X = healthy_value - X_value (paired within trial);
    deficit_additive = deficit_syn + deficit_neu;
    interaction = deficit_combined - deficit_additive
    (> 0: worse than additive / synergistic; < 0: better than additive).
    Deficits are not clipped, so deficit_additive can exceed the healthy value.
    """
    pt = (df.assign(**{value_col: df[value_col].astype(float)})
          .groupby(["condition", "synaptic_severity", "neuron_loss", "trial"])[value_col]
          .mean().reset_index())
    healthy = pt[pt.condition == "healthy"].set_index("trial")[value_col]
    syn = {(r.synaptic_severity, r.trial): healthy[r.trial] - getattr(r, value_col)
           for r in pt[pt.condition == "synaptic_only"].itertuples()}
    neu = {(r.neuron_loss, r.trial): healthy[r.trial] - getattr(r, value_col)
           for r in pt[pt.condition == "neuron_only"].itertuples()}
    per_trial = []
    for r in pt[pt.condition == "combined"].itertuples():
        ds = syn.get((r.synaptic_severity, r.trial), 0.0 if r.synaptic_severity == 0 else np.nan)
        dn = neu.get((r.neuron_loss, r.trial), 0.0 if r.neuron_loss == 0 else np.nan)
        dc = healthy[r.trial] - getattr(r, value_col)
        per_trial.append((r.synaptic_severity, r.neuron_loss, ds, dn, ds + dn, dc, dc - (ds + dn)))
    t = pd.DataFrame(per_trial, columns=["synaptic_severity", "neuron_loss", "deficit_syn",
                                         "deficit_neu", "deficit_additive", "deficit_combined",
                                         "interaction"])
    rows = []
    for (s, f), g in t.groupby(["synaptic_severity", "neuron_loss"], sort=True):
        m, lo, hi = mean_ci(g["interaction"].to_numpy())
        rows.append({"synaptic_severity": s, "neuron_loss": f,
                     **{c: float(g[c].mean()) for c in
                        ["deficit_syn", "deficit_neu", "deficit_additive", "deficit_combined"]},
                     "interaction": m, "interaction_lo": lo, "interaction_hi": hi,
                     "n_trials": len(g)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- E
def _kind_spec(kind: str, level: float, mode: str) -> DamageSpec:
    level = float(level)
    if kind == "synaptic":
        return DamageSpec(level, mode, 0.0)
    if kind == "neuron":
        return DamageSpec(0.0, mode, level)
    if kind == "combined":
        return DamageSpec(level, mode, level)
    raise ValueError("kind must be 'synaptic', 'neuron' or 'combined'")


def exp_new_learning(net_cfg, exp_cfg, ret_cfg, severities, kind="synaptic", mode="deletion",
                     progress: Progress = None) -> pd.DataFrame:
    """E: learning of new patterns before/after damage.

    healthy_learn      : healthy net stores new patterns; test new+old.
    damage_then_learn  : damage, then new patterns learned with ``plasticity_efficacy``.
    learn_then_damage  : new patterns learned (eta=1), then damage (control).
    Note: in this abstraction W_eff = W_raw*E*alive, so with eta=1 the two damage
    orders give identical W_eff; they differ only when plasticity_efficacy < 1.
    """
    n = net_cfg.n_neurons
    P, K, eta = exp_cfg.n_patterns, exp_cfg.n_new_patterns, exp_cfg.plasticity_efficacy
    rows: list[dict] = []
    for t in range(exp_cfg.n_trials):
        rngs = trial_rngs(exp_cfg.base_seed, t)
        old = random_patterns(P, n, rngs.patterns)
        new = random_patterns(K, n, rngs.new_patterns)
        draw = draw_damage_randomness(n, rngs.damage)
        tests = ([(i, "old", old[i]) for i in range(P)] +
                 [(P + k, "new", new[k]) for k in range(K)])
        net_old = HopfieldNetwork.from_patterns(old)
        net_hl = net_old.copy()
        net_hl.store(new, 1.0)
        ex = lambda lvl, e: {"experiment": "E", "n_patterns": P, "plasticity_efficacy": e,
                             "level": float(lvl), "damage_kind": kind}
        rows += run_condition_trial(net_hl, DamageSpec(), draw, tests, ret_cfg, rngs,
                                    "healthy_learn@0", ex(0, 1.0))
        for lvl in severities:
            spec = _kind_spec(kind, lvl, mode)
            net_dl = apply_damage(net_old, spec, draw)
            net_dl.store(new, eta)
            rows += run_condition_trial(net_dl, DamageSpec(), draw, tests, ret_cfg, rngs,
                                        f"damage_then_learn@{lvl}", ex(lvl, eta), reported_spec=spec)
            rows += run_condition_trial(net_hl, spec, draw, tests, ret_cfg, rngs,
                                        f"learn_then_damage@{lvl}", ex(lvl, 1.0))
        if progress:
            progress((t + 1) / exp_cfg.n_trials)
    return _finish(rows, "E", _meta(net_cfg, exp_cfg, ret_cfg,
                                    severities=list(map(float, severities)), kind=kind, mode=mode))


def sequential_learning_curve(net_cfg, exp_cfg, ret_cfg, spec: DamageSpec,
                              progress: Progress = None) -> pd.DataFrame:
    """E (sequential): add new patterns one at a time to a healthy vs a damaged net.

    After learning pattern k its retrieval is recorded (stage 'immediate', step k);
    after all K are learned every new pattern is tested again (stage 'final',
    step K), which exposes interference. Damaged net learns with plasticity_efficacy.
    """
    n = net_cfg.n_neurons
    P, K, eta = exp_cfg.n_patterns, exp_cfg.n_new_patterns, exp_cfg.plasticity_efficacy
    rows: list[dict] = []
    for t in range(exp_cfg.n_trials):
        rngs = trial_rngs(exp_cfg.base_seed, t)
        old = random_patterns(P, n, rngs.patterns)
        new = random_patterns(K, n, rngs.new_patterns)
        draw = draw_damage_randomness(n, rngs.damage)
        base_net = HopfieldNetwork.from_patterns(old)
        nets = {"healthy": (base_net.copy(), 1.0, DamageSpec()),
                "damaged": (apply_damage(base_net, spec, draw), eta, spec)}
        for name, (net, e, rep) in nets.items():
            ex = lambda stage, step: {"experiment": "E_seq", "n_patterns": P, "plasticity_efficacy": e,
                                      "stage": stage, "step": step}
            for k in range(K):
                net.store(new[k:k + 1], e)
                rows += _evaluate_net(net, [(P + k, "new", new[k])], ret_cfg, rngs, name, rep,
                                      ex("immediate", k + 1))
            rows += _evaluate_net(net, [(P + k, "new", new[k]) for k in range(K)], ret_cfg, rngs,
                                  name, rep, ex("final", K))
        if progress:
            progress((t + 1) / exp_cfg.n_trials)
    return _finish(rows, "E_seq", _meta(net_cfg, exp_cfg, ret_cfg, spec=spec.to_dict()))


# ----------------------------------------------------------------------- F
def default_load_values(n: int, steps: int = 12) -> list[int]:
    """Integers from 1 to ~0.25 N in ``steps`` roughly even steps."""
    return sorted({int(round(v)) for v in np.linspace(1, 0.25 * n, steps)})


def default_load_conditions(syn: float = 0.3, neu: float = 0.2, mode: str = "deletion") -> dict:
    return {"healthy": DamageSpec(), "synaptic": DamageSpec(syn, mode, 0.0),
            "neuron": DamageSpec(0.0, mode, neu), "combined": DamageSpec(syn, mode, neu)}


def exp_memory_load(net_cfg, exp_cfg, ret_cfg, load_values=None, conditions=None,
                    progress: Progress = None, max_test_patterns: int = 10) -> pd.DataFrame:
    """F: success vs number of stored patterns P (alpha = P/N).

    Each trial draws max(P) patterns once; load P uses the first P (nested loads).
    At most ``max_test_patterns`` of the stored patterns are tested per load.
    """
    n = net_cfg.n_neurons
    loads = list(load_values) if load_values is not None else default_load_values(n)
    conditions = conditions or default_load_conditions()
    rows: list[dict] = []
    for t in range(exp_cfg.n_trials):
        rngs = trial_rngs(exp_cfg.base_seed, t)
        allp = random_patterns(max(loads), n, rngs.patterns)
        draw = draw_damage_randomness(n, rngs.damage)
        for P in loads:
            net = HopfieldNetwork.from_patterns(allp[:P])
            tests = [(i, "all", allp[i]) for i in range(min(P, max_test_patterns))]
            for name, spec in conditions.items():
                rows += run_condition_trial(net, spec, draw, tests, ret_cfg, rngs, name,
                                            {"experiment": "F", "n_patterns": P,
                                             "plasticity_efficacy": 1.0, "load_alpha": P / n})
        if progress:
            progress((t + 1) / exp_cfg.n_trials)
    return _finish(rows, "F", _meta(net_cfg, exp_cfg, ret_cfg, loads=loads,
                                    conditions={k: v.to_dict() for k, v in conditions.items()},
                                    max_test_patterns=max_test_patterns))
