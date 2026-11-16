#!/usr/bin/env python3

"""Neurolab core: one cue shown to a healthy and a damaged network side by side.

Illustrative abstraction. Drug toggles are NOT pharmacological models:
  donepezil -> recovers a fraction of lost synaptic efficacy on surviving neurons
  memantine -> reduces the effective input noise reaching the network

``connectivity`` c is the mean synaptic efficacy kept in the damaged brain, exactly, in
both damage modes:
  deletion     E_ij = 1 if U_ij >= 1-c else 0                (mean c)
  attenuation  E_ij = 1 - 2(1-c) U_ij           for c >= 1/2  (mean c)
               E_ij = max(0, 1 - U_ij / (2c))   for c <  1/2  (mean c)
(The plain DamageSpec attenuation caps at s = 1, i.e. mean 1/2, so it cannot reach c < 1/2.)
E is monotone in c for a fixed U, so damage stays nested across connectivity levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import DamageSpec
from .damage import DamageDraw, apply_damage, draw_damage_randomness
from .metrics import evaluate_retrieval
from .network import HopfieldNetwork
from .patterns import random_patterns
from .retrieval import make_orders, retrieve
from .rng import trial_rngs

DONEPEZIL_RECOVERY = 0.5          # fraction of lost efficacy restored
MEMANTINE_NOISE_REDUCTION = 0.5   # fraction of input noise removed
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
EDGE_NEIGHBOURS = 4


def build_cue(pattern: np.ndarray, cue_given: float, noise: float, seed: int) -> np.ndarray:
    """Partial cue: round(cue_given*N) bits shown (rest = 0, unknown); then round(noise*n_shown)
    of the shown bits are flipped. Same seed -> same shown set and nested flips, so cues with
    different noise levels are paired."""
    n = pattern.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    shown = order[: int(round(cue_given * n))]
    cue = np.zeros(n, dtype=np.int8)
    cue[shown] = pattern[shown]
    k = int(round(noise * shown.size))
    cue[shown[:k]] = -cue[shown[:k]]
    return cue


def connectivity_efficacy(U: np.ndarray, connectivity: float, mode: str) -> np.ndarray:
    """Per-synapse efficacy whose mean over U ~ Uniform(0,1) is exactly ``connectivity``."""
    c = float(np.clip(connectivity, 0.0, 1.0))
    if mode == "deletion":
        E = np.where(U >= 1.0 - c, 1.0, 0.0)
    elif c >= 0.5:
        E = 1.0 - 2.0 * (1.0 - c) * U
    elif c > 0.0:
        E = np.maximum(0.0, 1.0 - U / (2.0 * c))
    else:
        E = np.zeros_like(U)
    E = np.asarray(E, dtype=np.float64)
    np.fill_diagonal(E, 0.0)
    return E


def damage_network(healthy: HopfieldNetwork, draw: DamageDraw, connectivity: float, mode: str,
                   neuron_loss: float) -> HopfieldNetwork:
    """Damaged copy: synaptic efficacy from :func:`connectivity_efficacy`, plus neuron loss."""
    sick = apply_damage(healthy, DamageSpec(0.0, mode, float(neuron_loss)), draw)
    sick.efficacy = sick.efficacy * connectivity_efficacy(draw.U, connectivity, mode)
    return sick


def restore_efficacy(net: HopfieldNetwork, recovery: float) -> HopfieldNetwork:
    """Donepezil analogue: E <- E + recovery*(1-E) (copy). Alive mask untouched."""
    out = net.copy()
    out.efficacy = out.efficacy + recovery * (1.0 - out.efficacy)
    np.fill_diagonal(out.efficacy, 0.0)
    return out


@lru_cache(maxsize=8)
def neuron_layout(n: int, k: int = EDGE_NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    """Sunflower (Vogel) layout in the unit disc and its k-nearest-neighbour edge list.
    Purely visual: the model itself is fully connected."""
    i = np.arange(n)
    r = np.sqrt((i + 0.5) / n)
    th = i * GOLDEN_ANGLE
    xy = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    edges = {(min(a, int(b)), max(a, int(b))) for a in range(n) for b in nn[a]}
    return xy, np.array(sorted(edges), dtype=np.int64)


@dataclass
class BrainRun:
    net: HopfieldNetwork
    cue: np.ndarray
    states: list          # state at start (cue) and after each sweep
    weights: np.ndarray   # effective weights
    metrics: dict
    update_pos: np.ndarray  # (sweeps, N): when each neuron is visited within each sweep, in [0,1)


def _run(net, cue, target, patterns, max_sweeps, thr, rngs) -> BrainRun:
    W = net.effective_weights()
    res = retrieve(net, cue, max_sweeps, rngs.fresh_dynamics(), record_states=True, W_eff=W)
    m = evaluate_retrieval(res, target, net.alive, patterns, thr)
    # Same generator stream as retrieve() -> the exact asynchronous update order it used.
    orders = make_orders(net.alive, max_sweeps, rngs.fresh_dynamics())[: res.sweeps]
    pos = np.full((res.sweeps, net.n), np.nan)
    if orders.shape[1]:
        ranks = np.arange(orders.shape[1]) / orders.shape[1]
        for k in range(res.sweeps):
            pos[k, orders[k]] = ranks
    return BrainRun(net, np.where(net.alive, cue, 0).astype(np.int8), res.state_history, W, m, pos)


def _build(n, n_patterns, connectivity, donepezil, neuron_loss, mode, base_seed):
    rngs = trial_rngs(base_seed, 0)
    pats = random_patterns(n_patterns, n, rngs.patterns)
    healthy = HopfieldNetwork.from_patterns(pats)
    sick = damage_network(healthy, draw_damage_randomness(n, rngs.damage), connectivity, mode,
                          neuron_loss)
    if donepezil:
        sick = restore_efficacy(sick, DONEPEZIL_RECOVERY)
    return rngs, pats, healthy, sick


def _noise_for_damaged(noise: float, memantine: bool) -> float:
    return noise * (1 - MEMANTINE_NOISE_REDUCTION) if memantine else noise


def run_stimulator(n: int, n_patterns: int, connectivity: float, noise: float, cue_given: float,
                   donepezil: bool = False, memantine: bool = False, neuron_loss: float = 0.0,
                   mode: str = "deletion", max_sweeps: int = 30, threshold: float = 0.95,
                   base_seed: int = 12345, pattern_idx: int = 0) -> dict:
    """Healthy vs damaged retrieval of one stored memory from the same (paired) cue."""
    rngs, pats, healthy, sick = _build(n, n_patterns, connectivity, donepezil, neuron_loss, mode,
                                       base_seed)
    target = pats[pattern_idx % n_patterns]
    cue_seed = base_seed * 7919 + pattern_idx
    cue_h = build_cue(target, cue_given, noise, cue_seed)
    cue_s = build_cue(target, cue_given, _noise_for_damaged(noise, memantine), cue_seed)
    return {"target": target, "patterns": pats,
            "healthy": _run(healthy, cue_h, target, pats, max_sweeps, threshold, rngs),
            "damaged": _run(sick, cue_s, target, pats, max_sweeps, threshold, rngs)}


def recall_reliability(n: int, n_patterns: int, connectivity: float, noise: float, cue_given: float,
                       donepezil: bool = False, memantine: bool = False, neuron_loss: float = 0.0,
                       mode: str = "deletion", max_sweeps: int = 30, threshold: float = 0.95,
                       base_seed: int = 12345, n_cues: int = 40) -> dict:
    """Success counts over ``n_cues`` fresh cues (cycling through every stored memory), same two
    networks as :func:`run_stimulator`. Shows whether the single animated run is typical."""
    _, pats, healthy, sick = _build(n, n_patterns, connectivity, donepezil, neuron_loss, mode,
                                    base_seed)
    Wh, Ws = healthy.effective_weights(), sick.effective_weights()
    noise_s = _noise_for_damaged(noise, memantine)
    ok = {"healthy": 0, "damaged": 0}
    for k in range(n_cues):
        idx = k % n_patterns
        seed = base_seed * 7919 + 100_003 * (k + 1) + idx
        target = pats[idx]
        for name, net, W, nz in (("healthy", healthy, Wh, noise), ("damaged", sick, Ws, noise_s)):
            res = retrieve(net, build_cue(target, cue_given, nz, seed), max_sweeps,
                           np.random.default_rng([base_seed & 0xFFFFFFFF, k]), W_eff=W)
            ok[name] += evaluate_retrieval(res, target, net.alive, pats, threshold)["success"]
    return {**ok, "n": n_cues}
