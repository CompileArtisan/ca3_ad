"""RECALL4 NEUROLAB core: the CA3 Hopfield reference model, with the values and settings of the
two Monte Carlo reports (S. Janardhan, 18 Nov 2026):

  * "CA3 Hopfield Monte Carlo Experiment: Connectivity Loss, Noise and Simulated Pharmacology"
    (Experiment 1: A connectivity loss, B simulated pharmacology, C connectivity x noise)
  * "Experiment 2: Monte Carlo Validation of Donepezil- and Memantine-Inspired Interventions"

Model (identical in both reports)
---------------------------------
  network      25 neurons, fully recurrent, bipolar states +-1 (0 = silent, not in the cue)
  memories     3 random +-1 patterns, ``default_rng(20261118).choice([-1, 1], (3, 25))``;
               the target is memory #1. One-memory sensitivity: same seed, P = 1.
  weights      W = (1/N) sum_mu xi^mu xi^mu^T, zero diagonal (Hebbian outer product)
  connectivity c keeps exactly round(c * 300) of the 300 neuron pairs. Pairs are ranked once per
               trial and the top ones kept, so lower c removes a superset of pairs. A removed pair
               deletes both W_ij and W_ji; surviving weights keep full strength (no scaling).
  cue          the same 5 neurons (20%), ``sorted(default_rng(20261119).choice(25, 5))`` =
               2, 4, 6, 7, 11, receive their stored value; the other 20 are silent (0).
               Cue strength above 20% reveals round(s * 25) neurons: the 5, then uncued neurons in the
               trial's "extra cue neurons" order (Exp 2's comparison run). Below 20% the first
               round(s * 25) of the 5 are used (not in the reports).
               Random-fill sensitivity: the 20 uncued neurons start at random +-1 instead.
  input noise  each neuron is corrupted with probability p (corrupted if u_i < p, one u_i per
               neuron per trial, so lower p corrupts a subset): a cue neuron has its sign flipped
               (amplitude kept), a silent one becomes spuriously active with a random sign.
  dynamics     asynchronous, a fresh random order each sweep (``permutation(25)`` per sweep);
               h_i = W[i] @ s, s_i = sign(h_i), h_i = 0 keeps the state. Converged = one full sweep
               with no change (that sweep is counted); at most 100 sweeps.
               h is compared with exact 0 in floating point, with no tolerance. With 3 memories,
               many fields are a tie in exact arithmetic, so float rounding decides them. This is
               what reproduces the reports' numbers (51.6% intact baseline, 63.9% at 4% noise);
               a tolerance (true ties keep the state) gives about 55% and 69% instead.
  accuracy     overlap m = (1/N) sum xi_i s_i and % neurons correct. Success = converged and
               m >= 0.8. Outcomes: correct (m >= 0.8), mirror (m <= -0.8), different memory
               (overlap >= 0.8 with another stored memory), mixture (any other converged state),
               not converged (a failure).
  trials       trial t uses seed 8,000,000 + t; ``SeedSequence(seed).spawn(5)`` gives five
               independent streams: mask, noise, spurious signs, extra cue neurons, update order.
               Every condition uses the same trials, so all comparisons are paired.

Interventions (computational stand-ins, not pharmacology)
---------------------------------------------------------
  Experiment 2 (used by the simulator by default, per its "Note for the app"):
    donepezil-inspired  effective cue strength s = 20 (baseline), 25, 30, 35%: the same 5 cue
                        neurons start at amplitude A = s / 0.20 (1.00, 1.25, 1.50, 1.75), x_i = A xi_i.
                        No extra neurons are revealed and the weights are untouched.
    memantine-inspired  input noise p = 10 (baseline), 8, 6, 4%.
    strengths           low = cue 25% / noise 8%, moderate = 30% / 6%, high = 35% / 4%.
    comparison only     "reveal more neurons": round(s * 25) = 5, 6, 8, 9 neurons at amplitude 1.
  Experiment 1 (Experiment B):
    donepezil-inspired  the noisy cue keeps driving retrieval, h_i += g * cue_i, g = 0.1
    memantine-inspired  activity gate: a neuron fires only when |h_i| > theta, else falls silent,
                        theta = 0.04 (one Hebbian unit, 1/N)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

# ----- network and memories
GRID = (5, 5)
N_NEURONS = GRID[0] * GRID[1]
N_PAIRS = N_NEURONS * (N_NEURONS - 1) // 2          # 300
N_STORED = 3
PATTERN_SEED = 20261118
CUE_SEED = 20261119
CUE_SIZE = 5                                         # 20% of 25

# ----- retrieval
MAX_SWEEPS = 100
SUCCESS_OVERLAP = 0.8

# ----- trials (Experiment 2 design)
TRIAL_SEED_BASE = 8_000_000
N_TRIALS = 2000
N_STREAMS = 5                                        # mask, noise, signs, extra cue, order

# ----- baseline and levels from the reports
BASELINE_CUE = 0.20
BASELINE_NOISE = 0.10
CONNECTIVITY_LEVELS = (100, 90, 80, 70, 60, 50, 40, 30, 20)   # %; Exp A 100-30, Exp C down to 20
NOISE_LEVELS = (0, 4, 6, 8, 10, 20, 30, 40, 50)               # %; Exp C 0-50, Exp 2 10/8/6/4
CUE_STRENGTHS = (0.20, 0.25, 0.30, 0.35)                      # Exp 2 effective cue strength
REDUCED_NOISE = (0.10, 0.08, 0.06, 0.04)                      # Exp 2 memantine-inspired noise
DRUG_STRENGTHS = {"Low": (0.25, 0.08), "Moderate": (0.30, 0.06), "High": (0.35, 0.04)}
EXP1_CUE_GAIN = 0.1                                           # g, Experiment B main dose
EXP1_ACTIVITY_THRESHOLD = 0.04                                # theta, Experiment B main dose
EXP1_GAIN_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.4)                 # Experiment B dose-response
EXP1_THRESHOLD_LEVELS = (0.0, 0.02, 0.04, 0.08, 0.12)

# ----- simulator severity dial: severity 0-100% maps linearly onto the reports' connectivity range
MIN_CONNECTIVITY = 20                                         # %, lowest level in Experiment C
DEFAULT_SEVERITY = 75                                         # -> 40% connectivity (Experiment B)
CUE_STRENGTH_RANGE = (5, 100, 5)                              # % of the 25 neurons; 20% = the 5 cue

OUTCOMES = ("correct", "different memory", "mixture", "mirror", "not converged")
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
EDGE_NEIGHBOURS = 4
_TOL = 0.0                                           # exact comparison, see docstring


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5 + 1e-9))


@lru_cache(maxsize=4)
def _patterns(n_stored: int) -> np.ndarray:
    p = np.random.default_rng(PATTERN_SEED).choice([-1, 1], size=(n_stored, N_NEURONS))
    p = p.astype(np.int8)
    p.setflags(write=False)
    return p


def stored_patterns(n_stored: int = N_STORED) -> np.ndarray:
    """(P, 25) int8 +-1. Row 0 is the target (memory #1)."""
    return _patterns(int(n_stored)).copy()


def memory_pattern() -> np.ndarray:
    """The target memory #1, int8 +-1, row-major over the 5x5 grid."""
    return stored_patterns(1)[0]


def cue_neurons() -> np.ndarray:
    """The 5 cued neurons (2, 4, 6, 7, 11)."""
    return np.sort(np.random.default_rng(CUE_SEED).choice(N_NEURONS, size=CUE_SIZE, replace=False))


def hebbian_weights(patterns: np.ndarray) -> np.ndarray:
    """W = (1/N) sum_mu xi xi^T, zero diagonal."""
    X = np.asarray(patterns, dtype=np.float64)
    W = X.T @ X / X.shape[1]
    np.fill_diagonal(W, 0.0)
    return W


def pairs_kept(connectivity: float) -> int:
    """Exactly round(c * 300) neuron pairs survive."""
    return _half_up(float(np.clip(connectivity, 0.0, 1.0)) * N_PAIRS)


def connectivity_for_severity(severity_pct: float) -> int:
    """Severity % -> functional connectivity % (0% -> 100, 100% -> 20), rounded half up."""
    sev = float(np.clip(severity_pct, 0, 100))
    return _half_up(100 - (100 - MIN_CONNECTIVITY) * sev / 100)


def severity_for_connectivity(connectivity_pct: float) -> int:
    """Functional connectivity % -> the severity % that implies it."""
    c = float(np.clip(connectivity_pct, MIN_CONNECTIVITY, 100))
    return _half_up((100 - c) * 100 / (100 - MIN_CONNECTIVITY))


def trial_seed(trial: int) -> int:
    return TRIAL_SEED_BASE + int(trial)


@dataclass(frozen=True)
class TrialDraws:
    """Per-trial random numbers for ``n`` consecutive trials (shared by every condition)."""

    pair_rank: np.ndarray    # (T, 300) pair indices, most-kept first
    u: np.ndarray            # (T, 25) noise uniforms: neuron i corrupted if u_i < p
    signs: np.ndarray        # (T, 25) +-1 for spurious activity (and random fill)
    extra: np.ndarray        # (T, 20) uncued neurons in reveal order (comparison run)
    order_seeds: tuple       # (T,) SeedSequence of the update-order stream


@lru_cache(maxsize=4)
def trial_draws(first_trial: int = 0, n_trials: int = N_TRIALS) -> TrialDraws:
    cue = cue_neurons()
    uncued = np.setdiff1d(np.arange(N_NEURONS), cue)
    T = int(n_trials)
    rank = np.empty((T, N_PAIRS), dtype=np.int16)
    u = np.empty((T, N_NEURONS))
    signs = np.empty((T, N_NEURONS), dtype=np.int8)
    extra = np.empty((T, uncued.size), dtype=np.int8)
    order_seeds = []
    for k in range(T):
        mask_s, noise_s, sign_s, extra_s, order_s = (
            np.random.default_rng(c) if j < 4 else c
            for j, c in enumerate(np.random.SeedSequence(trial_seed(first_trial + k)).spawn(N_STREAMS)))
        rank[k] = np.argsort(mask_s.random(N_PAIRS), kind="stable")
        u[k] = noise_s.random(N_NEURONS)
        signs[k] = sign_s.choice([-1, 1], size=N_NEURONS)
        extra[k] = extra_s.permutation(uncued)
        order_seeds.append(order_s)
    return TrialDraws(rank, u, signs, extra, tuple(order_seeds))


@dataclass(frozen=True)
class Condition:
    """One simulated condition. Defaults = the Experiment 2 baseline at full connectivity."""

    connectivity: float = 1.0
    noise: float = BASELINE_NOISE
    cue_fraction: float = BASELINE_CUE   # share of the 25 neurons cued: round(f*25) (20% = 5)
    cue_amplitude: float = 1.0           # Exp 2 donepezil-inspired: A = s / 0.20 on cued neurons
    cue_gain: float = 0.0                # Exp 1 donepezil-inspired: h_i += g * cue_i
    activity_threshold: float = 0.0      # Exp 1 memantine-inspired gate (0 = off)
    n_stored: int = N_STORED
    random_fill: bool = False            # Exp A sensitivity: uncued neurons start at random +-1

    @property
    def amplitude(self) -> float:
        return self.cue_amplitude

    @property
    def effective_cue(self) -> float:
        """Exp 2 'effective cue strength': cued share x amplitude (20% x 1.75 = 35%)."""
        return self.cue_fraction * self.cue_amplitude

    @property
    def n_cued(self) -> int:
        return _half_up(self.cue_fraction * N_NEURONS)


def _pair_index() -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(N_NEURONS, k=1)


def connectivity_masks(connectivity: float, draws: TrialDraws) -> np.ndarray:
    """(T, 25, 25) symmetric 0/1 masks, zero diagonal; nested across connectivity levels."""
    T = draws.pair_rank.shape[0]
    keep = np.zeros((T, N_PAIRS), dtype=bool)
    k = pairs_kept(connectivity)
    np.put_along_axis(keep, draws.pair_rank[:, :k].astype(np.int64), True, axis=1)
    iu, ju = _pair_index()
    M = np.zeros((T, N_NEURONS, N_NEURONS))
    M[:, iu, ju] = keep
    M[:, ju, iu] = keep
    return M


def initial_input(cond: Condition, draws: TrialDraws) -> np.ndarray:
    """(T, 25) noisy cue x (before any update)."""
    xi = memory_pattern().astype(np.float64)
    T = draws.u.shape[0]
    x = np.zeros((T, N_NEURONS))
    cue = cue_neurons()
    k = cond.n_cued
    x[:, cue[:k]] = cond.amplitude * xi[cue[:k]]
    if k > CUE_SIZE:                                      # reveal more neurons (Exp 2 comparison)
        rows = np.arange(T)[:, None]
        idx = draws.extra[:, :k - CUE_SIZE].astype(np.int64)
        x[rows, idx] = cond.amplitude * xi[idx]
    if cond.random_fill:
        silent = x == 0
        x[silent] = draws.signs[silent]
    hit = draws.u < cond.noise
    active = x != 0
    x = np.where(hit & active, -x, x)                     # flip sign, amplitude kept
    x = np.where(hit & ~active, draws.signs, x)           # spurious activity, random sign
    return x


def settle(W: np.ndarray, x0: np.ndarray, inp: np.ndarray, theta: float,
           order_seed: np.random.SeedSequence, record: bool = False):
    """Asynchronous dynamics for one trial.

    W (25,25), x0 (25,) initial input, inp (25,) persistent afferent input (Exp 1 cue gain),
    theta > 0 turns on the Exp 1 activity gate. The update order is drawn sweep by sweep from
    ``order_seed``. Returns (final, sweeps, converged, history, orders); history holds the state at
    the start and after each sweep when ``record``.
    """
    rng = np.random.default_rng(order_seed)
    s = np.array(x0, dtype=np.float64)
    hist = [s.copy()] if record else None
    orders = []
    for sw in range(MAX_SWEEPS):
        order = rng.permutation(N_NEURONS)
        orders.append(order)
        changed = False
        for i in order:
            h = W[i] @ s + inp[i]
            new = 1.0 if h > _TOL else -1.0 if h < -_TOL else s[i]
            if theta > 0 and not abs(h) > theta:
                new = 0.0
            if new != s[i]:
                s[i] = new
                changed = True
        if record:
            hist.append(s.copy())
        if not changed:
            return s, sw + 1, True, hist, orders
    return s, MAX_SWEEPS, False, hist, orders


def classify(final: np.ndarray, converged: np.ndarray, patterns: np.ndarray) -> dict:
    """Overlaps, % correct and outcome class for (T,25) final states."""
    S = np.sign(final)
    P = np.asarray(patterns, dtype=np.float64)
    ov = S @ P.T / N_NEURONS                               # (T, P)
    m = ov[:, 0]
    other = ov[:, 1:].max(axis=1) if P.shape[0] > 1 else np.full(m.shape, -np.inf)
    outcome = np.where(~converged, "not converged",
              np.where(m >= SUCCESS_OVERLAP - _TOL, "correct",
              np.where(m <= -SUCCESS_OVERLAP + _TOL, "mirror",
              np.where(other >= SUCCESS_OVERLAP - _TOL, "different memory", "mixture"))))
    return {"m": m, "overlaps": ov, "pct_correct": (S == P[0]).mean(axis=1) * 100,
            "success": converged & (m >= SUCCESS_OVERLAP - _TOL), "outcome": outcome}


def _prepare(cond: Condition, draws: TrialDraws):
    patterns = stored_patterns(cond.n_stored)
    W = hebbian_weights(patterns)[None] * connectivity_masks(cond.connectivity, draws)
    x0 = initial_input(cond, draws)
    inp = cond.cue_gain * x0 if cond.cue_gain else np.zeros_like(x0)
    return patterns, W, x0, inp


def run_trials(cond: Condition, n_trials: int = N_TRIALS, first_trial: int = 0) -> dict:
    """Monte Carlo over trials first_trial .. first_trial + n_trials - 1 (seeds 8,000,000 + t)."""
    draws = trial_draws(int(first_trial), int(n_trials))
    patterns, W, x0, inp = _prepare(cond, draws)
    T = x0.shape[0]
    final = np.empty_like(x0)
    sweeps = np.empty(T, dtype=np.int64)
    conv = np.empty(T, dtype=bool)
    for t in range(T):
        final[t], sweeps[t], conv[t], _, _ = settle(W[t], x0[t], inp[t], cond.activity_threshold,
                                                    draws.order_seeds[t])
    out = classify(final, conv, patterns)
    out.update(final=final, sweeps=sweeps, converged=conv, x0=x0)
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarise(res: dict) -> dict:
    n = int(res["success"].size)
    k = int(res["success"].sum())
    lo, hi = wilson(k, n)
    return {"n": n, "successes": k, "success": k / n, "ci": (lo, hi),
            "mean_overlap": float(res["m"].mean()),
            "sd_overlap": float(res["m"].std(ddof=1)) if n > 1 else 0.0,
            "pct_correct": float(res["pct_correct"].mean()),
            "mean_sweeps": float(res["sweeps"].mean()),
            "outcomes": {o: int((res["outcome"] == o).sum()) for o in OUTCOMES}}


def apply_drugs(cond: Condition, donepezil: bool, memantine: bool, model: str = "exp2",
                strength: str = "High", gain: float = EXP1_CUE_GAIN,
                threshold: float = EXP1_ACTIVITY_THRESHOLD) -> Condition:
    """Simulator drug toggles.

    ``model="exp2"``: donepezil sets the effective cue strength (25/30/35%); memantine lowers input
    noise by 2/4/6 points (10% -> 8/6/4% at the baseline), for low/moderate/high strength.
    ``model="exp1"``: donepezil adds the persistent cue input g; memantine adds the gate theta.
    """
    if model == "exp1":
        return replace(cond, cue_gain=gain if donepezil else 0.0,
                       activity_threshold=threshold if memantine else 0.0)
    cue, noise = DRUG_STRENGTHS[strength]
    out = cond
    if donepezil:
        out = replace(out, cue_amplitude=cue / BASELINE_CUE)
    if memantine:
        out = replace(out, noise=max(0.0, round(cond.noise - (BASELINE_NOISE - noise), 6)))
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
class TrialRun:
    """One trial, recorded sweep by sweep for the animation."""

    cond: Condition
    trial: int
    patterns: np.ndarray
    mask: np.ndarray          # (25, 25) 0/1
    weights: np.ndarray       # effective weights W * mask
    x0: np.ndarray            # initial input (amplitudes kept)
    states: list              # sign of the state at the start and after each sweep (int8)
    update_pos: np.ndarray    # (sweeps, 25): when each neuron is visited within each sweep, [0,1)
    orders: list              # update order of each sweep
    metrics: dict


def run_trial(cond: Condition, trial: int = 0) -> TrialRun:
    """One trial (seed 8,000,000 + trial), with every sweep recorded."""
    draws = trial_draws(int(trial), 1)
    patterns, W, x0, inp = _prepare(cond, draws)
    final, n_sw, conv, hist, orders = settle(W[0], x0[0], inp[0], cond.activity_threshold,
                                             draws.order_seeds[0], record=True)
    c = classify(final[None], np.array([conv]), patterns)
    ranks = np.arange(N_NEURONS) / N_NEURONS
    pos = np.full((n_sw, N_NEURONS), np.nan)
    for k in range(n_sw):
        pos[k, orders[k]] = ranks
    metrics = {"m": float(c["m"][0]), "overlaps": c["overlaps"][0].tolist(),
               "pct_correct": float(c["pct_correct"][0]), "success": bool(c["success"][0]),
               "outcome": str(c["outcome"][0]), "converged": bool(conv), "sweeps": int(n_sw),
               "pairs_kept": pairs_kept(cond.connectivity),
               "n_corrupted": int((draws.u[0] < cond.noise).sum())}
    states = [np.sign(h).astype(np.int8) for h in hist]
    mask = connectivity_masks(cond.connectivity, draws)[0]
    return TrialRun(cond, int(trial), patterns, mask, W[0], x0[0], states, pos, orders, metrics)
