#!/usr/bin/env python3

"""RECALL4 NEUROLAB core: one brain, one stored memory (an X on a 5x5 grid), one severity knob.

Why this is not the classic Hopfield model used by the analysis tabs
--------------------------------------------------------------------
The analysis tabs use zero-threshold sign dynamics (s_i <- sign(h_i)). There, scaling the
weights down never changes recall, so synapse loss only hurts through the random *pattern* of
loss, and a lightly loaded network shrugs that off: at 256 neurons / 8 memories the old
simulator still recalled 40/40 cues with 70% of synaptic strength gone. That made a moderately
"Alzheimer's" brain look perfect.

Here neurons are firing (1) or silent (0) and fire only when their recurrent input beats a fixed
firing threshold. Synapse loss therefore weakens the drive a cue can deliver, so a damaged brain
needs a stronger cue to complete the same memory, and past a point no cue is enough. That is
the behaviour the simulator is meant to show.

Model (illustrative abstraction, not pharmacology):
  memory     xi in {0,1}^25 (an X), activity a = mean(xi)
  weights    W_ij = (xi_i - a)(xi_j - a) / N,  W_ii = 0   (covariance Hebbian rule)
  damage     W_eff = W * E * alive_i * alive_j
             E from :func:`connectivity_efficacy` with connectivity
             c = 1 - SYNAPSE_LOSS_AT_MAX * severity, plus NEURON_LOSS_AT_MAX * severity * N
             dead neurons, rounded up or down at random per brain (3-4 of 25 at 100% severity:
             synapse loss does most of the damage).
             The simulator uses attenuation (every synapse weakened by a random amount): with
             all-or-none deletion a single 25-neuron brain declines in lumpy cliffs.
  dynamics   asynchronous; V_i <- 1 if h_i > theta, 0 if h_i < theta, unchanged on a tie,
             theta = THRESHOLD_SYNAPSES * (weight between two memory neurons)
  cue        strength k: round(k * 9) of the X's 9 neurons fire; noise q: round(q * 16) of the
             16 other neurons fire as well (stray input). Everything else starts silent.
  recall     correct if the settled state matches the X on >= threshold of all 25 neurons
             (a lost neuron can neither fire nor be correctly silent, so it scores half)
  donepezil  recovers DONEPEZIL_RECOVERY of the lost synaptic efficacy on surviving neurons and
             multiplies cue strength by DONEPEZIL_CUE_GAIN (cholinergic boost to the cue signal)
  memantine  halves input noise (the UI moves the Input noise slider); fewer stray neurons fire in
             the cue, so a few more cues get through and recall rises modestly. It also spares
             MEMANTINE_NEURON_PROTECTION of the neurons that would die (less excitotoxic
             damage), so at the defaults a donepezil brain loses 2 neurons (a full recall
             scores 96%) and a donepezil + memantine brain loses 1 (98%)

Monte Carlo (:func:`monte_carlo_recall`): every trial draws a fresh brain at the chosen severity
(which synapses weaken, which neurons die), a fresh cue and a fresh update order, and reports
recall = mean share of the 25 neurons that match the X once the network settles. The animation
replays the most typical trial (the one whose recall is closest to the average), so a lucky or
unlucky single trial cannot misrepresent the scenario. With the defaults (57% severity, 40% cue, 40% input noise) it lands
near 64% / 96% / 69% / 98% for none / donepezil / memantine / both, and the replays show
64 / 96 / 66 / 98%. These are outputs of this toy
model with parameters chosen for teaching, not clinical efficacy figures.

Damage uses common random numbers: one draw per seed, so a higher severity damages a superset of
what a lower severity damages (the same brain getting worse).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .damage import draw_damage_randomness
from .retrieval import make_orders
from .rng import trial_rngs

MEMORY_ROWS = ("#...#",
               ".#.#.",
               "..#..",
               ".#.#.",
               "#...#")
GRID = (5, 5)
N_NEURONS = GRID[0] * GRID[1]

THRESHOLD_SYNAPSES = 1.75    # firing threshold, in units of one healthy memory-memory synapse
SYNAPSE_LOSS_AT_MAX = 0.9    # share of synaptic strength lost at 100% severity
NEURON_LOSS_AT_MAX = 0.14    # share of neurons lost at 100% severity (expected count)
DONEPEZIL_RECOVERY = 0.7     # fraction of lost efficacy restored
DONEPEZIL_CUE_GAIN = 2.5     # cue strength multiplier (capped at 100%)
MEMANTINE_NOISE_REDUCTION = 0.5  # fraction of input noise removed (applied by the UI)
MEMANTINE_NEURON_PROTECTION = 0.5  # fraction of neuron loss prevented (less excitotoxic death)
LOST_NEURON_CREDIT = 0.5     # recall credit for a lost neuron (neither right nor wrong)
SIMULATOR_MODE = "attenuation"
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
EDGE_NEIGHBOURS = 4
_TOL = 1e-12


def memory_pattern() -> np.ndarray:
    """The stored memory as int8 in {-1,+1} (+1 = firing), row-major over the 5x5 grid."""
    return np.array([1 if c == "#" else -1 for row in MEMORY_ROWS for c in row], dtype=np.int8)


def _half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def severity_damage(severity: float) -> tuple[float, float]:
    """Severity in [0,1] -> (synaptic connectivity kept, neuron-loss fraction)."""
    s = float(np.clip(severity, 0.0, 1.0))
    return 1.0 - SYNAPSE_LOSS_AT_MAX * s, NEURON_LOSS_AT_MAX * s


def build_cue(pattern: np.ndarray, strength: float, noise: float, seed: int) -> np.ndarray:
    """int8 cue: +1 on round(strength * n_on) memory neurons and on round(noise * n_off) stray
    neurons, 0 (not given) elsewhere. The same seed gives nested sets, so cues that differ only in
    strength or noise are paired."""
    rng = np.random.default_rng(seed)
    on, off = np.flatnonzero(pattern > 0), np.flatnonzero(pattern <= 0)
    on, off = rng.permutation(on), rng.permutation(off)
    cue = np.zeros(pattern.shape[0], dtype=np.int8)
    cue[on[: _half_up(np.clip(strength, 0, 1) * on.size)]] = 1
    cue[off[: _half_up(np.clip(noise, 0, 1) * off.size)]] = 1
    return cue


def cue_counts(strength: float, noise: float) -> tuple[int, int, int, int]:
    """(memory neurons lit, of n_on, stray neurons lit, of n_off) for the stored memory."""
    p = memory_pattern()
    n_on, n_off = int((p > 0).sum()), int((p <= 0).sum())
    return _half_up(strength * n_on), n_on, _half_up(noise * n_off), n_off


def connectivity_efficacy(U: np.ndarray, connectivity: float, mode: str) -> np.ndarray:
    """Per-synapse efficacy whose mean over U ~ Uniform(0,1) is exactly ``connectivity``.

    deletion     E = 1 if U >= 1-c else 0
    attenuation  E = 1 - 2(1-c) U          for c >= 1/2
                 E = max(0, 1 - U / (2c))  for c <  1/2
    E is monotone in c for fixed U, so damage is nested across levels."""
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


def covariance_weights(pattern: np.ndarray) -> np.ndarray:
    """W_ij = (xi_i - a)(xi_j - a)/N for xi in {0,1}, zero diagonal."""
    xi = (pattern > 0).astype(np.float64)
    d = xi - xi.mean()
    W = np.outer(d, d) / xi.size
    np.fill_diagonal(W, 0.0)
    return W


def connectivity_for_severity(severity_pct: int) -> int:
    """Slider link: severity % -> synaptic connectivity % (rounded half up)."""
    return _half_up(round(100.0 * severity_damage(severity_pct / 100.0)[0], 6))


def severity_for_connectivity(connectivity_pct: int) -> int:
    """Slider link: synaptic connectivity % -> the severity % that implies it."""
    return int(np.clip(_half_up((100.0 - connectivity_pct) / SYNAPSE_LOSS_AT_MAX), 0, 100))


def firing_threshold(pattern: np.ndarray) -> float:
    a = float((pattern > 0).mean())
    return THRESHOLD_SYNAPSES * (1.0 - a) ** 2 / pattern.size


def n_lost_neurons(loss: float, base_seed: int) -> int:
    """How many of the 25 neurons die: loss * N with stochastic rounding (floor(loss * N + u),
    u ~ U(0,1) fixed per brain seed). The expected count is exactly loss * N, so the Monte Carlo
    average is not pinned to one integer, and a fixed seed still loses a superset of neurons as
    severity rises."""
    u = np.random.default_rng([base_seed & 0xFFFFFFFF, 77]).random()
    return int(np.floor(loss * N_NEURONS + u))


@dataclass
class Brain:
    W: np.ndarray            # learned weights (healthy)
    efficacy: np.ndarray     # per-synapse efficacy after damage (and donepezil)
    alive: np.ndarray        # bool per neuron
    theta: float
    connectivity: float      # mean efficacy kept (before donepezil)
    neuron_loss: float

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def effective_weights(self) -> np.ndarray:
        a = self.alive.astype(np.float64)
        return self.W * self.efficacy * np.outer(a, a)

    def synaptic_strength(self) -> float:
        """Share of the healthy synaptic strength left among all neuron pairs."""
        off = ~np.eye(self.n, dtype=bool)
        return float((self.efficacy * np.outer(self.alive, self.alive))[off].mean())


def make_brain(severity: float, donepezil: bool = False, mode: str = SIMULATOR_MODE,
               base_seed: int = 12345, connectivity: float | None = None,
               memantine: bool = False) -> Brain:
    """``connectivity`` (mean synaptic efficacy kept) overrides the one implied by severity;
    severity still sets neuron loss. ``memantine`` spares MEMANTINE_NEURON_PROTECTION of the
    neurons that would die (a subset of the same brain's losses)."""
    pattern = memory_pattern()
    conn, loss = severity_damage(severity)
    if memantine:
        loss *= 1.0 - MEMANTINE_NEURON_PROTECTION
    if connectivity is not None:
        conn = float(np.clip(connectivity, 0.0, 1.0))
    draw = draw_damage_randomness(N_NEURONS, trial_rngs(base_seed, 0).damage)
    E = connectivity_efficacy(draw.U, conn, mode)
    if donepezil:
        E = E + DONEPEZIL_RECOVERY * (1.0 - E)
        np.fill_diagonal(E, 0.0)
    alive = np.ones(N_NEURONS, dtype=bool)
    alive[draw.neuron_order[: n_lost_neurons(loss, base_seed)]] = False
    return Brain(covariance_weights(pattern), E, alive, firing_threshold(pattern), conn, loss)


def settle(W: np.ndarray, cue: np.ndarray, alive: np.ndarray, theta: float,
           orders: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """Asynchronous threshold dynamics from ``cue``. Returns display states (+1 firing,
    -1 silent, 0 = not given yet / lost) at the start and after each sweep, and convergence."""
    V = ((cue > 0) & alive).astype(np.float64)
    h = W @ V
    disp = np.where(V > 0, 1, 0).astype(np.int8)
    states = [disp.copy()]
    for order in orders:
        flips = 0
        for i in order:
            new = 1.0 if h[i] > theta + _TOL else 0.0 if h[i] < theta - _TOL else V[i]
            if new != V[i]:
                h += W[:, i] * (new - V[i])
                V[i] = new
                flips += 1
            disp[i] = 1 if V[i] > 0 else -1
        states.append(disp.copy())
        if flips == 0:
            return states, True
    return states, False


def score(final: np.ndarray, alive: np.ndarray, target: np.ndarray, threshold: float,
          converged: bool) -> dict:
    shown = np.where(alive & (final > 0), 1, -1)
    on = target > 0
    # a lost neuron can neither fire for the memory nor stay correctly silent: half credit
    acc = float(np.mean(np.where(alive, (shown == target).astype(float), LOST_NEURON_CREDIT)))
    hit, stray = int((shown[on] > 0).sum()), int((shown[~on] > 0).sum())
    if not converged:
        kind = "unsettled"
    elif acc >= threshold:
        kind = "recalled"
    elif hit + stray == 0:
        kind = "silent"
    else:
        kind = "partial"
    return {"accuracy": acc, "success": bool(converged and acc >= threshold), "converged": converged,
            "kind": kind, "hit": hit, "n_on": int(on.sum()), "stray": stray}


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
    brain: Brain
    cue: np.ndarray
    states: list            # display state at start (cue) and after each sweep
    weights: np.ndarray     # effective weights
    metrics: dict
    update_pos: np.ndarray  # (sweeps, N): when each neuron is visited within each sweep, in [0,1)


def _orders(alive: np.ndarray, max_sweeps: int, seed) -> np.ndarray:
    return make_orders(alive, max_sweeps, np.random.default_rng(seed))


def effective_cue_strength(strength: float, donepezil: bool) -> float:
    """Cue strength the network actually receives (donepezil amplifies it)."""
    return float(min(1.0, strength * DONEPEZIL_CUE_GAIN)) if donepezil else float(strength)


def run_stimulator(severity: float, noise: float, strength: float, donepezil: bool = False,
                   mode: str = SIMULATOR_MODE, max_sweeps: int = 30, threshold: float = 0.95,
                   base_seed: int = 12345, connectivity: float | None = None,
                   memantine: bool = False) -> dict:
    """One brain at ``severity`` recalling the X from one cue."""
    target = memory_pattern()
    brain = make_brain(severity, donepezil, mode, base_seed, connectivity, memantine)
    W = brain.effective_weights()
    cue = build_cue(target, effective_cue_strength(strength, donepezil), noise,
                    (base_seed * 7919) & 0xFFFFFFFF)
    orders = _orders(brain.alive, max_sweeps, [base_seed & 0xFFFFFFFF, 0])
    states, conv = settle(W, cue, brain.alive, brain.theta, orders)
    sweeps = len(states) - 1
    pos = np.full((sweeps, brain.n), np.nan)
    if orders.shape[1]:
        ranks = np.arange(orders.shape[1]) / orders.shape[1]
        for k in range(sweeps):
            pos[k, orders[k]] = ranks
    m = score(states[-1], brain.alive, target, threshold, conv)
    m["sweeps"] = sweeps
    run = BrainRun(brain, np.where(brain.alive, cue, 0).astype(np.int8), states, W, m, pos)
    return {"target": target, "brain": run}


def recall_reliability(severity: float, noise: float, strength: float, donepezil: bool = False,
                       mode: str = SIMULATOR_MODE, max_sweeps: int = 30, threshold: float = 0.95,
                       base_seed: int = 12345, connectivity: float | None = None,
                       n_cues: int = 40, memantine: bool = False) -> dict:
    """Successes over ``n_cues`` fresh cues (different neurons lit) for the same brain as
    :func:`run_stimulator`. Shows whether the single animated run is typical."""
    target = memory_pattern()
    brain = make_brain(severity, donepezil, mode, base_seed, connectivity, memantine)
    W = brain.effective_weights()
    ok = 0
    for k in range(n_cues):
        cue = build_cue(target, effective_cue_strength(strength, donepezil), noise,
                        (base_seed * 7919 + 100_003 * (k + 1)) & 0xFFFFFFFF)
        orders = _orders(brain.alive, max_sweeps, [base_seed & 0xFFFFFFFF, k + 1])
        states, conv = settle(W, cue, brain.alive, brain.theta, orders)
        ok += score(states[-1], brain.alive, target, threshold, conv)["success"]
    return {"recalled": ok, "n": n_cues}


MC_TRIALS = 1000


def monte_carlo_recall(severity: float, noise: float, strength: float, donepezil: bool = False,
                       mode: str = SIMULATOR_MODE, max_sweeps: int = 30, threshold: float = 0.95,
                       base_seed: int = 12345, connectivity: float | None = None,
                       n_trials: int = MC_TRIALS, memantine: bool = False) -> dict:
    """Monte Carlo recall: trial k is :func:`run_stimulator` with seed ``base_seed + k`` (a fresh
    damage draw, cue and update order). Returns the mean accuracy (share of the 25 neurons
    matching the X after settling) with a 95% CI, how many trials met the success threshold, and
    ``rep``: the index of the trial whose accuracy is closest to the mean (the first such trial),
    which is the run to animate (``run_stimulator(..., base_seed=base_seed + rep)``)."""
    target = memory_pattern()
    s_eff = effective_cue_strength(strength, donepezil)
    acc = np.empty(n_trials)
    ok = 0
    for k in range(n_trials):
        seed = base_seed + k
        brain = make_brain(severity, donepezil, mode, seed, connectivity, memantine)
        cue = build_cue(target, s_eff, noise, (seed * 7919) & 0xFFFFFFFF)
        orders = _orders(brain.alive, max_sweeps, [seed & 0xFFFFFFFF, 0])
        states, conv = settle(brain.effective_weights(), cue, brain.alive, brain.theta, orders)
        m = score(states[-1], brain.alive, target, threshold, conv)
        acc[k] = m["accuracy"]
        ok += m["success"]
    mean = float(acc.mean())
    half = float(1.96 * acc.std(ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0
    rep = int(np.argmin(np.abs(acc - mean)))
    return {"recall": mean, "lo": mean - half, "hi": mean + half, "recalled": ok, "n": n_trials,
            "rep": rep, "rep_accuracy": float(acc[rep])}
