"""Retrieval metrics (spec 4.8, 6.6)."""
from __future__ import annotations

import numpy as np

from .retrieval import RetrievalResult

MIRROR_MIN_OVERLAP = 0.9


def _sel(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return x if mask is None else x[mask]


def overlap(state: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """m = (1/N_eval) sum_i s_i xi_i over ``mask`` (all neurons if None). NaN if empty."""
    s, t = _sel(state, mask), _sel(target, mask)
    if s.size == 0:
        return float("nan")
    return float(np.mean(s.astype(np.float64) * t.astype(np.float64)))


def hamming(state: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> int:
    """#{i : s_i != xi_i} over ``mask`` (dead neurons with s=0 count as errors if included)."""
    s, t = _sel(state, mask), _sel(target, mask)
    return int(np.sum(s != t))


def classify_attractor(state: np.ndarray, target: np.ndarray, alive: np.ndarray,
                       patterns, success_threshold: float) -> str:
    """Classify the final state among surviving neurons.

    Let m_mu be the overlap with each stored pattern, m_t with the target and
    m_thr = 2*threshold - 1 (the overlap equivalent of the accuracy threshold).
      target       : m_t >= m_thr
      mirror       : best |m| belongs to a NEGATIVE pattern with |m| >= 0.9
      other_stored : best match is another stored pattern with m >= m_thr
      spurious     : otherwise
    """
    m_thr = 2.0 * success_threshold - 1.0
    if not alive.any():
        return "spurious"
    m_t = overlap(state, target, alive)
    if m_t >= m_thr:
        return "target"
    P = np.asarray(patterns)
    if P.size == 0:
        return "spurious"
    ms = (P[:, alive].astype(np.float64) @ state[alive].astype(np.float64)) / alive.sum()
    b = int(np.argmax(np.abs(ms)))
    if ms[b] < 0 and abs(ms[b]) >= MIRROR_MIN_OVERLAP:
        return "mirror"
    if ms[b] >= m_thr:
        return "other_stored"
    return "spurious"


def evaluate_retrieval(result: RetrievalResult, target: np.ndarray, alive: np.ndarray,
                       patterns, success_threshold: float) -> dict:
    """Metrics for one retrieval.

    accuracy_all  = 1 - hamming_all/N          (dead neurons count as errors)
    accuracy_surv = 1 - hamming_surv/n_alive   (survivors only)
    success       = converged AND accuracy_surv >= threshold   (primary)
    success_all   = converged AND accuracy_all  >= threshold
    """
    s = result.final_state
    n = s.shape[0]
    n_alive = int(alive.sum())
    h_all = hamming(s, target)
    h_surv = hamming(s, target, alive)
    acc_all = 1.0 - h_all / n
    acc_surv = 1.0 - h_surv / n_alive if n_alive else float("nan")
    ok_surv = bool(n_alive and acc_surv >= success_threshold)
    return {
        "accuracy_all": acc_all,
        "accuracy_surv": acc_surv,
        "hamming_all": h_all,
        "hamming_surv": h_surv,
        "overlap_surv": overlap(s, target, alive),
        "sweeps": result.sweeps,
        "n_flips": result.n_flips,
        "converged": result.converged,
        "success": bool(result.converged and ok_surv),
        "success_all": bool(result.converged and acc_all >= success_threshold),
        "attractor_type": classify_attractor(s, target, alive, patterns, success_threshold),
        "final_energy": float(result.energy_trace[-1]),
    }
