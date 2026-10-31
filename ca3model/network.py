"""Hopfield network with Hebbian storage and a separate damage state (spec 4.1-4.2, 6.3)."""
from __future__ import annotations

import numpy as np


class HopfieldNetwork:
    """Symmetric recurrent network with zero self-connections.

    ``W_raw`` holds *learned* weights; damage lives in ``efficacy`` (per synapse)
    and ``alive`` (per neuron). Effective weights::

        W_eff[i,j] = W_raw[i,j] * efficacy[i,j] * alive[i] * alive[j]
    """

    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.W_raw = np.zeros((self.n, self.n), dtype=np.float64)
        self.efficacy = np.ones((self.n, self.n), dtype=np.float64)
        np.fill_diagonal(self.efficacy, 0.0)
        self.alive = np.ones(self.n, dtype=bool)
        self.n_stored = 0
        self.patterns: list[np.ndarray] = []

    @classmethod
    def from_patterns(cls, patterns: np.ndarray) -> "HopfieldNetwork":
        """Batch Hebbian storage: W_raw = (1/N) sum_mu xi xi^T, diagonal zeroed."""
        patterns = np.atleast_2d(patterns)
        net = cls(patterns.shape[1])
        net.store(patterns, 1.0)
        return net

    def store(self, patterns: np.ndarray, plasticity_efficacy: float = 1.0) -> None:
        """Incremental learning: W_raw <- W_raw + (eta/N) * sum_mu (xi xi^T - I).

        Mutates ``W_raw`` in place; never touches ``efficacy`` or ``alive``.
        ``n_stored``/``patterns`` count every presented pattern (even at eta=0).
        """
        patterns = np.atleast_2d(patterns)
        if patterns.shape[1] != self.n:
            raise ValueError("pattern length does not match network size")
        if not np.all(np.abs(patterns) == 1):
            raise ValueError("patterns must be bipolar {-1,+1}")
        eta = float(plasticity_efficacy)
        if not 0.0 <= eta <= 1.0:
            raise ValueError("plasticity_efficacy must be in [0, 1]")
        X = patterns.astype(np.float64)
        inc = X.T @ X
        inc[np.diag_indices(self.n)] = 0.0  # xi xi^T - I has zero diagonal
        W = self.W_raw + (eta / self.n) * inc
        W = 0.5 * (W + W.T)                  # enforce exact symmetry
        np.fill_diagonal(W, 0.0)
        self.W_raw[...] = W
        self.n_stored += patterns.shape[0]
        self.patterns.extend(p.astype(np.int8).copy() for p in patterns)

    def effective_weights(self) -> np.ndarray:
        """W_eff = W_raw * efficacy * outer(alive, alive)  (spec 4.2)."""
        a = self.alive.astype(np.float64)
        return self.W_raw * self.efficacy * np.outer(a, a)

    def copy(self) -> "HopfieldNetwork":
        new = HopfieldNetwork(self.n)
        new.W_raw = self.W_raw.copy()
        new.efficacy = self.efficacy.copy()
        new.alive = self.alive.copy()
        new.n_stored = self.n_stored
        new.patterns = [p.copy() for p in self.patterns]
        return new

    def energy(self, state: np.ndarray) -> float:
        """E(s) = -1/2 s^T W_eff s."""
        s = state.astype(np.float64)
        return float(-0.5 * s @ self.effective_weights() @ s)

    def connectivity_stats(self) -> dict:
        """Connectivity summary relative to a healthy N(N-1) off-diagonal network.

        frac_synapses_lost : 1 - sum(offdiag efficacy among alive pairs)/(N(N-1))
                             (efficacy-weighted, so attenuation counts partially)
        frac_synapses_zero : fraction of off-diagonal synapses with efficacy exactly 0
                             among all pairs (deleted or attached to dead neurons)
        n_alive, mean_efficacy_alive_pairs, mean_degree (nonzero links per alive neuron)
        """
        n = self.n
        off = ~np.eye(n, dtype=bool)
        a = self.alive
        pair = np.outer(a, a) & off
        eff_total = float((self.efficacy * pair).sum())
        n_alive = int(a.sum())
        n_pairs = int(pair.sum())
        live_links = (self.efficacy > 0) & pair
        return {
            "frac_synapses_lost": 1.0 - eff_total / (n * (n - 1)),
            "frac_synapses_zero": 1.0 - float(live_links.sum()) / (n * (n - 1)),
            "n_alive": n_alive,
            "mean_efficacy_alive_pairs": eff_total / n_pairs if n_pairs else float("nan"),
            "mean_degree": float(live_links.sum()) / n_alive if n_alive else 0.0,
        }
