"""Seeding policy (spec section 5).

``np.random.SeedSequence([base_seed, trial]).spawn(5)`` gives five independent
child seeds -> one ``Generator`` per stream. The global NumPy state is never used.
Conditions inside a trial call :meth:`TrialRNGs.fresh_cues` /
:meth:`TrialRNGs.fresh_dynamics` to re-create the cue and update-order streams
from the *same* child seed, so every condition sees identical cues/orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STREAMS = ("patterns", "new_patterns", "cues", "damage", "dynamics")


@dataclass
class TrialRNGs:
    """One generator per random stream for a single (base_seed, trial)."""

    base_seed: int
    trial: int
    patterns: np.random.Generator
    new_patterns: np.random.Generator
    cues: np.random.Generator
    damage: np.random.Generator
    dynamics: np.random.Generator
    children: tuple = field(repr=False, default=())

    def _fresh(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.children[STREAMS.index(name)])

    def fresh_cues(self) -> np.random.Generator:
        """New cue generator from the same child seed (identical stream)."""
        return self._fresh("cues")

    def fresh_dynamics(self) -> np.random.Generator:
        """New dynamics generator from the same child seed (identical stream)."""
        return self._fresh("dynamics")


def trial_rngs(base_seed: int, trial: int) -> TrialRNGs:
    """Build the five streams for ``(base_seed, trial)``; deterministic everywhere."""
    mask = (1 << 64) - 1  # allow negative seeds from the UI
    ss = np.random.SeedSequence([int(base_seed) & mask, int(trial) & mask])
    kids = tuple(ss.spawn(len(STREAMS)))
    gens = {name: np.random.default_rng(k) for name, k in zip(STREAMS, kids)}
    return TrialRNGs(base_seed=int(base_seed), trial=int(trial), children=kids, **gens)
