"""Damage randomness and application (spec sections 4.3-4.5, 6.4).

Randomness is drawn ONCE per trial (:func:`draw_damage_randomness`) and reused for
every severity (common random numbers), so damage is nested: larger s / f damages
a superset of what smaller s / f damages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from .config import DamageSpec
from .network import HopfieldNetwork


@dataclass(frozen=True)
class DamageDraw:
    """U: symmetric U(0,1) matrix with zero diagonal; neuron_order: permutation of N."""

    U: np.ndarray
    neuron_order: np.ndarray


def draw_damage_randomness(n: int, rng: Generator) -> DamageDraw:
    """Draw U (symmetric, zero diagonal) then a neuron permutation, in that order."""
    upper = np.triu(rng.random((n, n)), k=1)
    U = upper + upper.T
    order = rng.permutation(n)
    return DamageDraw(U=U, neuron_order=order)


def n_lost_neurons(n: int, fraction: float) -> int:
    """k = round(f * N)."""
    return int(round(fraction * n))


def build_damage(draw: DamageDraw, spec: DamageSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(efficacy, alive)`` for ``spec``.

    deletion:     E_ij = 0 if U_ij < s else 1
    attenuation:  E_ij = 1 - s * U_ij
    neuron loss:  alive[order[:round(f N)]] = False
    Diagonal of E is 0.
    """
    U = draw.U
    n = U.shape[0]
    s = spec.synaptic_severity
    if spec.synaptic_mode == "deletion":
        E = np.where(U < s, 0.0, 1.0)
    else:
        E = 1.0 - s * U
    np.fill_diagonal(E, 0.0)
    alive = np.ones(n, dtype=bool)
    alive[draw.neuron_order[: n_lost_neurons(n, spec.neuron_loss)]] = False
    return E, alive


def apply_damage(net: HopfieldNetwork, spec: DamageSpec, draw: DamageDraw) -> HopfieldNetwork:
    """Damaged COPY: efficacy <- efficacy * E_built ; alive <- alive & alive_built.

    Because damage is a mask separate from W_raw, it composes (order-independent)
    and later ``store`` calls cannot restore deleted synapses or use dead neurons.
    """
    E, alive = build_damage(draw, spec)
    out = net.copy()
    out.efficacy = out.efficacy * E
    out.alive = out.alive & alive
    return out
