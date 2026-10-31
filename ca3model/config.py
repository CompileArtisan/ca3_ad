"""Frozen configuration dataclasses (spec section 6.1).

Every config validates its ranges on construction and exposes ``to_dict()``
returning JSON-safe primitives (used for cache keys, CSV metadata, display).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class NetworkConfig:
    """Network size. N must satisfy 100 <= N <= 400."""

    n_neurons: int = 256

    def __post_init__(self) -> None:
        _check(100 <= int(self.n_neurons) <= 400, "n_neurons must be in [100, 400]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalConfig:
    """Cue corruption and retrieval-dynamics settings."""

    cue_noise: float = 0.20
    cue_mode: Literal["flip", "occlude"] = "flip"
    max_sweeps: int = 30
    success_threshold: float = 0.95

    def __post_init__(self) -> None:
        _check(0.0 <= self.cue_noise <= 1.0, "cue_noise must be in [0, 1]")
        _check(self.cue_mode in ("flip", "occlude"), "cue_mode must be 'flip' or 'occlude'")
        _check(int(self.max_sweeps) >= 1, "max_sweeps must be >= 1")
        _check(0.5 < self.success_threshold <= 1.0, "success_threshold must be in (0.5, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DamageSpec:
    """Abstract damage: synaptic severity s, mode, and neuron-loss fraction f."""

    synaptic_severity: float = 0.0
    synaptic_mode: Literal["deletion", "attenuation"] = "deletion"
    neuron_loss: float = 0.0

    def __post_init__(self) -> None:
        _check(0.0 <= self.synaptic_severity <= 1.0, "synaptic_severity must be in [0, 1]")
        _check(0.0 <= self.neuron_loss <= 1.0, "neuron_loss must be in [0, 1]")
        _check(self.synaptic_mode in ("deletion", "attenuation"),
               "synaptic_mode must be 'deletion' or 'attenuation'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Trial structure, seeding and learning parameters."""

    n_patterns: int = 8
    n_trials: int = 30
    base_seed: int = 12345
    n_new_patterns: int = 5
    plasticity_efficacy: float = 1.0

    def __post_init__(self) -> None:
        _check(int(self.n_patterns) >= 1, "n_patterns must be >= 1")
        _check(int(self.n_trials) >= 1, "n_trials must be >= 1")
        _check(int(self.n_new_patterns) >= 1, "n_new_patterns must be >= 1")
        _check(0.0 <= self.plasticity_efficacy <= 1.0, "plasticity_efficacy must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
