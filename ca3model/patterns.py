"""Pattern generation and cue corruption (spec sections 4.7, 6.2)."""
from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator
from scipy.ndimage import gaussian_filter


def random_patterns(n_patterns: int, n_neurons: int, rng: Generator) -> np.ndarray:
    """i.i.d. fair bipolar patterns xi in {-1,+1}; returns int8 array (P, N)."""
    bits = rng.integers(0, 2, size=(n_patterns, n_neurons), dtype=np.int8)
    return (bits * 2 - 1).astype(np.int8)


def corrupt_cue(pattern: np.ndarray, noise: float, mode: str, rng: Generator) -> np.ndarray:
    """Corrupt ``round(noise*N)`` randomly chosen bits.

    ``"flip"``: chosen bits are negated (exactly k errors).
    ``"occlude"``: chosen bits are replaced by independent random +-1, so the
    *expected* number of wrong bits is k/2 (documented in the UI).
    """
    n = pattern.shape[0]
    k = int(round(noise * n))
    idx = rng.choice(n, size=k, replace=False)
    out = pattern.astype(np.int8).copy()
    if mode == "flip":
        out[idx] = -out[idx]
    elif mode == "occlude":
        out[idx] = (rng.integers(0, 2, size=k, dtype=np.int8) * 2 - 1).astype(np.int8)
    else:
        raise ValueError(f"unknown cue mode {mode!r}")
    return out


def grid_shape(n: int) -> tuple[int, int]:
    """Square if N is a perfect square, else nearest (H, W) rectangle with H*W >= N."""
    s = math.isqrt(n)
    if s * s == n:
        return s, s
    h = int(math.ceil(math.sqrt(n)))
    return h, int(math.ceil(n / h))


def to_image(vec: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Reshape a length-N vector to a float (H, W) image; pads with NaN if needed."""
    n = vec.shape[0]
    h, w = shape if shape is not None else grid_shape(n)
    img = np.full(h * w, np.nan, dtype=float)
    img[:n] = vec
    return img.reshape(h, w)


def smooth_patterns(n_patterns: int, n_neurons: int, rng: Generator,
                    length_scale: float = 1.5) -> np.ndarray:
    """Spatially smooth bipolar patterns (sign of Gaussian-filtered noise). DEMO ONLY.

    Neighbouring pixels are correlated, so patterns are correlated and capacity
    is lower than for random patterns. Not used for statistics.
    """
    h, w = grid_shape(n_neurons)
    out = np.empty((n_patterns, n_neurons), dtype=np.int8)
    for p in range(n_patterns):
        field = gaussian_filter(rng.standard_normal((h, w)), sigma=length_scale, mode="wrap")
        out[p] = np.where(field.reshape(-1)[:n_neurons] >= 0, 1, -1)
    return out
