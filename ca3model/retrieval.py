"""Asynchronous zero-threshold dynamics (spec 4.6, 6.5).

Update rule for alive neuron i (random order, one sweep = one permutation of the
alive neurons)::

    h_i = sum_j W_eff[i,j] s_j ;  s_i <- +1 if h_i > 0, -1 if h_i < 0, else unchanged

``h`` is maintained incrementally: a flip of neuron i by delta = +-2 does
``h += W[i,:] * delta`` (W_eff is symmetric). Energy: E = -1/2 s^T W_eff s.

Update orders are pre-generated so Numba and the NumPy fallback consume the same
random numbers and return identical results. The random block is always drawn
over *all* N neurons and then restricted to alive ones, so the stream consumed
(and the relative order of surviving neurons) does not depend on the damage level
-> paired comparisons across conditions see identical update orders.

Numerical note: ``h`` is compared to a tiny tolerance (1e-9 * max|W|) instead of
exact 0 so that float round-off cannot break exact ties differently in the two
back-ends; this is scale-free (uniform scaling of W cannot change results).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from .network import HopfieldNetwork

try:  # Numba is optional: fall back to pure NumPy/Python automatically
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is missing
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


@njit(cache=True)
def _kernel_hist(W, state, alive, orders, max_sweeps, hist):
    """Async dynamics. Writes state after each sweep into ``hist`` rows 0..sweeps
    (row 0 = initial) when ``hist`` has enough rows. Returns (state, sweeps, flips, converged)."""
    n = state.shape[0]
    h = np.dot(W, state)
    tol = 1e-9 * np.max(np.abs(W))
    n_ord = orders.shape[1]
    total_flips = 0
    sweeps = 0
    converged = False
    if hist.shape[0] > 0:
        hist[0, :] = state
    for sw in range(max_sweeps):
        flips = 0
        for k in range(n_ord):
            i = orders[sw, k]
            hi = h[i]
            if hi > tol:
                new = 1.0
            elif hi < -tol:
                new = -1.0
            else:
                new = state[i]
            if new != state[i]:
                d = new - state[i]
                state[i] = new
                h += W[i, :] * d
                flips += 1
        sweeps = sw + 1
        total_flips += flips
        if hist.shape[0] > sweeps:
            hist[sweeps, :] = state
        if flips == 0:
            converged = True
            break
    return state, sweeps, total_flips, converged


@njit(cache=True)
def _async_kernel(W, state, alive, orders, max_sweeps):
    """Spec signature: (W, state, alive, orders, max_sweeps) -> (state, sweeps, flips, converged)."""
    hist = np.empty((0, state.shape[0]))
    return _kernel_hist(W, state, alive, orders, max_sweeps, hist)


# Pure-Python/NumPy twin of the same code (identical algorithm, no compilation).
_kernel_numpy = getattr(_kernel_hist, "py_func", _kernel_hist)


@dataclass
class RetrievalResult:
    final_state: np.ndarray          # int8 (N,), 0 at dead neurons
    sweeps: int
    n_flips: int
    converged: bool
    energy_trace: np.ndarray         # energy at start and after each sweep
    state_history: list[np.ndarray] | None  # states at start and after each sweep (if recorded)


def make_orders(alive: np.ndarray, max_sweeps: int, rng: Generator) -> np.ndarray:
    """(max_sweeps, n_alive) int64 matrix; each row a random permutation of alive indices."""
    n = alive.shape[0]
    perm = np.argsort(rng.random((max_sweeps, n)), axis=1)
    keep = alive[perm]
    return perm[keep].reshape(max_sweeps, int(alive.sum())).astype(np.int64)


def retrieve(net: HopfieldNetwork, cue: np.ndarray, max_sweeps: int, rng: Generator,
             record_states: bool = False, W_eff: np.ndarray | None = None,
             backend: str = "auto") -> RetrievalResult:
    """Run asynchronous retrieval from ``cue``.

    ``W_eff`` may be passed to avoid recomputing effective weights across many cues.
    ``backend``: "auto" (numba if available), "numba", or "numpy".
    """
    W = np.ascontiguousarray(net.effective_weights() if W_eff is None else W_eff,
                             dtype=np.float64)
    alive = net.alive
    n = net.n
    state = np.where(alive, cue, 0).astype(np.float64)
    orders = make_orders(alive, max_sweeps, rng)  # always consumes the same stream amount

    if orders.shape[1] == 0:  # every neuron lost
        e = np.zeros(2)
        hist = [np.zeros(n, dtype=np.int8)] if record_states else None
        return RetrievalResult(np.zeros(n, dtype=np.int8), 1, 0, True, e, hist)

    if backend == "numba" and not HAVE_NUMBA:
        raise RuntimeError("numba backend requested but numba is not importable")
    kernel = _kernel_numpy if backend == "numpy" or (backend == "auto" and not HAVE_NUMBA) \
        else _kernel_hist
    hist = np.empty((max_sweeps + 1, n), dtype=np.float64)
    final, sweeps, flips, conv = kernel(W, state, alive, orders, max_sweeps, hist)
    H = hist[: sweeps + 1]
    energies = -0.5 * np.einsum("ki,ij,kj->k", H, W, H)
    states = [H[k].astype(np.int8) for k in range(sweeps + 1)] if record_states else None
    return RetrievalResult(final.astype(np.int8), int(sweeps), int(flips), bool(conv),
                           energies, states)
