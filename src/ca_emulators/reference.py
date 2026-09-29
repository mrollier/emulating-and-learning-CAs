"""Framework-free reference simulators for ECAs and nuCAs.

A vectorised numpy implementation serves as ground truth for the neural
emulators (it is independent of TensorFlow), and a thin wrapper around
CellPyLib ties both to the package used in the paper.

Shapes and conventions used throughout the package:

- A configuration is an (N,) array of 0/1 states; a batch is (S, N).
- Boundaries are periodic.
- ``n_updates`` counts global updates. A spacetime diagram of ``n_updates``
  updates has ``n_updates + 1`` rows, the first being the initial
  configuration. (CellPyLib's ``timesteps`` argument counts rows instead.)
- A rule allocation ``alloc`` holds, per cell, the index of its rule in
  ``rules``. It is either static, shape (N,), or varies in time, shape
  (n_updates, N), with row t - 1 governing update t (CellPyLib's convention).
"""
from __future__ import annotations

import numpy as np

from .rules import check_rules, rule_table


def as_states(x) -> np.ndarray:
    """Return ``x`` as a uint8 array of 0/1 states, raising on other values."""
    x = np.asarray(x)
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("states must be 0 or 1")
    return x.astype(np.uint8)


def neighbourhood_index(x) -> np.ndarray:
    """Index 4 s_- + 2 s_o + s_+ of every cell's neighbourhood (periodic)."""
    x = as_states(x).astype(np.int64)
    return 4 * np.roll(x, 1, axis=-1) + 2 * x + np.roll(x, -1, axis=-1)


def eca_step(x, rule) -> np.ndarray:
    """One global update of an ECA."""
    return rule_table(rule)[neighbourhood_index(x)]


def nuca_step(x, rules, alloc) -> np.ndarray:
    """One global update of a nuCA with static allocation ``alloc`` (shape (N,))."""
    tables = np.stack([rule_table(r) for r in check_rules(rules)])
    alloc = _check_alloc(alloc, np.shape(x)[-1], len(tables))
    return tables[alloc, neighbourhood_index(x)]


def evolve(x0, rules, n_updates: int, alloc=None) -> np.ndarray:
    """Spacetime diagram of ``n_updates`` updates, shape (..., n_updates + 1, N).

    ``rules`` is a single rule (an ECA, ``alloc`` omitted) or a sequence of
    rules with an allocation, static (N,) or time-varying (n_updates, N).
    """
    x = as_states(x0)
    rules = check_rules(rules)
    n_cells = x.shape[-1]
    if alloc is None:
        if len(rules) != 1:
            raise ValueError("several rules need an allocation")
        alloc = np.zeros(n_cells, dtype=np.int64)
    alloc = np.asarray(alloc)
    if alloc.ndim == 1:
        alloc = np.broadcast_to(alloc, (n_updates, n_cells))
    if alloc.shape != (n_updates, n_cells):
        raise ValueError(f"alloc must have shape ({n_cells},) or ({n_updates}, {n_cells})")
    frames = [x]
    for t in range(n_updates):
        frames.append(nuca_step(frames[-1], rules, alloc[t]))
    return np.stack(frames, axis=-2)


def evolve_cellpylib(x0, rules, n_updates: int, alloc=None) -> np.ndarray:
    """Same as :func:`evolve`, computed with CellPyLib (one call per sample).

    Memoisation is switched off: CellPyLib memoises by neighbourhood only,
    which is wrong as soon as the rule depends on the cell.
    """
    import cellpylib as cpl

    x = as_states(x0)
    batch = x.reshape(-1, x.shape[-1])
    rules = check_rules(rules)
    n_cells = x.shape[-1]
    if alloc is None:
        alloc = np.zeros(n_cells, dtype=np.int64)
    alloc = np.asarray(alloc)
    if alloc.ndim == 1:
        alloc = np.broadcast_to(alloc, (max(n_updates, 1), n_cells))
    out = [
        cpl.evolve(row[np.newaxis, :].astype(int), timesteps=n_updates + 1,
                   apply_rule=lambda n, c, t: cpl.nks_rule(n, int(rules[alloc[t - 1, c]])),
                   memoize=False)
        for row in batch
    ]
    return np.asarray(out, dtype=np.uint8).reshape(*x.shape[:-1], n_updates + 1, n_cells)


def _check_alloc(alloc, n_cells: int, n_rules: int) -> np.ndarray:
    alloc = np.asarray(alloc)
    if alloc.shape != (n_cells,):
        raise ValueError(f"rule_alloc must have shape ({n_cells},), got {alloc.shape}")
    if alloc.min() < 0 or alloc.max() >= n_rules:
        raise ValueError(f"rule_alloc entries must lie in [0, {n_rules - 1}]")
    return alloc.astype(np.int64)
