"""Elementary rule tables, neighbourhoods and rule symmetries.

Conventions (Wolfram numbering, as in CellPyLib's ``nks_rule``): the
neighbourhood (s_-, s_o, s_+) of a cell has index i = 4 s_- + 2 s_o + s_+,
and the rule table of rule number R assigns to neighbourhood i the state
given by bit i of R.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

N_NEIGHBOURHOODS = 8

#: The binary de Bruijn sequence of order 3. Read cyclically, it contains each
#: of the eight neighbourhoods exactly once, so one periodic configuration of
#: eight cells exercises the whole rule table in a single global update.
DE_BRUIJN = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)


def check_rule(rule) -> int:
    """Return ``rule`` as an int, raising if it is not an elementary rule."""
    if isinstance(rule, (bool, np.bool_)) or int(rule) != rule or not 0 <= int(rule) <= 255:
        raise ValueError(f"an elementary rule is an integer in [0, 255], got {rule!r}")
    return int(rule)


def check_rules(rules) -> np.ndarray:
    """Return ``rules`` as a 1-D int array of elementary rules."""
    rules = np.atleast_1d(np.asarray(rules))
    if rules.ndim != 1 or rules.size == 0:
        raise ValueError("rules must be a non-empty 1-D sequence of elementary rules")
    return np.array([check_rule(r) for r in rules], dtype=np.int64)


def check_alloc(alloc, n_cells: int, n_rules: int, time_varying: bool = False) -> np.ndarray:
    """Return a rule allocation as an int64 array, raising if it is not valid.

    A static allocation has shape (N,); with ``time_varying`` a 2-D array
    (n_updates, N) is accepted too. Entries must be integers in [0, n_rules).
    """
    alloc = np.asarray(alloc)
    if not (alloc.ndim == 1 or (time_varying and alloc.ndim == 2)) or alloc.shape[-1] != n_cells:
        raise ValueError(f"rule_alloc must have shape ({n_cells},)"
                         + (f" or (n_updates, {n_cells})" if time_varying else "")
                         + f", got {alloc.shape}")
    if alloc.size and (not np.all(np.equal(np.mod(alloc, 1), 0))
                       or alloc.min() < 0 or alloc.max() >= n_rules):
        raise ValueError(f"rule_alloc entries must be integers in [0, {n_rules - 1}]")
    return alloc.astype(np.int64)


def neighbourhood_patterns() -> np.ndarray:
    """The eight neighbourhoods as rows of an (8, 3) array.

    Row i is the neighbourhood (s_-, s_o, s_+) whose binary representation is i.
    """
    i = np.arange(N_NEIGHBOURHOODS)
    return np.stack([(i >> 2) & 1, (i >> 1) & 1, i & 1], axis=1).astype(np.uint8)


def rule_table(rule) -> np.ndarray:
    """Rule table of an elementary rule as an (8,) uint8 array (entry i = bit i)."""
    return ((check_rule(rule) >> np.arange(N_NEIGHBOURHOODS)) & 1).astype(np.uint8)


def rule_number(table: Iterable[int]) -> int:
    """Inverse of :func:`rule_table`."""
    table = np.asarray(list(table), dtype=np.int64)
    if table.shape != (N_NEIGHBOURHOODS,) or not np.all((table == 0) | (table == 1)):
        raise ValueError("a rule table has eight binary entries")
    return int(np.sum(table << np.arange(N_NEIGHBOURHOODS)))


def reflect(rule) -> int:
    """Left-right reflection: the rule with s_- and s_+ exchanged."""
    p = neighbourhood_patterns()
    mirrored = 4 * p[:, 2] + 2 * p[:, 1] + p[:, 0]
    return rule_number(rule_table(rule)[mirrored])


def complement(rule) -> int:
    """Conjugation: the rule obtained by exchanging the roles of states 0 and 1."""
    return rule_number(1 - rule_table(rule)[::-1])


def equivalence_class(rule) -> list[int]:
    """Sorted rules related to ``rule`` by reflection and/or conjugation."""
    rule = check_rule(rule)
    return sorted({rule, reflect(rule), complement(rule), reflect(complement(rule))})


def representatives() -> list[int]:
    """The 88 non-equivalent elementary rules (smallest member of each class)."""
    return sorted({equivalence_class(r)[0] for r in range(256)})


def de_bruijn_configuration(n_cells: int = 8, shift: int = 0) -> np.ndarray:
    """The de Bruijn sequence tiled to ``n_cells`` cells and rotated by ``shift``.

    ``n_cells`` must be a multiple of 8 so that the tiling is itself periodic.
    For a rule applied to every cell (an ECA) one such configuration shows every
    neighbourhood; for a nuCA, the eight shifts 0-7 together show every cell
    every neighbourhood (see :func:`certificate_configurations`).
    """
    if n_cells % N_NEIGHBOURHOODS:
        raise ValueError("n_cells must be a multiple of 8")
    return np.roll(np.tile(DE_BRUIJN, n_cells // N_NEIGHBOURHOODS), -shift)


def certificate_configurations(n_cells: int = 8) -> np.ndarray:
    """(8, n_cells) configurations in which every cell meets all eight neighbourhoods.

    If a candidate emulator and the exact rule agree on these eight
    configurations after one update, they agree on every configuration: a
    global update is a cellwise function of the neighbourhood.
    """
    return np.stack([de_bruijn_configuration(n_cells, k) for k in range(N_NEIGHBOURHOODS)])
