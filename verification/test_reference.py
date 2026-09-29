"""The numpy reference simulator and the rule utilities.

The reference is the ground truth of the other tests, so it is itself tied to
CellPyLib (the simulator used in the paper) for every rule.
"""
import numpy as np
import pytest

from ca_emulators import rules as R
from ca_emulators.reference import eca_step, evolve, evolve_cellpylib, neighbourhood_index


def test_numpy_equals_cellpylib_for_all_rules(rng):
    x = rng.integers(2, size=(4, 16))
    for rule in range(256):
        assert np.array_equal(evolve(x, rule, 8), evolve_cellpylib(x, rule, 8)), rule


def test_time_varying_allocation_equals_cellpylib(rng):
    rules = [30, 90, 110]
    alloc = rng.integers(3, size=(10, 20))
    x = rng.integers(2, size=(2, 20))
    assert np.array_equal(evolve(x, rules, 10, alloc), evolve_cellpylib(x, rules, 10, alloc))


def test_wolfram_numbering():
    assert R.rule_table(30).tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert R.rule_table(110).tolist() == [0, 1, 1, 1, 0, 1, 1, 0]
    assert all(R.rule_number(R.rule_table(r)) == r for r in range(256))
    # neighbourhood 100 (left neighbour alive) has index 4
    assert neighbourhood_index(np.array([0, 1, 0, 0]))[2] == 4


def test_rule_90_is_the_xor_of_the_neighbours(rng):
    x = rng.integers(2, size=(5, 21))
    assert np.array_equal(eca_step(x, 90), np.roll(x, 1, -1) ^ np.roll(x, -1, -1))


def test_symmetries():
    assert R.reflect(110) == 124 and R.complement(110) == 137
    assert all(R.reflect(R.reflect(r)) == r and R.complement(R.complement(r)) == r
               for r in range(256))
    assert len(R.representatives()) == 88


def test_de_bruijn_configuration_shows_every_neighbourhood():
    assert sorted(neighbourhood_index(R.DE_BRUIJN)) == list(range(8))
    cert = R.certificate_configurations(24)
    idx = neighbourhood_index(cert)  # (8 configurations, 24 cells)
    assert all(sorted(idx[:, c]) == list(range(8)) for c in range(24))


@pytest.mark.parametrize("bad", [-1, 256, 3.5, True])
def test_invalid_rules_are_rejected(bad):
    with pytest.raises(ValueError):
        R.check_rule(bad)
