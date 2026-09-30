"""C9: with the recommended recipe, training from random weights is reliable.

experiments/training/REPORT.md found the recipe (+-1 inputs, softplus
detectors, sigmoid/binary cross-entropy head, Adam 0.02) exact in all 262144
runs (every rule x 1024 seeds), all proved exact in unbinarised closed loop;
train_recipe's own configuration (full batch) in 262143 of them (a rare
plateau of rule 89).
Here the package's ca_emulators.training.train_recipe is checked on the rules
that are hardest for the 2024 recipe (rule 1, never learnt in 2024; the
parities 105 and 150) and on rules 30, 54 and 110.
"""
import numpy as np
import pytest

from ca_emulators import verify
from ca_emulators.rules import representatives
from ca_emulators.training import recipe_model, train_recipe

RULES = [1, 30, 54, 105, 110, 150]


def test_recipe_network_has_41_parameters():
    assert recipe_model().count_params() == 41


@pytest.mark.parametrize("rule", RULES)
def test_recipe_learns_the_rule_exactly(rng, rule):
    run = train_recipe(rule, seed=0)
    assert run.exact
    assert run.certificate > 0            # closed-loop exactness proved
    assert run.losses[-1] < run.losses[0]
    x = rng.integers(2, size=(4, 48))     # a size never seen in training
    assert verify.closed_loop_exact(run.model, rule, x, 100, logits=True)


@pytest.mark.slow
def test_recipe_every_equivalence_class():
    """One seed for each of the 88 non-equivalent rules (the study ran all 256 x 1024)."""
    failures = [r for r in representatives() if not train_recipe(r, seed=1).exact]
    assert failures == []
