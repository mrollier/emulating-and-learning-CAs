"""C2: with the analytic weights, the CNN performs ECA updates exactly.

Every rule 0-255 is checked against the numpy reference, for one update and
for many, on lattices from a single cell upwards. The de Bruijn configuration
certifies one-step exactness on its own; random configurations and long runs
check the iterated map.
"""
import numpy as np
import pytest
import tensorflow as tf

from ca_emulators import EcaEmulator, NucaEmulator
from ca_emulators.reference import eca_step, evolve
from ca_emulators.rules import de_bruijn_configuration


@pytest.mark.parametrize("n_cells", [1, 2, 3, 8, 32])
def test_all_rules_one_update(eca_runner, rng, n_cells):
    x = rng.integers(2, size=(16, n_cells))
    if n_cells % 8 == 0:
        x[0] = de_bruijn_configuration(n_cells)
    for rule in range(256):
        assert np.array_equal(eca_runner(n_cells, rule, x, 1)[:, 1], eca_step(x, rule)), rule


def test_all_rules_many_updates(eca_runner, rng):
    x = rng.integers(2, size=(8, 32))
    for rule in range(256):
        assert np.array_equal(eca_runner(32, rule, x, 64), evolve(x, rule, 64)), rule


def test_all_256_rules_in_one_pass(rng):
    """A rule-table layer with 256 channels computes every rule at once."""
    x = rng.integers(2, size=(16, 32))
    model = NucaEmulator(32, list(range(256)), rule_alloc=np.zeros(32, dtype=int)).model()
    candidates = tf.keras.Model(model.input, model.get_layer("rule_tables").output)
    out = candidates(x[:, :, np.newaxis].astype(np.float32)).numpy()  # (S, N, 256)
    expected = np.stack([eca_step(x, r) for r in range(256)], axis=-1)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("rule", [30, 54, 110])
def test_unrolled_model_equals_iteration(rng, rule):
    x = rng.integers(2, size=(4, 32))
    model = EcaEmulator(32, rule, timesteps=20, output_hidden=True).model()
    all_configs, outputs = model(x[:, :, np.newaxis].astype(np.float32))
    diagram = np.transpose(all_configs.numpy(), (0, 2, 1))
    assert np.array_equal(diagram, evolve(x, rule, 20))
    assert np.array_equal(outputs.numpy()[:, :, 0], diagram[:, -1])


@pytest.mark.parametrize("method", ["predict", "eager", "compiled", "while_loop"])
def test_simulation_methods_agree(rng, method):
    x = rng.integers(2, size=(4, 24))
    assert np.array_equal(EcaEmulator(24, 110).simulate(x, 12, method=method), evolve(x, 110, 12))


@pytest.mark.slow
def test_long_run_all_rules(eca_runner, rng):
    x = rng.integers(2, size=(4, 64))
    for rule in range(256):
        assert np.array_equal(eca_runner(64, rule, x, 256), evolve(x, rule, 256)), rule
