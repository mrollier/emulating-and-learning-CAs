"""The configurable architecture: +-1 inputs, detector and rule-table activations.

The defaults are the paper's network (0/1 inputs, ReLU everywhere), which the
golden and published-figure checks cover. With +-1 inputs the analytic
detectors are the sign patterns with bias -2, again exact for every rule.
"""
import numpy as np
import pytest

from ca_emulators import EcaEmulator, NucaEmulator
from ca_emulators.reference import eca_step, evolve
from ca_emulators.rules import de_bruijn_configuration, neighbourhood_patterns
from ca_emulators.simulate import compiled_step, spacetime, to_states
from ca_emulators.weights import detector_bias, detector_kernel, rule_table_kernel


def test_pm1_detectors_are_one_minus_twice_the_hamming_distance():
    patterns = neighbourhood_patterns().astype(float)
    pre = (2 * patterns - 1) @ detector_kernel(encoding="pm1")[:, 0, :] + detector_bias("pm1")
    hamming = np.abs(patterns[:, None, :] - patterns[None, :, :]).sum(-1)
    assert np.array_equal(pre, 1 - 2 * hamming)


def test_pm1_emulator_exact_for_all_rules(rng):
    model = EcaEmulator(32, 0, input_encoding="pm1").model()
    assert model.count_params() == 40
    step = compiled_step(model)
    x = rng.integers(2, size=(8, 32))
    x[0] = de_bruijn_configuration(32)
    for rule in range(256):
        model.get_layer("rule_tables").set_weights([rule_table_kernel([rule])])
        out = to_states(step(x[:, :, None].astype(np.float32)).numpy())
        assert np.array_equal(out, eca_step(x, rule)), rule


@pytest.mark.parametrize("variant", ["lc", "dense"])
def test_pm1_nuca_exact(rng, variant):
    rules, alloc = [30, 90, 110, 150], rng.integers(4, size=24)
    em = NucaEmulator(24, rules, rule_alloc=alloc, input_encoding="pm1")
    model = em.model_dense() if variant == "dense" else em.model()
    x = rng.integers(2, size=(4, 24))
    assert np.array_equal(spacetime(model, x, 16), evolve(x, rules, 16, alloc))


def test_linear_rule_table_is_also_exact(rng):
    x = rng.integers(2, size=(4, 16))
    model = EcaEmulator(16, 110, rule_activation=None).model()
    assert np.array_equal(spacetime(model, x, 8), evolve(x, 110, 8))


def test_analytic_detectors_need_relu():
    with pytest.raises(ValueError):
        EcaEmulator(16, 110, detector_activation="softplus").model()
    with pytest.raises(ValueError):
        EcaEmulator(16, 110, input_encoding="bipolar").model()


def test_model_for_any_number_of_cells(rng):
    model = EcaEmulator(None, 30).model()
    for n_cells in (5, 32, 101):
        x = rng.integers(2, size=(3, n_cells))
        assert np.array_equal(spacetime(model, x, 4), evolve(x, 30, 4))
