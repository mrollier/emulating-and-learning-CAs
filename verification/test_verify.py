"""The exactness checks of ca_emulators.verify."""
import numpy as np
import pytest

from ca_emulators import EcaEmulator, NucaEmulator, verify
from ca_emulators.reference import neighbourhood_index
from ca_emulators.rules import DE_BRUIJN
from ca_emulators.weights import rule_table_kernel


@pytest.mark.parametrize("encoding", ["01", "pm1"])
def test_analytic_emulators_are_exact(encoding):
    model = EcaEmulator(8, 0, input_encoding=encoding).model()
    for rule in range(256):
        model.get_layer("rule_tables").set_weights([rule_table_kernel([rule])])
        assert verify.is_exact(model, rule), rule


def test_one_step_states_match_the_model():
    """The map read from the weights equals the model's output on the de Bruijn configuration."""
    model = EcaEmulator(8, rule=None, activation="tanh").model()  # random weights
    out = model(DE_BRUIJN[None, :, None].astype(np.float32)).numpy()[0, :, 0]
    states = verify.one_step_states(model)
    assert np.allclose(states[neighbourhood_index(DE_BRUIJN)], out, atol=1e-6)


def test_a_wrong_rule_table_is_detected():
    model = EcaEmulator(8, 110).model()
    assert not verify.is_exact(model, 111)
    kernel = model.get_layer("rule_tables").get_weights()[0].copy()
    kernel[0, 3, 0] = 1 - kernel[0, 3, 0]  # flip neighbourhood 011
    model.get_layer("rule_tables").set_weights([kernel])
    assert not verify.is_exact(model, 110)
    assert verify.interval_certificate(model, 110) == 0.0


def test_closed_loop_of_the_analytic_emulator(rng):
    model = EcaEmulator(None, 54).model()
    assert verify.closed_loop_exact(model, 54, rng.integers(2, size=(4, 40)), 50)


def test_only_eca_networks():
    with pytest.raises(ValueError):
        verify.is_exact(NucaEmulator(8, [30, 90], rule_alloc=np.arange(8) % 2).model(), 30)
