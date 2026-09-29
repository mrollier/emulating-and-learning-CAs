"""C8: the rebuilt package reproduces the outputs of the 2024 code bit for bit.

data/golden/golden_2024.npz was produced by the acri-2024 copy of src/ in the
paper's environment (scripts/capture_golden_2024.py); every output in it was
checked against CellPyLib before it was stored.
"""
import numpy as np
import pytest

from ca_emulators import EcaEmulator, NucaEmulator
from ca_emulators.simulate import spacetime
from ca_emulators.weights import detector_bias, detector_kernel, rule_table_kernel


def test_analytic_weights(golden):
    assert np.array_equal(detector_kernel(), golden["tab71_kernel"])
    assert np.array_equal(detector_bias(), golden["tab71_bias"])
    assert np.array_equal(rule_table_kernel([54]), golden["tab71_rule54"])


def test_2024_default_emulator_was_not_exact(golden):
    """The 2024 default (train_triplet_id=True) left the 32 detector weights random;
    the rebuilt emulator is exact and frozen by default."""
    assert golden["default_trainable"] == 32
    assert len(EcaEmulator(32, 54).model().trainable_weights) == 0


def test_eca_one_update_all_rules(golden, eca_runner):
    for rule in range(256):
        out = eca_runner(32, rule, golden["eca_x"], 1)[:, 1]
        assert np.array_equal(out, golden["eca_y"][rule]), rule


def test_eca_many_updates(golden, eca_runner):
    for i, rule in enumerate(golden["eca_multi_rules"]):
        diagram = eca_runner(32, int(rule), golden["eca_x"], 32)
        assert np.array_equal(diagram, golden["eca_multi_unrolled"][i]), rule
        assert np.array_equal(diagram, golden["eca_multi_predict"][i]), rule


def test_eca_unrolled_model(golden):
    i = 6  # rule 54
    assert golden["eca_multi_rules"][i] == 54
    model = EcaEmulator(32, 54, timesteps=32, output_hidden=True).model()
    all_configs, _ = model(golden["eca_x"][:, :, None].astype(np.float32))
    assert np.array_equal(np.transpose(all_configs.numpy(), (0, 2, 1)), golden["eca_multi_unrolled"][i])


@pytest.mark.parametrize("case", ["nuca_32_8", "nuca_64_4",
                                  pytest.param("nuca_256_256", marks=pytest.mark.slow)])
@pytest.mark.parametrize("variant", ["lc", "dense"])
def test_nuca_both_variants(golden, case, variant):
    em = NucaEmulator(len(golden[f"{case}_alloc"]), golden[f"{case}_rules"],
                      rule_alloc=golden[f"{case}_alloc"])
    model = em.model_dense() if variant == "dense" else em.model()
    assert np.array_equal(spacetime(model, golden[f"{case}_x"], 32), golden[f"{case}_{variant}"])


def test_nuca_predict_protocol(golden):
    """The 2024 benchmark protocol (predict per update) gives the same diagram."""
    case = "nuca_32_8"
    em = NucaEmulator(32, golden[f"{case}_rules"], rule_alloc=golden[f"{case}_alloc"])
    out = spacetime(em.model(), golden[f"{case}_x"], 32, method="predict")
    assert np.array_equal(out, golden[f"{case}_lc"])
