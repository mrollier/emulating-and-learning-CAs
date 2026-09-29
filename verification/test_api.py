"""Behaviour of the public API: defaults, training modes, layers and weights."""
import numpy as np
import pytest
import tensorflow as tf

from ca_emulators import EcaEmulator, NucaEmulator, PeriodicPadding1D
from ca_emulators.reference import evolve, evolve_cellpylib
from ca_emulators.simulate import spacetime
from ca_emulators.weights import selector_kernel_lc


def test_exact_by_default():
    assert EcaEmulator(16, 30).model().trainable_weights == []
    em = NucaEmulator(16, [30, 90], rule_alloc=np.arange(16) % 2)
    assert em.model().trainable_weights == [] and em.model_dense().trainable_weights == []


def test_without_rule_the_network_is_trainable():
    model = EcaEmulator(16).model()
    assert {w.name.split("/")[0] for w in model.trainable_weights} == {"detectors", "rule_tables"}
    assert model.count_params() == 32 + 9  # the rule-table layer gets a bias


def test_trainable_keeps_the_analytic_initialisation(rng):
    model = EcaEmulator(16, 110, trainable=True).model()
    assert len(model.trainable_weights) == 3
    x = rng.integers(2, size=(4, 16))
    assert np.array_equal(spacetime(model, x, 5), evolve(x, 110, 5))


def test_nuca_with_unknown_allocation_has_a_trainable_selector():
    model = NucaEmulator(16, [30, 90]).model()
    assert {w.name.split("/")[0] for w in model.trainable_weights} == {"selector"}


def test_train_triplet_id_is_deprecated_but_honoured():
    with pytest.warns(DeprecationWarning):
        model = EcaEmulator(16, 54, train_triplet_id=True).model()
    assert {w.name.split("/")[0] for w in model.trainable_weights} == {"detectors"}
    with pytest.warns(DeprecationWarning):
        assert EcaEmulator(16, 54, train_triplet_id=False).model().trainable_weights == []


@pytest.mark.parametrize("kwargs", [dict(rule=256), dict(rule=-1)])
def test_invalid_rule(kwargs):
    with pytest.raises(ValueError):
        EcaEmulator(8, **kwargs)


def test_invalid_allocation():
    with pytest.raises(ValueError):
        NucaEmulator(8, [30, 90], rule_alloc=[0, 1])
    with pytest.raises(ValueError):
        NucaEmulator(8, [30, 90], rule_alloc=[0, 1, 2, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        NucaEmulator(4, [30, 90], rule_alloc=[0.0, 0.9, 1.0, 0.5])  # not integers
    with pytest.raises(ValueError):
        NucaEmulator(8)  # neither rules nor n_rules


def test_references_need_an_allocation_for_several_rules(rng):
    x = rng.integers(2, size=8)
    with pytest.raises(ValueError):
        evolve(x, [30, 90], 4)
    with pytest.raises(ValueError):
        evolve_cellpylib(x, [30, 90], 4)


def test_simulate_rejects_unknown_variant():
    em = NucaEmulator(8, [30, 90], rule_alloc=np.arange(8) % 2)
    with pytest.raises(ValueError):
        em.simulate(np.zeros(8, dtype=int), 2, variant="Dense")


def test_explicit_trainable_overrides_train_triplet_id():
    with pytest.warns(DeprecationWarning):
        model = EcaEmulator(16, 54, trainable=True, train_triplet_id=False).model()
    assert {w.name.split("/")[0] for w in model.trainable_weights} == {"detectors", "rule_tables"}


def test_removed_halfway_initialiser_is_reported():
    with pytest.raises(ValueError, match="halfway"):
        EcaEmulator(16, kernel_initializer="halfway")


def test_training_needs_at_least_one_restart():
    from ca_emulators.training import train_2024_recipe
    with pytest.raises(ValueError):
        train_2024_recipe(54, max_restarts=0)


def test_periodic_padding(rng):
    x = rng.random((3, 7, 2)).astype(np.float32)
    for p in (0, 1, 3):
        out = PeriodicPadding1D(p)(x).numpy()
        assert np.array_equal(out, np.pad(x, ((0, 0), (p, p), (0, 0)), mode="wrap"))


def test_sparse_selector_order_matches_keras(rng):
    """Implementation 3 stores the kernel in the order of the layer's kernel_idxs."""
    alloc = rng.integers(3, size=10)
    model = NucaEmulator(10, [30, 90, 110], rule_alloc=alloc, implementation=3).model()
    layer = model.get_layer("selector")
    values = selector_kernel_lc(alloc, 3, implementation=3)
    for (out_idx, in_idx), v in zip(layer.kernel_idxs, values):
        assert v == (in_idx == out_idx * 3 + alloc[out_idx])


def test_save_and_load(tmp_path, rng):
    model = NucaEmulator(12, [30, 90], rule_alloc=np.arange(12) % 2).model_dense()
    path = tmp_path / "nuca.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    x = rng.integers(2, size=(3, 12, 1)).astype(np.float32)
    assert np.array_equal(loaded(x).numpy(), model(x).numpy())
