"""C5: the emulators reproduce the published figures cell for cell.

The inputs were decoded from the published PDFs
(scripts/extract_published_inputs.py). Figs 1 and 4 are full spacetime
diagrams, Fig. 2 a single update of rule 54, and Fig. 3 shows a target
configuration of rule 54.
"""
import numpy as np
import pytest
import tensorflow as tf

from ca_emulators import EcaEmulator, NucaEmulator
from ca_emulators.reference import eca_step, evolve, evolve_cellpylib, neighbourhood_index
from ca_emulators.simulate import compiled_step, spacetime, to_states
from ca_emulators.weights import selector_kernel_dense


@pytest.mark.parametrize("variant", ["lc", "dense"])
def test_fig1_nuca_rules_30_and_90(published, variant):
    fig = published["fig1_nuca_rules30_90"]
    em = NucaEmulator(32, fig["rules"], rule_alloc=fig["alloc"])
    model = em.model_dense() if variant == "dense" else em.model()
    diagram = fig["diagram"]  # 32 rows: the initial configuration and 31 updates
    assert np.array_equal(spacetime(model, diagram[0], 31), diagram)


def test_fig1_matches_references(published):
    fig = published["fig1_nuca_rules30_90"]
    d = fig["diagram"]
    assert np.array_equal(evolve(d[0], fig["rules"], 31, fig["alloc"]), d)
    assert np.array_equal(evolve_cellpylib(d[0], fig["rules"], 31, fig["alloc"]), d)


def test_fig2_decomposition_rule_54(published):
    fig = published["fig2_decomposition_rule54"]
    model = EcaEmulator(32, 54).model()
    x = fig["x"][np.newaxis, :, np.newaxis].astype(np.float32)
    assert np.array_equal(to_states(model(x).numpy())[0], fig["y"])
    detectors = tf.keras.Model(model.input, model.get_layer("detectors").output)(x).numpy()[0]
    assert np.array_equal(np.argmax(detectors, axis=1), neighbourhood_index(fig["x"]))


def test_fig3_target_is_rule_54(published):
    fig = published["fig3_training_example_rule54"]
    assert np.array_equal(eca_step(fig["x"], 54), fig["y"])


def test_fig4_time_varying_allocation(published):
    """Fig. 4 shifts the allocation by one cell per update; the dense emulator
    reproduces it when its selector is swapped between updates."""
    fig = published["fig4_nuca_cellpylib_8rules"]
    rules, alloc, diagram = fig["rules"], fig["alloc"], fig["diagram"]
    n_updates = len(diagram) - 1
    assert np.array_equal(evolve(diagram[0], rules, n_updates, alloc[:n_updates]), diagram)
    model = NucaEmulator(32, rules, rule_alloc=alloc[0]).model_dense()
    step = compiled_step(model)
    frames = [diagram[0][np.newaxis]]
    for t in range(n_updates):
        model.get_layer("selector").set_weights([selector_kernel_dense(alloc[t], len(rules))])
        frames.append(to_states(step(frames[-1][:, :, None].astype(np.float32)).numpy()))
    assert np.array_equal(np.concatenate(frames), diagram)
