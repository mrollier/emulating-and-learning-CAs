"""C4: both nuCA emulators (Sec. 7.1.2) reproduce the nuCA exactly.

Checked against the numpy reference (and CellPyLib) for random rule sets and
allocations, for the locally connected selector in all three Keras
implementation modes and for the dense selector, up to one rule per cell
(N_R = N = 256, the first scenario of Tab. 7.2).
"""
import numpy as np
import pytest

from ca_emulators import NucaEmulator
from ca_emulators.reference import evolve, evolve_cellpylib, nuca_step
from ca_emulators.rules import certificate_configurations
from ca_emulators.simulate import compiled_step, spacetime, to_states
from ca_emulators.weights import selector_kernel_dense


def random_nuca(rng, n_cells, n_rules):
    rules = np.sort(rng.choice(256, size=n_rules, replace=False))
    alloc = rng.permutation(n_cells) % n_rules if n_rules <= n_cells else rng.integers(n_rules, size=n_cells)
    return rules, alloc


@pytest.mark.parametrize("n_cells,n_rules", [(8, 1), (16, 2), (32, 8), (64, 4), (40, 40)])
@pytest.mark.parametrize("variant", ["lc1", "lc2", "lc3", "dense"])
def test_random_nucas(rng, n_cells, n_rules, variant):
    rules, alloc = random_nuca(rng, n_cells, n_rules)
    implementation = int(variant[-1]) if variant.startswith("lc") else 1
    em = NucaEmulator(n_cells, rules, rule_alloc=alloc, implementation=implementation)
    model = em.model_dense() if variant == "dense" else em.model()
    x = rng.integers(2, size=(8, n_cells))
    assert np.array_equal(spacetime(model, x, 24), evolve(x, rules, 24, alloc))


@pytest.mark.parametrize("variant", ["lc", "dense"])
def test_certificate_configurations(rng, variant):
    """Eight rotations of the tiled de Bruijn sequence show every cell all neighbourhoods."""
    rules, alloc = random_nuca(rng, 32, 8)
    em = NucaEmulator(32, rules, rule_alloc=alloc)
    model = em.model_dense() if variant == "dense" else em.model()
    x = certificate_configurations(32)
    assert np.array_equal(spacetime(model, x, 1)[:, 1], nuca_step(x, rules, alloc))


def test_reference_equals_cellpylib_for_nucas(rng):
    for n_cells, n_rules in [(12, 3), (32, 8), (33, 33)]:
        rules, alloc = random_nuca(rng, n_cells, n_rules)
        x = rng.integers(2, size=(3, n_cells))
        assert np.array_equal(evolve(x, rules, 16, alloc), evolve_cellpylib(x, rules, 16, alloc))


def test_time_varying_allocation_by_swapping_the_selector(rng):
    """The emulators fix the allocation in time; swapping the selector weights
    between updates handles an allocation that varies in time (as in Fig. 4)."""
    rules, alloc0 = random_nuca(rng, 32, 8)
    alloc = np.stack([np.roll(alloc0, -t) for t in range(20)])
    model = NucaEmulator(32, rules, rule_alloc=alloc0).model_dense()
    step = compiled_step(model)
    x = rng.integers(2, size=(4, 32))
    frames = [x]
    for t in range(20):
        model.get_layer("selector").set_weights([selector_kernel_dense(alloc[t], 8)])
        frames.append(to_states(step(frames[-1][:, :, None].astype(np.float32)).numpy()))
    assert np.array_equal(np.stack(frames, axis=1), evolve(x, rules, 20, alloc))


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["lc1", "lc3", "dense"])
def test_one_rule_per_cell(rng, variant):
    """N_R = N = 256: every elementary rule allocated to exactly one cell."""
    rules, alloc = np.arange(256), rng.permutation(256)
    implementation = int(variant[-1]) if variant.startswith("lc") else 1
    em = NucaEmulator(256, rules, rule_alloc=alloc, implementation=implementation)
    model = em.model_dense() if variant == "dense" else em.model()
    x = rng.integers(2, size=(8, 256))
    assert np.array_equal(spacetime(model, x, 32), evolve(x, rules, 32, alloc))
