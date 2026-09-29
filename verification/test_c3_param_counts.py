"""C3: parameter counts of Sec. 7.1 (40 for an ECA; Eqs. 7.1 and 7.2 for nuCAs).

Eq. 7.1, (3 + 1) * 8 + 8 N_R + N_R N, counts the locally connected variant in
Keras's implementation modes 1 (the 2024 default) and 3. Mode 2 stores the
masked dense kernel and therefore has as many parameters as the dense variant,
Eq. 7.2: (3 + 1) * 8 + 8 N_R + N_R N^2.
"""
import numpy as np
import pytest

from ca_emulators import EcaEmulator, NucaEmulator
from ca_emulators.weights import parameter_count


def test_eca_has_40_parameters():
    assert parameter_count(32) == 40
    for n_cells in (8, 32, 100):
        assert EcaEmulator(n_cells, 54).model().count_params() == 40


def test_example_of_fig_7_4():
    """N = 32 cells and N_R = 8 rules: 352 and 8288 parameters."""
    em = NucaEmulator(32, range(8), rule_alloc=np.arange(32) % 8)
    assert em.model().count_params() == parameter_count(32, 8, "lc") == 352
    assert em.model_dense().count_params() == parameter_count(32, 8, "dense") == 8288


@pytest.mark.parametrize("n_cells,n_rules", [(8, 1), (16, 2), (64, 4), (100, 7)])
@pytest.mark.parametrize("implementation", [1, 2, 3])
def test_equations_7_1_and_7_2(n_cells, n_rules, implementation):
    alloc = np.arange(n_cells) % n_rules
    em = NucaEmulator(n_cells, range(n_rules), rule_alloc=alloc, implementation=implementation)
    variant = "dense" if implementation == 2 else "lc"
    assert em.model().count_params() == parameter_count(n_cells, n_rules, variant)
    assert em.model_dense().count_params() == parameter_count(n_cells, n_rules, "dense")


def test_counts_equal_the_2024_code(golden):
    em = NucaEmulator(32, golden["nuca_32_8_rules"], rule_alloc=golden["nuca_32_8_alloc"])
    ours = [EcaEmulator(32, 54).model().count_params(), em.model().count_params(),
            em.model_dense().count_params()]
    assert ours == list(golden["param_counts"])
