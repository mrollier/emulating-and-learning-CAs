"""C6: the statements of Sec. 7.1.4 hold for the archived 2024 timing data.

Thesis Sec. 7.1.4 (Fig. 7.5): the dense CNN's time is dominated by a fixed
cost and nearly independent of N, of N_S (up to a few hundred samples) and of
N_R (up to a few dozen rules); the locally connected CNN grows with N;
CellPyLib grows linearly with N and N_S and does not depend on N_R; all three
scale linearly with T with comparable slopes; the dense CNN is the fastest
from roughly a hundred cells, and from roughly a hundred samples, onwards.

The data show the second crossover between S = 64 and S = 128 samples, a
little earlier than the thesis's "a few hundred samples"; the test encodes
the data, and docs/provenance.md records the difference.
"""
import numpy as np
import pytest

from conftest import DATA

FILES = {
    "Nrules": "nuca-comparison-Nrules-N256_T32_S32_avg-from-10.npy",
    "T": "nuca-comparison-T-N64_Nrules4_S32_avg-from-10.npy",
    "N": "nuca-comparison-N-T32_Nrules4_S32_avg-from-10.npy",
    "S": "nuca-comparison-S-N32_T32_Nrules4_avg-from-10.npy",
}


def load(scenario):
    with open(DATA / "benchmarks_2024" / FILES[scenario], "rb") as f:
        x = np.load(f)
        cpl, lc, dense = (np.load(f).mean(axis=0) for _ in range(3))
    return x, cpl, lc, dense


def r_squared(x, y):
    fit = np.polyval(np.polyfit(x, y, 1), x)
    return 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)


def test_shapes_follow_table_7_2():
    expected = {"Nrules": [1, 2, 4, 8, 16, 32, 64, 128, 256], "T": list(range(10, 101, 10)),
                "N": list(range(32, 257, 32)), "S": [2**k for k in range(11)]}
    for scenario, xs in expected.items():
        x, *times = load(scenario)
        assert list(x) == xs
        with open(DATA / "benchmarks_2024" / FILES[scenario], "rb") as f:
            np.load(f)
            assert all(np.load(f).shape == (10, len(xs)) for _ in range(3))


def test_dense_cnn_is_nearly_constant():
    x, _, _, dense = load("N")
    assert dense.max() / dense.min() < 1.2
    x, _, _, dense = load("S")
    assert dense[x <= 256].max() / dense[x <= 256].min() < 1.4
    x, _, _, dense = load("Nrules")
    assert dense[x <= 32].max() / dense[x <= 32].min() < 1.2


def test_locally_connected_cnn_grows_with_n():
    _, _, lc, _ = load("N")
    assert np.all(np.diff(lc) > 0) and lc[-1] / lc[0] > 1.5


def test_cellpylib_linear_in_n_and_s_and_flat_in_rules():
    for scenario in ("N", "S"):
        x, cpl, _, _ = load(scenario)
        assert r_squared(x, cpl) > 0.99
    _, cpl, _, _ = load("Nrules")
    assert cpl.max() / cpl.min() < 1.05


def test_all_methods_linear_in_t_with_comparable_slopes():
    x, *times = load("T")
    slopes = [np.polyfit(x, t, 1)[0] for t in times]
    assert all(r_squared(x, t) > 0.99 for t in times)
    assert max(slopes) / min(slopes) < 1.5


@pytest.mark.parametrize("scenario,crossover", [("N", (64, 96)), ("S", (64, 128))])
def test_dense_cnn_fastest_beyond_crossover(scenario, crossover):
    below, above = crossover
    x, cpl, lc, dense = load(scenario)
    assert np.all(cpl[x <= below] < dense[x <= below])  # CellPyLib faster below
    assert np.all(dense[x >= above] < cpl[x >= above])  # dense CNN faster above
    assert np.all(dense[x >= above] <= lc[x >= above])


def test_dense_cnn_fastest_for_every_number_of_rules():
    _, cpl, lc, dense = load("Nrules")
    assert np.all(dense < lc) and np.all(lc < cpl)
