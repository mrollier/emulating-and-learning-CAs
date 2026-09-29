"""C1: the eight neighbourhood detectors of Tab. 7.1 give an exact one-hot encoding.

Thesis Sec. 7.1.1 and Tab. 7.1: channel i has weights +1 where neighbourhood i
holds a 1 and -omega where it holds a 0, and bias 1 - h; after the ReLU the
channels are one-hot for any omega >= 1. The index of the active channel is
the (4, 2, 1) integer encoding of the paper.
"""
import numpy as np
import pytest
import tensorflow as tf

from ca_emulators import EcaEmulator
from ca_emulators.reference import neighbourhood_index
from ca_emulators.rules import DE_BRUIJN, neighbourhood_patterns
from ca_emulators.weights import detector_bias, detector_kernel

W = -1.0  # placeholder for -omega in the table below

# Tab. 7.1 of the thesis: (s-, so, s+), weights, bias 1 - h, rule-54 column.
TAB_7_1 = [
    ((0, 0, 0), (W, W, W), 1, 0),
    ((0, 0, 1), (W, W, 1), 0, 1),
    ((0, 1, 0), (W, 1, W), 0, 1),
    ((0, 1, 1), (W, 1, 1), -1, 0),
    ((1, 0, 0), (1, W, W), 0, 1),
    ((1, 0, 1), (1, W, 1), -1, 1),
    ((1, 1, 0), (1, 1, W), -1, 0),
    ((1, 1, 1), (1, 1, 1), -2, 0),
]


def detector_activations(model, x):
    sub = tf.keras.Model(model.input, model.get_layer("detectors").output)
    return sub(np.asarray(x, dtype=np.float32)[:, :, np.newaxis]).numpy()


@pytest.mark.parametrize("omega", [1.0, 5.0])
def test_weights_match_table_7_1(omega):
    kernel, bias = detector_kernel(omega), detector_bias()
    assert kernel.shape == (3, 1, 8) and bias.shape == (8,)
    for i, (pattern, weights, b, _) in enumerate(TAB_7_1):
        assert tuple(neighbourhood_patterns()[i]) == pattern
        expected = [omega * w if w < 0 else w for w in weights]
        assert np.array_equal(kernel[:, 0, i], np.array(expected, dtype=np.float32))
        assert bias[i] == b


def test_rule_54_column_of_table_7_1():
    model = EcaEmulator(8, 54).model()
    column = model.get_layer("rule_tables").get_weights()[0][0, :, 0]
    assert np.array_equal(column, [row[3] for row in TAB_7_1])


@pytest.mark.parametrize("omega", [1.0, 1.5, 5.0, 100.0])
def test_one_hot_for_every_neighbourhood(omega):
    """On the de Bruijn configuration every cell sees a different neighbourhood."""
    model = EcaEmulator(8, 54, omega=omega).model()
    act = detector_activations(model, DE_BRUIJN[np.newaxis])[0]  # (8 cells, 8 channels)
    expected = np.eye(8)[neighbourhood_index(DE_BRUIJN)]
    assert np.array_equal(act, expected)
    # the active channel of every cell is its (4, 2, 1) integer encoding
    assert sorted(np.argmax(act, axis=1)) == list(range(8))


def test_omega_below_one_is_not_exact():
    """For omega < 1 some detector also fires on a wrong neighbourhood."""
    model = EcaEmulator(8, 54, omega=0.5).model()
    act = detector_activations(model, DE_BRUIJN[np.newaxis])[0]
    assert not np.array_equal(act, np.eye(8)[neighbourhood_index(DE_BRUIJN)])
    assert np.any((act > 0) & (act < 1))


def test_hamming_view_at_omega_one():
    """At omega = 1 the pre-activation equals 1 - Hamming distance to the pattern."""
    patterns = neighbourhood_patterns().astype(float)
    kernel, bias = detector_kernel(1.0)[:, 0, :], detector_bias()
    pre = patterns @ kernel + bias  # rows: input neighbourhood, columns: channel
    hamming = np.abs(patterns[:, None, :] - patterns[None, :, :]).sum(-1)
    assert np.array_equal(pre, 1 - hamming)
