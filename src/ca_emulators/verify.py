"""Checks that a network with the ECA emulator architecture is exact.

A network with receptive field 3 sees a configuration only through the
neighbourhoods of its cells, so its one-step map is fixed by what it does on
the eight binary neighbourhoods (equivalently, on the de Bruijn configuration
00010111). :func:`is_exact` checks those eight outputs after thresholding.

Exactness after one update, with binarisation between updates, implies exact
emulation forever. Without binarisation a trained network can drift off the
binary states. :func:`interval_certificate` proves that it cannot: if every
real-valued neighbourhood within eps (max norm) of a binary neighbourhood is
mapped within eps of the correct next state, then by induction the
unbinarised iteration stays within eps of the exact trajectory, for every
configuration, of any size, forever. The bound is computed by interval
propagation through the monotone activations; it is sufficient, not
necessary. :func:`closed_loop_exact` tests the same on given configurations.

The functions read the weights of the layers ``encoding`` (optional),
``detectors``, ``rule_tables`` and ``output`` of a model built by
:class:`~ca_emulators.EcaEmulator` (trained or not). With ``logits=True`` the
network's output is a logit, and the state is its sigmoid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf

from .reference import evolve
from .rules import neighbourhood_patterns, rule_table

#: Radii tried by :func:`interval_certificate` (as in experiments/training).
CERT_EPS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)


@dataclass(frozen=True)
class _Network:
    scale: float           # input encoding: u = scale * s + offset
    offset: float
    kernel: np.ndarray     # (3, H)
    bias: np.ndarray       # (H,)
    detector_activation: Callable
    w_out: np.ndarray      # (H,)
    b_out: float
    head: Callable         # rule-table pre-activation -> predicted state


def _network(model: tf.keras.Model, logits: bool) -> _Network:
    names = {layer.name for layer in model.layers}
    if "selector" in names or "detectors" not in names or "rule_tables" not in names:
        raise ValueError("verify handles networks built by EcaEmulator only")
    detectors, rule_tables = model.get_layer("detectors"), model.get_layer("rule_tables")
    d_weights, r_weights = detectors.get_weights(), rule_tables.get_weights()
    if r_weights[0].shape[-1] != 1:
        raise ValueError("verify handles networks with a single rule-table output")
    scale, offset = 1.0, 0.0
    if "encoding" in names:
        encoding = model.get_layer("encoding")
        scale, offset = float(encoding.scale), float(encoding.offset)
    output = model.get_layer("output").activation if "output" in names else tf.identity

    def head(z):
        y = output(rule_tables.activation(tf.constant(z, tf.float32)))
        return (tf.sigmoid(y) if logits else y).numpy()

    return _Network(
        scale=scale, offset=offset, kernel=d_weights[0][:, 0, :],
        bias=d_weights[1] if len(d_weights) > 1 else np.zeros(d_weights[0].shape[-1], np.float32),
        detector_activation=detectors.activation,
        w_out=r_weights[0][0, :, 0], b_out=float(r_weights[1][0]) if len(r_weights) > 1 else 0.0,
        head=head)


def one_step_states(model: tf.keras.Model, logits: bool = False) -> np.ndarray:
    """The network's next state (before thresholding) for each of the 8 neighbourhoods."""
    net = _network(model, logits)
    inputs = net.scale * neighbourhood_patterns().astype(np.float32) + net.offset  # (8, 3)
    hidden = net.detector_activation(tf.constant(inputs @ net.kernel + net.bias)).numpy()
    return net.head(hidden @ net.w_out + net.b_out)


def is_exact(model: tf.keras.Model, rule: int, logits: bool = False) -> bool:
    """True if the thresholded one-step map equals the rule on all 8 neighbourhoods."""
    return bool(np.array_equal(one_step_states(model, logits) > 0.5, rule_table(rule) == 1))


def interval_certificate(model: tf.keras.Model, rule: int, logits: bool = False,
                         eps_grid=CERT_EPS) -> float:
    """Largest radius in ``eps_grid`` for which closed-loop exactness is proved (0 if none)."""
    net = _network(model, logits)
    targets = rule_table(rule) == 1
    centre = net.scale * neighbourhood_patterns().astype(np.float32) + net.offset
    act = net.detector_activation
    best = 0.0
    for eps in eps_grid:
        pre_c = centre @ net.kernel + net.bias
        pre_r = abs(net.scale) * eps * np.abs(net.kernel).sum(axis=0)
        lo = act(tf.constant(pre_c - pre_r)).numpy()
        hi = act(tf.constant(pre_c + pre_r)).numpy()
        z_c = (lo + hi) / 2 @ net.w_out + net.b_out
        z_r = (hi - lo) / 2 @ np.abs(net.w_out)
        y_lo, y_hi = net.head(z_c - z_r), net.head(z_c + z_r)
        ok = np.where(targets, (y_lo >= 1 - eps) & (y_hi <= 1 + eps), (y_hi <= eps) & (y_lo >= -eps))
        if ok.all():
            best = float(eps)
    return best


def closed_loop_exact(model: tf.keras.Model, rule: int, x0, n_updates: int,
                      logits: bool = False) -> bool:
    """Iterate the network without binarisation and compare, after thresholding,
    every configuration with the exact diagram."""
    x = np.asarray(x0, dtype=np.float32)
    x = x[np.newaxis] if x.ndim == 1 else x
    expected = evolve(x.astype(np.uint8), rule, n_updates)
    state = x[:, :, np.newaxis]
    for t in range(1, n_updates + 1):
        y = model(state, training=False)
        state = (tf.sigmoid(y) if logits else y).numpy()
        if not np.array_equal(state[:, :, 0] > 0.5, expected[:, t] == 1):
            return False
    return True
