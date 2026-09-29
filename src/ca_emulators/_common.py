"""Building blocks shared by the ECA and nuCA emulators."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import tensorflow as tf

from .layers import PeriodicPadding1D
from .weights import detector_bias, detector_kernel


@dataclass(frozen=True)
class LayerModes:
    """Whether each part of the network is analytic (exact) and/or trainable."""

    detectors_analytic: bool
    detectors_trainable: bool
    rules_analytic: bool
    rules_trainable: bool


def resolve_modes(rules_known: bool, trainable, train_triplet_id) -> LayerModes:
    """Translate the constructor arguments into per-layer modes.

    By default, a network whose rules are known is exact and frozen, and a
    network without rules is randomly initialised and trainable.
    ``trainable=True`` keeps the analytic initialisation but makes every layer
    trainable. The 2024 argument ``train_triplet_id`` is still honoured for
    the detectors (True: random and trainable; False: analytic and frozen)
    when ``trainable`` is not given; an explicit ``trainable`` takes precedence.
    """
    rules_trainable = (not rules_known) if trainable is None else bool(trainable)
    if train_triplet_id is not None:
        warnings.warn("train_triplet_id is deprecated; exact emulators are now the default "
                      "and trainable=True makes every layer trainable",
                      DeprecationWarning, stacklevel=3)
        if trainable is None:
            return LayerModes(detectors_analytic=not train_triplet_id,
                              detectors_trainable=bool(train_triplet_id),
                              rules_analytic=rules_known, rules_trainable=rules_trainable)
    return LayerModes(detectors_analytic=rules_known, detectors_trainable=rules_trainable,
                      rules_analytic=rules_known, rules_trainable=rules_trainable)


def detector_layers(modes: LayerModes, kernel_initializer) -> list[tf.keras.layers.Layer]:
    """Periodic padding followed by the eight neighbourhood detectors."""
    return [
        PeriodicPadding1D(1, name="periodic_padding"),
        tf.keras.layers.Conv1D(
            8, 3, activation="relu", name="detectors",
            kernel_initializer=kernel_initializer, bias_initializer="zeros",
            trainable=modes.detectors_trainable),
    ]


def set_detector_weights(model: tf.keras.Model, omega: float) -> None:
    model.get_layer("detectors").set_weights([detector_kernel(omega), detector_bias()])


def unroll(inputs, step, timesteps: int, output_hidden: bool, activation):
    """Apply ``step`` ``timesteps`` times with shared weights (as in 2024).

    The output activation is applied after the last update only. With
    ``output_hidden`` the model returns ``[all_configs, outputs]``, where
    ``all_configs`` has shape (S, N, timesteps + 1) and starts with the input.
    """
    if int(timesteps) < 1:
        raise ValueError("timesteps must be at least 1")
    x = inputs
    frames = [inputs]
    for _ in range(int(timesteps) - 1):
        x = step(x)
        frames.append(x)
    x = step(x)
    outputs = tf.keras.layers.Activation(activation, name="output")(x)
    if not output_hidden:
        return outputs
    frames.append(outputs)
    all_configs = tf.keras.layers.Concatenate(axis=2, name="all_configs")(frames)
    return [all_configs, outputs]
