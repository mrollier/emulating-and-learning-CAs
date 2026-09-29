"""Custom Keras layer: periodic padding for 1-D convolutions."""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="ca_emulators")
class PeriodicPadding1D(tf.keras.layers.Layer):
    """Pad a (batch, N, channels) tensor periodically along the cell axis.

    Followed by a standard ``Conv1D`` with ``padding="valid"``, this gives a
    convolution with periodic boundary conditions: cell i sees cells i - p to
    i + p modulo N. ``padding`` must not exceed N.
    """

    def __init__(self, padding: int = 1, **kwargs):
        super().__init__(**kwargs)
        if int(padding) < 0:
            raise ValueError("padding must be non-negative")
        self.padding = int(padding)

    def call(self, inputs):
        p = self.padding
        if p == 0:
            return inputs
        return tf.concat([inputs[:, -p:, :], inputs, inputs[:, :p, :]], axis=1)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if input_shape[1] is not None:
            input_shape[1] += 2 * self.padding
        return tf.TensorShape(input_shape)

    def get_config(self):
        return {**super().get_config(), "padding": self.padding}
