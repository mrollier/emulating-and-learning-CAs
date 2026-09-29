"""Running a one-step emulator over many updates.

An emulator built with ``timesteps=1`` performs a single global update. A
spacetime diagram of T updates feeds its output back T times; since every
update needs the complete previous configuration, these passes are
sequential. This module offers several ways to do so, which give identical
states but differ greatly in overhead:

``"predict"``
    ``model.predict`` once per update: the protocol of the 2024 benchmarks
    (Fig. 5). Every call sets up Keras's data pipeline anew.
``"eager"``
    ``model(x, training=False)`` once per update.
``"compiled"``
    One ``tf.function`` for the update, traced once and called T times.
``"while_loop"``
    The whole diagram in a single ``tf.function`` using ``tf.while_loop``.

Alternatively, ``EcaEmulator(N, rule, timesteps=T, output_hidden=True)``
unrolls all T updates into one model.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

METHODS = ("predict", "eager", "compiled", "while_loop")


def as_input(x) -> np.ndarray:
    """Return a configuration or batch as a float32 array of shape (S, N, 1)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis]
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError("expected a configuration (N,), a batch (S, N) or (S, N, 1)")
    return x.astype(np.float32)


def to_states(y, threshold: float | None = None) -> np.ndarray:
    """Convert network output (S, N, 1) or (S, N) to uint8 states (S, N).

    Without a threshold the output must be exactly 0 or 1, which is what an
    exact emulator produces; with one, values above it become 1.
    """
    y = np.asarray(y)
    if y.ndim == 3:
        y = y[:, :, 0]
    if threshold is not None:
        return (y > threshold).astype(np.uint8)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("the network output is not exactly binary; pass a threshold")
    return y.astype(np.uint8)


def _check_one_step(model: tf.keras.Model) -> None:
    if isinstance(model.output, (list, tuple)) or len(model.output_shape) != 3:
        raise ValueError("spacetime needs a one-step model (timesteps=1, output_hidden=False)")


def compiled_step(model: tf.keras.Model, jit_compile: bool = False):
    """A traced ``tf.function`` performing one global update."""
    @tf.function(jit_compile=jit_compile, reduce_retracing=True)
    def step(x):
        return model(x, training=False)
    return step


def spacetime(model: tf.keras.Model, x0, n_updates: int, method: str = "compiled",
              threshold: float | None = None) -> np.ndarray:
    """Spacetime diagram of ``n_updates`` updates, shape (..., n_updates + 1, N).

    ``x0`` is a configuration (N,) or a batch (S, N); the output has the same
    leading shape. With ``threshold`` (for trained, inexact networks) the
    output of every update is binarised before it is fed back.
    """
    _check_one_step(model)
    squeeze = np.ndim(x0) == 1
    x = as_input(x0)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    def binarise(y):
        return y if threshold is None else tf.cast(y > threshold, tf.float32)

    if method == "while_loop":
        frames = _while_loop_diagram(model, tf.constant(x), int(n_updates), threshold).numpy()
    else:
        frames = [x]
        if method == "compiled":
            call = compiled_step(model)
        elif method == "eager":
            def call(z):
                return model(z, training=False)
        else:
            def call(z):
                return model.predict(z, verbose=0)
        for _ in range(int(n_updates)):
            frames.append(np.asarray(binarise(call(frames[-1]))))
        frames = np.stack(frames, axis=1)[..., 0]  # (S, T + 1, N)
    states = np.stack([to_states(f, threshold) for f in np.moveaxis(frames, 1, 0)], axis=1)
    return states[0] if squeeze else states


def _while_loop_diagram(model, x, n_updates: int, threshold):
    @tf.function
    def run(x):
        frames = tf.TensorArray(tf.float32, size=n_updates + 1)
        frames = frames.write(0, x)

        def body(t, z, frames):
            z = model(z, training=False)
            if threshold is not None:
                z = tf.cast(z > threshold, tf.float32)
            return t + 1, z, frames.write(t + 1, z)

        _, _, frames = tf.while_loop(lambda t, z, f: t < n_updates, body, (0, x, frames))
        return tf.transpose(frames.stack()[..., 0], (1, 0, 2))  # (S, T + 1, N)

    return run(x)


def predict_loop(model: tf.keras.Model, x0, n_updates: int) -> np.ndarray:
    """The 2024 benchmark protocol (``model.predict`` per update)."""
    return spacetime(model, x0, n_updates, method="predict")
