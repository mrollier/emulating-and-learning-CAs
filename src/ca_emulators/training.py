"""The 2024 training recipe behind Fig. 3 (thesis Fig. 7.3).

The published figure illustrates that the emulator architecture can also be
*trained*: starting from random weights, a network learns one update of rule
54 from pairs of configurations. This module reproduces that recipe
faithfully (``scripts/eca_optimisation.py`` at the tag ``acri-2024``), with
seeds. Better ways to train emulators are studied in ``experiments/training``.

The recipe:

1. Draw ``n_train`` and ``n_val`` random configurations of N cells; the
   targets are one update by the exact emulator.
2. Build the emulator architecture without a rule (random, trainable layers:
   He-normal kernels, zero-initialised detector biases, a biased rule-table
   layer) with a tanh output.
3. Pretraining: re-initialise and train for one epoch (batch size 128) until
   the training loss drops below ``pretrain_threshold``; keep the best network.
   The 2024 loop had no cap; here at most ``max_restarts`` networks are tried.
4. Train the best network for ``epochs`` epochs (batch size 64, Adam with
   learning rate 0.005, mean squared error), storing the weights after every
   epoch.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from .eca import EcaEmulator


class WeightsHistory(tf.keras.callbacks.Callback):
    """Store a copy of all model weights at the end of every epoch."""

    def __init__(self):
        super().__init__()
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([np.array(w) for w in self.model.get_weights()])


def fit(model: tf.keras.Model, x, y, x_val, y_val, *, batch_size: int, epochs: int,
        learning_rate: float, loss: str = "mse", callbacks=None, verbose: int = 0):
    """Compile with a fresh Adam optimiser and fit, as the 2024 ``Train1D`` did."""
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["mse"])
    return model.fit(x, np.asarray(y, dtype=np.float32), batch_size=batch_size, epochs=epochs,
                     verbose=verbose, shuffle=True, callbacks=callbacks or [],
                     validation_data=(x_val, np.asarray(y_val, dtype=np.float32)))


@dataclass
class TrainingRun:
    """Everything the Fig. 3 plot needs, plus the bookkeeping of the pretraining."""

    model: tf.keras.Model
    history: tf.keras.callbacks.History
    weights_history: list
    x_train: np.ndarray
    y_train: np.ndarray
    pretrain_losses: list = field(default_factory=list)

    @property
    def n_restarts(self) -> int:
        return len(self.pretrain_losses)


def train_2024_recipe(rule: int = 54, N: int = 32, *, seed: int = 0, n_train: int = 4096,
                      n_val: int = 4096, epochs: int = 40, batch_size: int = 64,
                      learning_rate: float = 0.005, pretrain_batch_size: int = 128,
                      pretrain_threshold: float = 0.1, max_restarts: int = 200,
                      example=None, verbose: int = 0) -> TrainingRun:
    """Train an ECA emulator from random weights with the 2024 recipe.

    ``example`` (a configuration of N cells) is placed first in the training
    set; Fig. 3 shows the network's output on it. The run is deterministic for
    a given ``seed`` when TensorFlow's op determinism is enabled.
    """
    if max_restarts < 1:
        raise ValueError("max_restarts must be at least 1")
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    x_train = rng.integers(2, size=(n_train, N, 1)).astype(np.int8)
    x_val = rng.integers(2, size=(n_val, N, 1)).astype(np.int8)
    if example is not None:
        x_train[0, :, 0] = np.asarray(example, dtype=np.int8)

    exact = EcaEmulator(N, rule).model()
    y_train = exact.predict(x_train.astype(np.float32), batch_size=n_train, verbose=0)
    y_val = exact.predict(x_val.astype(np.float32), batch_size=n_val, verbose=0)

    trainable = EcaEmulator(N, rule=None, activation="tanh")
    best_loss, best_model, losses = np.inf, None, []
    while best_loss > pretrain_threshold and len(losses) < max_restarts:
        candidate = trainable.model()
        history = fit(candidate, x_train, y_train, x_val, y_val, batch_size=pretrain_batch_size,
                      epochs=1, learning_rate=learning_rate)
        losses.append(float(history.history["loss"][0]))
        if losses[-1] < best_loss:
            best_loss, best_model = losses[-1], candidate
        if verbose:
            print(f"pretraining {len(losses)}: loss {losses[-1]:.4f} (best {best_loss:.4f})")
    if best_loss > pretrain_threshold:
        warnings.warn(f"no pretraining run reached a loss below {pretrain_threshold} in "
                      f"{max_restarts} restarts (best {best_loss:.4f}); training the best one, "
                      "which the 2024 recipe would not have done", RuntimeWarning, stacklevel=2)

    record = WeightsHistory()
    history = fit(best_model, x_train, y_train, x_val, y_val, batch_size=batch_size,
                  epochs=epochs, learning_rate=learning_rate, callbacks=[record], verbose=verbose)
    return TrainingRun(best_model, history, record.weights, x_train, y_train, losses)
