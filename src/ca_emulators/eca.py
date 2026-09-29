"""A convolutional neural network that emulates an elementary cellular automaton."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ._common import detector_layers, resolve_modes, set_detector_weights, unroll
from .rules import check_rule
from .weights import DEFAULT_OMEGA, rule_table_kernel


class EcaEmulator:
    """CNN whose forward pass performs global updates of an ECA.

    The network has three layers (thesis Tab. 7.1):

    1. periodic padding and a width-3 ``Conv1D`` with eight ReLU channels,
       the neighbourhood detectors (32 parameters);
    2. a width-1 ``Conv1D`` without bias whose eight weights are the rule
       table (8 parameters);
    3. an optional output activation (for training).

    With ``rule`` given, all 40 weights are set analytically and frozen, and
    the network reproduces the ECA exactly. Without a rule, the same
    architecture is initialised randomly and can be trained.

    Parameters
    ----------
    N : int
        Number of cells. The model takes inputs of shape (S, N, 1).
    rule : int or None
        Wolfram number of the rule. ``None`` gives a trainable network.
    timesteps : int
        Number of global updates in one forward pass (the layers are applied
        repeatedly with shared weights). Default 1.
    output_hidden : bool
        If True, the model also returns every intermediate configuration, as
        ``[all_configs, outputs]`` with ``all_configs`` of shape
        (S, N, timesteps + 1). Default False.
    activation : str, callable or None
        Activation applied to the final output. Default None.
    trainable : bool or None
        ``None`` (default): frozen if ``rule`` is given, trainable otherwise.
        ``True`` with a rule: analytic initialisation, but trainable.
    omega : float
        Weight of the detectors at positions that hold a 0; the detectors are
        exact for any omega >= 1. Default 5, as in 2024.
    kernel_initializer : str or initialiser
        Initialiser of the randomly initialised layers. Default ``"he_normal"``.
    train_triplet_id : bool or None
        Deprecated 2024 argument. True makes the detectors random and
        trainable, False makes them analytic and frozen.
    """

    def __init__(self, N: int, rule=None, timesteps: int = 1, output_hidden: bool = False, *,
                 activation=None, trainable=None, omega: float = DEFAULT_OMEGA,
                 kernel_initializer="he_normal", train_triplet_id=None):
        if isinstance(kernel_initializer, str) and kernel_initializer == "halfway":
            raise ValueError("the 2024 'halfway' initialiser was removed in 1.0 (it is still "
                             "available at the tag acri-2024)")
        self.N = int(N)
        self.rule = None if rule is None else check_rule(rule)
        self.timesteps = int(timesteps)
        self.output_hidden = bool(output_hidden)
        self.activation = activation
        self.trainable = trainable
        self.omega = float(omega)
        self.kernel_initializer = kernel_initializer
        self.train_triplet_id = train_triplet_id

    def model(self) -> tf.keras.Model:
        """Build a new Keras model (fresh weights for the random layers)."""
        modes = resolve_modes(self.rule is not None, self.trainable, self.train_triplet_id)
        padding, detectors = detector_layers(modes, self.kernel_initializer)
        rule_tables = tf.keras.layers.Conv1D(
            1, 1, activation="relu", name="rule_tables",
            use_bias=not modes.rules_analytic,
            kernel_initializer=self.kernel_initializer,
            trainable=modes.rules_trainable)

        def step(x):
            return rule_tables(detectors(padding(x)))

        inputs = tf.keras.Input((self.N, 1), dtype=tf.float32, name="configuration")
        outputs = unroll(inputs, step, self.timesteps, self.output_hidden, self.activation)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eca_emulator")
        if modes.detectors_analytic:
            set_detector_weights(model, self.omega)
        if modes.rules_analytic:
            rule_tables.set_weights([rule_table_kernel([self.rule])])
        return model

    def simulate(self, x0, n_updates: int, method: str = "compiled") -> np.ndarray:
        """Spacetime diagram (..., n_updates + 1, N) of the exact emulator."""
        from .simulate import spacetime

        if self.rule is None:
            raise ValueError("simulate needs a rule")
        one_step = EcaEmulator(self.N, self.rule, omega=self.omega).model()
        return spacetime(one_step, x0, n_updates, method=method)
