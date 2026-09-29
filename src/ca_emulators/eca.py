"""A convolutional neural network that emulates an elementary cellular automaton."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ._common import (check_architecture, detector_layers, resolve_modes, set_detector_weights,
                      unroll)
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
    N : int or None
        Number of cells. The model takes inputs of shape (S, N, 1). ``None``
        builds a model that accepts any number of cells.
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
    input_encoding : {"01", "pm1"}
        How the states reach the detectors: as 0/1 (default, the paper's
        network) or as -1/+1 through a rescaling layer named ``encoding``.
        Both are exact with the analytic weights; +-1 inputs train better
        (``ca_emulators.training.train_recipe``).
    detector_activation : str or callable
        Activation of the detectors. Default ``"relu"``, the only one for which
        the analytic weights are exact; e.g. ``"softplus"`` for training.
    rule_activation : str, callable or None
        Activation of the rule-table layer. Default ``"relu"``, as in 2024
        (harmless for the analytic 0/1 weights). ``None`` gives a linear
        output, e.g. the logit of a trained network; the ReLU is the main
        obstacle to training (see ``experiments/training``).
    kernel_initializer : str or initialiser
        Initialiser of the randomly initialised layers. Default ``"he_normal"``.
    train_triplet_id : bool or None
        Deprecated 2024 argument. True makes the detectors random and
        trainable, False makes them analytic and frozen.
    """

    def __init__(self, N, rule=None, timesteps: int = 1, output_hidden: bool = False, *,
                 activation=None, trainable=None, omega: float = DEFAULT_OMEGA,
                 input_encoding: str = "01", detector_activation="relu", rule_activation="relu",
                 kernel_initializer="he_normal", train_triplet_id=None):
        if isinstance(kernel_initializer, str) and kernel_initializer == "halfway":
            raise ValueError("the 2024 'halfway' initialiser was removed in 1.0 (it is still "
                             "available at the tag acri-2024)")
        self.N = None if N is None else int(N)
        self.rule = None if rule is None else check_rule(rule)
        self.timesteps = int(timesteps)
        self.output_hidden = bool(output_hidden)
        self.activation = activation
        self.trainable = trainable
        self.omega = float(omega)
        self.input_encoding = input_encoding
        self.detector_activation = detector_activation
        self.rule_activation = rule_activation
        self.kernel_initializer = kernel_initializer
        self.train_triplet_id = train_triplet_id

    def model(self) -> tf.keras.Model:
        """Build a new Keras model (fresh weights for the random layers)."""
        modes = resolve_modes(self.rule is not None, self.trainable, self.train_triplet_id)
        check_architecture(modes, self.input_encoding, self.detector_activation)
        front = detector_layers(modes, self.kernel_initializer, self.input_encoding,
                                self.detector_activation)
        rule_tables = tf.keras.layers.Conv1D(
            1, 1, activation=self.rule_activation, name="rule_tables",
            use_bias=not modes.rules_analytic,
            kernel_initializer=self.kernel_initializer,
            trainable=modes.rules_trainable)

        def step(x):
            for layer in front:
                x = layer(x)
            return rule_tables(x)

        inputs = tf.keras.Input((self.N, 1), dtype=tf.float32, name="configuration")
        outputs = unroll(inputs, step, self.timesteps, self.output_hidden, self.activation)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eca_emulator")
        if modes.detectors_analytic:
            set_detector_weights(model, self.omega, self.input_encoding)
        if modes.rules_analytic:
            rule_tables.set_weights([rule_table_kernel([self.rule])])
        return model

    def simulate(self, x0, n_updates: int, method: str = "compiled") -> np.ndarray:
        """Spacetime diagram (..., n_updates + 1, N) of the exact emulator."""
        from .simulate import spacetime

        if self.rule is None:
            raise ValueError("simulate needs a rule")
        n_cells = self.N if self.N is not None else np.shape(x0)[-1]
        one_step = EcaEmulator(n_cells, self.rule, omega=self.omega,
                               input_encoding=self.input_encoding).model()
        return spacetime(one_step, x0, n_updates, method=method)
