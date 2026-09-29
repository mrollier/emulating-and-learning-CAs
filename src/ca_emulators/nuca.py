"""Convolutional neural networks that emulate non-uniform cellular automata."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ._common import detector_layers, resolve_modes, set_detector_weights, unroll
from .reference import _check_alloc
from .rules import check_rules
from .weights import DEFAULT_OMEGA, rule_table_kernel, selector_kernel_dense, selector_kernel_lc


class NucaEmulator:
    """CNN whose forward pass performs global updates of a nuCA.

    A nuCA lets every cell follow its own elementary rule, chosen from
    ``rules`` by the allocation ``rule_alloc``. The network extends the ECA
    emulator (:class:`~ca_emulators.EcaEmulator`) in two places: the rule-table
    layer gets one output channel per rule, so it computes the N_R candidate
    updates of the whole configuration, and a third layer selects, per cell,
    the candidate of its allocated rule (thesis Sec. 7.1.2). Two versions of
    the selector are available:

    - :meth:`model`: a ``LocallyConnected1D`` layer (kernel size 1, one filter),
      a convolution whose weights differ between cells;
      (3 + 1) * 8 + 8 N_R + N_R N parameters (Eq. 7.1).
    - :meth:`model_dense`: a ``Dense`` layer on the flattened candidates, whose
      N_R N x N weight matrix is mostly zero;
      (3 + 1) * 8 + 8 N_R + N_R N^2 parameters (Eq. 7.2).

    With ``rules`` and ``rule_alloc`` given, every weight is analytic and
    frozen, and both networks reproduce the nuCA exactly. The allocation is
    fixed in time.

    Parameters
    ----------
    N : int
        Number of cells. The models take inputs of shape (S, N, 1).
    rules : sequence of int or None
        The N_R elementary rules. ``None`` gives a trainable rule-table layer
        with ``n_rules`` channels.
    timesteps, output_hidden, activation, trainable, omega, kernel_initializer
        As for :class:`~ca_emulators.EcaEmulator`.
    rule_alloc : sequence of int or None
        Index into ``rules`` for every cell (length N). ``None`` gives a
        trainable selector.
    n_rules : int or None
        Number of rule channels when ``rules`` is None.
    implementation : int
        Implementation mode of Keras's ``LocallyConnected1D`` (1, 2 or 3).
        Mode 1, the default and the one used in 2024, loops over the cells;
        mode 2 multiplies by a masked dense kernel; mode 3 by a sparse one.
    train_triplet_id : bool or None
        Deprecated 2024 argument, see :class:`~ca_emulators.EcaEmulator`.
    """

    def __init__(self, N: int, rules=None, timesteps: int = 1, output_hidden: bool = False, *,
                 rule_alloc=None, activation=None, trainable=None, n_rules=None,
                 implementation: int = 1, omega: float = DEFAULT_OMEGA,
                 kernel_initializer="he_normal", train_triplet_id=None):
        self.N = int(N)
        self.rules = None if rules is None else check_rules(rules)
        if self.rules is None and n_rules is None:
            raise ValueError("without rules, give the number of rule channels n_rules")
        self.n_rules = len(self.rules) if self.rules is not None else int(n_rules)
        self.rule_alloc = None if rule_alloc is None else _check_alloc(rule_alloc, self.N, self.n_rules)
        self.timesteps = int(timesteps)
        self.output_hidden = bool(output_hidden)
        self.activation = activation
        self.trainable = trainable
        self.implementation = int(implementation)
        self.omega = float(omega)
        self.kernel_initializer = kernel_initializer
        self.train_triplet_id = train_triplet_id

    def model(self) -> tf.keras.Model:
        """Build the network with a locally connected selector."""
        selector = tf.keras.layers.LocallyConnected1D(
            1, 1, activation="relu", name="selector", implementation=self.implementation,
            use_bias=self.rule_alloc is None, kernel_initializer=self.kernel_initializer,
            trainable=self._selector_trainable())
        model = self._build(lambda x: selector(x), "nuca_emulator_lc")
        if self.rule_alloc is not None:
            selector.set_weights([selector_kernel_lc(self.rule_alloc, self.n_rules,
                                                     self.implementation)])
        return model

    def model_dense(self) -> tf.keras.Model:
        """Build the network with a dense selector."""
        to_rule_major = tf.keras.layers.Permute((2, 1), name="candidates_by_rule")
        flatten = tf.keras.layers.Flatten(name="flatten")
        selector = tf.keras.layers.Dense(
            self.N, activation="relu", name="selector", use_bias=self.rule_alloc is None,
            kernel_initializer=self.kernel_initializer, trainable=self._selector_trainable())
        to_cells = tf.keras.layers.Reshape((self.N, 1), name="configuration_out")

        def select(x):
            return to_cells(selector(flatten(to_rule_major(x))))

        model = self._build(select, "nuca_emulator_dense")
        if self.rule_alloc is not None:
            selector.set_weights([selector_kernel_dense(self.rule_alloc, self.n_rules)])
        return model

    def simulate(self, x0, n_updates: int, variant: str = "dense",
                 method: str = "compiled") -> np.ndarray:
        """Spacetime diagram (..., n_updates + 1, N) of the exact emulator."""
        from .simulate import spacetime

        if self.rules is None or self.rule_alloc is None:
            raise ValueError("simulate needs rules and rule_alloc")
        one_step = NucaEmulator(self.N, self.rules, rule_alloc=self.rule_alloc,
                                implementation=self.implementation, omega=self.omega)
        model = one_step.model_dense() if variant == "dense" else one_step.model()
        return spacetime(model, x0, n_updates, method=method)

    def _selector_trainable(self) -> bool:
        if self.trainable is not None:
            return bool(self.trainable)
        return self.rule_alloc is None

    def _build(self, select, name: str) -> tf.keras.Model:
        modes = resolve_modes(self.rules is not None, self.trainable, self.train_triplet_id)
        padding, detectors = detector_layers(modes, self.kernel_initializer)
        rule_tables = tf.keras.layers.Conv1D(
            self.n_rules, 1, activation="relu", name="rule_tables",
            use_bias=not modes.rules_analytic, kernel_initializer=self.kernel_initializer,
            trainable=modes.rules_trainable)

        def step(x):
            return select(rule_tables(detectors(padding(x))))

        inputs = tf.keras.Input((self.N, 1), dtype=tf.float32, name="configuration")
        outputs = unroll(inputs, step, self.timesteps, self.output_hidden, self.activation)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        if modes.detectors_analytic:
            set_detector_weights(model, self.omega)
        if modes.rules_analytic:
            rule_tables.set_weights([rule_table_kernel(self.rules)])
        return model
