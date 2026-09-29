"""The simulation methods compared by the emulation benchmark.

Every method computes the complete spacetime diagram of S independent nuCAs,
all with the same N_R rules and the same static rule allocation, from S
initial configurations of N cells. A method is set up once per benchmark
case (for the neural emulators: build the network and set its analytic
weights) and returns a :class:`Prepared` pair:

``run()``
    The timed part. It returns the diagram as a numpy array in the method's
    own layout and dtype; for TensorFlow methods the result is copied back
    to numpy, so the work is guaranteed to be finished.
``to_states(result)``
    Untimed. Converts that result to uint8 states of shape (S, T, N), the
    layout of :func:`ca_emulators.reference.evolve`, raising if a network
    output is not exactly 0 or 1.

Methods are named ``"<driver>:<selector>"`` for the neural emulators and by a
single word for the reference implementations:

Reference implementations
    ``cellpylib``   CellPyLib, one ``cpl.evolve`` per sample, per-cell rule,
                    ``memoize=False`` (:func:`ca_emulators.reference.evolve_cellpylib`).
    ``numpy``       The package's vectorised reference, :func:`ca_emulators.reference.evolve`.
    ``numpy_lut``   A tighter numpy loop written for this benchmark: the
                    per-cell rule tables are looked up once, and every update
                    is one neighbourhood index plus one ``np.take``.

Drivers (how the T - 1 updates are run)
    ``predict``         ``model.predict`` once per update (the 2024 protocol,
                        here with ``batch_size=S``: one batch per call).
    ``eager``           ``model(x, training=False)`` once per update.
    ``compiled``        One ``tf.function`` per update (:func:`ca_emulators.simulate.compiled_step`).
    ``xla``             The same with ``jit_compile=True``.
    ``while_loop``      The whole diagram in one ``tf.function`` using ``tf.while_loop``.
    ``while_loop_xla``  The same with ``jit_compile=True``.
    ``unrolled``        A model with all T - 1 updates unrolled
                        (``NucaEmulator(..., timesteps=T - 1, output_hidden=True)``),
                        called once inside a ``tf.function``.

Selectors (the third layer of the nuCA emulator)
    ``lc1``, ``lc2``, ``lc3``  ``LocallyConnected1D`` with ``implementation`` 1, 2 or 3
                               (:meth:`ca_emulators.NucaEmulator.model`; 1 is the 2024 default).
    ``dense``                  Dense selector (:meth:`ca_emulators.NucaEmulator.model_dense`).
    ``elem``                   Elementwise selector, defined here: the N_R candidate
                               channels are multiplied by the one-hot allocation and
                               summed over the rules.

``xla:lc3`` and ``while_loop_xla:lc3`` do not exist: XLA cannot compile the
sparse matrix product of implementation 3 (``SparseTensorDenseMatMul``).

Conventions follow the 2024 benchmark scripts: T counts the rows of the
spacetime diagram (CellPyLib's ``timesteps``), so a case performs T - 1
global updates; the N_R rules are drawn without replacement from 0-255 and
sorted; the allocation and the initial states are uniform at random.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

#: Base seed of every benchmark case; the data of a case depend only on this
#: seed and the case's (N, N_R, T, S), not on the order in which cases run.
BASE_SEED = 20240912

DRIVERS = ("predict", "eager", "compiled", "xla", "while_loop", "while_loop_xla", "unrolled")
SELECTORS = ("lc1", "lc2", "lc3", "dense", "elem")
REFERENCE_METHODS = ("cellpylib", "numpy", "numpy_lut")
#: Driver/selector pairs that cannot run (XLA has no kernel for sparse matmul).
UNSUPPORTED = frozenset({("xla", "lc3"), ("while_loop_xla", "lc3")})


# --------------------------------------------------------------------------
# Benchmark cases
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """One benchmark point: S nuCAs of N cells, N_R rules, T rows (T - 1 updates)."""

    N: int
    n_rules: int
    T: int
    S: int
    seed: int = BASE_SEED

    def __post_init__(self):
        if self.N < 3:
            raise ValueError("N must be at least 3")
        if not 1 <= self.n_rules <= 256:
            raise ValueError("n_rules must lie in [1, 256]")
        if self.T < 2:
            raise ValueError("T counts rows of the diagram and must be at least 2")
        if self.S < 1:
            raise ValueError("S must be at least 1")

    @property
    def n_updates(self) -> int:
        return self.T - 1

    def data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rules, alloc, x0): sorted rules (N_R,), allocation (N,), states (S, N) uint8."""
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.N, self.n_rules, self.T, self.S]))
        rules = np.sort(rng.choice(256, size=self.n_rules, replace=False)).astype(np.int64)
        alloc = rng.integers(self.n_rules, size=self.N).astype(np.int64)
        x0 = rng.integers(2, size=(self.S, self.N), dtype=np.uint8)
        return rules, alloc, x0


@dataclass
class Prepared:
    """A method set up for one case (see the module docstring)."""

    run: Callable[[], Any]
    to_states: Callable[[Any], np.ndarray]
    info: dict = field(default_factory=dict)


def _binary(y) -> np.ndarray:
    """Float network output to uint8 states, refusing anything but exact 0/1."""
    y = np.asarray(y)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("network output is not exactly binary")
    return y.astype(np.uint8)


def _from_cells_last(y) -> np.ndarray:
    """(S, N, T) float -> (S, T, N) uint8."""
    return _binary(np.transpose(y, (0, 2, 1)))


def _from_time_major(y) -> np.ndarray:
    """(T, S, N, 1) float -> (S, T, N) uint8."""
    return _binary(np.transpose(y[..., 0], (1, 0, 2)))


# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------

def cellpylib(case: Case, rules, alloc, x0) -> Prepared:
    """CellPyLib, one call per sample, per-cell rule, no memoisation."""
    from ca_emulators import reference

    def run():
        return reference.evolve_cellpylib(x0, rules, case.n_updates, alloc)

    return Prepared(run, np.asarray)


def numpy_reference(case: Case, rules, alloc, x0) -> Prepared:
    """The package's vectorised numpy reference (ground truth of every check)."""
    from ca_emulators import reference

    def run():
        return reference.evolve(x0, rules, case.n_updates, alloc)

    return Prepared(run, np.asarray)


def numpy_lut(case: Case, rules, alloc, x0) -> Prepared:
    """Hand-vectorised numpy with a per-cell lookup table.

    ``lut[8 c + i]`` is the state that cell c's rule assigns to neighbourhood
    i = 4 s_- + 2 s_o + s_+; an update computes i for every cell of every
    sample and gathers from ``lut``. The diagram is stored time-major, so
    each update writes one contiguous (S, N) block.
    """
    from ca_emulators.rules import rule_table

    tables = np.stack([rule_table(r) for r in rules])            # (N_R, 8)
    lut = np.ascontiguousarray(tables[alloc]).reshape(-1)         # (8 N,)
    offsets = 8 * np.arange(case.N, dtype=np.intp)
    x_init = np.ascontiguousarray(x0, dtype=np.uint8)

    def run():
        out = np.empty((case.T, case.S, case.N), dtype=np.uint8)
        out[0] = x_init
        for t in range(1, case.T):
            x = out[t - 1]
            idx = (np.roll(x, 1, axis=1) << 2) | (x << 1) | np.roll(x, -1, axis=1)
            np.take(lut, offsets + idx, out=out[t], mode="clip")  # indices are in range
        return out

    return Prepared(run, lambda y: np.ascontiguousarray(np.transpose(y, (1, 0, 2))))


# --------------------------------------------------------------------------
# Selectors: Keras models of one update (or of all updates, unrolled)
# --------------------------------------------------------------------------

def _elementwise_selector_layer():
    import tensorflow as tf

    class ElementwiseSelector(tf.keras.layers.Layer):
        """Per cell, sum over rules of candidate x one-hot allocation (no ReLU needed)."""

        def __init__(self, one_hot, **kwargs):
            super().__init__(trainable=False, **kwargs)
            self._one_hot = np.asarray(one_hot, dtype=np.float32)  # (N, N_R)

        def build(self, input_shape):
            self.one_hot = self.add_weight(
                name="one_hot", shape=self._one_hot.shape, trainable=False,
                initializer=tf.constant_initializer(self._one_hot))
            super().build(input_shape)

        def call(self, candidates):  # (S, N, N_R) -> (S, N, 1)
            return tf.reduce_sum(candidates * self.one_hot, axis=-1, keepdims=True)

    return ElementwiseSelector


def elementwise_model(case: Case, rules, alloc, timesteps: int = 1, output_hidden: bool = False):
    """The nuCA emulator with an elementwise selector, built from plain Keras layers.

    Layers 1 and 2 (periodic padding, the eight neighbourhood detectors, the
    N_R rule tables) are those of :class:`ca_emulators.NucaEmulator`, with
    the package's analytic weights; layer 3 multiplies the N_R candidate
    channels by the one-hot allocation (N, N_R) and sums over the rules. The
    updates are unrolled like the package's ``unroll``: with ``output_hidden``
    the model returns ``[all_configs (S, N, timesteps + 1), outputs]``.
    """
    import tensorflow as tf

    from ca_emulators import weights
    from ca_emulators.layers import PeriodicPadding1D

    padding = PeriodicPadding1D(1, name="periodic_padding")
    detectors = tf.keras.layers.Conv1D(8, 3, activation="relu", name="detectors", trainable=False)
    rule_tables = tf.keras.layers.Conv1D(case.n_rules, 1, activation="relu", use_bias=False,
                                         name="rule_tables", trainable=False)
    selector = _elementwise_selector_layer()(weights.allocation_one_hot(alloc, case.n_rules),
                                             name="selector")

    inputs = tf.keras.Input((case.N, 1), dtype=tf.float32, name="configuration")
    x, frames = inputs, [inputs]
    for _ in range(int(timesteps)):
        x = selector(rule_tables(detectors(padding(x))))
        frames.append(x)
    if output_hidden:
        outputs = [tf.keras.layers.Concatenate(axis=2, name="all_configs")(frames), x]
    else:
        outputs = x
    model = tf.keras.Model(inputs, outputs, name="nuca_emulator_elementwise")
    detectors.set_weights([weights.detector_kernel(), weights.detector_bias()])
    rule_tables.set_weights([weights.rule_table_kernel(rules)])
    return model


def selector_model(case: Case, rules, alloc, selector: str, timesteps: int = 1,
                   output_hidden: bool = False):
    """Exact nuCA emulator (Keras model) with the given selector."""
    from ca_emulators import NucaEmulator

    if selector == "elem":
        return elementwise_model(case, rules, alloc, timesteps, output_hidden)
    implementation = {"lc1": 1, "lc2": 2, "lc3": 3, "dense": 1}[selector]
    emulator = NucaEmulator(case.N, rules, timesteps=timesteps, output_hidden=output_hidden,
                            rule_alloc=alloc, implementation=implementation)
    return emulator.model_dense() if selector == "dense" else emulator.model()


def selector_bytes(case: Case, selector: str) -> int:
    """Bytes of the selector's float32 kernel (what limits the size of a case).

    ``dense`` and ``lc2`` store N_R N^2 weights (Eq. 7.2 of the thesis;
    Keras's implementation 2 of ``LocallyConnected1D`` uses a masked dense
    kernel and keeps a mask of the same size); the others store N_R N.
    """
    if selector in ("dense", "lc2"):
        return 4 * case.n_rules * case.N ** 2
    return 4 * case.n_rules * case.N


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------

def _input(x0):
    import tensorflow as tf
    return tf.constant(np.asarray(x0, dtype=np.float32)[:, :, np.newaxis])


def predict(model, case: Case, x0) -> Prepared:
    """``model.predict`` once per update (2024 protocol; here one batch of S per call)."""
    x = np.asarray(x0, dtype=np.float32)[:, :, np.newaxis]

    def run():
        frames = [x]
        for _ in range(case.n_updates):
            frames.append(model.predict(frames[-1], batch_size=case.S, verbose=0))
        return np.concatenate(frames, axis=2)

    return Prepared(run, _from_cells_last)


def eager(model, case: Case, x0) -> Prepared:
    """``model(x, training=False)`` once per update, in eager mode."""
    import tensorflow as tf

    x = _input(x0)

    def run():
        frames = [x]
        for _ in range(case.n_updates):
            frames.append(model(frames[-1], training=False))
        return tf.concat(frames, axis=2).numpy()

    return Prepared(run, _from_cells_last)


def compiled(model, case: Case, x0, jit_compile: bool = False) -> Prepared:
    """One traced ``tf.function`` per update (optionally XLA-compiled), called T - 1 times."""
    import tensorflow as tf

    from ca_emulators.simulate import compiled_step

    step = compiled_step(model, jit_compile=jit_compile)
    x = _input(x0)

    def run():
        frames = [x]
        for _ in range(case.n_updates):
            frames.append(step(frames[-1]))
        return tf.concat(frames, axis=2).numpy()

    return Prepared(run, _from_cells_last)


def xla(model, case: Case, x0) -> Prepared:
    """:func:`compiled` with ``jit_compile=True``."""
    return compiled(model, case, x0, jit_compile=True)


def while_loop(model, case: Case, x0, jit_compile: bool = False) -> Prepared:
    """The whole diagram in one ``tf.function`` built around ``tf.while_loop``.

    Unlike :func:`ca_emulators.simulate.spacetime` (which builds a new
    function per call), the function is created once, so warm calls reuse
    its trace.
    """
    import tensorflow as tf

    n_updates = case.n_updates
    x = _input(x0)

    @tf.function(jit_compile=jit_compile)
    def diagram(z0):
        frames = tf.TensorArray(tf.float32, size=n_updates + 1, element_shape=z0.shape)
        frames = frames.write(0, z0)

        def body(t, z, frames):
            z = model(z, training=False)
            return t + 1, z, frames.write(t + 1, z)

        _, _, frames = tf.while_loop(lambda t, z, f: t < n_updates, body,
                                     (tf.constant(0), z0, frames))
        return frames.stack()  # (T, S, N, 1)

    return Prepared(lambda: diagram(x).numpy(), _from_time_major)


def while_loop_xla(model, case: Case, x0) -> Prepared:
    """:func:`while_loop` with ``jit_compile=True``."""
    return while_loop(model, case, x0, jit_compile=True)


def unrolled(model, case: Case, x0) -> Prepared:
    """One call of a model with all updates unrolled, inside a ``tf.function``."""
    import tensorflow as tf

    x = _input(x0)

    @tf.function
    def diagram(z0):
        return model(z0, training=False)[0]  # all_configs, (S, N, T)

    return Prepared(lambda: diagram(x).numpy(), _from_cells_last)


_DRIVER_FUNCTIONS = {"predict": predict, "eager": eager, "compiled": compiled, "xla": xla,
                     "while_loop": while_loop, "while_loop_xla": while_loop_xla,
                     "unrolled": unrolled}


def neural(driver: str, selector: str) -> Callable[[Case, Any, Any, Any], Prepared]:
    """Setup function of the method ``driver:selector``."""
    if (driver, selector) in UNSUPPORTED:
        raise ValueError(f"{driver}:{selector} is not supported (XLA cannot compile sparse matmul)")
    drive = _DRIVER_FUNCTIONS[driver]

    def setup(case: Case, rules, alloc, x0) -> Prepared:
        if driver == "unrolled":
            model = selector_model(case, rules, alloc, selector, timesteps=case.n_updates,
                                   output_hidden=True)
        else:
            model = selector_model(case, rules, alloc, selector)
        prepared = drive(model, case, x0)
        prepared.info["n_params"] = int(model.count_params())
        return prepared

    return setup


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Method:
    name: str
    setup: Callable[[Case, Any, Any, Any], Prepared]
    driver: str
    selector: str
    uses_tf: bool

    def kernel_bytes(self, case: Case) -> int:
        return selector_bytes(case, self.selector) if self.uses_tf else 0


def _registry() -> dict[str, Method]:
    methods = {
        "cellpylib": Method("cellpylib", cellpylib, "cellpylib", "", False),
        "numpy": Method("numpy", numpy_reference, "numpy", "", False),
        "numpy_lut": Method("numpy_lut", numpy_lut, "numpy_lut", "", False),
    }
    for driver in DRIVERS:
        for selector in SELECTORS:
            if (driver, selector) not in UNSUPPORTED:
                name = f"{driver}:{selector}"
                methods[name] = Method(name, neural(driver, selector), driver, selector, True)
    return methods


#: Every available method (3 reference implementations and 33 neural ones).
METHODS: dict[str, Method] = _registry()

#: The methods of a full run (see README.md): every driver with the two
#: selectors of 2024 (lc1, dense), every other selector compiled with and
#: without XLA, and the elementwise selector in one XLA-compiled while loop.
DEFAULT_METHODS: tuple[str, ...] = (
    *REFERENCE_METHODS,
    *(f"{d}:{s}" for s in ("lc1", "dense") for d in DRIVERS),
    "compiled:lc2", "xla:lc2", "compiled:lc3", "compiled:elem", "xla:elem",
    "while_loop_xla:elem",
)


def expected_diagram(case: Case, rules, alloc, x0) -> np.ndarray:
    """Ground truth: :func:`ca_emulators.reference.evolve`, shape (S, T, N) uint8."""
    from ca_emulators import reference
    return reference.evolve(x0, rules, case.n_updates, alloc)
