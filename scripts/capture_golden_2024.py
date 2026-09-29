"""Capture reference ("golden") outputs of the 2024 emulator code.

This script is run once, in the environment of the ACRI 2024 paper
(Python 3.11.8, TensorFlow 2.14.0, Keras 2.14.0, CellPyLib 2.4.0), against a
copy of the ``src/`` package as it was at the tag ``acri-2024``::

    git archive acri-2024 src | tar -x -C <legacy-root> --exclude='*__pycache__*'
    <paper-env>/python.exe -B scripts/capture_golden_2024.py --legacy-root <legacy-root>

It stores the *inputs* together with the outputs, so the refactored package can
be checked against them without re-drawing random numbers
(``verification/test_golden_2024.py``). Before anything is written, every CNN
output is checked against CellPyLib, so the files also document that the 2024
code was exact.

Contents of ``data/golden/golden_2024.npz`` (states stored as uint8):

- ``eca_x`` (64, 32): seeded random configurations, the first row being the
  de Bruijn configuration 00010111 tiled four times.
- ``eca_y`` (256, 64, 32): one global update for every rule 0-255.
- ``eca_multi_rules`` (16,), ``eca_multi_unrolled`` (16, 64, 33, 32) and
  ``eca_multi_predict`` (16, 64, 33, 32): 32 updates, once with the unrolled
  model (``timesteps=32, output_hidden=True``) and once by feeding the output
  of a one-step model back 32 times through ``predict`` (the protocol of the
  2024 benchmarks). Axis 2 is time, including the initial configuration.
- ``nuca_<N>_<NR>_{rules,alloc,x,lc,dense}``: non-uniform cases
  (N, N_R) in {(32, 8), (64, 4), (256, 256)}; ``lc`` and ``dense`` hold 32
  updates, shape (S, 33, N), for the locally connected and dense variants.
- ``tab71_kernel`` (3, 1, 8), ``tab71_bias`` (8,), ``tab71_rule54`` (1, 8, 1):
  the analytic weights of Tab. 7.1 as produced by the 2024 initialisers.
- ``param_counts``: [ECA, nuCA LC (32, 8), nuCA dense (32, 8)].
- ``default_trainable``: number of trainable weights of
  ``EcaEmulator(32, rule=54).model()`` with the 2024 default
  ``train_triplet_id=True`` (non-zero: the default emulator is not exact).

A plain-text ``ENV.txt`` records the versions used.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

SEED = 20240424  # date of the acri-2024 commit
DE_BRUIJN = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)


def cellpylib_evolve(x0: np.ndarray, rule_of_cell, n_updates: int) -> np.ndarray:
    """Evolve each row of ``x0`` with CellPyLib; returns (S, n_updates + 1, N)."""
    import cellpylib as cpl

    out = []
    for row in x0:
        diagram = cpl.evolve(
            row[np.newaxis, :].astype(int),
            timesteps=n_updates + 1,  # CellPyLib counts the initial row
            apply_rule=lambda n, c, t: cpl.nks_rule(n, rule_of_cell(c)),
            memoize=False,  # memoisation is wrong for cell-dependent rules
        )
        out.append(diagram)
    return np.asarray(out, dtype=np.uint8)


def as_states(y) -> np.ndarray:
    """Convert a float CNN output to uint8, asserting it is exactly binary."""
    y = np.asarray(y)
    if not np.all((y == 0) | (y == 1)):
        raise AssertionError("CNN output is not exactly binary")
    return y.astype(np.uint8)


def predict_loop(model, x0: np.ndarray, n_updates: int) -> np.ndarray:
    """The 2024 benchmark protocol: feed a one-step model back via predict."""
    frames = [x0.astype(np.float32)]
    for _ in range(n_updates):
        y = model.predict(frames[-1][:, :, np.newaxis], verbose=0)
        frames.append(np.asarray(y)[:, :, 0])
    return as_states(np.stack(frames, axis=1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--legacy-root", required=True, type=Path,
                   help="folder containing the acri-2024 copy of src/")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "golden")
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.legacy_root.resolve()))
    from importlib.metadata import version

    import keras
    import tensorflow as tf
    from src.nn.eca import EcaEmulator
    from src.nn.nuca import NucaEmulator
    from src.custom_tf_classes.initializers import (
        BiasesTripletFinder, WeightsLocalUpdate, WeightsTripletFinder)

    t_start = time.time()
    rng = np.random.default_rng(SEED)
    store: dict[str, np.ndarray] = {}

    # --- Tab. 7.1 and parameter counts -----------------------------------
    store["tab71_kernel"] = WeightsTripletFinder()(None).numpy()
    store["tab71_bias"] = BiasesTripletFinder()(None).numpy()
    store["tab71_rule54"] = WeightsLocalUpdate([54])(None).numpy()

    n_rules, n_cells = 8, 32
    alloc = rng.integers(n_rules, size=n_cells)
    rules = np.sort(rng.choice(256, size=n_rules, replace=False))
    nuca = NucaEmulator(n_cells, rules=rules, rule_alloc=alloc, train_triplet_id=False)
    store["param_counts"] = np.array([
        EcaEmulator(32, rule=54, train_triplet_id=False).model().count_params(),
        nuca.model().count_params(),
        nuca.model_dense().count_params(),
    ])
    default_model = EcaEmulator(32, rule=54).model()
    store["default_trainable"] = np.array(
        sum(int(np.prod(w.shape)) for w in default_model.trainable_weights))
    print("parameter counts", store["param_counts"],
          "| trainable weights of the default emulator", store["default_trainable"])

    # --- ECAs: one update for all 256 rules --------------------------------
    n_samples = 64
    eca_x = rng.integers(2, size=(n_samples, 32)).astype(np.uint8)
    eca_x[0] = np.tile(DE_BRUIJN, 4)
    store["eca_x"] = eca_x
    eca_y = np.empty((256, n_samples, 32), dtype=np.uint8)
    for rule in range(256):
        model = EcaEmulator(32, rule=rule, train_triplet_id=False).model()
        y = model.predict(eca_x[:, :, np.newaxis].astype(np.float32), verbose=0)
        eca_y[rule] = as_states(y)[:, :, 0]
        reference = cellpylib_evolve(eca_x, lambda c, r=rule: r, 1)[:, 1]
        assert np.array_equal(eca_y[rule], reference), f"rule {rule} differs from CellPyLib"
        print(f"ECA rule {rule:3d} matches CellPyLib", end="\r")
    print()
    store["eca_y"] = eca_y

    # --- ECAs: 32 updates, unrolled and via the predict loop ---------------
    n_updates = 32
    multi_rules = np.array([0, 1, 18, 22, 30, 45, 54, 60, 73, 90, 105, 110, 126, 150, 184, 255])
    unrolled = np.empty((len(multi_rules), n_samples, n_updates + 1, 32), dtype=np.uint8)
    looped = np.empty_like(unrolled)
    for i, rule in enumerate(multi_rules):
        model = EcaEmulator(32, rule=int(rule), timesteps=n_updates, output_hidden=True,
                            train_triplet_id=False).model()
        all_configs, _ = model.predict(eca_x[:, :, np.newaxis].astype(np.float32), verbose=0)
        unrolled[i] = np.transpose(as_states(all_configs), (0, 2, 1))
        one_step = EcaEmulator(32, rule=int(rule), train_triplet_id=False).model()
        looped[i] = predict_loop(one_step, eca_x, n_updates)
        reference = cellpylib_evolve(eca_x, lambda c, r=int(rule): r, n_updates)
        assert np.array_equal(unrolled[i], reference), f"unrolled rule {rule} differs"
        assert np.array_equal(looped[i], reference), f"looped rule {rule} differs"
        print(f"32 updates of rule {rule:3d} match CellPyLib", end="\r")
    print()
    store["eca_multi_rules"] = multi_rules
    store["eca_multi_unrolled"] = unrolled
    store["eca_multi_predict"] = looped

    # --- nuCAs: both variants, 32 updates ----------------------------------
    for n_cells, n_rules, n_samples_nu in [(32, 8, 16), (64, 4, 16), (256, 256, 8)]:
        rules = np.sort(rng.choice(256, size=n_rules, replace=False))
        alloc = rng.integers(n_rules, size=n_cells)
        if n_rules == n_cells:  # every rule allocated exactly once
            alloc = rng.permutation(n_cells)
        x = rng.integers(2, size=(n_samples_nu, n_cells)).astype(np.uint8)
        emulator = NucaEmulator(n_cells, rules=rules, rule_alloc=alloc, train_triplet_id=False)
        lc = predict_loop(emulator.model(), x, n_updates)
        dense = predict_loop(emulator.model_dense(), x, n_updates)
        reference = cellpylib_evolve(x, lambda c: int(rules[alloc[c]]), n_updates)
        assert np.array_equal(lc, reference), f"LC nuCA ({n_cells}, {n_rules}) differs"
        assert np.array_equal(dense, reference), f"dense nuCA ({n_cells}, {n_rules}) differs"
        key = f"nuca_{n_cells}_{n_rules}"
        store.update({f"{key}_rules": rules, f"{key}_alloc": alloc, f"{key}_x": x,
                      f"{key}_lc": lc, f"{key}_dense": dense})
        print(f"nuCA N={n_cells}, N_R={n_rules}: both variants match CellPyLib over {n_updates} updates")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "golden_2024.npz", **store)
    env = [
        "Golden outputs of the acri-2024 emulator code (scripts/capture_golden_2024.py)",
        f"captured: {time.strftime('%Y-%m-%d %H:%M:%S')} in {time.time() - t_start:.0f} s",
        f"seed: {SEED}",
        f"python: {platform.python_version()} ({sys.executable})",
        f"platform: {platform.platform()}; processor: {platform.processor()}",
        f"tensorflow: {tf.__version__}; keras: {keras.__version__}",
        f"numpy: {np.__version__}; cellpylib: {version('cellpylib')}",
        "check: every stored CNN output equals CellPyLib (memoize=False)",
    ]
    (args.out / "ENV.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    print("\n".join(env))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
