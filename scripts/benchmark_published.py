"""Re-run the benchmark of the ACRI 2024 paper with its original protocol.

Reproduces the four scenarios of Tab. 1 of the paper (thesis Tab. 7.2) as
the 2024 scripts ``tests/performance_comparisons/comparison_*.py`` ran them:

- CellPyLib: ``cpl.evolve`` once per sample, ``timesteps=T`` (T rows),
  ``memoize=False``, the whole loop over samples timed;
- CNN (locally connected, Keras implementation mode 1) and CNN (dense): a
  new model per point, built outside the timed region, then ``model.predict``
  once per update (T - 1 updates) on the whole batch, appending the output to
  the diagram with ``np.append``; the loop is timed;
- 10 repeats; per repeat all points in order; CellPyLib first, then the
  locally connected CNN, then the dense CNN.

The one deliberate change: ``time.perf_counter`` instead of ``time.time``
(better resolution, same quantity). Every diagram is checked against the
numpy reference outside the timed region. Results are written in the 2024
format (x-values, then CellPyLib, locally connected and dense times, each
(repeats, n), saved in sequence) with the 2024 file names, next to a JSON file
with the machine and software versions.

Usage (about 80 minutes for all scenarios on an i7-9850H)::

    python scripts/benchmark_published.py --scenario all
    python scripts/benchmark_published.py --scenario S --repeats 2 --quick   # smoke test

The package is imported from ``src/`` if it is not installed, so the script
also runs in the 2024 conda environment of the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import ca_emulators  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ca_emulators import NucaEmulator  # noqa: E402
from ca_emulators.benchmarks import SCENARIOS, save_timings  # noqa: E402
from ca_emulators.reference import evolve  # noqa: E402


def points(scenario: str, rng: np.random.Generator, quick: bool):
    """The (params, rules, alloc, init_configs) of every point, drawn as in 2024.

    Rules and allocation are drawn once per scenario, except where the varied
    parameter changes them (N_R: new rules and allocation per point; N: new
    allocation per point); initial configurations are drawn per point where
    their shape changes.
    """
    fixed, varied, values = (SCENARIOS[scenario].fixed, SCENARIOS[scenario].varied,
                             list(SCENARIOS[scenario].values))
    if quick:
        values = values[:3]
    rules = alloc = x0 = None
    if varied != "Nrules":
        rules = np.sort(rng.choice(256, size=fixed["Nrules"], replace=False))
    if varied in ("T", "S"):
        alloc = rng.integers(fixed["Nrules"], size=fixed["N"])
    if varied in ("T", "Nrules"):
        x0 = rng.integers(2, size=(fixed["S"], fixed["N"]))
    out = []
    for value in values:
        p = {**fixed, varied: value}
        r = np.sort(rng.choice(256, size=value, replace=False)) if varied == "Nrules" else rules
        a = rng.integers(p["Nrules"], size=p["N"]) if varied in ("Nrules", "N") else alloc
        x = rng.integers(2, size=(p["S"], p["N"])) if varied in ("N", "S") else x0
        out.append((p, r, a, x))
    return values, out


def time_cellpylib(p, rules, alloc, x) -> tuple[float, np.ndarray]:
    import cellpylib as cpl

    start = time.perf_counter()
    for row in x:
        diagram = cpl.evolve(row[np.newaxis, :], timesteps=p["T"],
                             apply_rule=lambda n, c, t: cpl.nks_rule(n, rules[alloc[c]]),
                             memoize=False)
    elapsed = time.perf_counter() - start
    return elapsed, np.asarray(diagram)  # the last sample's diagram, (T, N)


def time_cnn(p, rules, alloc, x, variant: str) -> tuple[float, np.ndarray]:
    em = NucaEmulator(p["N"], rules, rule_alloc=alloc)
    model = em.model() if variant == "lc" else em.model_dense()
    diagram = x[:, :, np.newaxis].astype(np.float64)  # (S, N, 1), as np.transpose gave in 2024
    start = time.perf_counter()
    for _ in range(p["T"] - 1):
        output = model.predict(diagram[:, :, -1], verbose=False)
        diagram = np.append(diagram, output, axis=2)
    elapsed = time.perf_counter() - start
    return elapsed, np.transpose(diagram, (0, 2, 1))  # (S, T, N)


def run(scenario: str, repeats: int, rng, quick: bool, verbose: bool):
    values, pts = points(scenario, rng, quick)
    # reference diagrams, computed once per point (outside every timed region)
    expected = [evolve(x, rules, p["T"] - 1, alloc) for p, rules, alloc, x in pts]
    times = {m: np.zeros((repeats, len(pts))) for m in ("cpl", "lc", "dense")}
    for method in times:
        for rep in range(repeats):
            for j, (p, rules, alloc, x) in enumerate(pts):
                if method == "cpl":
                    elapsed, diagram = time_cellpylib(p, rules, alloc, x)
                    reference = expected[j][-1]  # CellPyLib returns the last sample's diagram
                else:
                    elapsed, diagram = time_cnn(p, rules, alloc, x, method)
                    reference = expected[j]
                if not np.array_equal(np.asarray(diagram).astype(np.uint8), reference):
                    raise AssertionError(f"{scenario}/{method}: wrong diagram at {p}")
                times[method][rep, j] = elapsed
                if verbose:
                    print(f"{scenario:6s} {method:5s} repeat {rep + 1}/{repeats} "
                          f"{SCENARIOS[scenario].varied}={values[j]}: {elapsed:.3f} s", flush=True)
    return np.array(values), times


def metadata(args) -> dict:
    import cellpylib  # noqa: F401
    import tensorflow as tf
    from importlib.metadata import version

    return {
        "script": "scripts/benchmark_published.py",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed, "repeats": args.repeats, "quick": args.quick,
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "tensorflow": tf.__version__, "numpy": np.__version__, "cellpylib": version("cellpylib"),
        "clock": "time.perf_counter",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--quick", action="store_true", help="first three points only (smoke test)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks_2026")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    meta = metadata(args)
    for scenario in scenarios:
        rng = np.random.default_rng([args.seed, list(SCENARIOS).index(scenario)])
        x, times = run(scenario, args.repeats, rng, args.quick, not args.quiet)
        stem = SCENARIOS[scenario].stem + ("-quick" if args.quick else "")
        save_timings(args.out / f"{stem}.npy", x, [times["cpl"], times["lc"], times["dense"]])
        (args.out / f"{stem}.json").write_text(json.dumps({**meta, "scenario": scenario}, indent=2) + "\n")
        print(f"wrote {args.out / stem}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
