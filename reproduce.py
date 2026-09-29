#!/usr/bin/env python
"""Single entry point to reproduce the figures and checks of the repository.

Usage
-----
    python reproduce.py quick                     # checks + fast figures + notebook (CI)
    python reproduce.py all                       # everything, including the slow checks
    python reproduce.py all --with-benchmarks     # ... and re-time the published benchmark

``quick`` runs the fast verification suite, draws Figs 1, 2, 4 and 5 (each
script also asserts that it reproduces the published data) and executes the
walkthrough notebook. ``all`` adds the slow checks, the seeded re-run of the
Fig. 3 training (a few minutes) and the talk grids. ``--with-benchmarks``
re-runs the published benchmark protocol (about 80 minutes;
``scripts/benchmark_published.py``) and draws Fig. 5 with the re-run on top.
The new benchmarks and the training study have their own drivers in
``experiments/``.

Figures are written to ``output/``. Every step runs as a subprocess; the exit
code is non-zero if any step fails.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

FAST_FIGURES = [
    ("Fig. 1 nuCA with rules 30 and 90", [PY, "figures/fig1_nuca_example.py"]),
    ("Fig. 2 decomposition of one update", [PY, "figures/fig2_decomposition.py"]),
    ("Fig. 4 nuCA with 8 rules (CellPyLib)", [PY, "figures/fig4_nuca_cellpylib.py"]),
    ("Fig. 5 benchmarks (2024 data)", [PY, "figures/fig5_benchmarks.py"]),
]
SLOW_FIGURES = [
    ("Fig. 3 training re-run (seeded)", [PY, "figures/fig3_training.py"]),
    ("talk grids of example ECAs and nuCAs", [PY, "figures/extra_talk_grids.py"]),
]
NOTEBOOK = [("walkthrough notebook", [PY, "notebooks/execute.py"])]
CHECKS_QUICK = [("verification suite (fast)", [PY, "-m", "pytest", "-m", "not slow"])]
CHECKS_ALL = [("verification suite (all, incl. slow)", [PY, "-m", "pytest"])]
BENCHMARKS = [
    ("published benchmark protocol, re-run", [PY, "scripts/benchmark_published.py", "--quiet"]),
    ("Fig. 5 with the re-run overlaid", [PY, "figures/fig5_benchmarks.py", "--overlay", "2026"]),
]


def run(label: str, argv: list[str]) -> tuple[str, bool, float]:
    print(f"\n=== {label}: {' '.join(argv[1:])}", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"}
    start = time.time()
    proc = subprocess.run(argv, cwd=ROOT, env=env)
    return label, proc.returncode == 0, time.time() - start


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["quick", "all"])
    p.add_argument("--with-benchmarks", action="store_true")
    args = p.parse_args(argv)

    if args.mode == "quick":
        steps = CHECKS_QUICK + FAST_FIGURES + NOTEBOOK
    else:
        steps = FAST_FIGURES + SLOW_FIGURES + NOTEBOOK + CHECKS_ALL
    if args.with_benchmarks:
        steps += BENCHMARKS

    results = [run(label, cmd) for label, cmd in steps]
    print("\n=== summary")
    for label, ok, seconds in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {seconds:7.1f} s  {label}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
