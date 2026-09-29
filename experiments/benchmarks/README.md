# How fast can exact nuCA emulation be? A fair benchmark

This folder holds a new benchmark of ways to compute spacetime diagrams of
non-uniform cellular automata (nuCAs) exactly: CellPyLib, numpy, and the exact
CNN emulators of `ca_emulators` run in several ways. It is separate from the
ACRI 2024 benchmark (Fig. 5 of the paper, thesis Sec. 7.1.4 and Tab. 7.2),
whose protocol is reproduced by `scripts/benchmark_published.py`. The aim here
is to measure each method at its best, with its one-off (cold) costs kept
separate from its steady-state (warm) cost. The questions, protocol and
results are written up in [REPORT.md](REPORT.md).

| file | content |
|---|---|
| `methods.py` | the methods: one function per reference implementation, per selector and per driver, and the registry `METHODS` |
| `run.py` | the driver: plans the jobs, runs every job in a fresh subprocess, validates, times, writes the CSV and metadata |
| `plot.py` | figures (PNG and PDF) and a summary table (Markdown) from a results CSV |
| `tests/` | fast pytest checks: every method equals the numpy reference; the driver's logic |
| `REPORT.md` | the write-up (skeleton until the full run is done) |
| `results/` | `benchmark_full.csv`, `benchmark_full_meta.json`, figures, summary (after the full run) |
| `results/raw/` | smoke runs and logs of failed jobs (ignored by git) |

## Methods

A method is `cellpylib`, `numpy`, `numpy_lut`, or `<driver>:<selector>` for a CNN:

- **reference implementations**: `cellpylib` (CellPyLib's `evolve`, one call
  per sample, per-cell rule, `memoize=False`); `numpy`
  (`ca_emulators.reference.evolve`, the ground truth); `numpy_lut` (a tighter
  numpy loop with a per-cell lookup table, written for this benchmark);
- **drivers**: `predict` (`model.predict` per update, as in 2024), `eager`
  (`model(x)` per update), `compiled` (`tf.function` per update), `xla` (the
  same with `jit_compile=True`), `while_loop` and `while_loop_xla` (the whole
  diagram in one `tf.function` with `tf.while_loop`), `unrolled`
  (`NucaEmulator(..., timesteps=T-1, output_hidden=True)`, called once);
- **selectors**: `lc1`, `lc2`, `lc3` (`LocallyConnected1D` with
  `implementation=1, 2, 3`; `lc1` is the 2024 model), `dense` (the dense
  selector), `elem` (candidates times the one-hot allocation, summed over the
  rules; plain TensorFlow ops, defined in `methods.py`).

All 36 combinations are available (`--methods all`); XLA cannot compile
`lc3` (no XLA kernel for the sparse matrix product), so `xla:lc3` and
`while_loop_xla:lc3` do not exist. A full run uses 23 of them
(`methods.DEFAULT_METHODS`): the three reference implementations, all seven
drivers with `lc1` and with `dense`, and `compiled:lc2`, `xla:lc2`,
`compiled:lc3`, `compiled:elem`, `xla:elem` and `while_loop_xla:elem`.

## Protocol in brief

- **Correctness first.** Every job compares its diagram with
  `reference.evolve` (bit for bit, after the cold call and after the last
  warm call). A timing only counts if both match.
- **Isolation.** One subprocess per (method, scenario point, thread setting).
- **Thread settings.** `default` (the libraries' defaults) and `single`
  (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS` = 1, and
  `tf.config.threading` intra/inter = 1). CellPyLib is pure Python and runs in
  the default setting only; the plots show it in both rows.
- **Cold and warm.** Cold = set-up (model build and analytic weights) + first
  call (tracing, XLA compilation, Keras's predict function). The imports are
  timed separately (`import_s`). Warm = later calls, timed with
  `time.perf_counter`: 10 repeats of `number` calls each, with `number` chosen
  timeit-style so that one repeat lasts at least 0.1 s. The CSV reports the
  median, the quartiles, the minimum and the maximum per call.
- **Scenarios.** Those of thesis Tab. 7.2 (the *core* range) and their
  extensions (T counts rows of the diagram, as in 2024, so there are T - 1
  updates):

  | scenario | varies | fixed | core (Tab. 7.2) | extended |
  |---|---|---|---|---|
  | `NR` | N_R | N=256, T=32, S=32 | 1, 2, 4, ..., 256 | – |
  | `T` | T | N=64, N_R=4, S=32 | 10, 20, ..., 100 | 200, 500, 1000 |
  | `N` | N | N_R=4, T=32, S=32 | 32, 64, ..., 256 | 512, ..., 16384 |
  | `S` | S | N=32, N_R=4, T=32 | 1, 2, 4, ..., 1024 | 2048, ..., 16384 |

- **Budgets and skips** (all recorded in the CSV `status` and `note` columns):
  - Core points get 600 s for the warm phase and 600 s for the cold phase.
    Extended points get 60 s and 120 s, and a point is skipped
    (`skipped_predicted`) if linear extrapolation from the previous good point
    of the same method would exceed the budget.
  - A worker abandons a point whose cold phase, or 10 times its first warm
    call, exceeds the budget (`over_budget`).
  - A subprocess is killed after budget + cold budget + 180 s (`timeout`).
  - After `over_budget`, `timeout`, `error` or `mismatch`, the larger points of
    that method, scenario and thread setting are skipped
    (`skipped_after_failure`).
  - Selectors whose kernel exceeds 512 MiB (dense and `lc2` store
    N_R N² float32 weights) are skipped (`skipped_memory`).
- **Seeds.** The rules, allocation and initial states of a point depend only
  on `methods.BASE_SEED` and (N, N_R, T, S).
- **Metadata** (`*_meta.json`): platform, CPU, cores, RAM, AC power, CPU load
  before the run, package versions, BLAS, TensorFlow build, git commit, the
  settings, and a fixed calibration workload timed before and after the run.
  A large difference between the two calibrations means that the machine was
  not equally quiet throughout.

## Running

Use the project's environment (TensorFlow 2.14, Keras 2, numpy 1.26,
CellPyLib 2.4, with `ca_emulators` installed via `pip install -e .`). All
commands are run from the repository root; on Windows Git Bash, prefix them
with `PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2`.

```bash
# 1. correctness of every method (about 1-2 minutes)
python -m pytest experiments/benchmarks/tests

# 2. smoke run: tiny sizes, 2 repeats, tiny budgets; all default methods and both
#    thread settings; exercises every skip path. Timings are meaningless.
#    About 25 minutes on a quiet machine (270 jobs; most of the time goes on
#    starting TensorFlow in each subprocess). Output: results/raw/smoke/
python experiments/benchmarks/run.py --smoke --overwrite
python experiments/benchmarks/plot.py --smoke

# 3. full run (6-8 hours; see below), then figures and summary
python experiments/benchmarks/run.py --all --dry-run      # lists the 2295 jobs
python experiments/benchmarks/run.py --all
python experiments/benchmarks/plot.py
```

The results file is written row by row, so an interrupted run loses at most
one job. `--resume` continues it (and keeps the earlier session's metadata):

```bash
python experiments/benchmarks/run.py --all --resume
```

The full run can also be split, for example by thread setting:

```bash
python experiments/benchmarks/run.py --all --threads default
python experiments/benchmarks/run.py --all --threads single --resume
```

Other selections write to `results/benchmark_custom.csv` (or `--tag NAME`):

```bash
python experiments/benchmarks/run.py --scenarios N --methods numpy,numpy_lut,xla:elem --threads single
python experiments/benchmarks/run.py --scenarios S --ranges core --methods all --tag s_all
```

The protocol options (`--repeats`, `--min-time`, `--budget`, `--cold-budget`,
`--core-budget`, `--core-cold-budget`, `--max-selector-mb`) are listed by
`python experiments/benchmarks/run.py --help`.

### Expected duration of the full run

There are 2295 jobs: 1710 at core points and 585 at extended points, some
of which will be skipped without being started. This is a rough estimate for
the i7-9850H laptop, idle and on AC power:

- **Per-job overhead: about 3.5-4.5 hours.** Each job starts a subprocess and
  imports TensorFlow (about 4-6 s when the machine is idle), builds and
  traces its model (0.5-3 s), and spends about 1.5 s on the warm timings of a
  fast method. That is 7-8 s for each of about 2000 jobs that run.
- **CellPyLib at the core points: about 35 minutes.** It takes about 27 µs
  per cell update, and each point is timed 11 times (10 warm + 1 cold). The
  largest point, S = 1024, takes about 40 s per run.
- **`predict`: about 30-50 minutes.** Four series of about 1400 calls, times
  11 timings each, at 30-50 ms per call.
- **Large cases: up to about 1 hour.** These are the slow cold phases of
  `lc1` at large N, and of the unrolled models at large T or N, before they
  are cut off by the budget.

In total, **about 6-8 hours**. Run it overnight.

### Before the full run: a quiet machine

- **Nothing else running.** The driver prints a warning if the CPU is more
  than 15% busy when it starts. It records the CPU load before the run and
  the calibration timings before and after it.
- **AC power, a high-performance power plan, and the lid open** (thermals).
- **OneDrive sync paused.** This folder is inside OneDrive, and the results
  file is appended to after every job.
