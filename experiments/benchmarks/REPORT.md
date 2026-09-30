# How fast can exact nuCA emulation be? Benchmark report

> **Status: complete** for the CPU run of 29-30 September 2026 (laptop,
> 2,295 jobs, 8 hours). The numbers come from `results/benchmark_full.csv`,
> `results/full_summary.md` and the figures in `results/`. This report does
> not edit the thesis; Section 6 lists the thesis statements that the results
> qualify, with proposed wording.

## Summary

- **Every method is exact.** All 2,182 timed jobs produced diagrams
  bit-identical to the numpy reference; the other 113 were skipped by design
  (time budget or memory cap), none failed.
- **The 2024 CNN timings measure `model.predict`, not the network.** Each
  `predict` call costs about 40 ms (36-50 ms) whatever the size. Driven by a
  compiled loop, the same networks need 25 µs per update at the smallest case
  and 0.25-0.3 ms per update at N = 64, S = 32. In the ranges of the thesis,
  the fastest exact CNN beats the 2024 protocol by 13-2000 times (median 130)
  and CellPyLib by 30-480 times. Even the simplest fix (one `tf.function` per
  update) beats CellPyLib at every point measured.
- **What is one-off is small, apart from importing TensorFlow.** Import 3.3 s;
  build and first call a median of 0.2-0.8 s for the per-update and
  while-loop drivers. The unrolled model is the exception (4-60 s, growing
  with T and N), as is the `LocallyConnected1D` graph at large N (46 s at
  N = 4096).
- **Hand-vectorised numpy is the fastest exact method** at 97 of 102 (point,
  thread setting) pairs. The package's `reference.evolve` is 14-2100 times
  faster than CellPyLib and 1.3-15 times faster than the fastest CNN. The CNN
  wins only for tiny diagrams (S ≤ 8 samples of 32 cells), by up to 2.2
  times, and needs 400-1900 diagrams there to recover its compile cost.
- **Selectors:** the elementwise selector (N_R N weights) scales best; the
  dense and `implementation=2` selectors store N_R N² weights and stop at
  N = 4096 (memory cap), `implementation=1` builds a graph of N slices and
  stops there too (cold time).

## 1. Why a new benchmark

The ACRI 2024 paper (Fig. 5) and thesis Sec. 7.1.4 (Tab. 7.2) compare three
ways to simulate nuCAs: CellPyLib, a CNN with a `LocallyConnected1D`
selector, and a CNN with a dense selector. The CNNs were run with one
`model.predict` call per time step, and the timings covered one pass with
`time.time` (mean ± std of 10 repeats, model build excluded, first call
included). That protocol, reproduced by `scripts/benchmark_published.py`,
measures a particular way of using the networks. It does not measure how
fast exact emulation can be. The thesis draws two conclusions that go
beyond what was measured (Section 6).

This benchmark keeps the 2024 scenarios and adds:

- the ways of running a TensorFlow model that avoid per-call overhead;
- selectors that avoid the N_R N² dense kernel;
- plain numpy;
- a strict separation of one-off (cold) and steady-state (warm) costs;
- a thread-controlled comparison.

## 2. Questions

- **Q1 (overhead).** How much of the 2024 CNN time is per-call overhead of
  `model.predict`, rather than computation? What remains with `model(x)`, a
  `tf.function` step, XLA, a `tf.while_loop` or an unrolled model?
- **Q2 (fixed cost).** What does a user pay once, split into importing
  TensorFlow, building the model, and tracing or compiling on the first call?
  After how many diagrams does it amortise?
- **Q3 (selectors).** How do `LocallyConnected1D` (implementations 1, 2 and
  3), the dense selector and an elementwise selector scale with N and N_R?
  Where does the N_R N² memory of the dense selector (and of implementation
  2) stop it?
- **Q4 (numpy).** Does hand-vectorised numpy outperform CellPyLib and the
  CNNs, as the thesis states without having measured it? Over which ranges?
- **Q5 (best case).** What is the fastest exact method at each scenario
  point, and by what factor does it beat the 2024 protocol and CellPyLib?
- **Q6 (threads).** How much of each method's speed comes from using several
  cores? Are the rankings the same with a single thread?

## 3. Methods

Every method returns the complete spacetime diagram of S nuCAs (N cells, N_R
rules, static allocation, T rows = T - 1 updates, periodic boundaries). Each
must be bit-identical to `ca_emulators.reference.evolve` before its timing
counts. `tests/test_methods.py` checks all 36 methods on de Bruijn
certificate configurations: every cell meets all eight neighbourhoods, so
every cell's rule table is exercised in full. Each job checks again at its
own size.

| method | what is timed |
|---|---|
| `cellpylib` | `cpl.evolve` per sample, per-cell rule via `nks_rule`, `memoize=False` (`reference.evolve_cellpylib`) |
| `numpy` | `reference.evolve` (the package's vectorised reference; its speed depends on the package version, see the metadata's git commit) |
| `numpy_lut` | numpy with a precomputed per-cell lookup table: per update one neighbourhood index and one `np.take` (written for this benchmark) |
| `predict:<sel>` | `model.predict(x, batch_size=S)` per update, output fed back (2024 protocol, but one batch per call) |
| `eager:<sel>` | `model(x, training=False)` per update |
| `compiled:<sel>` | a `tf.function` of one update (`simulate.compiled_step`), traced once, called T - 1 times |
| `xla:<sel>` | the same with `jit_compile=True` |
| `while_loop:<sel>` | the whole diagram in one `tf.function` using `tf.while_loop` and a `TensorArray` |
| `while_loop_xla:<sel>` | the same with `jit_compile=True` |
| `unrolled:<sel>` | one call of a model with the T - 1 updates unrolled (`timesteps=T-1, output_hidden=True`), inside a `tf.function` |

Selectors: `lc1`/`lc2`/`lc3` = `LocallyConnected1D` with `implementation=1/2/3`
(`NucaEmulator.model()`; 1 is the 2024 network); `dense` =
`NucaEmulator.model_dense()`; `elem` = the N_R candidate channels
multiplied by the one-hot allocation (N, N_R) and summed over the rules
(plain TensorFlow ops; the detector and rule-table layers and their weights
are the package's). XLA cannot compile `lc3`.

## 4. Protocol

The settings of a full run are listed below; see README.md for the options.

- **Isolation.** Each (method, scenario point, thread setting) runs in its
  own Python subprocess (fresh TensorFlow runtime, no shared traces or
  caches).
- **Threads.**
  - `default`: library defaults (TensorFlow's own thread pools; numpy's
    elementwise operations are single-threaded anyway).
  - `single`: `tf.config.threading` intra = inter = 1, and
    `OMP/MKL/OPENBLAS_NUM_THREADS = 1`.
  - CellPyLib (pure Python) runs in the default setting only.
- **Cold.** Set-up + first call, measured once. It is reported with the
  import time (`import_s`: TensorFlow and `ca_emulators`, plus CellPyLib for
  `cellpylib`), which is excluded from cold.
- **Warm.** `time.perf_counter`, garbage collector off during a repeat. There
  are 10 repeats of `number` calls, where `number` (1, 2, 5, 10, ...) makes a
  repeat last at least 0.1 s. We report the median and the interquartile
  range per call.
- **Scenarios.** Tab. 7.2 (the *core* range: N_R 1-256; T 10-100; N 32-256;
  S 1-1024, with the 2024 fixed values), extended to T ≤ 1000, N ≤ 16384 and
  S ≤ 16384.
- **Budgets.**
  - Core points: 600 s warm and 600 s cold.
  - Extended points: 60 s warm and 120 s cold, and skipped when a linear
    extrapolation from the previous good point exceeds the budget.
  - After a failure (over budget, time-out, error, mismatch), the larger
    points of the chain are skipped.
  - Selector kernels above 512 MiB are skipped.
  - Every skip is recorded with its reason.
- **Data.** Rules drawn without replacement from 0-255 and sorted; uniform
  random allocation and initial states. The seed depends only on (N, N_R, T,
  S) and `BASE_SEED = 20240912`.
- **Recorded context.** Machine, software, git commit and dirty flag, CPU
  load before the run, and a calibration workload timed before and after
  (`results/benchmark_full_meta.json`).

**Differences from the 2024 protocol**, all deliberate:

- warm and cold are separated (2024 included the first `predict` call in
  the timed loop, but not the model build);
- `perf_counter` and median/IQR instead of `time.time` and mean ± std;
- `predict` gets `batch_size=S`, one batch per call (the Keras default of 32
  split S > 32 into several batches);
- the frames are collected in a list and stacked once, instead of
  `np.append` at every update;
- CellPyLib runs through `reference.evolve_cellpylib` (the same per-sample
  loop, rule looked up per cell).

## 5. Results

**Machine and session** (`results/benchmark_full_meta.json`).

- **Machine.** Laptop with an Intel Core i7-9850H (6 cores, 12 threads,
  2.6 GHz), 15.8 GB RAM, on mains power; Windows 11 Enterprise (build 26200).
  It is the machine of the 2024 measurements.
- **Software.** Python 3.11.5, TensorFlow 2.14.0 (CPU build; the GPU hidden
  with `CUDA_VISIBLE_DEVICES=-1`), Keras 2.14.0, numpy 1.26.4 (OpenBLAS),
  CellPyLib 2.4.0, `ca_emulators` 1.0.0 at commit `a7648d6`. The metadata
  flags the tree as dirty: the modified files were training summaries that
  the preceding training sweeps had just rewritten, not code.
- **Session.** Started on 29 September 2026 at 23:15, after the training
  sweeps and a 5-minute idle check (CPU below 20%); finished at 07:14
  (28,715 s). CPU load just before: 13.9%. Calibration (a 256 × 256 matrix
  product × 50; a Python sum over 10⁶ numbers) 28.7 / 32.4 ms before and
  27.4 / 32.3 ms after, so the machine's speed did not drift.
- **Jobs.** 2,295 in total:
  - 2,182 timed, every one bit-identical to the reference;
  - 53 skipped because the extrapolated time exceeded the budget;
  - 34 skipped for memory (selector kernel above 512 MiB);
  - 12 over budget;
  - 14 skipped after an over-budget point.

  No errors, time-outs or mismatches.

### 5.1 Overview

![overview](results/full_overview.png)

Times are warm medians per complete diagram, default threads unless stated.
"Best CNN" is the fastest exact CNN variant at each point. "2024 protocol"
is the faster of `predict:lc1` and `predict:dense`.

**Number of rules** (N = 256, T = 32, S = 32; N_R = 1-256).
- `numpy_lut` is the fastest at every point (2.8-4.0 ms); `numpy` takes
  3.6-6.0 ms.
- Best CNN: `while_loop_xla:elem`, 15-87 ms. That is 16-79 times faster than
  the 2024 protocol (1.1-1.9 s) and 68-360 times faster than CellPyLib
  (5.1-5.9 s).
- CellPyLib is the slowest throughout and does not depend on N_R.
- Only the CNNs grow with N_R, because every cell evaluates all N_R rule
  tables before the selector picks one. numpy looks up a single table per
  cell (`numpy` × 1.6 from N_R = 1 to 256, `numpy_lut` flat).

**Rows** (N = 64, N_R = 4, S = 32; T = 10-100, extended to 1000).
- Every method is linear in T. The slope of warm time against T − 1 (per
  update):

  | method | per update |
  |---|---|
  | CellPyLib | 43 ms |
  | `predict` | 36-37 ms |
  | compiled per-update step | 0.8-1.0 ms |
  | XLA-compiled while loop | 0.25-0.29 ms |
  | `numpy` | 67 µs |
  | `numpy_lut` | 52 µs |

- At T = 100: CellPyLib 4.3 s and the 2024 protocol 3.7 s, against 22 ms for
  the best CNN and 5-6 ms for numpy.
- The best CNN is 134-180 times faster than the 2024 protocol.
- `numpy_lut` is the fastest at every point.
- In the extended range (T up to 1000) only the compiled drivers and numpy
  remain (best CNN 0.25 s, `numpy_lut` 53 ms at T = 1000).

**Cells** (N_R = 4, T = 32, S = 32; N = 32-256, extended to 16384).
- CellPyLib is linear in N: 0.65 s at N = 32, 5.2 s at N = 256.
- The 2024 protocol is flat: 1.1-1.4 s up to N = 1024, 1.8-2.4 s at
  N = 4096.
- CellPyLib beats warm `predict` only at N = 32.
- Best CNN: `while_loop_xla:elem`, 4.0 ms (N = 32) to 16 ms (N = 256) to
  0.59 s (N = 16384). `numpy_lut`: 1.5 ms, 2.8 ms and 0.14 s.
- Only the elementwise and `implementation=3` selectors reach N = 16384.
  The others stop at N = 4096 (Section 5.4).

**Samples** (N = 32, N_R = 4, T = 32; S = 1-1024, extended to 16384).
- CellPyLib is linear in S: 19 ms at S = 1, 23 s at S = 1024.
- The 2024 protocol is flat up to S = 1024 (1.1-1.6 s), then grows (2.0 s at
  S = 16384).
- CellPyLib beats warm `predict` up to S = 32; the crossover lies between 32
  and 64.
- The best CNN runs one sample in 0.64 ms (`while_loop_xla`) and is the
  fastest method of all at S ≤ 4 (Section 5.5).
- At S = 16384: `numpy_lut` 0.18 s, `numpy` 0.39 s, best CNN 0.55 s
  (default threads) or 1.75 s (single thread).

### 5.2 Where the 2024 CNN time goes (Q1)

![drivers, LocallyConnected1D](results/full_drivers_lc1.png)
![drivers, dense](results/full_drivers_dense.png)

**Per-call overhead dominates.** Warm time per update at the smallest case
(N = 32, N_R = 4, T = 32, S = 1), where the arithmetic is negligible
(summary table A; default threads, single thread in brackets where it
differs):

| driver | `LocallyConnected1D` | dense |
|---|---|---|
| `predict` (2024) | 42.5 ms (40.9) | 40.9 ms (50.3) |
| eager `model(x)` | 18.5 ms | 3.7 ms (4.7) |
| `tf.function` per update | 0.58 ms | 0.54 ms (0.38) |
| + XLA | 0.41 ms | 0.42 ms (0.38) |
| `tf.while_loop` | 0.18 ms (0.12) | 0.07 ms (0.05) |
| `tf.while_loop` + XLA | 26 µs | 25 µs (20) |
| unrolled model | 0.17 ms | 44 µs |
| `numpy` / `numpy_lut` (for reference) | 44 / 43 µs (45 / 28) | |

- Of the roughly 40 ms that a `predict` call costs, about 25 µs is needed
  for the update itself. The rest is per-call overhead, mostly Keras's
  set-up: in Keras 2.14 each call builds a data handler with a `tf.data`
  iterator and a callback list.
- The eager call of the `LocallyConnected1D` model is slow for a different
  reason. `implementation=1` assembles its input by slicing once per cell,
  N separate operations that eager mode dispatches one by one and a compiled
  graph does not.

**Ratios in the core range** (all 38 core points, default threads; single
thread similar):

| ratio | `LocallyConnected1D` | dense |
|---|---|---|
| `predict` / `tf.function` per update | 5-73 (median 40) | 3.8-86 (median 44) |
| `predict` / XLA per update | 6-103 (median 46) | 3.7-99 (median 44) |
| `predict` / `tf.while_loop` | 6-231 (median 66) | 3.9-562 (median 79) |
| `predict` / `tf.while_loop` + XLA | 9-1640 (median 114) | 4.4-1610 (median 108) |
| `predict` / unrolled model | 6-249 (median 79) | 3.7-931 (median 93) |

The smallest ratios are at N_R = 256, where the arithmetic finally matters;
the largest at S = 1.

**`predict` is flat in N, N_R and S, and linear in T.**

| scenario (core range) | warm `predict`, default threads |
|---|---|
| N_R = 1-256 | 1.14-1.86 s |
| N = 32-256 | 1.14-1.37 s |
| S = 1-1024 | 1.12-1.59 s |

Against T, both selectors take 36-37 ms per update with default threads and
36-38 ms single-threaded (linear fits, R² 0.93-0.99). This is the signature of a per-call cost: 31 calls × about
40 ms ≈ 1.2 s at T = 32. The 2024 data show 48-51 ms per update (Section 6).
The difference plausibly comes from the 2024 protocol's `np.append` per step
and its default batching, but this benchmark does not separate the two.

### 5.3 One-off costs (Q2)

![cold](results/full_cold.png)

**Import.** TensorFlow plus the package take 3.3 s (median; interquartile
range 3.2-3.5 s over 1,940 CNN jobs); with CellPyLib 3.9 s.
- The numpy methods show the same 3.3 s only because the benchmark imports
  them from `ca_emulators`, whose `__init__` imports TensorFlow.
- A script with numpy alone would start in a fraction of that.
- The horizontal line in the cold figure marks the import time; it exceeds
  most cold times.

**Build, first call, one-off cost** (medians over the core points, default
threads; one-off = cold − warm, i.e. what the first diagram costs extra):

| driver | build, lc1 / dense | first call, lc1 / dense | one-off, lc1 / dense / elem |
|---|---|---|---|
| `predict` | 0.24 / 0.13 s | 2.0 / 1.6 s | 0.82 / 0.37 s / – |
| eager | 0.23 / 0.13 s | 1.9 / 0.22 s | 0.32 / 0.17 s / – |
| `tf.function` per update | 0.23 / 0.12 s | 0.39 / 0.13 s | 0.59 / 0.21 / 0.23 s |
| + XLA | 0.26 / 0.12 s | 0.46 / 0.18 s | 0.67 / 0.27 / 0.26 s |
| `tf.while_loop` | 0.25 / 0.13 s | 0.50 / 0.20 s | 0.71 / 0.31 s / – |
| `tf.while_loop` + XLA | 0.25 / 0.12 s | 0.60 / 0.25 s | 0.84 / 0.36 / 0.36 s |
| unrolled model | 8.1 / 0.88 s | 16 / 1.2 s | 24 / 2.1 s / – |

- **`predict`'s first call is itself expensive.** It exceeds a warm call by
  0.04-1.6 s (median 0.5 s for `lc1`, 0.23 s for dense). The 2024 timings
  included it in every measurement.
- **Two things grow.**
  - The unrolled model's cold time is linear in T. For `lc1`: 4.0 s at
    T = 10, 42 s at T = 100, 84 s at T = 200. It grows with N as well:
    7.4 s at N = 32, 60 s at N = 256, 110 s at N = 512.
  - The cold time of every `lc1` driver grows with N, because its graph has
    N slices. `tf.function`: 0.35 s at N = 32, 2.2 s at 256, 46 s at 4096,
    and over the 120 s budget at 8192 for all `lc1` drivers.
  - By contrast, the elementwise XLA while loop stays below 1 s up to
    N = 16384 (default threads).
- **Amortisation.**
  - Against `predict` (same selector; `lc1` for the other selectors), the
    per-update, XLA and while-loop drivers are ahead from the first diagram
    at almost every point: their cold time and their warm time are both
    lower.
  - The exceptions:
    - the `lc1` drivers at N = 4096, which need 3-8 diagrams;
    - the unrolled `lc1` model, which needs more than one diagram at every
      point (up to 86 at N = 512) because of its build time;
    - `implementation=2` at its largest points (N_R = 256 single-threaded;
      N = 4096), which is slower than `predict:lc1` even warm.
  - Against numpy, a CNN never catches up where numpy is faster per diagram,
    which is 94 of the 102 pairs. At the 8 small-S pairs where the XLA while
    loop is faster, it recovers its extra cold time only after 400-1900
    diagrams (e.g. 520 at S = 1, default threads).

### 5.4 Selectors (Q3)

![selectors](results/full_selectors.png)

**Scaling with N_R** (N = 256, T = 32, S = 32; warm, from N_R = 1 to 256):

| selector | `tf.function`, default | single thread | XLA while loop, default |
|---|---|---|---|
| `lc1` (graph of N slices) | 53 → 261 ms (× 4.9) | × 7.7 | 16 → 147 ms (× 9.4) |
| `lc2` (masked N_R N² kernel) | 36 → 831 ms (× 23) | × 57 | – |
| `lc3` (sparse) | 29 → 328 ms (× 11) | × 10 | – |
| dense (N_R N²) | 29 → 444 ms (× 16) | × 19 | 17 → 379 ms (× 22) |
| elementwise (N_R N) | 28 → 125 ms (× 4.4) | × 4.7 | 15 → 87 ms (× 5.9) |
| `numpy` / `numpy_lut` | 3.8 → 6.0 / 3.3 → 2.8 ms | | |

**Scaling with N** (N_R = 4, T = 32, S = 32; `tf.function` per update,
default threads):

| selector | N = 32 | N = 256 | N = 4096 | N = 16384 | stops at |
|---|---|---|---|---|---|
| `lc1` | 24 ms | 65 ms | 0.39 s | – | N = 8192: cold time over 120 s |
| `lc2` | 22 ms | 47 ms | 4.6 s | – | N = 8192: kernel 1 GiB > 512 MiB (single thread: N = 4096 over the warm budget) |
| `lc3` | 20 ms | 33 ms | 0.21 s | 0.80 s | – |
| dense | 23 ms | 47 ms | 0.97 s | – | N = 8192: kernel 1 GiB > 512 MiB |
| elementwise | 20 ms | 32 ms | 0.16 s | 0.60 s | – |
| elementwise, XLA while loop | 4.0 ms | 16 ms | 0.16 s | 0.59 s | – |
| `numpy_lut` | 1.5 ms | 2.8 ms | 45 ms | 0.14 s | – |

**Peak memory of the job's process** (Windows peak working set; about
250 MB is TensorFlow itself):

| selector | N = 4096 |
|---|---|
| elementwise | 0.35-0.37 GB |
| `lc3` | 0.35 GB |
| `lc1` | 0.8-1.2 GB |
| dense | 1.0 GB |
| `lc2` | 1.6 GB |

For comparison, `numpy` peaks at 0.3 GB at N = 16384 (again mostly the
TensorFlow import). The dense and `lc2` kernels alone are 256 MiB at
N = 4096 (N_R N² float32).

**In short.** The elementwise selector (the allocation as an (N, N_R) one-hot
mask, multiplied into the N_R candidate channels) scales best. It is the
fastest CNN at most points and the dense selector at most of the rest, and
it uses the least memory. It stores N_R N numbers, like `lc1`/`lc3`, and has
none of their graph or sparse overheads.
`implementation=2` is the worst at large N_R and N; the dense selector is
fine up to a few hundred cells.

### 5.5 numpy against CellPyLib and the CNNs (Q4, Q5)

**numpy against the fastest exact CNN at each point** (summary table C;
ratio = time of the best CNN / time of `numpy`):

| scenario | threads | points | `numpy` faster | `numpy_lut` faster | ratio min-max |
|---|---|---|---|---|---|
| N_R | default / single | 9 / 9 | 9 / 9 | 9 / 9 | 3.5-15 / 5.8-11 |
| T | default / single | 13 / 13 | 13 / 13 | 13 / 13 | 2.7-4.0 / 2.6-4.3 |
| N | default / single | 14 / 14 | 14 / 14 | 14 / 14 | 1.5-5.5 / 1.9-11 |
| S | default / single | 15 / 15 | 11 / 11 | 12 / 13 | 0.47-4.0 / 0.46-11 |

- **Overall.** `numpy` is faster than every CNN variant at 94 of the 102
  (point, thread setting) pairs, by 1.3-15 times; `numpy_lut` at 97, by up
  to 31 times.
- **Where a CNN wins.** Only at the smallest sample counts: S = 1, 2, 4 and
  8 at N = 32, T = 32. There the XLA-compiled while loop (elementwise or
  dense selector) beats `numpy` by 1.1-2.2 times; at S = 1, 0.64 ms against
  1.35 ms.
  - These diagrams have at most 8 × 32 × 31 ≈ 8000 cell updates.
  - numpy's cost there is its per-update Python overhead (about 40 µs),
    which the single compiled loop does not pay.
  - The CNN needs 400-1900 such diagrams to recover its one-off cost
    (Section 5.3).
  - No CNN wins at large N or S: at S = 16384, `numpy_lut` takes 0.18 s
    against 0.55 s for the best CNN.
- **Fastest exact method overall.** `numpy_lut` at 91 pairs, `numpy` at 6,
  a CNN at 5 (S ≤ 4).
- **numpy against CellPyLib.** The ratio grows with the work per diagram:
  14 times faster at S = 1, 130 at S = 8, and 290-2100 times at every point
  with S ≥ 32. Against the 2024 protocol, 5-1130 times.
- **The best CNN against CellPyLib and against the 2024 protocol.** 30-480
  times faster than CellPyLib at every point, and 13-2000 times faster than
  `predict` in the core range.
- **Which CNN is fastest.** An XLA-compiled while loop at 74 of the 102
  pairs (otherwise the unrolled model, a plain while loop or a compiled
  step), with the elementwise selector at 69 and the dense one at 30.

### 5.6 Threads (Q6)

Ratio single-thread / default-thread warm time (above 1: the thread pool
helps):

| point | `numpy` | `numpy_lut` | `predict:lc1` | `tf.function` (lc1 / dense / elem) | elementwise, XLA while loop |
|---|---|---|---|---|---|
| N_R = 1 | 1.0 | 0.9 | 1.0 | 1.0 / 1.4 / 1.4 | 1.8 |
| N_R = 256 | 1.6 | 1.5 | 1.1 | 1.6 / 1.7 / 1.5 | 1.1 |
| T = 10 | 1.2 | 0.9 | 0.9 | 1.1 / 0.8 / 1.0 | 1.1 |
| T = 1000 | 1.2 | 1.1 | – | 1.0 / 0.8 / 0.8 | 0.9 |
| N = 32 | 0.9 | 1.1 | 1.0 | 0.9 / 0.9 / 0.8 | 0.9 |
| N = 16384 | 1.1 | 1.3 | – | – / – / 3.1 | 3.6 |
| S = 1 | 1.0 | 0.7 | 1.0 | 1.0 / 0.7 / 0.9 | 1.0 |
| S = 16384 | 1.0 | 0.9 | 1.6 | 2.8 / 3.2 / 3.5 | 3.4 |

- **numpy barely changes.** Its elementwise operations are single-threaded
  anyway. The deviations of up to 1.6 (N_R = 256: 6.0 against 9.3 ms, with
  non-overlapping interquartile ranges) cannot come from threading. They
  must come from conditions that differed between the two jobs, such as
  turbo or background activity. They show how much a separate measurement
  of a few milliseconds can vary on this laptop.
- **TensorFlow's thread pool only pays off for large tensors.** It gives
  about 3-3.6 times at N = 16384 or S = 16384 (6 physical cores). For small
  diagrams it does not help, and it sometimes hurts (ratios 0.7-0.9).
- **Rankings.**
  - numpy stays ahead of the CNNs at 47 of the 51 points in both settings,
    and single-threaded its lead grows (up to 11 times at N = 2048, against
    4.5 with default threads).
  - The fastest CNN variant differs between the settings at 30 of 51
    points. The two candidates are close: in the other setting, one is a
    median of 1.3 times (at most 2.2 times) slower than the other. This is
    a reshuffle among while loop, unrolled model and XLA, and between the
    dense and elementwise selectors, not a change of conclusion.
  - CellPyLib is pure Python and was run in the default setting only.

## 6. Thesis statements this benchmark qualifies

The thesis is not edited here. The quotations are from
`ch07-efficient_simulation.tex` (lines 112, 134 and 198 of the version in
`personal-website/_source/thesis/mainmatter/`, 30 September 2026).

**S1: the CNNs carry a "fixed cost of a few seconds".** Line 134: "the
computation time of the densely connected CNN is dominated by a fixed cost of
a few seconds, and is nearly independent of N, of N_S [...] and of N_R [...].
The computation time of the locally connected CNN carries a similar fixed
cost but grows with N, in line with the loop over cell positions in its
implementation. [...] all three methods scale linearly with T with comparable
slopes [...] the densely connected CNN is consequently the fastest of the
three whenever the fixed cost is compensated for: from roughly a hundred
cells onwards in the third scenario, and from a few hundred samples onwards
in the fourth." Line 112 expects a speed-up for the CNNs "as soon as the
number of cells or samples is large enough for the compiled kernels to
outweigh their fixed overhead", and line 198 repeats that the CNN "is faster
once its fixed overhead is amortised over a hundred cells or a few hundred
samples".

What the 2024 timing arrays (`data/benchmarks_2024/`) already show:

- **The cost is proportional to T, not fixed.** In the T scenario (N = 64,
  S = 32) the CNN time grows linearly with the number of updates:
  - 51 ms per update with a near-zero intercept (0.06 s) for the dense
    selector;
  - 48 ms per update plus 0.6 s for the locally connected one.

  It is flat in N, N_R and S only because those scenarios kept T = 32
  (31 `predict` calls).
- **It depends on the session.** The same configuration (N = 256, N_R = 4,
  T = 32, S = 32) was measured in two scenarios:

  | selector | N_R scenario | N scenario |
  |---|---|---|
  | dense | 1.51 s | 3.58 s |
  | locally connected | 3.44 s | 6.51 s |
  | CellPyLib | 5.82 s | 8.90 s |

  At S = 1 (N = 32) the CNN costs 126-142 ms per update, against about
  50 ms in the T scenario.

  The CNN baseline therefore varied by a factor of 1.5-2.5 between sessions.

The new benchmark measures the parts directly: import, build, first call,
and warm time per update for each driver.

**The measured decomposition.** At T = 32, the time the 2024 protocol
charged to a CNN consists of three parts.

1. **Per-call overhead of `predict`: about 1.2 s.** 31 calls of about 40 ms
   each (36-42 ms here; 48-51 ms in 2024). This part is flat in N, N_R and
   S (warm `predict` 1.1-1.9 s over the core ranges) and linear in T. It is
   the "fixed cost".
2. **The first `predict` call: 0.04-1.6 s more than a warm call.** It traces
   the predict function. For the locally connected network it also builds
   the graph of N slices, whose cost grows with N (cold `predict:lc1`
   1.7 s at N = 32, 2.9 s at N = 256, 48 s at N = 4096). This, not the
   warm computation, is what grows with N. The warm `predict` time of the
   locally connected network is flat in N up to N ≈ 1000 (1.15-1.35 s).
3. **Not timed in 2024:** importing TensorFlow (3.3 s) and building the
   model (0.1-0.25 s).

**The computation itself is small.** The same networks need 25 µs per
update at the smallest case, and 0.25-0.3 ms per update at N = 64, S = 32,
when the whole diagram runs in one XLA-compiled `tf.while_loop`. A plain
`tf.function` per update needs 0.5-1 ms.

**How each part of the statement fares.**
- **"Nearly independent of N, N_S and N_R".** Confirmed for `predict`, and
  explained by the per-call overhead.
- **"The locally connected CNN [...] grows with N, in line with the loop over
  cell positions".** Confirmed as far as the cause goes (the per-position
  slicing of `implementation=1`), but the growth is in the one-off graph
  construction and tracing, which the 2024 timings included through the
  first call.
- **"All three methods scale linearly with T with comparable slopes".**
  True under the 2024 protocol (CellPyLib 43 ms, `predict` 36-37 ms per
  update). With a compiled update the CNN slope is 40-150 times smaller
  (0.25-1 ms).
- **The crossovers ("roughly a hundred cells", "a few hundred samples").**
  They are a property of `predict`. With warm `predict` the crossovers lie
  between N = 32 and 64 and between S = 32 and 64. With a compiled
  per-update step, the same networks beat CellPyLib at every point
  measured, from S = 1 (1.05-1.6 times) to S = 1024 (195-350 times).

**Proposed rewording of S1** (for line 134, and correspondingly lines 112
and 198):

> The CNN timings of Fig. 7.5 are dominated by the overhead of calling
> `model.predict` once per update, about 40 ms per call on this CPU,
> rather than by computation. This makes them nearly independent of N,
> N_S and N_R, and linear in T. The locally connected CNN additionally pays
> a one-off cost for building its graph of N slices, which grows with N.
> Compiled with `tf.function`, the same networks need about 0.5 ms per
> update, and 25-300 µs when the whole diagram runs in one XLA-compiled
> loop, after a one-off cost of 0.2-0.8 s (plus about 3 s to import
> TensorFlow). They are then faster than CellPyLib in every scenario
> tested.

**S2: "hand-vectorised numpy code would outperform both" (untested in the
thesis).** Line 112: "A dedicated nuCA simulator, or hand-vectorised
`numpy` code, would outperform both". Line 198: "no corresponding `numpy`
baseline was included for the CNN emulator". `numpy` (the package's
reference) and `numpy_lut` measure this directly, against CellPyLib and
against every CNN variant, not just the two of 2024.

**Confirmed, with one small qualification.**

| comparison | numpy faster at | by |
|---|---|---|
| against CellPyLib | every point | 14 times (S = 1) to 2100 times |
| against the 2024 CNNs (`predict`) | every point | 5 to 1130 times |
| against the fastest of the 20 CNN variants | 94 of 102 (point, thread) pairs | 1.3 to 15 times; `numpy_lut` 97 pairs, up to 31 times |

This holds in every scenario and both thread settings, in the core and the
extended ranges. The exceptions are the smallest diagrams (S ≤ 8 samples of
32 cells), where an XLA-compiled while loop avoids numpy's per-update Python
overhead and wins by 1.1-2.2 times. There it needs 400-1900 diagrams to
recover its compile cost.

**Proposed rewording of S2**, for line 112:

> Hand-vectorised `numpy` code outperforms both: in our measurements it is
> 14-2100 times faster than CellPyLib and 1.3-15 times faster than the
> fastest compiled CNN variant, except for diagrams of a few thousand cell
> updates, where a single compiled loop avoids numpy's per-update overhead.

Line 198 would then read that a `numpy` baseline for the CNN emulator is
faster by 1.3-15 times on CPU, which parallels the `scipy.sparse` result
for the GNN emulator (two to four times).

## 7. Limitations

- **One machine.** A laptop (i7-9850H, Windows, CPU only), the same one as
  in 2024. Turbo and thermals make single-thread and multi-thread results
  hard to compare; separate millisecond measurements of the same method can
  differ by up to about 1.6 times (Section 5.6). No GPU; the GPU variant
  (`--device gpu`, docs/workstation.md) has not been run.
- **Import time.** The numpy methods were imported through `ca_emulators`,
  so their import time includes TensorFlow's (Section 5.3). Cold and warm
  times are unaffected.
- **One software stack.** TensorFlow 2.14 and Keras 2 (the versions of the
  paper); Keras 3 dropped `LocallyConnected1D`.
- **Budgets.** The extended ranges are cut by budgets, so absent points mean
  "too slow or too large under this protocol", not "impossible". Linear
  extrapolation can skip a point that would have fitted, if a method scales
  sublinearly.
- **Methods written here.** `numpy_lut` and the elementwise selector were
  written for this benchmark. Neither is optimal (Numba, bit packing, or a
  gather-based TF step would be faster); they are an attainable baseline.
- **The `numpy` baseline moves with the package.** It is the package's
  `reference.evolve`, whose implementation changed during the v1 clean-up
  (rule tables now computed once per diagram, not once per update).
- **Scope.** Static allocations only; ECAs and time-varying allocations are
  not covered.

## 8. Reproduction

```bash
python -m pytest experiments/benchmarks/tests
python experiments/benchmarks/run.py --all         # results/benchmark_full.csv (+ _meta.json)
python experiments/benchmarks/plot.py              # figures + results/full_summary.md
```

The results in this report come from one run with
`run.py --all --resume` (a fresh start; `--resume` only guards against
interruptions), from 29 September 2026, 23:15, to 30 September, 07:14
(7 h 59 min), at commit `a7648d6` (`ca_emulators` 1.0.0), followed by
`plot.py`. On this laptop the run needs the machine to itself for the whole
night; see docs/workstation.md for running it elsewhere.
