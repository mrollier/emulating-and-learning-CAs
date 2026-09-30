# Changelog

## 1.0.1 (2026-09-30)

Results and documentation; the package code is unchanged apart from its version number.

- **Benchmark** (`experiments/benchmarks`): the full CPU run (2,295 jobs, all timed
  outputs exact) and its report. The CNN "fixed cost" of Fig. 5 is `model.predict`
  overhead (about 40 ms per call); compiled, the same networks beat CellPyLib at every
  point, and hand-vectorised numpy beats every CNN variant except on the smallest
  diagrams. Section 6 of the report proposes wording for the two thesis statements this
  qualifies.
- **Training study** (`experiments/training`): the high-seed sweeps (`laptop.sh`; the
  recommended recipe exact in 262144/262144 runs), the missing wide-network points, and
  the configuration of `train_recipe` itself (`recipe_package`, 262143/262144). Its one
  failure is a rare plateau of rule 89 (about 1 in 65,000 runs of that rule, with random
  batches as well), now documented. `workstation.sh` runs only the two sweeps that need
  a GPU.
- **Fixed** in `experiments/training/sweep.py`: parallel workers starting on a fresh
  output folder could read a half-written `meta.json` and crash. The parent now writes
  it before starting them.
- Docs: `docs/workstation.md`, `CLAUDE.md` and the provenance table updated; the C9 row
  now names the slow test's actual scope (the 88 non-equivalent rules).

## 1.0.0 (2026-09-29)

A rebuild of the repository as a reproducibility package for the ACRI 2024 paper
and Sec. 7.1 of the PhD thesis.

- **Package** `ca_emulators` (src layout) replaces `src/nn`, `src/custom_tf_classes`,
  `src/train`, `src/utils` and `src/visual`. `EcaEmulator` and `NucaEmulator` keep
  their names and architecture, but are exact and frozen by default; the 2024
  argument `train_triplet_id` is deprecated. Analytic weights are plain numpy
  functions (`weights.py`), set with `set_weights`, so models can be saved and
  loaded. New: a numpy and CellPyLib reference (`reference.py`), rule utilities and
  de Bruijn certificates (`rules.py`), fast simulation paths (`simulate.py`), the
  2024 training recipe (`training.py`) and the figure code (`plotting.py`).
- **Fixed**: the 2024 default emulator was not exact (random detector weights);
  `rules.all() is not None` was always true; the nuCA rule-table layer's
  trainability was tied to `train_triplet_id`.
- **Data**: the 2024 timing data (the only copy lived in a local, gitignored folder),
  golden outputs of the 2024 code, and the inputs of Figs 1-4 decoded from the
  published PDFs.
- **Figures**: one script per published figure; Figs 1, 2 and 4 are reproduced
  cell for cell, Fig. 3 by a seeded re-run of its training recipe, Fig. 5 from the
  archived data.
- **Verification**: a pytest suite for claims C1-C8 (docs/provenance.md).
- **Notebook**: `notebooks/walkthrough.ipynb`.
- **Experiments**: a training study (`experiments/training/`) and new benchmarks of
  exact emulation (`experiments/benchmarks/`).
- **Training**: `training.train_recipe`, the recipe the study found exact for every rule
  (+-1 inputs, softplus detectors, sigmoid/BCE head), next to the faithful
  `train_2024_recipe`; `verify` (`is_exact` via the de Bruijn certificate, an interval
  certificate of closed-loop exactness, `closed_loop_exact`); the emulators take
  `input_encoding` ("01"/"pm1", with analytic +-1 weights), `detector_activation` and
  `rule_activation`, and `EcaEmulator(None, ...)` accepts any number of cells. Defaults are
  unchanged, so the golden outputs and figures are too.
- **Removed**: bytecode, `old/`, the Windows-only `environment.yml`, the unused
  training exploration (generalised sigmoid, constraints, grid search, the
  `kernel_initializer='halfway'` option, which now raises a clear error). All of it
  remains available at the tag `acri-2024`.
- Licence (MIT), `CITATION.cff`, CI, `reproduce.py`.

## 0.1.0 (2024-04-24), tag `acri-2024`

The state of the repository when the ACRI 2024 paper was written (commit 9db72c1),
plus the branch `bigger_grid_search-consistent_learning` (August 2024: the
spatially non-uniform Fig. 1 and the example grids), merged in 1.0.0.
