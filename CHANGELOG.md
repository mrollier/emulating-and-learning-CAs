# Changelog

## 1.0.0 (unreleased)

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
- **Removed**: bytecode, `old/`, the Windows-only `environment.yml`, the unused
  training exploration (generalised sigmoid, constraints, grid search). All of it
  remains available at the tag `acri-2024`.
- Licence (MIT), `CITATION.cff`, CI, `reproduce.py`.

## 0.1.0 (2024-04-24), tag `acri-2024`

The state of the repository when the ACRI 2024 paper was written (commit 9db72c1),
plus the branch `bigger_grid_search-consistent_learning` (August 2024: the
spatially non-uniform Fig. 1 and the example grids), merged in 1.0.0.
