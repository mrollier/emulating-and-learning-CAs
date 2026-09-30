# Orientation for future Claude Code sessions

This repository is the reproducibility package of the ACRI 2024 paper *Efficient
Simulation of Non-uniform Cellular Automata with a Convolutional Neural Network*
(Rollier, Daly, Bruno, Baetens; LNCS 14978, pp. 121-131; arXiv:2409.02722) and of
Sec. 7.1 of Michiel Rollier's PhD thesis (Ghent University, 2026). The main message:
an ECA or nuCA can be emulated *exactly* by a CNN with analytically set weights (40 for
an ECA). Efficiency is secondary. Sec. 7.2 of the thesis (LLNAs with a GNN) lives in the
separate repository `network_automata_robustness` and is deliberately not included.

## Layout
- `src/ca_emulators/`: the package; everything else imports from here and never
  duplicates the construction. `weights.py` holds the analytic weights (Tab. 7.1,
  Eqs. 7.1-7.2) as numpy functions; the emulators copy them in with `set_weights`.
- `figures/`: one script per published figure, writing to `output/` (gitignored).
- `scripts/`: one-off generators (golden outputs of the 2024 code, inputs decoded from
  the published PDFs, the published benchmark protocol).
- `data/`: golden/, published_inputs/, benchmarks_2024/ (irreplaceable), benchmarks_2026/.
- `verification/`: pytest for claims C1-C9 (see docs/provenance.md); `-m slow` for the
  long ones. Must exit zero.
- `training.train_recipe` + `verify` implement the recipe of `experiments/training`
  (approved by the owner on 2026-09-29); `train_2024_recipe` stays the faithful Fig. 3
  reproduction. Trained recipe models output logits: pass `logits=True` to `verify`.
- `experiments/training`, `experiments/benchmarks`: new work beyond the paper, each with
  its own README/REPORT; not imported by the package.
- `reproduce.py quick|all [--with-benchmarks]`.

## Hazards
- **Keras 3** (TensorFlow >= 2.16) removed `LocallyConnected1D`. The package pins
  TF 2.14 / Keras 2 and refuses to import under Keras 3.
- **`n_updates` vs `timesteps`.** New code counts updates (`n_updates`); a diagram has
  `n_updates + 1` rows. CellPyLib's `timesteps` counts rows (initial row included); the
  emulators' `timesteps` counts updates per forward pass.
- **CellPyLib `memoize`** caches by neighbourhood only and is wrong for nuCAs; always
  `memoize=False`.
- **Parameter counts** depend on the `LocallyConnected1D` implementation mode: Eq. 7.1
  holds for modes 1 (default, as in 2024) and 3; mode 2 stores N_R N^2 weights.
- **uint8 arithmetic**: `rules.neighbourhood_patterns()` is uint8; cast before
  subtracting (the detector bias 1 - h once wrapped to 255).
- **Op determinism** (`tf.config.experimental.enable_op_determinism`) is process-global;
  only `figures/fig3_training.py` switches it on, never the tests.
- **Windows**: the console is cp1252, so set `PYTHONIOENCODING=utf-8` for output with
  Greek letters or arrows; the repository sits in OneDrive with a long path, so git
  worktrees inside the scratchpad fail (path too long); `core.autocrlf=true` is set,
  data files are marked binary in `.gitattributes`.
- MiKTeX prints "luatex: critical issue: ... unsupported version of Windows" when
  figures use LaTeX; harmless.

## Environment
Development venv: `%LOCALAPPDATA%\venvs\ca-emulators` (Python 3.11.5, TF 2.14.0,
package installed editable). The paper's original conda environment lived in the local
clone `../learning_automata`, which was used once to capture `data/golden` and then
deleted (2026-09-29, with the owner's approval); its package list is
`docs/env/paper-env-2024.txt`, and its gitignored outputs (figures, the 85 trained 2024
models, the timing data) are archived in `../learning_automata-outputs-2024.zip`.
Long runs: the CPU benchmark (`experiments/benchmarks`, 8 h) and the high-seed training
sweeps (`experiments/training/laptop.sh`, 2.7 h) ran overnight on the laptop on
29-30 September 2026. The GPU benchmark variant and the two GPU-sized training sweeps
(`workstation.sh`) are left for the owner's Linux workstation, see `docs/workstation.md`.

## Conventions
- UK English in prose and comments (behaviour, neighbourhood, initialise).
- Every RNG is seeded; the seed is written in the script.
- Tests encode the claims of the paper and thesis; never loosen a failing check,
  report the discrepancy with the actual value.
- Ask the owner before changing existing core modules in `src/ca_emulators` beyond
  bug fixes; stage files by explicit path.
