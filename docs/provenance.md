# Provenance: figure / claim → script → command → expected → status

Figure numbers: ACRI 2024 paper first, thesis in brackets. "Observed" is a run on the
development laptop (Intel i7-9850H, 6 cores, Windows 11, Python 3.11.5, TensorFlow
2.14.0, CellPyLib 2.4.0), the same CPU as the 2024 benchmarks. Commands run from the
repository root after `pip install -r requirements.txt && pip install -e .`.

## Figures

| Fig | Output stem (thesis source) | Script | Expected | Status |
|---|---|---|---|---|
| 1 [7.1] | `spacetime_diagram-nuca-30_90-vertical` | `figures/fig1_nuca_example.py` | the published allocation (rules 30/90, uniform in time) and diagram, 32 × 32 | ✅ reproduced cell for cell (asserted in the script and in C5) |
| 2 [7.2] | `eca-N32-rule54-decomposition` | `figures/fig2_decomposition.py` | the published input, one-hot and output strips for rule 54 | ✅ reproduced cell for cell (the one-hot panel is the detector layer's activation) |
| 3 [7.3] | `plot_configs-ECA-32cells-rule54-40epochs_bs64_lr0p005` | `figures/fig3_training.py` | a trained network converging to rule 54 on the published example | ✅ qualitatively reproduced by a seeded re-run (seed 2024, fixed in advance): 132 pretraining restarts, final validation MSE 3.0e-6, exact after thresholding, margin 0.497 (`output/fig3_training.json`); the published run was unseeded, so the heatmap differs in detail |
| 4 [7.4] | `cellpylib-spacetime_diagram-Nrules8` | `figures/fig4_nuca_cellpylib.py` | rules 41, 46, 72, 105, 158, 193, 232, 254, allocation shifted by one cell per step, CellPyLib diagram | ✅ reproduced cell for cell |
| 5 [7.5] | `nuca-comparison-{Nrules-N256_T32_S32, T-N64_Nrules4_S32, N-T32_Nrules4_S32, S-N32_T32_Nrules4}_avg-from-10` | `figures/fig5_benchmarks.py` | the four panels from the archived 2024 timings | ✅ redrawn from `data/benchmarks_2024` (same data, same layout) |
| 5, re-run | `…-overlay2026` | `scripts/benchmark_published.py` (about 85 min), then `figures/fig5_benchmarks.py --overlay 2026` | the 2024 protocol re-timed on the same CPU | ✅ qualitatively reproduced on 29 September 2026 (details below) |

### The 2026 re-run of the published benchmark

`scripts/benchmark_published.py` ran the 2024 protocol again on the same laptop CPU
(Python 3.11.5, TensorFlow 2.14.0, idle machine; every diagram checked against the
numpy reference). The qualitative findings of Fig. 5 hold:

- the dense CNN overtakes CellPyLib between 64 and 96 cells, and between 64 and 128
  samples, the same crossovers as in 2024;
- for every number of rules the dense CNN is fastest and CellPyLib slowest;
- all three methods scale linearly with the number of time steps.

Absolute times differ from 2024 by factors 0.6 to 1.9. The CNNs are mostly faster now
(ratio 0.6-0.9 in the N and S scenarios). CellPyLib matches 2024 within 5% in the N and
S scenarios, but is 1.7 times slower in the T scenario. The 2024 data themselves vary this
much between scenarios measured in different sessions: the same configuration took
1.51 s and 3.58 s for the dense CNN in two 2024 scenarios. Session-to-session variation
of this size is the reason the report of `experiments/benchmarks` separates warm and
cold timings and records a calibration workload.
| talks | `examples-of-ecas_6x6`, `examples-of-nucas_6x6` | `figures/extra_talk_grids.py` | 36 random ECAs / two-rule nuCAs (seeded) | ✅ |

## Claims

| Claim | Source | Test | Status |
|---|---|---|---|
| C1 The eight detectors of Tab. 7.1 give an exact one-hot encoding for any ω ≥ 1 (not for ω < 1) | Sec. 7.1.1, Tab. 7.1 | `verification/test_c1_detectors.py` | ✅ |
| C2 With analytic weights, the 40-parameter CNN performs ECA updates exactly (all 256 rules, any N, any number of updates) | Sec. 1.2 / 7.1.1 | `verification/test_c2_eca_exact.py` | ✅ (N = 1 to 64, up to 256 updates) |
| C3 Parameter counts 40, (3+1)·8 + 8N_R + N_R N (Eq. 1 [7.1]) and (3+1)·8 + 8N_R + N_R N² (Eq. 2 [7.2]); 352 and 8288 for N = 32, N_R = 8 | Sec. 2.2 / 7.1.2 | `verification/test_c3_param_counts.py` | ✅ (Eq. 7.1 holds for `LocallyConnected1D` modes 1 and 3; mode 2 has the dense count) |
| C4 Both νCA networks reproduce the νCA exactly, up to one rule per cell | Sec. 2.2 / 7.1.2 | `verification/test_c4_nuca_exact.py` | ✅ (also for a time-varying allocation, by swapping the selector) |
| C5 The published figures are reproduced | Figs 1-4 | `verification/test_c5_published_figures.py` | ✅ |
| C6 The statements of Sec. 7.1.4 about Fig. 7.5 hold for the archived timings | Sec. 3 / 7.1.4 | `verification/test_c6_benchmark_2024.py` | ✅, with one nuance: in the samples scenario the dense CNN overtakes CellPyLib between S = 64 and 128 (4.11 s vs 4.52 s at S = 128), not only from "a few hundred samples" |
| C7 The architecture can be trained from random weights to emulate rule 54 | Sec. 1.2 / 7.1.1, Fig. 3 | `verification/test_c7_training_recipe.py` (slow) | ✅ with the seeded 2024 recipe |
| C8 The rebuilt package gives the same outputs as the 2024 code | – | `verification/test_golden_2024.py` | ✅ bit for bit |
| C9 (new, not in the paper) With the recommended recipe, training from random weights is exact for every rule and stays exact in unbinarised closed loop | `experiments/training/REPORT.md` (32768/32768 runs) | `verification/test_c9_training_recipe.py` (rules 1, 30, 54, 105, 110, 150; all 256 rules in the slow suite) | ✅ |
| "The output of both CNNs was verified to be bit-identical to that of CellPyLib" | Sec. 7.1.4 | C2, C4 and `verification/test_reference.py` (numpy reference = CellPyLib for all 256 rules) | ✅ |

Run the fast checks with `python -m pytest -m "not slow"` (about a minute) and all of
them with `python -m pytest` (a few more minutes, most of it the Fig. 3 training).

## Data

| File(s) | Produced by | Notes |
|---|---|---|
| `data/benchmarks_2024/*.npy` | the 2024 scripts `tests/performance_comparisons/comparison_*.py` (tag `acri-2024`), April 2024 | copied byte for byte from the only copy (`learning_automata/data/nuca/`); sha256 below |
| `data/golden/golden_2024.npz`, `ENV.txt` | `scripts/capture_golden_2024.py`, run with the acri-2024 `src/` in the paper's conda environment (Python 3.11.8, TF 2.14.0) | every stored output checked against CellPyLib before saving |
| `data/published_inputs/*.npz`, `SOURCES.txt` | `scripts/extract_published_inputs.py` from the figure PDFs of the paper's LaTeX source | each decoding verified by re-evolving it; Figs 1 and 4 cross-checked against the published Springer PDF |
| `data/benchmarks_2026/` | `scripts/benchmark_published.py` | the 2024 protocol re-timed |

sha256 of the 2024 timing files:

```
b7bd8ed1c54f9293cd6d6ed30ea91e9344cd7ca95bca9e6a34f69c446b16bacb  nuca-comparison-N-T32_Nrules4_S32_avg-from-10.npy
ac95ab8a248ee95e321fd9bf5862eebce7970d9fb1e2aa0b9838a39f0f498f5e  nuca-comparison-Nrules-N256_T32_S32_avg-from-10.npy
7e53ea3db21a09e437d45f36498c685447bafb9e17a7016773652b95e22caef3  nuca-comparison-S-N32_T32_Nrules4_avg-from-10.npy
5b6832bd8fac21a722c49cf58f9c0eee4122118c73e24aafc9a9da59ef5edd7a  nuca-comparison-T-N64_Nrules4_S32_avg-from-10.npy
```

The sha256 hashes of the source figure PDFs are in `data/published_inputs/SOURCES.txt`.
