# Emulating and learning cellular automata

A global update of an elementary cellular automaton (ECA) *is* a small convolutional
neural network: two convolutions with 40 fixed weights reproduce it exactly, for every
rule and every configuration. A non-uniform cellular automaton (νCA), in which every
cell follows its own rule, is the same network with one extra layer whose weights are
not shared between cells. Nothing has to be learned, and the output is bit-identical
to a direct simulation.

This repository accompanies

- M. Rollier, A. J. Daly, O. M. Bruno and J. M. Baetens, *Efficient Simulation of
  Non-uniform Cellular Automata with a Convolutional Neural Network*, in *Cellular
  Automata (ACRI 2024)*, LNCS 14978, pp. 121–131, Springer, 2024.
  [doi:10.1007/978-3-031-71552-5_11](https://doi.org/10.1007/978-3-031-71552-5_11),
  [arXiv:2409.02722](https://arxiv.org/abs/2409.02722);
- Sec. 7.1 of M. Rollier, *The rich landscape of iterated simplicity: cellular automata
  and network automata*, PhD thesis, Ghent University, 2026 (Ch. 7, "Neural network
  architectures as exact CA and NA simulators").

Every figure of the paper is regenerated from this repository, every claim about the
construction is checked by a test, and the outputs of the 2024 code are kept as a
reference. The state of the repository at the time of the paper is the tag
[`acri-2024`](https://github.com/mrollier/emulating-and-learning-CAs/tree/acri-2024).

## Coming from the paper or the thesis? Start here

Paper figure numbers first, thesis numbers in brackets.

| Where the text points | What to open or run |
|---|---|
| Sec. 1.2 [7.1.1], Fig. 2 [7.2] and Tab. 7.1: the 40-parameter ECA network | [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb), sections 1–2; the weights are in [`src/ca_emulators/weights.py`](src/ca_emulators/weights.py), the network in [`src/ca_emulators/eca.py`](src/ca_emulators/eca.py); `python figures/fig2_decomposition.py` |
| Sec. 2.2 [7.1.2]: "the code and annotations of the `NucaEmulator` class" | [`src/ca_emulators/nuca.py`](src/ca_emulators/nuca.py): `NucaEmulator(...).model()` (locally connected, Eq. 1 [7.1]) and `.model_dense()` (dense, Eq. 2 [7.2]); walkthrough section 3 |
| Fig. 1 [7.1]: a νCA with rules 30 and 90 | `python figures/fig1_nuca_example.py` |
| Fig. 3 [7.3]: a CNN trained to emulate rule 54 | `python figures/fig3_training.py` (seeded re-run of the 2024 recipe, [`src/ca_emulators/training.py`](src/ca_emulators/training.py)); better recipes in [`experiments/training/`](experiments/training/) |
| Fig. 4 [7.4]: an 8-rule νCA in CellPyLib | `python figures/fig4_nuca_cellpylib.py` |
| Tab. 1 [7.2] and Fig. 5 [7.5]: the four benchmark scenarios | `python figures/fig5_benchmarks.py` (archived 2024 data in [`data/benchmarks_2024/`](data/benchmarks_2024/)); re-run the protocol with `python scripts/benchmark_published.py`; faster ways to run the CNNs in [`experiments/benchmarks/`](experiments/benchmarks/) |
| Sec. 7.1.4: "the output of both CNNs was verified to be bit-identical to that of CellPyLib" | `python -m pytest` ([`verification/`](verification/), claims C1–C8 in [`docs/provenance.md`](docs/provenance.md)) |
| Sec. 7.2: Life-like network automata as graph neural networks | not in this repository: the `LLNA` class in [`mrollier/network_automata_robustness`](https://github.com/mrollier/network_automata_robustness) (`src/automata.py`) |

## The construction in brief

Layer 1 is a width-3 convolution with periodic padding and eight ReLU channels, the
*neighbourhood detectors*. Channel *i* has weight +1 where the neighbourhood with binary
representation *i* holds a 1, weight −ω where it holds a 0, and bias 1 − *h* (*h* the
number of ones). For any ω ≥ 1 exactly one channel fires per cell: a one-hot encoding
of the neighbourhood. Layer 2 is a width-1 convolution whose eight weights are the rule
table. In total, 32 + 8 = 40 parameters.

For a νCA with *N*<sub>R</sub> rules, layer 2 gets *N*<sub>R</sub> output channels (the
candidate updates of the whole configuration under every rule) and a third layer selects,
per cell, the channel of its allocated rule: either a `LocallyConnected1D` layer (a
convolution without weight sharing) or a dense layer with a mostly-zero weight matrix.

```python
import numpy as np
from ca_emulators import EcaEmulator, NucaEmulator

model = EcaEmulator(64, rule=110).model()              # a Keras model with 40 frozen weights
diagram = EcaEmulator(64, rule=110).simulate(np.random.randint(2, size=64), n_updates=63)

nuca = NucaEmulator(32, rules=[30, 90], rule_alloc=np.random.randint(2, size=32))
lc_model, dense_model = nuca.model(), nuca.model_dense()   # 32 + 16 + 64 and 32 + 16 + 2048 weights
```

Both classes are exact and frozen by default. Without a rule (`EcaEmulator(N)`), or with
`trainable=True`, the same architecture can be trained.

## Install

The package pins the versions of the paper: Python 3.9–3.11 and TensorFlow 2.14 (Keras 2).
Keras 3, shipped with TensorFlow 2.16 and later, no longer provides `LocallyConnected1D`.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`requirements-lock-windows.txt` freezes the exact environment the repository was
developed and tested in (Windows 11, Python 3.11.5). On a Linux machine with an NVIDIA
GPU, `tensorflow[and-cuda]==2.14.0` or the Docker image `tensorflow/tensorflow:2.14.0-gpu`
provide GPU support (only the training study benefits from it).

## Reproduce everything

```bash
python reproduce.py quick                   # fast checks, Figs 1, 2, 4, 5, the notebook (about 3 min)
python reproduce.py all                     # plus the slow checks, Fig. 3 and the talk grids
python reproduce.py all --with-benchmarks   # plus a re-run of the published benchmark (80 min)
```

Figures are written to `output/` under the file names of the thesis source.
[`docs/provenance.md`](docs/provenance.md) maps every figure and claim to its script,
command, expected output and status.

## Repository layout

```
src/ca_emulators/   the package: eca.py, nuca.py (the emulators), weights.py (analytic weights,
                    Eqs. 7.1-7.2), layers.py (periodic padding), reference.py (numpy and CellPyLib
                    reference simulators), rules.py, simulate.py, training.py (Fig. 3 recipe), plotting.py
figures/            one script per published figure, writing to output/
scripts/            one-off generators: golden outputs of the 2024 code, inputs decoded from the
                    published figures, re-run of the published benchmark
data/               golden/, published_inputs/, benchmarks_2024/ (the paper's timings), benchmarks_2026/
verification/       pytest: the claims C1-C8 of docs/provenance.md
notebooks/          walkthrough.ipynb, the guided tour
experiments/        training/ and benchmarks/: new work beyond the paper, each with a REPORT.md
docs/provenance.md  figure/claim -> script -> command -> expected -> status
```

## Differences from the paper

- **Layer 1.** The paper describes the first layer as a convolution with weights
  (4, 2, 1) giving an integer neighbourhood code, followed by a one-hot encoding. The
  code (in 2024 and now) fuses both steps into the eight ReLU detectors of thesis
  Tab. 7.1; the index of the active detector is that integer. The (4, 2, 1) code appears
  only as the "Integer neighbourhood encoding" strip of Fig. 2. The walkthrough shows
  that a literal (4, 2, 1)-plus-one-hot network gives the same result with an extra
  hidden layer.
- **Fig. 4** uses an allocation that shifts by one cell per time step, so that νCA is
  non-uniform in time as well as in space; the text of the paper otherwise considers
  spatial non-uniformity only. The emulators fix the allocation in time; the test suite
  shows that swapping the selector weights between updates handles the Fig. 4 case too.
- **The 2024 emulators were not exact by default.** Their default `train_triplet_id=True`
  left the detector weights random; every published result used `train_triplet_id=False`.
  The rebuilt classes are exact by default (`train_triplet_id` still works, with a
  deprecation warning).
- **Benchmarks.** The 2024 benchmark ran the CNNs through `model.predict`, once per time
  step. Most of the few seconds of "fixed cost" of the CNNs in Fig. 5 is the overhead of
  those calls, not of the networks; see [`experiments/benchmarks/`](experiments/benchmarks/).
  In the samples scenario the dense CNN overtakes CellPyLib between 64 and 128 samples,
  a little earlier than the thesis's "a few hundred samples" (checked in
  `verification/test_c6_benchmark_2024.py`).

## Where the 2024 files went

| 2024 | now |
|---|---|
| `src/nn/eca.py`, `src/nn/nuca.py` | `src/ca_emulators/eca.py`, `nuca.py` |
| `src/custom_tf_classes/initializers.py` | `src/ca_emulators/weights.py` |
| `src/custom_tf_classes/layers.py` (`PeriodicConv1D`) | `src/ca_emulators/layers.py` (`PeriodicPadding1D` + `Conv1D`) |
| `src/train/train.py`, `custom_tf_classes/callbacks.py`, `scripts/eca_optimisation.py` | `src/ca_emulators/training.py` |
| `src/visual/decomposition.py`, `histories.py` | `src/ca_emulators/plotting.py` |
| `tests/test_show-nuca.py`, `test_decomposition_eca.py`, `test_nuca_cellpylib.py`, `test_eca.py` | `figures/fig1_…`, `fig2_…`, `fig4_…`, `fig3_…` |
| `tests/performance_comparisons/comparison_*.py` | `scripts/benchmark_published.py`, `figures/fig5_benchmarks.py` |
| `tests/test_equivalence_cpl_cnn.py` | `verification/` |
| `tests/generate-eca.py`, `old/generate-nuca.py` | `figures/extra_talk_grids.py` |
| `old/`, `scripts/eca_gridsearch_lr-bs.py`, activations, constraints | removed; see the tag `acri-2024` |

## Related work

W. Gilpin, *Cellular automata as convolutional neural networks*, Phys. Rev. E 100,
032402 (2019), showed that any CA can be represented by a CNN and studied how such
networks learn; code at [williamgilpin/convoca](https://github.com/williamgilpin/convoca).
Simulations in the paper are compared with [CellPyLib](https://github.com/lantunes/cellpylib)
(L. M. Antunes, JOSS 6, 3608, 2021).

## Citation and licence

Please cite the paper (see [`CITATION.cff`](CITATION.cff)). The code is released under the
MIT licence ([`LICENSE`](LICENSE)).
