# Reliable training of ECA emulators

This study asks how to train, from random weights, a small convolutional
network that emulates an elementary cellular automaton (ECA) **exactly**, for
every one of the 256 rules and (nearly) every random seed. It starts from the
architecture of the ACRI 2024 paper (`ca_emulators.EcaEmulator`: a width-3
convolution with 8 ReLU "detectors" and a 1x1 "rule-table" convolution) and
from the 2024 training recipe (`ca_emulators.training.train_2024_recipe`),
which Fig. 3 of the paper used for rule 54 and which, according to the 2024
notes, had "persistent problems with i.a. rule 1".

The findings and the recommended recipe are in **[REPORT.md](REPORT.md)**.

## Layout

| path | content |
|---|---|
| `ensemble.py` | the engine: trains thousands of independent networks at once (weights with a leading member axis, plain TF ops, a Keras-identical Adam); all metrics |
| `sweep.py` | command line: runs a sweep file over rules x seeds, in resumable parts, optionally chunked and in parallel |
| `analyse.py` | tables and figures from the raw output, into `results/<sweep>/` |
| `validate_2024.py` | cross-check: the real Keras 2024 recipe on a few rules and seeds |
| `templates.py` | retrains a sample of minimal networks with their weights kept and compares their detector responses with the analytic one-hot template |
| `workstation.sh` | the full sweeps for the GPU workstation |
| `configs/*.json` | the sweep definitions (one per phase of the study) |
| `tests/` | fast pytest checks, incl. "one ensemble member = one standalone Keras network" |
| `results/<sweep>/` | committed summaries (CSV/Markdown) and figures (PNG) |
| `results/raw/` | per-run output (one CSV row per trained network); gitignored |

## How the engine works

Every network in an ensemble has receptive field 3 and is translation
invariant, so the mean loss over the B x N cells of a batch of configurations
equals a weighted sum of its losses on the 8 neighbourhoods, weighted by
their frequencies in the batch. One-step training (`mode: "patterns"`)
therefore evaluates each network on the 8 neighbourhood patterns only, with
the frequencies of a random batch (drawn from a precomputed pool; `data:
"random"`) or uniform weights (`data: "full"`). This is exact, and
`tests/test_ensemble.py` checks it against training on the configurations
themselves. Spacetime training (`mode: "configs"`) rolls real configurations
out T times through the network with backpropagation through time.

The ensemble loss is the sum of the members' mean losses, so each member's
gradient is its own, and the elementwise Adam keeps members independent; the
tests check that an ensemble member follows the same trajectory as a Keras
model with the same weights, for the 2024 architecture and for variants.
Each member's initialisation and data stream come from
`numpy.random.SeedSequence([base_seed, rule, seed, crc32(config name)])`, so a
run does not depend on the part, chunk or machine it runs in (up to
floating-point differences between CPU/GPU and XLA/non-XLA kernels).

## Metrics (one CSV row per trained network)

- `exact_final`: after thresholding at 0.5, the one-step output is correct on
  all 8 neighbourhoods (equivalently on the cyclic de Bruijn configuration
  00010111, `ca_emulators.rules.DE_BRUIJN`). The headline "success".
- `margin_final`: min over neighbourhoods of (y - 0.5)(2t - 1); positive iff
  exact, then equal to min |y - 0.5|.
- `exact_ever`, `first_exact_step`, `persist`, `n_flips`: exactness checked
  every `eval_every` steps.
- `cl_steps`, `cl_exact`, `cl_maxdev`: closed loop. The trained network is
  iterated 100 steps **without** binarisation from 8 test configurations of 32
  cells (the first is the tiled de Bruijn configuration, the others random)
  and compared with the exact ECA after thresholding.
- `cl_cert_eps`: a proof of closed-loop exactness. Interval bound propagation
  shows that every real-valued neighbourhood within eps of a binary one is
  mapped within eps of the correct state, so the unbinarised iteration stays
  exact from every configuration, of any size, forever. 0 if not certified
  for any eps in {0.01, ..., 0.45}.
- `dead_l1_*`, `dead_hidden_final`: units whose pre-activation is <= 0 on all
  8 neighbourhoods; `stuck_out_*`: neighbourhoods with target 1 whose output
  pre-activation is <= 0 under the 2024 ReLU->tanh head (zero gradient);
  `g000_init`, `g000_l1_init`: norm of the gradient of the loss on
  neighbourhood 000 at initialisation (the "dead origin").
- `n_pure_units`, `cover_pure`, `template_strict`: template recovery. A unit
  is pure if it fires (pre-activation > 0) on exactly one neighbourhood;
  `template_strict` means the 8 x 8 firing matrix of the minimal network is a
  permutation matrix, i.e. the analytic one-hot template up to a permutation
  of channels (and scale).
- `pretrain_*`: restarts, loss and number of passing candidates of the 2024
  pretraining loop.
- `learned_rule`: the rule the thresholded network implements; `analyse.py`
  uses it to decide whether a spacetime-trained network's T-step map is exact.

## Running it

Use the project's environment (Python 3.11, TensorFlow 2.14, `pip install -e .`
in the repository root). From `experiments/training/`:

```bash
# tests (about a minute)
python -m pytest tests -q -p no:cacheprovider

# smoke version of a sweep: a few rules, seeds and steps (one to two minutes)
python sweep.py configs/onestep_ablations.json --smoke
python analyse.py onestep_ablations --smoke

# a local sweep, 5 single-core worker processes; interrupted runs resume
python sweep.py configs/minimal_grid.json --workers 5
python analyse.py minimal_grid --grid sigmoid_bce_pm1_b0.0_softplus,relu_tanh_mse_01_b0.0_relu

# list configurations and work units without running
python sweep.py configs/width_depth.json --list
```

The sweeps of the study, in order, with their approximate cost on a laptop
CPU (i7-9850H, 5 workers):

| sweep | question | runs | laptop |
|---|---|---|---|
| `onestep_ablations` | one-factor ablations around the 2024 recipe | 16 x 8192 | 10 min |
| `minimal_grid` | head x encoding x bias x activation, minimal network | 48 x 8192 | 25 min |
| `width_depth` | width H x extra 1x1 depth D, two heads (run locally with `--seeds 8`) | 24 x 2048 | 60 min |
| `recipes` | candidate recipes at 128 seeds per rule | 6 x 32768 | 15 min |
| `recipe_robustness` | optimiser/data sensitivity of the recipe | 19 x 8192 | 15 min |
| `spacetime` | multi-step (BPTT) training, 88 representatives x 8 seeds (`--max-members 352`) | 19 x 704 | 60 min |

`python validate_2024.py` runs the real Keras recipe for 6 rules x 6 seeds
(about an hour) and writes `results/validate_2024.csv`.

### Options of `sweep.py`

- `--chunk i/n`: run every n-th work unit starting at i (split a sweep over
  machines or nights); `--workers K`: run the chunk in K processes, one core
  each (on a CPU the tiny networks are dispatch-bound, so processes scale
  almost linearly while threads do not).
- `--seeds N` or `--seeds a:b`, `--rules all|representatives|1,30,110`:
  override the sweep file; `--tag ws` writes to `results/raw/<sweep>-ws/`.
- `--max-members` (default 1024) and `--mem-mb` (default 2000) set how many
  networks train together in one part; the part size is stored in
  `meta.json` and reused on resumption once a part of that configuration
  exists.
- `--require-gpu`: abort unless TensorFlow sees a GPU.
- `--jit auto|on|off`: XLA compilation (auto: on for one-step training of
  networks without 1x1 layers, where it is 3-8x faster on a CPU).

To stop a run started with `--workers`, stop the worker processes too (on
Windows they outlive their launcher): every process whose command line
contains the sweep file. A part that was being trained is simply recomputed
on the next run; completed parts are never redone.

## Workstation (NVIDIA T400, 2 GB, Docker)

The image `tensorflow/tensorflow:2.14.0-gpu` (Python 3.11, CUDA) has the
pinned TensorFlow; `pip install -e .` adds the package and its pinned
dependencies. The repository is mounted at `/work`. Use one process for the
GPU. TensorFlow allocates memory on demand (`TF_FORCE_GPU_ALLOW_GROWTH`),
and parts are sized to a 1000 MB budget (`MEM_MB`), which leaves room for the
CUDA context on the 2 GB T400. Minimal networks then train 8192 at a time;
wide networks and spacetime runs get smaller parts automatically
(`sweep.py --list --tag ws --max-members 8192 --mem-mb 1000` shows them).
If a part still runs out of memory, the run stops with a message; rerun with
`MEM_MB=600`. A configuration's part size may change as long as none of its
parts exists yet.

**1. Check that the container sees the GPU** (this should print one
`PhysicalDevice ... GPU`):

```bash
docker run --gpus all --rm tensorflow/tensorflow:2.14.0-gpu \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**2. Install, test and run everything** (detached; about a day or more in
total, depending on the GPU):

```bash
cd /path/to/emulating-and-learning-CAs
docker run --gpus all -d --name ca-training -v "$PWD":/work -w /work \
  -e PYTHONDONTWRITEBYTECODE=1 -e TF_CPP_MIN_LOG_LEVEL=2 -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  tensorflow/tensorflow:2.14.0-gpu \
  bash -c 'pip install -q -e ".[test]" && cd experiments/training && \
           python -m pytest tests -q -p no:cacheprovider && bash workstation.sh; \
           chown -R "$HOST_UID:$HOST_GID" /work/experiments/training/results /work/src'
docker logs -f ca-training          # progress; also results/raw/<sweep>-ws/log_*.txt
```

Every sweep log starts with the GPUs TensorFlow sees. `workstation.sh`
passes `--require-gpu`, so it aborts rather than silently falling back to the
CPU.

**3. Split over several nights / resume.** Work units are parts of
configurations; each is written atomically and skipped when it exists. So
simply rerun the same command after an interruption. To split deliberately,
run `CHUNK=0/3`, `CHUNK=1/3` and `CHUNK=2/3` on different nights (add
`-e CHUNK=1/3` to `docker run`). To run only some sweeps, add e.g.
`-e ONLY="recipes spacetime_full"`. A single sweep by hand, inside the
container:

```bash
cd /work/experiments/training
python sweep.py configs/spacetime_full.json --tag ws --seeds 32 --chunk 0/2 \
  --max-members 8192 --mem-mb 1000 --require-gpu
python analyse.py spacetime_full-ws
```

`workstation.sh` runs, all with `--tag ws` so that they never mix with the
laptop results:

| sweep | seeds per rule | what it adds |
|---|---|---|
| `recipes` | 1024 | failure rate of the recipes below about 1e-5 (262,144 runs each) |
| `minimal_grid` | 128 | per-rule rates to about +-4 % |
| `onestep_ablations` | 128 | idem, incl. the 2024 recipe |
| `recipe_robustness` | 128 | idem |
| `width_depth_full` | 64 | H up to 128, D up to 2, three heads, 2024 pretraining filter at every width |
| `spacetime_full` | 32 | all 256 rules, 16 configurations of 32 cells, T up to 16, 5120 steps, wide net |

**4. Bring the results back.** The summaries and figures land in
`experiments/training/results/<sweep>-ws/` (small CSV/Markdown/PNG files).
Commit and push those from the workstation. The raw per-network CSVs stay
in `results/raw/<sweep>-ws/`, which is gitignored; archive them separately
if wanted (`tar czf training-raw-ws.tgz experiments/training/results/raw`).

```bash
git add experiments/training/results/*-ws
git commit -m "Training study: workstation sweeps"
git push
```

The GPU path has not been run by the author of this study (the laptop has no
NVIDIA GPU). The code uses only standard TF ops (no custom kernels), and the
test suite runs on the GPU too. If XLA causes trouble there, pass
`--jit off`.
