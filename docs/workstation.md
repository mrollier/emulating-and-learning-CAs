# Running the long jobs on the Linux workstation

Two jobs are too long for the laptop and run on the workstation (Linux,
NVIDIA T400 with 2 GB, driver 570 / CUDA 12.8):

- the new emulation benchmark, `experiments/benchmarks` (about 6-8 hours on
  the laptop; CPU, with an optional GPU extra);
- the large sweeps of the training study, `experiments/training` (see its
  README for the sweep commands; the ensemble engine uses the GPU).

The re-run of the *published* benchmark protocol
(`scripts/benchmark_published.py`) stays on the laptop: it has the same CPU as
the 2024 measurements, which is what makes the comparison meaningful.

## 1. Get the code

```bash
git clone https://github.com/mrollier/emulating-and-learning-CAs.git
cd emulating-and-learning-CAs      # main (release 1.0.0 and later)
```

A clone of the branch `cleanup/v1`, from before the 1.0.0 merge, works as well; push
its results to that branch.

## 2. Start the container

The official image has TensorFlow 2.14 (the paper's version) with CUDA 11.8,
which the 570 driver runs. The NVIDIA Container Toolkit must be installed
(`docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
tests it).

```bash
docker run --gpus all -it --rm \
    -v "$PWD":/work -w /work \
    -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
    tensorflow/tensorflow:2.14.0-gpu bash
```

Inside the container:

```bash
pip install -r requirements.txt && pip install -e .
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # expect one GPU
python -m pytest -m "not slow"                      # the verification suite, about a minute
python -m pytest experiments/benchmarks/tests experiments/training/tests
```

The container runs as root, so files it writes into the repository belong to
root. Before leaving it, give them back: `chown -R $HOST_UID:$HOST_GID /work`.

For the long runs, start the container detached so that they survive logging
out, and follow the log:

```bash
docker run --gpus all -d --name ca-bench -v "$PWD":/work -w /work \
    -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) tensorflow/tensorflow:2.14.0-gpu \
    bash -lc "pip install -q -r requirements.txt && pip install -q -e . && \
              python experiments/benchmarks/run.py --all; chown -R \$HOST_UID:\$HOST_GID /work"
docker logs -f ca-bench
```

## 3. The emulation benchmark

```bash
python experiments/benchmarks/run.py --all --dry-run | head     # 2295 jobs
python experiments/benchmarks/run.py --all                      # CPU, GPU hidden; hours
python experiments/benchmarks/run.py --all --resume             # after an interruption
python experiments/benchmarks/plot.py                           # figures + summary table
# optional: the same with the GPU visible (separate results file)
python experiments/benchmarks/run.py --all --device gpu --threads default
python experiments/benchmarks/plot.py --csv experiments/benchmarks/results/benchmark_full_gpu.csv
```

Nothing else should run on the machine meanwhile; the metadata file records
the CPU load before the run and a calibration workload before and after it.

## 4. The training study

Follow `experiments/training/README.md` (section "Workstation"). The sweeps
are chunked and resumable; the ensemble sizes are chosen to fit in 2 GB of
GPU memory.

## 5. Bring the results back

Raw per-job output stays in `results/raw/` (ignored by git). Commit the
summaries on the branch and push them:

```bash
git add experiments/benchmarks/results experiments/training/results
git commit -m "Benchmark and training results from the workstation"
git push                  # to main, or to cleanup/v1 if you cloned that branch
```

On the laptop, `git pull` brings them in; the reports (`REPORT.md` in both
folders) are then completed from them.
