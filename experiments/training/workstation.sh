#!/usr/bin/env bash
# Full sweeps of the training study for the Linux workstation (NVIDIA T400,
# 2 GB), run inside tensorflow/tensorflow:2.14.0-gpu (see README.md).
#
#   bash workstation.sh                 # both sweeps, in order
#   CHUNK=1/3 bash workstation.sh       # one third of every sweep (other nights: 0/3, 2/3)
#   ONLY="spacetime_full" bash workstation.sh
#
# Output goes to results/raw/<sweep>-ws/ (resumable: existing parts are
# skipped) and the summaries to results/<sweep>-ws/.
set -euo pipefail
cd "$(dirname "$0")"

CHUNK="${CHUNK:-0/1}"
GPU_OPTS="--tag ws --chunk $CHUNK --max-members 8192 --mem-mb ${MEM_MB:-1000} --require-gpu"

# sweep file | seeds | question it answers at scale
# The high-seed sweeps ran on the laptop CPU instead (laptop.sh, 29 September
# 2026; summaries in results/<sweep>-large/). To repeat them on the GPU, add:
#   "recipes 1024" "minimal_grid 128" "onestep_ablations 128" "recipe_robustness 128"
SWEEPS=(
  "width_depth_full 64"      # H up to 128, three heads, 2024 pretraining filter
  "spacetime_full 32"        # all 256 rules, T up to 16, larger configurations
)

for entry in "${SWEEPS[@]}"; do
  read -r sweep seeds <<<"$entry"
  if [[ -n "${ONLY:-}" && " $ONLY " != *" $sweep "* ]]; then
    continue
  fi
  echo "=== $sweep ($seeds seeds per rule, chunk $CHUNK) ==="
  python sweep.py "configs/$sweep.json" --seeds "$seeds" $GPU_OPTS
done

for entry in "${SWEEPS[@]}"; do
  read -r sweep _ <<<"$entry"
  if [[ -d "results/raw/$sweep-ws" ]]; then
    python analyse.py "$sweep-ws" || echo "analysis of $sweep-ws failed"
  fi
done
