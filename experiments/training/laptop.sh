#!/usr/bin/env bash
# The large sweeps of the training study, sized for a laptop CPU (no GPU).
#
# Used instead of workstation.sh when no GPU machine is available. The
# high-seed sweeps run in full (about 14 CPU hours); the wide-network and
# spacetime sweeps of workstation.sh would need hundreds of CPU hours, so only
# the points missing from REPORT.md are completed, in the local sweeps
# themselves (resumed in place).
#
#   bash laptop.sh                      # WORKERS=4 by default; PYTHON=python
#   WORKERS=6 bash laptop.sh
#
# Summaries of the high-seed sweeps go to results/<sweep>-large/.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
OPTS="--workers ${WORKERS:-4}"

echo "=== gaps: wide networks (8 seeds) and wide spacetime at T = 8 ==="
"$PY" sweep.py configs/width_depth.json --seeds 8 \
    --only sigmoid_bce_H32_D2,sigmoid_bce_H64_D1,sigmoid_bce_H64_D2 $OPTS
"$PY" sweep.py configs/spacetime.json --only st_wide_T8_all,st_wide_T8_final --max-members 352 $OPTS
"$PY" analyse.py width_depth
"$PY" analyse.py spacetime

# sweep | seeds per rule | configurations shown in the success grid
SWEEPS=(
  "recipes 1024 recipe_minimal,head_only_2024"
  "minimal_grid 128 -"
  "onestep_ablations 128 baseline_2024,nopre,head_sigmoid_bce"
  "recipe_robustness 128 -"
)
for entry in "${SWEEPS[@]}"; do
  read -r sweep seeds grid <<<"$entry"
  echo "=== $sweep ($seeds seeds per rule) ==="
  "$PY" sweep.py "configs/$sweep.json" --seeds "$seeds" --tag large $OPTS
  if [[ "$grid" == "-" ]]; then
    "$PY" analyse.py "$sweep-large"
  else
    "$PY" analyse.py "$sweep-large" --grid "$grid"
  fi
done
