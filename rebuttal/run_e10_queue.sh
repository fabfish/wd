#!/usr/bin/env bash
# E10 (reviewer xkCF follow-up, 2026-08-03): held-out test of the constant C.
# Phases are chained in order so that the prediction is written before any
# held-out grid is trained; that ordering is what makes the test blind.
set -euo pipefail
cd /home/yzy/GitHub/wd
PY=/home/yzy/.conda/envs/trace/bin/python
GPUS=${GPUS:-1,2,3}
WPG=${WPG:-6}
LOGDIR=outputs/logs
mkdir -p "$LOGDIR"

run_ds () {
  local ds="$1"
  local phases="$2"
  echo "=== $(date '+%F %T') E10 $ds phases=$phases ==="
  PYTHONUNBUFFERED=1 "$PY" -m mlp_wd.scripts.run_e10_c_width \
    --dataset "$ds" --phases "$phases" --gpus "$GPUS" --workers_per_gpu "$WPG" \
    2>&1 | tee "$LOGDIR/e10_${ds}_${phases//,/_}.out"
  echo "=== $(date '+%F %T') E10 $ds phases=$phases done ==="
  "$PY" -m mlp_wd.analysis.report_e10_c_width --dataset "$ds" \
    2>&1 | tee "$LOGDIR/e10_${ds}_report.out"
}

run_ds "${1:-mnist}" "${2:-ladder,predict,heldout}"
echo "=== $(date '+%F %T') E10 queue done ==="
