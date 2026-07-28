#!/usr/bin/env bash
# E8 queue: SGD phase -> SGDM phase -> analysis.
set -euo pipefail
cd /home/yzy/GitHub/wd
PY=/home/yzy/.conda/envs/trace/bin/python
GPUS=${GPUS:-0,2,3}
WPG=${WPG:-2}
LOGDIR=outputs/logs
mkdir -p "$LOGDIR"

run_phase () {
  local phase="$1"
  echo "=== $(date '+%F %T') starting E8 phase=$phase gpus=$GPUS wpg=$WPG ==="
  PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26_wd_sched.py \
    --phase "$phase" --gpus "$GPUS" --workers_per_gpu "$WPG" \
    2>&1 | tee "$LOGDIR/nips26_e8_${phase}.out"
  echo "=== $(date '+%F %T') finished E8 phase=$phase ==="
}

run_phase sgd
run_phase sgdm

echo "=== $(date '+%F %T') analyzing E8 ==="
"$PY" -m analysis.nips26_e8_wd_sched 2>&1 | tee "$LOGDIR/nips26_e8_analyze.out"
echo "=== $(date '+%F %T') E8 queue done ==="
