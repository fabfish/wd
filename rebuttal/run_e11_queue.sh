#!/usr/bin/env bash
# E11 queue: raise-up lambda shapes under cosine LR (SGDM) -> analysis.
set -euo pipefail
cd /home/yzy/Documents/GitHub/wd
PY=/home/yzy/anaconda3/envs/nm/bin/python
GPUS=${GPUS:-6}
WPG=${WPG:-3}
LOGDIR=outputs/logs
mkdir -p "$LOGDIR"

echo "=== $(date '+%F %T') starting E11 sweep=raise phase=sgdm gpus=$GPUS wpg=$WPG ==="
PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26_wd_sched.py \
  --sweep raise --phase sgdm --gpus "$GPUS" --workers_per_gpu "$WPG" \
  2>&1 | tee "$LOGDIR/nips26_e11_raise_sgdm.out"
echo "=== $(date '+%F %T') finished E11 sweep ==="

echo "=== $(date '+%F %T') analyzing E11 ==="
"$PY" -m analysis.nips26_e11_wd_raise 2>&1 | tee "$LOGDIR/nips26_e11_analyze.out"
echo "=== $(date '+%F %T') E11 queue done ==="
