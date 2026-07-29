#!/usr/bin/env bash
# E8 follow-up (section 5): joint multiplier -> long T=200 -> e4 baselines -> analyze.
set -euo pipefail
cd /home/yzy/GitHub/wd
PY=/home/yzy/.conda/envs/trace/bin/python
GPUS=${GPUS:-0,1,2,3}
WPG=${WPG:-2}
LOGDIR=outputs/logs
mkdir -p "$LOGDIR"

run_sweep () {
  local sweep="$1"
  echo "=== $(date '+%F %T') starting E8 sweep=$sweep gpus=$GPUS wpg=$WPG ==="
  PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26_wd_sched.py \
    --sweep "$sweep" --phase sgdm --gpus "$GPUS" --workers_per_gpu "$WPG" \
    2>&1 | tee "$LOGDIR/nips26_e8_${sweep}_sgdm.out"
  echo "=== $(date '+%F %T') finished E8 sweep=$sweep ==="
}

run_sweep joint
run_sweep long
run_sweep e4_baselines

echo "=== $(date '+%F %T') analyzing E8 follow-up ==="
"$PY" -m analysis.nips26_e8_wd_sched 2>&1 | tee "$LOGDIR/nips26_e8_followup_analyze.out"
echo "=== $(date '+%F %T') E8 follow-up queue done ==="
