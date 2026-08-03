#!/usr/bin/env bash
# E9 (reviewer xkCF follow-up, 2026-08-03):
#   iso     -> lambda_t = lambda_0 * eta_0/eta_t, so eta_t*lambda_t stays constant
#   matched -> every lambda shape rescaled to a common sum_t eta_t*lambda_t budget
# Both use a cosine learning rate, so they are directly comparable to the main
# protocol (cosine LR + constant lambda) rather than to the const-LR E8 arms.
set -euo pipefail
cd /home/yzy/GitHub/wd
PY=/home/yzy/.conda/envs/trace/bin/python
GPUS=${GPUS:-1,2,3}
WPG=${WPG:-2}
LOGDIR=outputs/logs
mkdir -p "$LOGDIR"

run_sweep () {
  local sweep="$1"
  echo "=== $(date '+%F %T') starting E9 sweep=$sweep gpus=$GPUS wpg=$WPG ==="
  PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26_wd_sched.py \
    --sweep "$sweep" --phase sgdm --gpus "$GPUS" --workers_per_gpu "$WPG" \
    2>&1 | tee "$LOGDIR/nips26_e9_${sweep}_sgdm.out"
  echo "=== $(date '+%F %T') finished E9 sweep=$sweep ==="
}

run_sweep iso
run_sweep matched

echo "=== $(date '+%F %T') analyzing E9 ==="
"$PY" -m analysis.nips26_e9_iso_sched 2>&1 | tee "$LOGDIR/nips26_e9_analyze.out"
echo "=== $(date '+%F %T') E9 queue done ==="
