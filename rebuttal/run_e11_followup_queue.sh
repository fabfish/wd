#!/usr/bin/env bash
# E11 follow-up queue: SGD phase, matched budgets, multi-seed -> analysis.
# All runs write to the E11-dedicated CSV (nips26_runs.csv stays frozen).
set -euo pipefail
cd /home/yzy/Documents/GitHub/wd
PY=/home/yzy/anaconda3/envs/nm/bin/python
GPUS=${GPUS:-1,2}
WPG=${WPG:-3}
CSV=rebuttal/results/nips26_e11_runs.csv
LOGDIR=outputs/logs
mkdir -p "$LOGDIR"

run_sweep () {
  local sweep="$1"
  local phase="$2"
  local extra="${3:-}"
  echo "=== $(date '+%F %T') starting E11-followup sweep=$sweep phase=$phase $extra gpus=$GPUS wpg=$WPG ==="
  PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26_wd_sched.py \
    --sweep "$sweep" --phase "$phase" --gpus "$GPUS" --workers_per_gpu "$WPG" \
    --csv "$CSV" $extra \
    2>&1 | tee "$LOGDIR/nips26_e11f_${sweep}_${phase}.out"
  echo "=== $(date '+%F %T') finished sweep=$sweep phase=$phase ==="
}

# 1. SGD fixed-lambda control grid under cosine LR (only 3 legacy rows exist)
run_sweep e4_baselines sgd
# 2. SGD raise-up curves
run_sweep raise sgd
run_sweep raise_big sgd
# 3. matched-budget raise-up shapes
run_sweep raise_matched sgdm
run_sweep raise_matched sgd
# 4. multi-seed on the SGDM peak configs
run_sweep raise_ms sgdm "--seeds 42,123,2024"

echo "=== $(date '+%F %T') analyzing E11 ==="
"$PY" -m analysis.nips26_e11_wd_raise 2>&1 | tee "$LOGDIR/nips26_e11_analyze.out"
echo "=== $(date '+%F %T') E11 follow-up queue done ==="
