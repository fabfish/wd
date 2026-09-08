#!/usr/bin/env bash
# E11 dense-budget ladder for iso_product (theory-predicted hyperbolic
# raise-up), SGDM + SGD -> analysis. Writes to the E11-dedicated CSV.
set -euo pipefail
cd /home/yzy/Documents/GitHub/wd
PY=/home/yzy/anaconda3/envs/nm/bin/python
GPUS=${GPUS:-6}
WPG=${WPG:-2}
CSV=e11_raise_up/results/nips26_e11_runs.csv
LOGDIR=outputs/logs
mkdir -p "$LOGDIR"

for phase in sgdm sgd; do
  echo "=== $(date '+%F %T') starting E11 raise_matched_iso phase=$phase gpus=$GPUS ==="
  PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26_wd_sched.py \
    --sweep raise_matched_iso --phase "$phase" --gpus "$GPUS" --workers_per_gpu "$WPG" \
    --csv "$CSV" 2>&1 | tee "$LOGDIR/nips26_e11f_matched_iso_${phase}.out"
  echo "=== $(date '+%F %T') finished phase=$phase ==="
done

echo "=== $(date '+%F %T') analyzing E11 ==="
"$PY" -m analysis.nips26_e11_wd_raise 2>&1 | tee "$LOGDIR/nips26_e11_analyze.out"
echo "=== $(date '+%F %T') E11 iso dense queue done ==="
