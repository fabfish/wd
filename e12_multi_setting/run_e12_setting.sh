#!/usr/bin/env bash
# E12 queue for ONE (model, dataset) setting:
#   e12_fixed (both phases) -> make_anchors.py -> e12_matched (sgdm, sgd)
# All rows go to the E12-dedicated CSV; e11_raise_up/ stays frozen.
# Usage: run_e12_setting.sh <model> <dataset>   (env GPUS, WPG override)
set -euo pipefail
cd /home/yzy/Documents/GitHub/wd
PY=${PY:-/home/yzy/anaconda3/envs/nm/bin/python}
MODEL=${1:?usage: run_e12_setting.sh <model> <dataset>}
DATASET=${2:?usage: run_e12_setting.sh <model> <dataset>}
GPUS=${GPUS:-1,2}
WPG=${WPG:-3}
CSV=e12_multi_setting/results/e12_runs.csv
ANCHORS=e12_multi_setting/anchors.json
LOGDIR=e12_multi_setting/logs
mkdir -p "$LOGDIR"

run_sweep () {
  local sweep="$1" phase="$2" extra="${3:-}"
  echo "=== $(date '+%F %T') E12 sweep=$sweep phase=$phase model=$MODEL dataset=$DATASET gpus=$GPUS wpg=$WPG ==="
  PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26_wd_sched.py \
    --sweep "$sweep" --phase "$phase" --model "$MODEL" --dataset "$DATASET" \
    --gpus "$GPUS" --workers_per_gpu "$WPG" \
    --anchors "$ANCHORS" --csv "$CSV" $extra \
    2>&1 | tee "$LOGDIR/e12_${MODEL}_${DATASET}_${sweep}_${phase}.out"
  echo "=== $(date '+%F %T') finished sweep=$sweep phase=$phase ==="
}

run_sweep e12_fixed all
"$PY" e12_multi_setting/make_anchors.py "$CSV" "$ANCHORS"
run_sweep e12_matched sgdm
run_sweep e12_matched sgd
echo "=== $(date '+%F %T') E12 setting $MODEL/$DATASET done ==="
