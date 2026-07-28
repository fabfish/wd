#!/usr/bin/env bash
# Continue rebuttal queue after E5c: E4 fill → e6b_lambda → e1_sched → report.
# E7 / Hessian are launched separately on spare GPUs.
set -euo pipefail
cd /home/yzy/GitHub/wd
PY=/home/yzy/.conda/envs/trace/bin/python
export PYTHONUNBUFFERED=1
LOG=outputs/logs/nips26_continue_queue.log
mkdir -p outputs/logs

run() {
  local exp="$1" gpus="$2" workers="$3"
  echo "=== $(date '+%F %T') starting $exp gpus=$gpus workers/gpu=$workers ===" | tee -a "$LOG"
  $PY rebuttal/run_nips26.py --exp "$exp" --gpus "$gpus" --workers_per_gpu "$workers" \
    2>&1 | tee -a "outputs/logs/nips26_${exp}_continue.out"
  local code=${PIPESTATUS[0]}
  echo "=== $(date '+%F %T') finished $exp exit=$code ===" | tee -a "$LOG"
  $PY -m analysis.nips26_report > "outputs/logs/nips26_report_after_${exp}.out" 2>&1 || true
  return $code
}

# Prefer free cards; include 0,2 if lightly used.
run e4 0,1,2,3 2
run e6b_lambda 0,1,2,3 3
run e1_sched 0,1,2,3 3

echo "=== $(date '+%F %T') queue done ===" | tee -a "$LOG"
$PY -m analysis.nips26_report | tee outputs/logs/nips26_report_after_continue.out
