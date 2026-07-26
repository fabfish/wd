#!/usr/bin/env bash
# Rebuttal experiment queue, in priority order.
#
# Ordering rationale: E1 settles whether the optimal weight decay moves with the
# training length, which is the claim that separates our account from the
# rotational-equilibrium one and the point three of the four reviews turn on. It
# runs on the dense lambda ladder first (e1_fine), because the coarse ladder in
# e1_prelim cannot resolve the predicted shift. E2b and E3 answer the remaining
# two main criticisms; everything after that is supporting evidence.
#
# Every experiment appends to one CSV and skips configurations already present,
# so this is safe to interrupt and restart.
set -u

PY=${PY:-/home/yzy/.conda/envs/trace/bin/python}
GPUS=${GPUS:-1,2,3}
WPG=${WPG:-3}
NW=${NW:-5}
LOGDIR=outputs/logs

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"

# Do not oversubscribe the GPUs if an earlier queue is still training. The
# pattern is anchored so it matches the interpreter process itself rather than
# any shell whose command line happens to mention the script.
while pgrep -f "^${PY} rebuttal/run_nips26.py" >/dev/null 2>&1; do
    echo "$(date '+%F %T')  waiting for the running experiment to finish"
    sleep 120
done

run() {
    local exp=$1
    local wpg=${2:-$WPG}
    echo "=== $(date '+%F %T')  starting $exp (gpus=$GPUS workers/gpu=$wpg) ==="
    PYTHONUNBUFFERED=1 "$PY" rebuttal/run_nips26.py --exp "$exp" --gpus "$GPUS" \
        --workers_per_gpu "$wpg" --num_workers "$NW" \
        >"$LOGDIR/nips26_${exp}.out" 2>&1
    echo "=== $(date '+%F %T')  finished $exp with exit code $? ==="
    # Refresh the resolved numbers so the drafts can be filled in as results land.
    PYTHONUNBUFFERED=1 "$PY" -m analysis.nips26_report \
        >"$LOGDIR/nips26_report_after_${exp}.out" 2>&1
}

run e1_fine     # the discriminating measurement, dense lambda ladder
run e2b         # accuracy is not flat along a constant-product line
run e3 4        # divergence boundary, short runs so more fit per GPU
run e4          # zero-tuning transfer against four alternative rules
run e5b         # cost of a mis-specified constant
run e1_full     # second learning rate, second seed, constant-LR arm
run e6b         # momentum and the generalization gap

echo "=== $(date '+%F %T')  queue complete ==="
