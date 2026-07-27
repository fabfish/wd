#!/bin/bash
# Re-renders rebuttal/figures/response_to_reviewer_9i84.png every 10 min
# while rebuttal.run_exp2_fill is running, so you can refresh the IDE preview
# and watch the curves grow as cells land.
cd "$(dirname "$0")/.."
PY=${PY:-/home/yzy/.conda/envs/trace/bin/python}
LOG=/tmp/auto_replot.log
echo "[auto_replot] started at $(date)" > "$LOG"
while pgrep -f "run_exp2_fill" >/dev/null 2>&1; do
    fill_csv="rebuttal/results/results_resnet18_seed42_exp2_fill.csv"
    if [ -f "$fill_csv" ]; then
        n=$(($(wc -l < "$fill_csv") - 1))
    else
        n=0
    fi
    echo "[auto_replot] $(date '+%H:%M:%S')  fill rows=$n  rendering ..." >> "$LOG"
    "$PY" rebuttal/generate_figures.py 2>&1 | tail -3 >> "$LOG"
    sleep 600
done
echo "[auto_replot] grid finished, last render ..." >> "$LOG"
"$PY" rebuttal/generate_figures.py 2>&1 | tail -3 >> "$LOG"
echo "[auto_replot] done at $(date)" >> "$LOG"
