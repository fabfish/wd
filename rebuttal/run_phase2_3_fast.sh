#!/bin/bash
# Phase 2 + Phase 3 with 8 workers per GPU for maximum throughput
# 3 GPUs * 8 workers = 24 parallel tasks
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

GPUS="0,1,3"
EPOCHS=100
WPG=8
RUNNER="rebuttal/run_rebuttal.py"

echo "=========================================="
echo "FAST REBUTTAL (${WPG} workers/GPU): $(date)"
echo "=========================================="

# Phase 2: VGG-16 seed=42
echo ""
echo ">>> PHASE 2: VGG-16 seed=42"
for EXP in 1 2 3; do
    echo "--- Exp $EXP start: $(date) ---"
    python "$RUNNER" --model vgg16 --seed 42 --experiment $EXP --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG
    echo "--- Exp $EXP done: $(date) ---"
done

# Phase 3: VGG-16 seed=123
echo ""
echo ">>> PHASE 3: VGG-16 seed=123"
for EXP in 1 2 3; do
    echo "--- Exp $EXP start: $(date) ---"
    python "$RUNNER" --model vgg16 --seed 123 --experiment $EXP --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG
    echo "--- Exp $EXP done: $(date) ---"
done

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
