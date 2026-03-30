#!/bin/bash
# ResNet-50 experiments: seed=42 and seed=123, all 3 experiment sets
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

GPUS="0,1,3"
EPOCHS=100
WPG=6
RUNNER="rebuttal/run_rebuttal.py"

echo "=========================================="
echo "ResNet-50 REBUTTAL (${WPG} workers/GPU): $(date)"
echo "=========================================="

# ResNet-50 seed=42
echo ""
echo ">>> ResNet-50 seed=42"
for EXP in 1 2 3; do
    echo "--- Exp $EXP start: $(date) ---"
    python "$RUNNER" --model resnet50 --seed 42 --experiment $EXP --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG
    echo "--- Exp $EXP done: $(date) ---"
done

# ResNet-50 seed=123
echo ""
echo ">>> ResNet-50 seed=123"
for EXP in 1 2 3; do
    echo "--- Exp $EXP start: $(date) ---"
    python "$RUNNER" --model resnet50 --seed 123 --experiment $EXP --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG
    echo "--- Exp $EXP done: $(date) ---"
done

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
