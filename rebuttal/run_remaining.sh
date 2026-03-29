#!/bin/bash
# Continue from Phase 1 Exp 2 (Exp 1 already running separately)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

GPUS="0,1,3"
EPOCHS=100
RUNNER="rebuttal/run_rebuttal.py"

echo "=========================================="
echo "REMAINING REBUTTAL EXPERIMENTS: $(date)"
echo "=========================================="

# Phase 1 continued: ResNet-18 seed=123 Exp 2 & 3
echo ">>> PHASE 1 (continued): ResNet-18 seed=123 Exp 2"
python "$RUNNER" --model resnet18 --seed 123 --experiment 2 --gpus $GPUS --epochs $EPOCHS
echo "--- Exp 2 done: $(date) ---"

echo ">>> PHASE 1 (continued): ResNet-18 seed=123 Exp 3"
python "$RUNNER" --model resnet18 --seed 123 --experiment 3 --gpus $GPUS --epochs $EPOCHS
echo "--- Exp 3 done: $(date) ---"

# Phase 2: VGG-16 seed=42
echo ""
echo ">>> PHASE 2: VGG-16 seed=42"
for EXP in 1 2 3; do
    echo "--- Exp $EXP start: $(date) ---"
    python "$RUNNER" --model vgg16 --seed 42 --experiment $EXP --gpus $GPUS --epochs $EPOCHS
    echo "--- Exp $EXP done: $(date) ---"
done

# Phase 3: VGG-16 seed=123
echo ""
echo ">>> PHASE 3: VGG-16 seed=123"
for EXP in 1 2 3; do
    echo "--- Exp $EXP start: $(date) ---"
    python "$RUNNER" --model vgg16 --seed 123 --experiment $EXP --gpus $GPUS --epochs $EPOCHS
    echo "--- Exp $EXP done: $(date) ---"
done

echo ""
echo "=========================================="
echo "ALL REMAINING EXPERIMENTS DONE: $(date)"
echo "=========================================="
