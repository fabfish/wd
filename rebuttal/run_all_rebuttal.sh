#!/bin/bash
# Rebuttal: run all phases sequentially on GPUs 0,1,3
# Phase 1: ResNet-18 seed=123 (Exp 1/2/3)
# Phase 2: VGG-16 seed=42 (Exp 1/2/3)
# Phase 3: VGG-16 seed=123 (Exp 1/2/3)

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

GPUS="0,1,3"
EPOCHS=100
RUNNER="rebuttal/run_rebuttal.py"

echo "=========================================="
echo "REBUTTAL EXPERIMENTS START: $(date)"
echo "=========================================="

# Phase 1: ResNet-18 seed=123
echo ""
echo ">>> PHASE 1: ResNet-18 seed=123"
for EXP in 1 2 3; do
    echo "--- Exp $EXP start: $(date) ---"
    python "$RUNNER" --model resnet18 --seed 123 --experiment $EXP --gpus $GPUS --epochs $EPOCHS
    echo "--- Exp $EXP done: $(date) ---"
done

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
echo "ALL REBUTTAL EXPERIMENTS DONE: $(date)"
echo "=========================================="
