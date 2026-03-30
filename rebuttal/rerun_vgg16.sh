#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

GPUS="0,1"
EPOCHS=100
WPG=8

echo "=========================================="
echo "  VGG-16 Rerun (MODEL_NAME bug fixed)"
echo "  GPUs: $GPUS | Workers/GPU: $WPG"
echo "  Start: $(date)"
echo "=========================================="

for SEED in 42 123; do
  for EXP in 1 2 3; do
    echo "[$(date)] VGG-16 seed=$SEED Exp$EXP START"
    python3 rebuttal/run_rebuttal.py --model vgg16 --experiment $EXP --seed $SEED --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG
    echo "[$(date)] VGG-16 seed=$SEED Exp$EXP DONE"
  done
done

echo "=========================================="
echo "  VGG-16 ALL DONE: $(date)"
echo "=========================================="
