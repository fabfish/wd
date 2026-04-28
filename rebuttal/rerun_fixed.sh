#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

GPUS="0,1"
EPOCHS=100
WPG_VGG=8
WPG_R50=4

echo "=========================================="
echo "  RERUN: Bug fix - MODEL_NAME via args"
echo "  GPUs: $GPUS"
echo "=========================================="

# --- VGG-16 seed=42 ---
echo "[$(date)] VGG-16 seed=42 Exp1"
python3 rebuttal/run_rebuttal.py --model vgg16 --experiment 1 --seed 42 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_VGG
echo "[$(date)] VGG-16 seed=42 Exp2"
python3 rebuttal/run_rebuttal.py --model vgg16 --experiment 2 --seed 42 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_VGG
echo "[$(date)] VGG-16 seed=42 Exp3"
python3 rebuttal/run_rebuttal.py --model vgg16 --experiment 3 --seed 42 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_VGG

# --- VGG-16 seed=123 ---
echo "[$(date)] VGG-16 seed=123 Exp1"
python3 rebuttal/run_rebuttal.py --model vgg16 --experiment 1 --seed 123 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_VGG
echo "[$(date)] VGG-16 seed=123 Exp2"
python3 rebuttal/run_rebuttal.py --model vgg16 --experiment 2 --seed 123 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_VGG
echo "[$(date)] VGG-16 seed=123 Exp3"
python3 rebuttal/run_rebuttal.py --model vgg16 --experiment 3 --seed 123 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_VGG

echo "[$(date)] VGG-16 ALL DONE"

# --- ResNet-50 seed=42 ---
echo "[$(date)] ResNet-50 seed=42 Exp1"
python3 rebuttal/run_rebuttal.py --model resnet50 --experiment 1 --seed 42 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_R50
echo "[$(date)] ResNet-50 seed=42 Exp2"
python3 rebuttal/run_rebuttal.py --model resnet50 --experiment 2 --seed 42 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_R50
echo "[$(date)] ResNet-50 seed=42 Exp3"
python3 rebuttal/run_rebuttal.py --model resnet50 --experiment 3 --seed 42 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_R50

# --- ResNet-50 seed=123 ---
echo "[$(date)] ResNet-50 seed=123 Exp1"
python3 rebuttal/run_rebuttal.py --model resnet50 --experiment 1 --seed 123 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_R50
echo "[$(date)] ResNet-50 seed=123 Exp2"
python3 rebuttal/run_rebuttal.py --model resnet50 --experiment 2 --seed 123 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_R50
echo "[$(date)] ResNet-50 seed=123 Exp3"
python3 rebuttal/run_rebuttal.py --model resnet50 --experiment 3 --seed 123 --gpus $GPUS --epochs $EPOCHS --workers_per_gpu $WPG_R50

echo "[$(date)] ALL EXPERIMENTS DONE"
