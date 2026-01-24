#!/bin/bash
# Run SGD+WD supplementary experiment (low LR)
# Fills in missing data points: lr=[0.01, 0.02, 0.03, 0.07] with wd=0.002
# Total: 4 experiments

echo "=========================================="
echo "SGD+WD Supplementary Experiment (Low LR)"
echo "=========================================="
echo "Missing LRs: 0.01, 0.02, 0.03, 0.07"
echo "WD: 0.002 (fixed)"
echo "Total experiments: 4"
echo ""
echo "Using 2 workers per GPU (for A6000 Pro 48GB)"
echo "=========================================="

python3 run_sgdwd_supplement.py --gpus 0,1 --epochs 100 --workers-per-gpu 10 --log-interval 25

echo ""
echo "Results saved to: outputs/results/sgdwd_supplement.csv"
