#!/bin/bash
# Run momentum search experiment with multi-worker per GPU
# Tests momentum values [0.5, 0.7, 0.8, 0.95, 0.99] with wd=0.002, lr=[0.01-0.1]
# Total: 30 experiments (5 momentums × 6 LRs)

echo "=========================================="
echo "SGDM+WD Momentum Search Experiment"
echo "=========================================="
echo "Momentum values: 0.5, 0.7, 0.8, 0.95, 0.99"
echo "LRs: 0.01, 0.02, 0.03, 0.05, 0.07, 0.1"
echo "WD: 0.002 (fixed)"
echo "Total experiments: 30"
echo ""
echo "Using 2 workers per GPU (for A6000 Pro 48GB)"
echo "=========================================="

python3 run_momentum_search.py --gpus all --epochs 100 --workers-per-gpu 10

echo ""
echo "Results saved to: outputs/results/momentum_search.csv"
