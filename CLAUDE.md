# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a PyTorch experiment framework for investigating SGD weight decay and generalization theory on CIFAR-100 with ResNet-18. It tests three theoretical hypotheses about learning rate stability bounds, learning rate/weight decay interaction, and batch size scaling.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single experiment
python main.py --batch_size 128 --lr 0.1 --wd 5e-4 --momentum 0.9 --epochs 100

# Run experiment sets with multi-GPU
python run_experiments.py --experiment 1 --epochs 100 --gpus all
python run_experiments.py --experiment 2 --epochs 100 --gpus 0,1,2,3
python run_experiments.py --experiment 3 --epochs 100 --gpus 0-3

# Run all experiments at once
./run_all_experiments.sh
./run_all_experiments.sh --parallel  # Run experiment sets in parallel

# Generate plots from results
python plot_results.py
python plot_results.py --experiment 1 --stats
```

## Architecture

### Core Training Pipeline
- `main.py` - Single experiment entry point with argparse for all hyperparameters
- `models.py` - ResNet-18 implementation adapted for CIFAR (32x32 input, no initial maxpool)
- `utils.py` - Training loop (`train_model`), evaluation, checkpointing, and seed management

### Multi-GPU Experiment Runner
- `run_experiments.py` - Orchestrates grid searches for three experiment sets
- `gpu_scheduler.py` - `GPUScheduler` class distributes tasks across GPUs using multiprocessing with spawn method
- `logger.py` - Timestamped logging to `outputs/logs/`

### Three Experiment Sets
1. **Exp 1**: Optimal LR ordering across SGD variants (SGD vs SGD+WD vs SGDM+WD)
2. **Exp 2**: LR-WD interaction heatmap (tests inverse relationship)
3. **Exp 3**: Batch size scaling with linear LR rule

### Output Structure
All outputs go to `outputs/` (gitignored):
- `outputs/results/results.csv` - Experiment results (appended, supports resume)
- `outputs/logs/` - Timestamped log files
- `outputs/checkpoints/` - Model checkpoints (when `--save_checkpoints` used)
- `outputs/plots/` - Generated visualizations

## Key Implementation Details

- Uses PyTorch native SGD with CosineAnnealingLR scheduler
- Mixed precision training via `torch.amp` enabled by default
- GPU scheduler uses `mp.set_start_method('spawn')` for CUDA compatibility
- CIFAR-100 normalization: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
- Data stored in `./data/` (auto-downloaded)
