# Weight Decay & Learning Rate Experiments

Investigating the relationship between Learning Rate ($\eta$), Weight Decay ($\lambda$), and Batch Size in SGD optimization dynamics.

## Environment Setup
- **Python**: 3.13
- **PyTorch**: 2.4+ (CUDA 12.x recommended)
- **GPUs**: Supports multi-GPU parallel execution (A6000/A100 recommended)

### Dependency Management with uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

## Key Scripts

### 1. V3 Supplementary: Batch Size Scaling
Compares optimal hyperparameter landscapes between Batch Size 32 and 128.
- **Script**: `run_experiments_v3_supplementary.py`
- **Features**: 
  - Multi-GPU parallel execution
  - Real-time progress monitoring (tqdm)
  - Customizable concurrency per GPU
  - Automatic checkpointing and resumption

**Usage:**
```bash
# Run on specific GPUs with default settings
python run_experiments_v3_supplementary.py --gpus 0,1,2,3

# Run with optimized concurrency (recommended for high-end GPUs like A6000/A100)
# Default is 6 workers per GPU. Increase to 8-10 for small batch sizes on A100.
python run_experiments_v3_supplementary.py --gpus 0,1,2,3 --workers_per_gpu 8

# Check existing results without running
python run_experiments_v3_supplementary.py --check
```

### 2. Three Methods Comparison
Compares optimal learning rates across SGD, SGD+WD, and SGDM+WD.
- **Script**: `run_three_methods_comparison.py`
- **Usage**:
```bash
python run_three_methods_comparison.py --gpus 0,1
```

## Optimization Tips
- **Concurrency**: The experiments are lightweight on GPU memory (~600MB for ResNet-18 on CIFAR-100).
- **Workers**: Use `--workers_per_gpu` to maximize throughput. 
  - For **A6000/A100**: Set to `6` or `8`.
  - Ensure CPU cores are sufficient (total workers < total CPU cores).
- **DataLoader**: `num_workers` is set to `0` to avoid multiprocessing deadlocks in `spawn` mode.

## Results
Results are saved to:
- `outputs/results/`: aggregated CSV files
- `outputs/history/`: per-experiment training logs (JSON)


python run_experiments_v3_supplementary.py --gpus 0,1,2,3,4,5,6,7 --workers_per_gpu 8