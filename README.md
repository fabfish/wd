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

## Initial Experiment Designs (Original)

### Experiment 1: LR Ordering (Stability Bound Verification)

**Goal**: Verify `η_SGD > η_SGD+WD > η_SGDM+WD`

| Method | Momentum | Weight Decay |
|--------|----------|--------------|
| SGD | 0 | 0 |
| SGD+WD | 0 | 5e-4 |
| SGDM+WD | 0.9 | 5e-4 |

- **Search Space**: LR ∈ [0.01, 3.0], 15 points
- **Expected Result**: In Test Accuracy vs. LR curve, SGD peak is rightmost, SGDM+WD peak is leftmost.

### Experiment 2: η-λ Interaction (Heatmap Analysis)

**Goal**: Verify optimal η and λ inverse relationship (`λ_opt ∝ 1/η`)

- **Config**: SGDM (momentum=0.9), Batch Size = 128
- **Search Space**:
  - LR: [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
  - WD: [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
- **Expected Result**: High accuracy region in 2D heatmap follows a "diagonal" trend.

### Experiment 3: Batch Size Scaling Law

**Goal**: Verify `λ_opt ∝ B` linear relationship.

- **Config**: SGDM (momentum=0.9), Linear LR Scaling: `η = 0.1 × (B / 128)`
- **Search Space**:
  - Batch Size: [32, 64, 128, 256, 512]
  - WD: [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
- **Expected Result**: Optimal λ vs. Batch Size curve is linear.

## Detailed Experiment Breakdown (New Findings)

### 1. LR Ordering Analysis (v18)
**Script**: `analysis/plot_exp1_v18.py` -> `outputs/plots/exp1_lr_ordering_v18.png`

This experiment visualizes the optimal Learning Rate (LR) shift across four configurations to verifying the stability bound theory.

*   **Configurations**:
    1.  **SGD** (Baseline): Batch Size 128.
    2.  **SGDM (no WD)**: Pure momentum effect.
    3.  **SGD+WD**: Showing how Weight Decay shifts optimal LR leftwards.
    4.  **SGDM+WD**: The combination that yields the highest accuracy but requires the smallest LR.
*   **Key Design Elements**:
    *   **Data Aggregation**: Combines data from multiple search runs (`three_methods_comparison.csv`, `wd_shift_search.csv`).
    *   **Visualization**: Uses a "Star & Bar" design to explicitly mark the peak accuracy and its corresponding LR.
    *   **Color Coding**: SGD (Deep Blue), SGDM (Orange), SGD+WD (Mid Blue), SGDM+WD (Yellow).

### 2. $\eta$-$\lambda$ Interaction Heatmap
**Script**: `analysis/plot_exp2_fixed.py` -> `outputs/plots/exp2_heatmap_analysis.png`

Explores the coupling between Learning Rate ($\eta$) and Weight Decay ($\lambda$) in SGDM.

*   **Layout**:
    *   **Left (Heatmap)**: Test Accuracy vs. $(\eta, \lambda)$. Y-axis is inverted (large WD on top). High accuracy region (72-79%) uses a specialized colormap.
    *   **Center (Inverse Law)**: Fits the curve $\lambda \propto \eta^{-b}$ to observed optimal points. Verifies $b \approx 1$ (Theory: $\lambda \propto 1/\eta$).
    *   **Right (Effective Regularization)**: Plots Accuracy vs. $\eta \lambda$. Verifies that performance depends largely on the product $\eta \lambda$.
*   **Findings**: The high-accuracy region forms a diagonal, confirming that as LR increases, optimal WD must decrease proportionally.

### 3. Batch Size Scaling Laws (V3 Supplementary)
**Script**: `analysis/plot_v3_supplementary_v2.py` -> `outputs/plots/v3_supplementary_analysis_v2.png`

Investigates how optimal hyperparameters scale when Batch Size (BS) increases from 32 to 128.

*   **Design**: 3×2 Grid Comparison (Left: BS=32, Right: BS=128).
    *   **Row 1 (Fixed $\lambda$)**: Varying LR. Finds optimal $\eta$ for each BS.
    *   **Row 2 (Heatmap)**: Full landscape shift.
    *   **Row 3 (Fixed $\eta$)**: Varying WD. Finds optimal $\lambda$ for each BS.
*   **Key Finding**:
    *   **LR Scaling**: When BS increases $4\times$ (32->128), optimal LR increases $\approx 2.5\times$ (0.02->0.05).
    *   **WD Stability**: Optimal WD remains relatively stable ($\approx 1\times 10^{-3}$) across batch sizes.