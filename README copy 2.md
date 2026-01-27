
# Weight Decay Experiments

This repository contains experiments analyzing the interaction between Learning Rate (LR) and Weight Decay (WD) in SGD.

## Project Structure

*   **`wd_core/`**: Core package containing models, utility functions, and logging.
*   **`scripts/`**: Experiment running scripts (e.g., `train.py`, `run_all_experiments.py`).
*   **`analysis/`**: Plotting and analysis scripts.
*   **`outputs/`**: Experiment results and plots.
*   **`archive/`**: Archived experiment scripts.

## Setup

This project uses a local package `wd_core`. You should install it in editable mode (or ensure the root directory is in your PYTHONPATH).

```bash
# If using uv / pip
uv pip install -e .
# OR
pip install -e .
```

## Running Experiments

Example: Train a single model

```bash
python scripts/train.py --lr 0.1 --wd 5e-4
```

Example: Run Momentum Search

```bash
python scripts/run_momentum_search.py
```

## Analysis

Plotting scripts are located in `analysis/`.

```bash
python analysis/plot_exp1_v18.py
```