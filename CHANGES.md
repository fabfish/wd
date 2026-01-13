# Logging and Output Management Updates

## Summary

The experimental framework has been enhanced with comprehensive logging and organized file management:

### New Features

1. **Structured Logging System** (`logger.py`)
   - Timestamped log files for each experiment
   - Console and file output
   - Organized logging with experiment context
   - Log files saved to `outputs/logs/`

2. **Checkpoint Management** (updated `utils.py`)
   - Save/load model checkpoints
   - Best and final checkpoint saving
   - Complete state preservation (model, optimizer, scheduler, metrics)
   - Checkpoints saved to `outputs/checkpoints/`

3. **Organized Output Directory Structure**
   ```
   outputs/
   ├── logs/            # Timestamped experiment logs
   ├── results/         # CSV result files
   ├── checkpoints/     # Model checkpoints
   └── plots/           # Generated visualizations
   ```

4. **.gitignore Configuration**
   - Ignores all output files, data, and Python artifacts
   - Keeps repository clean

### Updated Files

1. **logger.py** (NEW)
   - `ExperimentLogger` class for structured logging
   - Directory management
   - Context-aware logging methods

2. **utils.py** (UPDATED)
   - Added `save_checkpoint()` function
   - Added `load_checkpoint()` function
   - Added `train_model_with_checkpoints()` function

3. **run_experiments.py** (UPDATED)
   - Integrated logging system
   - Added `--save_checkpoints` flag
   - Added `--no_log` flag
   - Changed default output to `outputs/results/results.csv`
   - Progress tracking and timing

4. **plot_results.py** (UPDATED)
   - Changed default input to `outputs/results/results.csv`
   - Changed default output to `outputs/plots/`
   - Added `--output_dir` argument

5. **.gitignore** (NEW)
   - Comprehensive ignore patterns for Python, PyTorch, and outputs

6. **README.md** (UPDATED)
   - Added "Output Organization" section
   - Updated file structure diagram
   - Updated usage examples with new paths
   - Added checkpoint loading example

### Usage Examples

**Run with logging and checkpoints:**
```bash
python run_experiments.py --experiment 1 --gpus all --save_checkpoints
```

**Run without file logging:**
```bash
python run_experiments.py --experiment 1 --gpus all --no_log
```

**Generate plots:**
```bash
python plot_results.py --stats
```

**Load a checkpoint:**
```python
from utils import load_checkpoint
from models import resnet18

model = resnet18(num_classes=100)
checkpoint = load_checkpoint('outputs/checkpoints/exp1_SGD_bs128_lr0.1_wd0.0005_mom0_best.pth', model)
print(f"Best accuracy: {checkpoint['metrics']['best_test_acc']:.2f}%")
```

### Benefits

1. **Better Organization**: All outputs in one place, easy to find
2. **Reproducibility**: Complete experiment logs with timestamps
3. **Resume Capability**: Save and load model checkpoints
4. **Clean Repository**: Git ignores all generated files
5. **Debugging**: Detailed logs for troubleshooting
6. **Professionalism**: Industry-standard output organization

### Migration Notes

If you have existing results:
- Move `results.csv` → `outputs/results/results.csv`
- Move `*.png` plots → `outputs/plots/`
- Old checkpoints will need to be moved to `outputs/checkpoints/`

The new default paths are:
- Results: `outputs/results/results.csv`
- Logs: `outputs/logs/exp<N>_YYYYMMDD_HHMMSS.log`
- Checkpoints: `outputs/checkpoints/`
- Plots: `outputs/plots/`
