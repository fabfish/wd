"""
Supplementary experiment: SGD+WD with different WD values
Goal: Find WD where optimal LR is between SGD (0.1) and SGDM+WD (0.1)
"""
import subprocess
import sys
from pathlib import Path

# Configuration
batch_size = 128
momentum = 0
epochs = 100
seed = 42

# Test different WD values (larger than the existing 5e-4)
wd_values = [1e-3, 2e-3, 5e-3, 1e-2]

# Test different LR values around the expected optimal range
lr_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

output_dir = Path("outputs/logs")
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / "sgd_wd_supplementary.log"
with open(log_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Supplementary Experiment: SGD+WD with varying WD\n")
    f.write("="*80 + "\n")
    f.write(f"Batch size: {batch_size}, Momentum: {momentum}, Epochs: {epochs}\n")
    f.write(f"WD values: {wd_values}\n")
    f.write(f"LR values: {lr_values}\n")
    f.write("="*80 + "\n\n")

print("="*80)
print("SGD+WD Supplementary Experiments")
print("="*80)
print(f"WD values: {wd_values}")
print(f"LR values: {lr_values}")
print(f"Total experiments: {len(wd_values) * len(lr_values)}")
print()

# Run experiments
count = 0
total = len(wd_values) * len(lr_values)

for wd in wd_values:
    for lr in lr_values:
        count += 1
        print(f"[{count}/{total}] Running: SGD+WD | LR={lr} | WD={wd}")

        cmd = [
            sys.executable, "main.py",
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--wd", str(wd),
            "--momentum", str(momentum),
            "--epochs", str(epochs),
            "--seed", str(seed)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            with open(log_file, 'a') as f:
                f.write(f"--- Run {count}/{total}: SGD+WD | LR={lr} | WD={wd} ---\n")
                f.write(result.stdout)
                if result.returncode != 0:
                    f.write(f"ERROR: {result.stderr}\n")
        except subprocess.TimeoutExpired:
            with open(log_file, 'a') as f:
                f.write(f"--- Run {count}/{total}: TIMEOUT ---\n")
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"--- Run {count}/{total}: ERROR {e} ---\n")

print("\n" + "="*80)
print(f"All experiments completed! Log saved to: {log_file}")
print("="*80)
