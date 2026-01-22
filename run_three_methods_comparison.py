"""
Three-Method Optimizer Comparison Experiment
Compare optimal learning rates across:
1. SGD (no wd, no momentum)
2. SGD+WD (various wd values, no momentum)
3. SGDM+WD (wd=5e-4, momentum=0.9)

Goal: Demonstrate decreasing optimal LR trend: SGD > SGD+WD > SGDM+WD

Usage:
    python run_three_methods_comparison.py --device 0
    python run_three_methods_comparison.py --device 0,1
"""
import subprocess
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser(description='Three-Method Optimizer Comparison')
parser.add_argument('--device', type=str, default="0", help='GPU device(s) to use, e.g., "0" or "0,1"')
args = parser.parse_args()

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# Configuration
batch_size = 128
epochs = 100
seed = 42

# Experiment definitions
experiments = []

# ========== 1. SGD (no wd, no momentum) ==========
# From existing results, optimal LR is around 0.3
# Add more granular search around 0.15-0.5
sgd_lrs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
for lr in sgd_lrs:
    experiments.append({
        "method": "SGD",
        "lr": lr,
        "wd": 0.0,
        "momentum": 0.0
    })

# ========== 2. SGD+WD (various wd, no momentum) ==========
# Test different WD values to find one where optimal LR is between SGD and SGDM+WD
sgd_wd_lrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
sgd_wd_wds = [1e-3, 2e-3, 5e-3, 1e-2]  # Larger WD values
for wd in sgd_wd_wds:
    for lr in sgd_wd_lrs:
        experiments.append({
            "method": "SGD+WD",
            "lr": lr,
            "wd": wd,
            "momentum": 0.0
        })

# ========== 3. SGDM+WD (wd=5e-4, momentum=0.9) ==========
# From existing results, optimal LR is around 0.1
# Add more granular search around 0.05-0.2
sgdm_lrs = [0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2]
for lr in sgdm_lrs:
    experiments.append({
        "method": "SGDM+WD",
        "lr": lr,
        "wd": 5e-4,
        "momentum": 0.9
    })

# Output setup
output_dir = Path("outputs/logs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = output_dir / f"three_methods_comparison_{timestamp}.log"
results_file = output_dir / f"three_methods_results_{timestamp}.csv"

# Write CSV header
with open(results_file, 'w') as f:
    f.write("method,batch_size,lr,wd,momentum,final_test_acc,final_train_loss,best_test_acc\n")

# Log experiment info
with open(log_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Three-Method Optimizer Comparison Experiment\n")
    f.write(f"Started: {timestamp}\n")
    f.write("="*80 + "\n")
    f.write(f"Batch size: {batch_size}, Epochs: {epochs}\n")
    f.write(f"Total experiments: {len(experiments)}\n")
    f.write("="*80 + "\n\n")

print("="*80)
print("Three-Method Optimizer Comparison Experiment")
print("="*80)
print(f"Total experiments: {len(experiments)}")
print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
print()

# Count by method
from collections import Counter
method_counts = Counter(exp["method"] for exp in experiments)
for method, count in sorted(method_counts.items()):
    print(f"  {method}: {count} experiments")
print()

# Run experiments
for i, exp in enumerate(experiments, 1):
    method = exp["method"]
    lr = exp["lr"]
    wd = exp["wd"]
    momentum = exp["momentum"]
    
    print(f"[{i}/{len(experiments)}] {method} | LR={lr} | WD={wd} | M={momentum}")
    
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
        
        # Parse results from output
        best_acc = final_acc = train_loss = "N/A"
        for line in result.stdout.split('\n'):
            if "Best Test Accuracy:" in line:
                best_acc = line.split(':')[-1].strip().replace('%', '')
            elif "Final Test Accuracy:" in line:
                final_acc = line.split(':')[-1].strip().replace('%', '')
            elif "Final Train Loss:" in line:
                train_loss = line.split(':')[-1].strip()
        
        # Write to CSV
        with open(results_file, 'a') as f:
            f.write(f"{method},{batch_size},{lr},{wd},{momentum},{final_acc},{train_loss},{best_acc}\n")
        
        # Write to log
        with open(log_file, 'a') as f:
            f.write(f"--- [{i}/{len(experiments)}] {method} | LR={lr} | WD={wd} | M={momentum} ---\n")
            f.write(f"Best Acc: {best_acc}%, Final Acc: {final_acc}%, Train Loss: {train_loss}\n")
            if result.returncode != 0:
                f.write(f"ERROR: {result.stderr}\n")
            f.write("\n")
        
        print(f"    -> Best: {best_acc}%, Final: {final_acc}%")
        
    except subprocess.TimeoutExpired:
        with open(log_file, 'a') as f:
            f.write(f"--- [{i}/{len(experiments)}] TIMEOUT ---\n")
        print("    -> TIMEOUT")
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"--- [{i}/{len(experiments)}] ERROR: {e} ---\n")
        print(f"    -> ERROR: {e}")

print("\n" + "="*80)
print(f"Experiments completed!")
print(f"Results: {results_file}")
print(f"Log: {log_file}")
print("="*80)
