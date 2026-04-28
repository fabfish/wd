"""
Supplementary Exp2 runs: extend specific curves in the eta*lambda scaling plot.

1. Blue (wd=0.0002) extra seeds (7, 2024) at all existing eta values
2. Purple (wd=0.0001) extend right: eta=[1.0, 2.0, 3.0, 5.0], seeds 42+123
3. Blue (wd=0.0002) extend right: eta=[0.75, 1.0, 1.5, 2.5], seeds 42+123+7+2024
4. Dark red (wd=0.05) extend left: eta=[0.0002, 0.0005, 0.001, 0.002], seeds 42+123
"""
import csv
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rebuttal.run_rebuttal import run_single_experiment_worker
from wd_core.gpu_scheduler import GPUScheduler
from wd_core.logger import get_logger

WORKERS_PER_GPU = 4


def build_tasks():
    tasks = []

    # Group 1: blue (wd=0.0002) extra seeds
    for seed in [7, 2024]:
        for lr in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
            tasks.append(('resnet18', 'SGDM', 128, lr, 0.0002, 0.9, 100, seed, True))

    # Group 2: purple (wd=0.0001) extend right
    for seed in [42, 123]:
        for lr in [1.0, 2.0, 3.0, 5.0]:
            tasks.append(('resnet18', 'SGDM', 128, lr, 0.0001, 0.9, 100, seed, True))

    # Group 3: blue (wd=0.0002) extend right (all 4 seeds)
    for seed in [42, 123, 7, 2024]:
        for lr in [0.75, 1.0, 1.5, 2.5]:
            tasks.append(('resnet18', 'SGDM', 128, lr, 0.0002, 0.9, 100, seed, True))

    # Group 4: dark red (wd=0.05) extend left
    for seed in [42, 123]:
        for lr in [0.0002, 0.0005, 0.001, 0.002]:
            tasks.append(('resnet18', 'SGDM', 128, lr, 0.05, 0.9, 100, seed, True))

    return tasks


def main():
    logger = get_logger("exp2_supplement")

    tasks = build_tasks()
    logger.info(f"Total supplement tasks: {len(tasks)}")

    gpu_ids = None  # use all GPUs
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True, workers_per_gpu=WORKERS_PER_GPU)

    start = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed = time.time() - start

    output_file = Path(__file__).resolve().parent / 'results' / 'results_resnet18_exp2_supplement.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc',
                  'final_test_loss']
    file_exists = os.path.exists(output_file)
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            if r is not None:
                writer.writerow(r)

    ok = sum(1 for r in results if r is not None)
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Done: {ok}/{len(tasks)} succeeded in {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
