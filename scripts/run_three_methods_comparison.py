"""
Three-Method Optimizer Comparison Experiment
Compare optimal learning rates across:
1. SGD (no wd, no momentum)
2. SGD+WD (various wd values, no momentum)
3. SGDM+WD (wd=5e-4, momentum=0.9)

Goal: Demonstrate decreasing optimal LR trend: SGD > SGD+WD > SGDM+WD

Usage:
    python run_three_methods_comparison.py --gpus 0,1
    python run_three_methods_comparison.py --gpus all
"""
import argparse
import csv
import os
import time
from itertools import product
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from wd_core.models import resnet18
from wd_core.utils import set_seed, train_model
from wd_core.gpu_scheduler import GPUScheduler, parse_gpu_ids
from wd_core.logger import get_logger


def get_cifar100_loaders(batch_size=128, num_workers=2):
    """Load CIFAR-100 dataset with standard augmentation"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def run_single_experiment_worker(method, batch_size, lr, wd, momentum, epochs, seed, use_amp, save_checkpoints=False):
    """Worker function for running a single experiment."""
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(batch_size, num_workers=2)
    model = resnet18(num_classes=100).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Running: {method} | BS={batch_size} | LR={lr} | WD={wd} | Mom={momentum}")

    best_test_acc, final_test_acc, final_train_loss = train_model(
        model, train_loader, test_loader, optimizer, scheduler,
        device, epochs=epochs, use_amp=use_amp
    )

    return {
        'method': method,
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'momentum': momentum,
        'final_test_acc': final_test_acc,
        'final_train_loss': final_train_loss,
        'best_test_acc': best_test_acc
    }


def run_three_methods_comparison(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """
    三种方法对比实验：
    1. SGD (wd=0, m=0)
    2. SGD+WD (多种 wd, m=0)
    3. SGDM+WD (wd=5e-4, m=0.9)

    目标：验证最优 LR 递减：η_SGD > η_SGD+WD > η_SGDM+WD
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("三种方法最优LR对比实验")
        logger.info("="*80)

    batch_size = 128
    tasks = []

    # ========== 1. SGD (no wd, no momentum) ==========
    # Optimal LR around 0.3 based on existing results
    sgd_lrs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    for lr in sgd_lrs:
        task = ("SGD", batch_size, lr, 0.0, 0.0, epochs, seed, use_amp, False)
        tasks.append(task)

    # ========== 2. SGD+WD (various wd, no momentum) ==========
    # Test larger WD values to bring optimal LR down
    sgd_wd_lrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    sgd_wd_wds = [1e-3, 2e-3, 5e-3, 1e-2]
    for wd in sgd_wd_wds:
        for lr in sgd_wd_lrs:
            task = ("SGD+WD", batch_size, lr, wd, 0.0, epochs, seed, use_amp, False)
            tasks.append(task)

    # ========== 3. SGDM+WD (wd=5e-4, momentum=0.9) ==========
    # Optimal LR around 0.1 based on existing results
    sgdm_lrs = [0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2]
    for lr in sgdm_lrs:
        task = ("SGDM+WD", batch_size, lr, 5e-4, 0.9, epochs, seed, use_amp, False)
        tasks.append(task)

    total_runs = len(tasks)
    if logger:
        logger.info(f"总实验数: {total_runs}")
        logger.info(f"  SGD: {len(sgd_lrs)} runs (LRs: {sgd_lrs})")
        logger.info(f"  SGD+WD: {len(sgd_wd_lrs) * len(sgd_wd_wds)} runs (LRs: {sgd_wd_lrs}, WDs: {sgd_wd_wds})")
        logger.info(f"  SGDM+WD: {len(sgdm_lrs)} runs (LRs: {sgdm_lrs})")

    # Use GPU scheduler for multi-GPU parallel execution
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\n完成 {total_runs} 实验，耗时 {elapsed_time/60:.2f} 分钟")

        # Analyze results: find optimal config for each method
        logger.info("\n" + "="*80)
        logger.info("各方法最优配置分析")
        logger.info("="*80)

        for method in ['SGD', 'SGD+WD', 'SGDM+WD']:
            method_results = [r for r in results if r and r['method'] == method]
            if method_results:
                best = max(method_results, key=lambda x: x['best_test_acc'])
                logger.info(f"{method}:")
                logger.info(f"  最优 LR: {best['lr']}, 最优 WD: {best['wd']}")
                logger.info(f"  最佳准确率: {best['best_test_acc']:.2f}%")

    return results


def save_results(results, output_file, logger=None):
    """Save results to CSV file"""
    if not results:
        return

    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc']

    file_exists = os.path.exists(output_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for result in results:
            if result is not None:
                writer.writerow(result)

    if logger:
        logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='三种方法最优LR对比实验')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/results/three_methods_comparison.csv')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpus', type=str, default=None, help='GPU IDs to use, e.g., "0,1" or "all"')
    parser.add_argument('--no_log', action='store_true')
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus == 'all':
        gpu_ids = None  # Use all available GPUs
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []  # CPU only

    exp_name = "three_methods_comparison"
    logger = get_logger(exp_name, log_to_file=not args.no_log, log_to_console=True)

    logger.log_config({
        'Experiment': 'Three Methods Comparison',
        'Epochs': args.epochs,
        'Seed': args.seed,
        'GPUs': gpu_ids if gpu_ids else 'all' if args.gpus == 'all' else 'CPU',
    })

    start_time = time.time()
    results = run_three_methods_comparison(gpu_ids, args.epochs, args.seed, args.use_amp, logger)
    elapsed_time = time.time() - start_time

    save_results(results, args.output, logger)

    successful = sum(1 for r in results if r is not None)
    logger.log_experiment_end(successful, len(results), elapsed_time)


if __name__ == '__main__':
    main()
