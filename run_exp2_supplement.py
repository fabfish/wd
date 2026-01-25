"""
Supplementary experiments for Exp2: Add smaller η×λ values to show drop trend

Current η×λ range: 1e-6 to 3e-3
Target: Add points in 1e-8 to 1e-6 range to show accuracy drop on the left side

Strategy:
- Use very small lr and/or very small wd combinations
- Focus on SGDM method with batch_size=128 to match existing exp2 data
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import resnet18
from utils import set_seed, train_model
from gpu_scheduler import GPUScheduler, parse_gpu_ids
from logger import get_logger


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
    
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def run_single_experiment_worker(method, batch_size, lr, wd, momentum, epochs, seed, use_amp, save_checkpoints=False):
    """Worker function for running a single experiment."""
    set_seed(seed)
    
    train_loader, test_loader = get_cifar100_loaders(batch_size)
    
    model = resnet18(num_classes=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if method == 'SGDM':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_test_acc, final_test_acc, final_train_loss = train_model(
        model, train_loader, test_loader, optimizer, scheduler,
        device=device, epochs=epochs, use_amp=use_amp
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


def run_supplement_experiments(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None, workers_per_gpu=3):
    """
    Run supplementary experiments with small η×λ values.
    
    Target η×λ range: 1e-8 to 1e-6 (to show drop on left side of plot)
    
    Combinations to test:
    - lr=0.001, wd=1e-5, 2e-5, 5e-5 -> η×λ = 1e-8, 2e-8, 5e-8
    - lr=0.005, wd=1e-5, 2e-5 -> η×λ = 5e-8, 1e-7
    - lr=0.01, wd=1e-5, 2e-5, 5e-5 -> η×λ = 1e-7, 2e-7, 5e-7
    """
    if logger:
        logger.info("=" * 60)
        logger.info("Exp2 Supplement: Small η×λ values for drop trend")
        logger.info("=" * 60)
    
    # Define experiment configurations
    # Focus on small wd values with various lr
    tasks = []
    
    # Very small η×λ combinations (target: 1e-8 to 1e-6)
    small_wd_values = [1e-5, 2e-5, 5e-5]
    lr_values = [0.001, 0.005, 0.01]
    
    for lr in lr_values:
        for wd in small_wd_values:
            eta_lambda = lr * wd
            if eta_lambda < 1e-6:  # Only add if smaller than current minimum
                tasks.append(('SGDM', 128, lr, wd, 0.9, epochs, seed, use_amp))
    
    if logger:
        logger.info(f"Total experiments to run: {len(tasks)}")
        logger.info(f"Workers per GPU: {workers_per_gpu}")
        for method, bs, lr, wd, mom, ep, sd, amp in tasks:
            eta_lambda = lr * wd
            logger.info(f"  lr={lr}, wd={wd:.1e} -> η×λ={eta_lambda:.1e}")
    
    # Use GPUScheduler for parallel execution
    scheduler = GPUScheduler(gpu_ids, verbose=True, workers_per_gpu=workers_per_gpu)
    
    def on_complete(result):
        if logger and result:
            logger.info(f"Completed: lr={result['lr']}, wd={result['wd']:.1e}, "
                       f"acc={result['best_test_acc']:.2f}%")
    
    results = scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=on_complete)
    
    return results


def save_results(results, output_file, logger=None):
    """Save results to CSV file, appending to existing data"""
    df_new = pd.DataFrame(results)
    
    # Load existing results and append
    existing_file = Path('outputs/results/results.csv')
    if existing_file.exists():
        df_existing = pd.read_csv(existing_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Remove duplicates based on key columns
        df_combined = df_combined.drop_duplicates(
            subset=['method', 'batch_size', 'lr', 'wd', 'momentum'],
            keep='last'
        )
    else:
        df_combined = df_new
    
    df_combined.to_csv(output_file, index=False)
    
    if logger:
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Total entries: {len(df_combined)}")


def main():
    parser = argparse.ArgumentParser(description='Exp2 Supplement: Small η×λ experiments')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs (e.g., 0,1,2)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP')
    args = parser.parse_args()
    
    gpu_ids = parse_gpu_ids(args.gpus)
    use_amp = not args.no_amp
    
    # Setup output
    output_dir = Path('outputs/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'results.csv'
    
    # Setup logger
    logger = get_logger('exp2_supplement')
    
    logger.info("=" * 60)
    logger.info("Exp2 Supplement: Extending η×λ range to smaller values")
    logger.info("=" * 60)
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"AMP: {use_amp}")
    
    # Run experiments
    results = run_supplement_experiments(
        gpu_ids, args.epochs, args.seed, use_amp, logger
    )
    
    # Save results
    save_results(results, output_file, logger)
    
    logger.info("=" * 60)
    logger.info("Experiments completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
