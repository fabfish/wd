"""
Supplementary SGD+WD Experiments
Fill in missing low-LR data points for SGD+WD with wd=0.002

Missing LRs: 0.01, 0.02, 0.03, 0.07

Usage:
    python run_sgdwd_supplement.py --gpus 0,1
    python run_sgdwd_supplement.py --gpus all
"""
import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def run_sgdwd_supplement(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None, workers_per_gpu=1):
    """
    补充 SGD+WD (wd=0.002) 在低 LR 的实验数据
    
    已有数据: lr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    需要补充: lr in [0.01, 0.02, 0.03, 0.07]
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("SGD+WD Supplementary Experiments (Low LR)")
        logger.info("="*80)

    batch_size = 128
    wd = 0.002  # Fixed optimal wd
    momentum = 0.0  # SGD+WD has no momentum
    
    # Missing low LR values
    missing_lrs = [0.01, 0.02, 0.03, 0.07]
    
    tasks = []
    for lr in missing_lrs:
        task = ("SGD+WD", batch_size, lr, wd, momentum, epochs, seed, use_amp, False)
        tasks.append(task)

    total_runs = len(tasks)
    if logger:
        logger.info(f"Total experiments: {total_runs}")
        logger.info(f"  Fixed: wd={wd}, batch_size={batch_size}, momentum={momentum}")
        logger.info(f"  Missing LRs to fill: {missing_lrs}")

    # Use GPU scheduler for multi-GPU parallel execution
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True, workers_per_gpu=workers_per_gpu)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\nCompleted {total_runs} experiments in {elapsed_time/60:.2f} minutes")

        # Show results
        logger.info("\n" + "="*80)
        logger.info("Supplementary Results")
        logger.info("="*80)
        for r in results:
            if r:
                logger.info(f"LR={r['lr']:.3f}: Acc={r['best_test_acc']:.2f}%")

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
    parser = argparse.ArgumentParser(description='SGD+WD Supplementary Experiments')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/results/sgdwd_supplement.csv')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpus', type=str, default=None, help='GPU IDs to use, e.g., "0,1" or "all"')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Workers per GPU (default 2)')
    parser.add_argument('--no_log', action='store_true')
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus == 'all':
        gpu_ids = None
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []

    exp_name = "sgdwd_supplement"
    logger = get_logger(exp_name, log_to_file=not args.no_log, log_to_console=True)

    logger.log_config({
        'Experiment': 'SGD+WD Supplement (Low LR)',
        'Epochs': args.epochs,
        'Seed': args.seed,
        'GPUs': gpu_ids if gpu_ids else 'all' if args.gpus == 'all' else 'CPU',
        'Weight Decay': 0.002,
        'Missing LRs': [0.01, 0.02, 0.03, 0.07],
    })

    start_time = time.time()
    results = run_sgdwd_supplement(gpu_ids, args.epochs, args.seed, args.use_amp, logger, args.workers_per_gpu)
    elapsed_time = time.time() - start_time

    save_results(results, args.output, logger)

    successful = sum(1 for r in results if r is not None)
    logger.log_experiment_end(successful, len(results), elapsed_time)


if __name__ == '__main__':
    main()
