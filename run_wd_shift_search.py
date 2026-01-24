"""
Search for parameters to achieve the desired LR ordering trend:
SGDM+WD > SGD+WD > SGD in accuracy
SGDM+WD < SGD+WD < SGD in optimal LR

Experiments:
1. SGD+WD with higher WD (0.005, 0.01, 0.02) to shift optimal LR leftwards.
2. SGDM+WD with WD=0.001/0.0015 and mom=0.8/0.85/0.9 to boost accuracy > 79%.
"""
import argparse
import csv
import os
import time
from pathlib import Path
import threading

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import resnet18
from utils import set_seed, train_model
from gpu_scheduler import GPUScheduler, parse_gpu_ids
from logger import get_logger

# Thread-safe lock for incremental CSV writing
csv_lock = threading.Lock()


def get_cifar100_loaders(batch_size=128, num_workers=2):
    """Load CIFAR-100 dataset"""
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


def run_single_experiment_worker(method, batch_size, lr, wd, momentum, epochs, seed, use_amp, log_interval=10):
    """Worker function for running a single experiment."""
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(batch_size, num_workers=2)
    model = resnet18(num_classes=100).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Simplified logging for worker
    print(f"Running: {method} | LR={lr} | WD={wd} | Mom={momentum}")

    best_test_acc, final_test_acc, final_train_loss = train_model(
        model, train_loader, test_loader, optimizer, scheduler,
        device, epochs=epochs, use_amp=use_amp, log_interval=log_interval
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


def save_single_result(result, output_file):
    """Save a single result to CSV file (thread-safe)"""
    if result is None:
        return

    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc']

    with csv_lock:
        file_exists = os.path.exists(output_file)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)


def run_search(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None, workers_per_gpu=1, log_interval=10, output_file=None):
    batch_size = 128
    
    tasks = []
    
    # Plan A: Shift SGD+WD Left
    # Test higher WDs to lower optimal LR
    # Expected optimal LR: around 0.1-0.2
    for wd in [0.005, 0.01]:
        for lr in [0.05, 0.1, 0.15, 0.2, 0.25]:
             # method, batch_size, lr, wd, momentum, epochs, seed, use_amp, log_interval
             # Note: momentum=0.0 for SGD+WD
             tasks.append(("SGD+WD", batch_size, lr, wd, 0.0, epochs, seed, use_amp, log_interval))

    # Plan B: Boost SGDM+WD Accuracy
    # Test various mom/wd combos to beat 78.95%
    # Expected optimal LR: < 0.1
    # wd=0.001 might be better than 0.002 for higher accuracy?
    for wd in [0.001, 0.0015]:
        for momentum in [0.8, 0.85, 0.9]:
            for lr in [0.03, 0.05, 0.07, 0.1]:
                tasks.append(("SGDM+WD", batch_size, lr, wd, momentum, epochs, seed, use_amp, log_interval))

    total_runs = len(tasks)
    if logger:
        logger.info(f"Total experiments: {total_runs}")
        logger.info("Groups:")
        logger.info("  1. SGD+WD (shift left): wd=[0.005, 0.01], lr=[0.05-0.25]")
        logger.info("  2. SGDM+WD (boost acc): wd=[0.001, 0.0015], mom=[0.8, 0.85, 0.9], lr=[0.03-0.1]")

    # Create incremental save callback
    def save_result_incrementally(result):
        if result is not None and output_file:
            save_single_result(result, output_file)
            if logger:
                logger.info(f"  Saved: Method={result['method']}, WD={result['wd']}, Mom={result['momentum']}, LR={result['lr']}, Acc={result['best_test_acc']:.2f}%")

    # Use GPU scheduler
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=False, workers_per_gpu=workers_per_gpu)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=save_result_incrementally)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\nCompleted {total_runs} experiments in {elapsed_time/60:.2f} minutes")
        
    return results


def main():
    parser = argparse.ArgumentParser(description='Refining LR Ordering Search')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/results/wd_shift_search.csv')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpus', type=str, default=None, help='GPU IDs to use')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=25)
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus == 'all':
        gpu_ids = None
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []

    exp_name = "wd_shift_search"
    logger = get_logger(exp_name, log_to_file=True, log_to_console=True)

    logger.log_config(vars(args))

    run_search(
        gpu_ids, args.epochs, args.seed, args.use_amp, logger, args.workers_per_gpu,
        log_interval=args.log_interval, output_file=args.output
    )

if __name__ == '__main__':
    main()
