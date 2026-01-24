"""
Extend LR range for SGD+WD and SGDM+WD to show convex shape (drop at high LR).
Target configurations:
- SGD+WD (wd=0.005): Extend to [0.3, 0.4, 0.5]
- SGDM+WD (wd=0.0015, mom=0.8): Extend to [0.15, 0.2, 0.25, 0.3]
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
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(batch_size, num_workers=2)
    model = resnet18(num_classes=100).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

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

def run_extension(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None, workers_per_gpu=1, log_interval=10, output_file=None):
    batch_size = 128
    tasks = []
    
    # 1. Extend SGD+WD (wd=0.005)
    # Existing: [0.05, 0.1, 0.15, 0.2, 0.25]
    # Add: [0.3, 0.4, 0.5]
    for lr in [0.3, 0.4, 0.5]:
        tasks.append(("SGD+WD", batch_size, lr, 0.005, 0.0, epochs, seed, use_amp, log_interval))
        
    # 2. Extend SGDM+WD (wd=0.0015, mom=0.8)
    # Existing: [0.03, 0.05, 0.07, 0.1]
    # Add: [0.15, 0.2, 0.25, 0.3]
    for lr in [0.15, 0.2, 0.25, 0.3]:
        tasks.append(("SGDM+WD", batch_size, lr, 0.0015, 0.8, epochs, seed, use_amp, log_interval))
        
    def save_result_incrementally(result):
        if result is not None and output_file:
            save_single_result(result, output_file)
            if logger:
                logger.info(f"  Saved: Method={result['method']}, Mom={result['momentum']}, LR={result['lr']}, Acc={result['best_test_acc']:.2f}%")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=False, workers_per_gpu=workers_per_gpu)
    scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=save_result_incrementally)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/results/lr_extension.csv')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=25)
    args = parser.parse_args()

    if args.gpus: gpu_ids = parse_gpu_ids(args.gpus)
    else: gpu_ids = []
    
    logger = get_logger("lr_extension", log_to_file=True, log_to_console=True)
    run_extension(gpu_ids, args.epochs, args.seed, True, logger, args.workers_per_gpu, args.log_interval, args.output)

if __name__ == '__main__':
    main()
