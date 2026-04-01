"""
Rebuttal experiment runner: supports multiple models and seeds.
Runs the same 3 experiment sets as the original paper with configurable model and seed.
"""
import argparse
import csv
import os
import sys
import time
from itertools import product
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wd_core.models import get_model
from wd_core.utils import set_seed, train_model
from wd_core.gpu_scheduler import GPUScheduler, parse_gpu_ids
from wd_core.logger import get_logger


MODEL_NAME = 'resnet18'
WORKERS_PER_GPU = 1


def get_cifar100_loaders(batch_size=128, num_workers=2):
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
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def run_single_experiment_worker(model_name, method, batch_size, lr, wd, momentum, epochs, seed, use_amp):
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(batch_size, num_workers=2)
    model = get_model(model_name, num_classes=100).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"[{model_name}] {method} | BS={batch_size} | LR={lr} | WD={wd} | Mom={momentum}")

    best_test_acc, final_test_acc, final_train_loss, final_test_loss = train_model(
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
        'best_test_acc': best_test_acc,
        'final_test_loss': final_test_loss
    }


def experiment_set_1(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """Experiment 1: Optimal LR Ordering — SGD vs SGD+WD vs SGDM+WD"""
    if logger:
        logger.info("EXPERIMENT SET 1: Optimal LR Ordering")

    batch_size = 128
    lr_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    conditions = [
        ('SGD', 0, 0),
        ('SGD+WD', 0, 5e-4),
        ('SGDM+WD', 0.9, 5e-4),
    ]

    tasks = []
    for (method, momentum, wd), lr in product(conditions, lr_values):
        tasks.append((MODEL_NAME, method, batch_size, lr, wd, momentum, epochs, seed, use_amp))

    if logger:
        logger.info(f"Total experiments: {len(tasks)}, GPUs: {gpu_ids}")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True, workers_per_gpu=WORKERS_PER_GPU)
    start = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    if logger:
        logger.info(f"Exp1 done in {(time.time()-start)/60:.1f} min")
    return results


def experiment_set_2(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """Experiment 2: LR-WD Interaction Heatmap (SGDM)"""
    if logger:
        logger.info("EXPERIMENT SET 2: Eta-Lambda Interaction (Heatmap)")

    batch_size = 128
    momentum = 0.9
    method = 'SGDM'
    lr_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    wd_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]

    tasks = []
    for lr, wd in product(lr_values, wd_values):
        tasks.append((MODEL_NAME, method, batch_size, lr, wd, momentum, epochs, seed, use_amp))

    if logger:
        logger.info(f"Total experiments: {len(tasks)}, GPUs: {gpu_ids}")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True, workers_per_gpu=WORKERS_PER_GPU)
    start = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    if logger:
        logger.info(f"Exp2 done in {(time.time()-start)/60:.1f} min")
    return results


def experiment_set_3(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """Experiment 3: Batch Size Scaling with linear LR rule"""
    if logger:
        logger.info("EXPERIMENT SET 3: Batch Size Scaling")

    momentum = 0.9
    method = 'SGDM'
    base_lr = 0.1
    base_batch_size = 128
    batch_sizes = [64, 128, 256, 512]
    wd_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

    tasks = []
    for batch_size, wd in product(batch_sizes, wd_values):
        lr = base_lr * (batch_size / base_batch_size)
        tasks.append((MODEL_NAME, method, batch_size, lr, wd, momentum, epochs, seed, use_amp))

    if logger:
        logger.info(f"Total experiments: {len(tasks)}, GPUs: {gpu_ids}")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True, workers_per_gpu=WORKERS_PER_GPU)
    start = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    if logger:
        logger.info(f"Exp3 done in {(time.time()-start)/60:.1f} min")
    return results


def save_results(results, output_file, logger=None):
    if not results:
        return
    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc',
                  'final_test_loss']
    file_exists = os.path.exists(output_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            if r is not None:
                writer.writerow(r)
    if logger:
        logger.info(f"Results saved to {output_file}")


def main():
    global MODEL_NAME

    parser = argparse.ArgumentParser(description='Rebuttal experiments with multi-model/seed support')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg16'],
                        help='Model architecture')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], required=True,
                        help='Which experiment set to run')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs (e.g., "0,1,2" or "all")')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--workers_per_gpu', type=int, default=1,
                        help='Number of parallel workers per GPU (increase for low-utilization models)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix appended to output filename (e.g. "_ext" -> results_resnet18_seed42_ext.csv)')
    args = parser.parse_args()

    global WORKERS_PER_GPU
    MODEL_NAME = args.model
    WORKERS_PER_GPU = args.workers_per_gpu

    if args.gpus == 'all':
        gpu_ids = None
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []

    output_dir = Path(__file__).resolve().parent / 'results'
    output_file = output_dir / f'results_{args.model}_seed{args.seed}{args.suffix}.csv'

    logger = get_logger(f"rebuttal_{args.model}_s{args.seed}_exp{args.experiment}")
    logger.info(f"Model: {args.model} | Seed: {args.seed} | Experiment: {args.experiment}")
    logger.info(f"Output: {output_file}")

    start = time.time()
    exp_funcs = {1: experiment_set_1, 2: experiment_set_2, 3: experiment_set_3}
    results = exp_funcs[args.experiment](gpu_ids, args.epochs, args.seed, args.use_amp, logger)
    elapsed = time.time() - start

    save_results(results, str(output_file), logger)

    ok = sum(1 for r in results if r is not None)
    logger.info(f"Done: {ok}/{len(results)} succeeded in {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
