"""
实验一改进版 V3：针对 SGDM+WD 高学习率不收敛问题的修正

核心问题分析：
- SGDM+WD 在 LR>0.5 时发散，因为 momentum=0.9 放大了有效步长约 1/(1-0.9)=10 倍
- 固定 WD=5e-4 对于高 LR 的 SGDM 来说太小，无法提供足够的正则化

解决方案：
1. 方案A：为每种方法使用不同的 WD（基于理论最优配置）
2. 方案B：为 SGDM+WD 使用更小的 LR 搜索范围
3. 方案C：联合搜索 LR 和 WD，找到每种方法的最优配置

本脚本实现方案C：对每种方法进行 LR-WD 联合搜索
"""
import argparse
import csv
import os
import time
from itertools import product
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


def experiment_1_joint_search(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """
    实验一改进：LR-WD 联合搜索

    为每种方法分别搜索最优的 (LR, WD) 组合，然后比较各方法的最优 LR

    理论预期：
    - 在各自最优 WD 下，最优 LR 应满足 η_SGD > η_SGD+WD > η_SGDM+WD
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("实验一 V3：LR-WD 联合搜索 - 验证最优LR排序")
        logger.info("="*80)

    batch_size = 128

    # 为每种方法定义合适的搜索空间
    experiments = {
        'SGD': {
            'momentum': 0,
            'lr_values': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            'wd_values': [0],  # Pure SGD 不使用 WD
        },
        'SGD+WD': {
            'momentum': 0,
            'lr_values': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            'wd_values': [1e-4, 5e-4, 1e-3, 2e-3],
        },
        'SGDM+WD': {
            'momentum': 0.9,
            # SGDM 需要更小的 LR 范围（因为 momentum 放大有效步长）
            'lr_values': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
            # SGDM 需要更大的 WD 范围来稳定高 LR
            'wd_values': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        },
    }

    tasks = []
    for method, config in experiments.items():
        momentum = config['momentum']
        for lr, wd in product(config['lr_values'], config['wd_values']):
            task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp, False)
            tasks.append(task)

    total_runs = len(tasks)
    if logger:
        logger.info(f"总实验数: {total_runs}")
        for method, config in experiments.items():
            n = len(config['lr_values']) * len(config['wd_values'])
            logger.info(f"  {method}: {n} runs (LR: {config['lr_values']}, WD: {config['wd_values']})")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\n完成 {total_runs} 实验，耗时 {elapsed_time/60:.2f} 分钟")

        # 分析结果：找出每种方法的最优配置
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


def experiment_1_sgdm_wd_sweep(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """
    专门针对 SGDM+WD 的细粒度 LR-WD 联合搜索

    目的：找到 SGDM+WD 在不同 LR 下的最优 WD，验证 WD 需要随 LR 增加而增加
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("SGDM+WD 细粒度 LR-WD 联合搜索")
        logger.info("="*80)

    batch_size = 128
    momentum = 0.9
    method = 'SGDM+WD'

    # 细粒度搜索
    lr_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
    wd_values = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    tasks = []
    for lr, wd in product(lr_values, wd_values):
        task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp, False)
        tasks.append(task)

    total_runs = len(tasks)
    if logger:
        logger.info(f"总实验数: {total_runs} ({len(lr_values)} LRs × {len(wd_values)} WDs)")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\n完成 {total_runs} 实验，耗时 {elapsed_time/60:.2f} 分钟")

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
    parser = argparse.ArgumentParser(description='实验一改进版 V3')
    parser.add_argument('--experiment', type=str, choices=['joint', 'sgdm_sweep'], default='joint',
                        help='joint=联合搜索, sgdm_sweep=SGDM专项搜索')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/results/results_v3.csv')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--no_log', action='store_true')
    args = parser.parse_args()

    if args.gpus == 'all':
        gpu_ids = None
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []

    exp_name = f"exp1_v3_{args.experiment}"
    logger = get_logger(exp_name, log_to_file=not args.no_log, log_to_console=True)

    logger.log_config({
        'Experiment': args.experiment,
        'Epochs': args.epochs,
        'Seed': args.seed,
        'GPUs': gpu_ids if gpu_ids else 'CPU',
    })

    start_time = time.time()

    if args.experiment == 'joint':
        results = experiment_1_joint_search(gpu_ids, args.epochs, args.seed, args.use_amp, logger)
    else:
        results = experiment_1_sgdm_wd_sweep(gpu_ids, args.epochs, args.seed, args.use_amp, logger)

    elapsed_time = time.time() - start_time
    save_results(results, args.output, logger)

    successful = sum(1 for r in results if r is not None)
    logger.log_experiment_end(successful, len(results), elapsed_time)


if __name__ == '__main__':
    main()
