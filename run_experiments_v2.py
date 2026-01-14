"""
改进的实验设计 - 用于验证三个核心理论假设
实验一：验证最优学习率递减 SGD > SGD+WD > SGDM+WD (稳定性边界)
实验二：验证 λ 与 η 成反比例关系
实验三：验证 B 与 λ 成正比例关系
"""
import argparse
import csv
import os
import time
from itertools import product
from pathlib import Path
import numpy as np

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


def run_single_experiment_worker(method, batch_size, lr, wd, momentum, epochs, seed, use_amp, save_checkpoints=False):
    """Worker function for running a single experiment."""
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(batch_size, num_workers=2)
    model = resnet18(num_classes=100).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd
    )

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


def experiment_1_refined(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """
    实验一改进：更细粒度的学习率搜索
    目标：精确定位每种方法的最优学习率，验证 η_SGD > η_SGD+WD > η_SGDM+WD

    改进点：
    1. 更密集的学习率采样 (15个点)
    2. 扩展搜索范围到更高的学习率
    3. 分别测量"稳定性边界"和"最优泛化点"
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("实验一改进：细粒度学习率搜索 - 验证最优LR排序")
        logger.info("="*80)
    else:
        print("\n" + "="*80)
        print("实验一改进：细粒度学习率搜索 - 验证最优LR排序")
        print("="*80)

    batch_size = 128
    # 更细粒度的学习率搜索：15个点，覆盖 0.01 到 3.0
    lr_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    conditions = [
        ('SGD', 0, 0),           # Pure SGD
        ('SGD+WD', 0, 5e-4),     # SGD + Weight Decay
        ('SGDM+WD', 0.9, 5e-4),  # SGD + Momentum + Weight Decay
    ]

    tasks = []
    for (method, momentum, wd), lr in product(conditions, lr_values):
        task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp, False)
        tasks.append(task)

    total_runs = len(tasks)
    msg = f"实验一改进: {len(conditions)} methods × {len(lr_values)} LRs = {total_runs} runs"
    if logger:
        logger.info(msg)
        logger.info(f"LR range: {lr_values}")
    else:
        print(msg)

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\n完成 {total_runs} 实验，耗时 {elapsed_time/60:.2f} 分钟")

    return results


def experiment_2_eta_lambda(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """
    实验二：验证 λ (weight decay) 与 η (learning rate) 成反比例关系

    理论预期：最优配置应该满足 λ * η ≈ const，即热力图中高准确率区域呈反比例曲线

    设计：
    - 固定 SGDM (momentum=0.9)
    - 2D 网格搜索 η × λ
    - 绘制热力图验证反比例关系
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("实验二：η-λ 反比例关系验证 (热力图)")
        logger.info("="*80)
    else:
        print("\n" + "="*80)
        print("实验二：η-λ 反比例关系验证 (热力图)")
        print("="*80)

    batch_size = 128
    momentum = 0.9
    method = 'SGDM'

    # 更细粒度的网格
    lr_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]  # 7个点
    wd_values = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]  # 8个点

    tasks = []
    for lr, wd in product(lr_values, wd_values):
        task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp, False)
        tasks.append(task)

    total_runs = len(tasks)
    msg = f"实验二: {len(lr_values)} LRs × {len(wd_values)} WDs = {total_runs} runs"
    if logger:
        logger.info(msg)
        logger.info(f"LR range: {lr_values}")
        logger.info(f"WD range: {wd_values}")
    else:
        print(msg)

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\n完成 {total_runs} 实验，耗时 {elapsed_time/60:.2f} 分钟")

    return results


def experiment_3_batch_lambda(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    """
    实验三：验证 B (batch size) 与 λ (weight decay) 成正比例关系

    理论预期：λ_optimal ∝ B，即更大的 batch size 需要更大的 weight decay

    设计：
    - 固定 SGDM (momentum=0.9)
    - 使用 Linear LR Scaling: η = 0.1 × (B / 128)
    - 对每个 B，搜索最优 λ
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("实验三：B-λ 正比例关系验证")
        logger.info("="*80)
    else:
        print("\n" + "="*80)
        print("实验三：B-λ 正比例关系验证")
        print("="*80)

    momentum = 0.9
    method = 'SGDM'
    base_lr = 0.1
    base_batch_size = 128

    # 更多的 batch size 选项
    batch_sizes = [32, 64, 128, 256, 512]
    # 更细粒度的 weight decay 搜索
    wd_values = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    tasks = []
    for batch_size, wd in product(batch_sizes, wd_values):
        # Linear LR Scaling
        lr = base_lr * (batch_size / base_batch_size)
        task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp, False)
        tasks.append(task)

    total_runs = len(tasks)
    msg = f"实验三: {len(batch_sizes)} BSs × {len(wd_values)} WDs = {total_runs} runs"
    if logger:
        logger.info(msg)
        logger.info(f"Batch sizes: {batch_sizes}")
        logger.info(f"WD range: {wd_values}")
        logger.info("Using Linear LR Scaling: η = 0.1 × (B / 128)")
    else:
        print(msg)

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
        msg = "No results to save!"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
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

    msg = f"Results saved to {output_file}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def main():
    parser = argparse.ArgumentParser(
        description='改进的实验设计 - 验证三个核心理论假设',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验说明:
  实验1: 细粒度LR搜索 - 验证 η_SGD > η_SGD+WD > η_SGDM+WD
  实验2: η-λ 网格搜索 - 验证反比例关系 (热力图)
  实验3: B-λ 搜索 - 验证正比例关系

示例:
  python run_experiments_v2.py --experiment 1 --gpus all
  python run_experiments_v2.py --experiment 2 --gpus 0,1,2,3
  python run_experiments_v2.py --experiment 3 --gpus all --epochs 200
        """
    )
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], required=True,
                        help='实验编号 (1=LR排序, 2=η-λ关系, 3=B-λ关系)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default='outputs/results/results_v2.csv',
                        help='输出CSV文件')
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs (e.g., "0,1,2" or "0-3" or "all")')
    parser.add_argument('--no_log', action='store_true', help='禁用日志文件')
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus == 'all':
        gpu_ids = None
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []

    # Create logger
    exp_name = f"exp{args.experiment}_v2"
    logger = get_logger(exp_name, log_to_file=not args.no_log, log_to_console=True)

    # Log configuration
    config = {
        'Experiment': args.experiment,
        'Epochs': args.epochs,
        'Seed': args.seed,
        'GPUs': gpu_ids if gpu_ids else 'CPU',
        'Output': args.output,
    }
    logger.log_config(config)

    start_time = time.time()

    if args.experiment == 1:
        results = experiment_1_refined(gpu_ids, args.epochs, args.seed, args.use_amp, logger)
    elif args.experiment == 2:
        results = experiment_2_eta_lambda(gpu_ids, args.epochs, args.seed, args.use_amp, logger)
    elif args.experiment == 3:
        results = experiment_3_batch_lambda(gpu_ids, args.epochs, args.seed, args.use_amp, logger)

    elapsed_time = time.time() - start_time

    save_results(results, args.output, logger)

    successful_results = sum(1 for r in results if r is not None)
    logger.log_experiment_end(successful_results, len(results), elapsed_time)
    logger.info(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
