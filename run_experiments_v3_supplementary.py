"""
V3 补充实验：Batch Size 缩放规律探究

实验目标：
- 对比 BS=32 vs BS=128 下，η(LR) 和 λ(WD) 对训练动态的影响
- 验证 Batch Size 缩放规律

可视化设计（3×2布局）：
- 第一排：固定λ，η变化的训练曲线
- 第二排：最优(η,λ)热力图
- 第三排：固定η，λ变化的训练曲线

Usage:
    python run_experiments_v3_supplementary.py --gpus 0,1,3,4
    python run_experiments_v3_supplementary.py --gpus all
"""
import argparse
import csv
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from datetime import datetime
import filelock
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import resnet18
from utils import set_seed
from gpu_scheduler import GPUScheduler, parse_gpu_ids
from logger import get_logger


def get_cifar100_loaders(batch_size=128, num_workers=0):
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


def train_epoch_with_acc(model, train_loader, optimizer, scheduler, device, use_amp=True):
    """Train for one epoch and return both loss and accuracy."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None

    total_loss = 0.0
    correct = 0
    total_samples = 0

    # Use tqdm with position=0, leave=True for multiprocessing compatibility
    pbar = tqdm(train_loader, desc="Training", leave=False, 
                file=sys.stdout, dynamic_ncols=True)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'acc': f'{100.0*correct/total_samples:.1f}%'
        })

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy, avg_loss


def train_model_with_history(model, train_loader, test_loader, optimizer, scheduler,
                              device, epochs=100, use_amp=True):
    """
    Train model and record per-epoch history.

    Returns:
        result: dict with final metrics
        history: list of per-epoch records
    """
    history = []
    best_test_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_with_acc(model, train_loader, optimizer, scheduler, device, use_amp)
        test_acc, test_loss = evaluate(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Record epoch history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%", flush=True)

    final_record = history[-1]
    result = {
        'final_test_acc': final_record['test_acc'],
        'final_train_loss': final_record['train_loss'],
        'best_test_acc': best_test_acc
    }

    return result, history


def run_single_experiment_worker(batch_size, lr, wd, momentum, epochs, seed, use_amp, history_dir):
    """
    Worker function for running a single experiment.
    Saves per-epoch history to a JSON file.
    """
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(batch_size, num_workers=2)
    model = resnet18(num_classes=100).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Running: BS={batch_size} | LR={lr} | WD={wd} | Mom={momentum}", flush=True)

    result, history = train_model_with_history(
        model, train_loader, test_loader, optimizer, scheduler,
        device, epochs=epochs, use_amp=use_amp
    )

    # Save history to JSON file
    if history_dir:
        history_path = Path(history_dir)
        history_path.mkdir(parents=True, exist_ok=True)
        history_file = history_path / f"bs{batch_size}_lr{lr}_wd{wd}_m{momentum}.json"
        with open(history_file, 'w') as f:
            json.dump({
                'config': {
                    'batch_size': batch_size,
                    'lr': lr,
                    'wd': wd,
                    'momentum': momentum,
                    'epochs': epochs,
                    'seed': seed
                },
                'history': history
            }, f, indent=2)

    return {
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'momentum': momentum,
        'final_test_acc': result['final_test_acc'],
        'final_train_loss': result['final_train_loss'],
        'best_test_acc': result['best_test_acc']
    }


def get_experiment_key(batch_size, lr, wd, momentum):
    """Generate a unique key for an experiment configuration."""
    return f"bs{batch_size}_lr{lr}_wd{wd}_m{momentum}"


def load_completed_experiments(output_file):
    """
    Load completed experiments from existing results CSV.
    Returns a set of experiment keys that are already completed.
    """
    completed = set()
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            for _, row in df.iterrows():
                key = get_experiment_key(
                    row['batch_size'], row['lr'], row['wd'], row['momentum']
                )
                completed.add(key)
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    return completed


def save_single_result(result, output_file, logger=None):
    """Save a single result to CSV with file locking for thread safety."""
    if result is None:
        return

    fieldnames = ['batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc']

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    lock_file = output_file + ".lock"

    with filelock.FileLock(lock_file, timeout=30):
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

    if logger:
        key = get_experiment_key(
            result['batch_size'], result['lr'], result['wd'], result['momentum']
        )
        logger.info(f"Saved result: {key} -> {result['best_test_acc']:.2f}%")


def check_and_display_results(output_file, history_dir=None):
    """
    Check existing results and display a summary.
    Returns the DataFrame of existing results or None.
    """
    print("\n" + "=" * 80)
    print("检查现有实验结果")
    print("=" * 80)

    if not os.path.exists(output_file):
        print(f"结果文件不存在: {output_file}")
        print("尚未有任何已完成的实验。")
        return None

    try:
        df = pd.read_csv(output_file)
        total_experiments = len(df)
        print(f"\n已完成实验数量: {total_experiments}")

        if total_experiments == 0:
            return None

        # Group by batch size
        for bs in df['batch_size'].unique():
            bs_df = df[df['batch_size'] == bs]
            print(f"\n--- Batch Size = {bs} ---")
            print(f"  完成实验: {len(bs_df)}")

            # Best configuration
            best_idx = bs_df['best_test_acc'].idxmax()
            best = bs_df.loc[best_idx]
            print(f"  最优配置: LR={best['lr']}, WD={best['wd']}")
            print(f"  最佳准确率: {best['best_test_acc']:.2f}%")

            # Show top 5 configurations
            print(f"  Top 5 配置:")
            top5 = bs_df.nlargest(5, 'best_test_acc')[['lr', 'wd', 'best_test_acc']]
            for _, row in top5.iterrows():
                print(f"    LR={row['lr']}, WD={row['wd']} -> {row['best_test_acc']:.2f}%")

        # Check history files
        if history_dir and os.path.exists(history_dir):
            history_files = list(Path(history_dir).glob("*.json"))
            print(f"\n历史文件数量: {len(history_files)}")

        print("\n" + "=" * 80)
        return df

    except Exception as e:
        print(f"读取结果文件出错: {e}")
        return None


def run_batch_size_comparison(gpu_ids, epochs=100, seed=42, use_amp=True,
                               history_dir=None, output_file=None, logger=None,
                               workers_per_gpu=1):
    """
    Run batch size comparison experiments for BS=32 and BS=128.
    Supports checkpoint/resume: skips already completed experiments.

    Searches over (LR, WD) grid for each batch size.
    """
    if logger:
        logger.info("\n" + "="*80)
        logger.info("V3 补充实验：Batch Size 缩放规律")
        logger.info("="*80)

    momentum = 0.9  # SGDM

    # Experiment grid
    batch_sizes = [32, 128]

    # LR range (log scale, adjusted for both batch sizes)
    lr_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

    # WD range (log scale)
    wd_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

    # Load completed experiments for resume functionality
    completed_keys = set()
    if output_file:
        completed_keys = load_completed_experiments(output_file)
        if completed_keys and logger:
            logger.info(f"发现 {len(completed_keys)} 个已完成的实验，将跳过这些实验")

    # Build task list, skipping completed experiments
    tasks = []
    skipped = 0
    for bs in batch_sizes:
        for lr in lr_values:
            for wd in wd_values:
                key = get_experiment_key(bs, lr, wd, momentum)
                if key in completed_keys:
                    skipped += 1
                    continue
                task = (bs, lr, wd, momentum, epochs, seed, use_amp, history_dir)
                tasks.append(task)

    total_possible = len(batch_sizes) * len(lr_values) * len(wd_values)
    remaining = len(tasks)

    if logger:
        logger.info(f"Batch Sizes: {batch_sizes}")
        logger.info(f"LR values: {lr_values}")
        logger.info(f"WD values: {wd_values}")
        logger.info(f"总实验数: {total_possible} ({len(batch_sizes)} BS × {len(lr_values)} LR × {len(wd_values)} WD)")
        logger.info(f"已完成: {skipped}, 待运行: {remaining}")

    if remaining == 0:
        if logger:
            logger.info("所有实验已完成，无需运行。")
        return []

    # Callback to save results incrementally
    def on_task_complete(result):
        if result is not None and output_file:
            save_single_result(result, output_file, logger)

    # Use GPU scheduler for multi-GPU parallel execution
    scheduler = GPUScheduler(gpu_ids=gpu_ids, workers_per_gpu=workers_per_gpu, verbose=True)
    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=on_task_complete)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\n完成 {remaining} 实验，耗时 {elapsed_time/60:.2f} 分钟")

        # Analyze ALL results (including previously completed)
        # Reload from file to get complete picture
        if output_file and os.path.exists(output_file):
            df = pd.read_csv(output_file)
            logger.info("\n" + "="*80)
            logger.info("各 Batch Size 最优配置（含已完成实验）")
            logger.info("="*80)

            for bs in batch_sizes:
                bs_df = df[df['batch_size'] == bs]
                if len(bs_df) > 0:
                    best_idx = bs_df['best_test_acc'].idxmax()
                    best = bs_df.loc[best_idx]
                    logger.info(f"BS={bs}:")
                    logger.info(f"  最优 LR: {best['lr']}, 最优 WD: {best['wd']}")
                    logger.info(f"  最佳准确率: {best['best_test_acc']:.2f}%")

    return results


def save_results(results, output_file, logger=None):
    """Save results to CSV file"""
    if not results:
        return

    fieldnames = ['batch_size', 'lr', 'wd', 'momentum',
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
    parser = argparse.ArgumentParser(description='V3 补充实验：Batch Size 缩放规律')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/results/v3_supplementary.csv')
    parser.add_argument('--history_dir', type=str, default='outputs/history/v3_supplementary')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpus', type=str, default=None, help='GPU IDs to use, e.g., "0,1" or "all"')
    parser.add_argument('--workers_per_gpu', type=int, default=6, help='Number of workers per GPU (default: 6)')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--check', action='store_true', help='仅检查并显示现有结果，不运行实验')
    args = parser.parse_args()

    # Check mode: only display existing results
    if args.check:
        check_and_display_results(args.output, args.history_dir)
        return

    # Parse GPU IDs
    if args.gpus == 'all':
        gpu_ids = None  # Use all available GPUs
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []  # CPU only

    exp_name = "v3_supplementary"
    logger = get_logger(exp_name, log_to_file=not args.no_log, log_to_console=True)

    logger.log_config({
        'Experiment': 'V3 Supplementary - Batch Size Scaling',
        'Epochs': args.epochs,
        'Seed': args.seed,
        'GPUs': gpu_ids if gpu_ids else 'all' if args.gpus == 'all' else 'CPU',
        'History Dir': args.history_dir,
        'Output File': args.output,
    })

    # Show existing results before running
    check_and_display_results(args.output, args.history_dir)

    start_time = time.time()
    results = run_batch_size_comparison(
        gpu_ids, args.epochs, args.seed, args.use_amp,
        history_dir=args.history_dir,
        output_file=args.output,  # Enable incremental saving
        logger=logger,
        workers_per_gpu=args.workers_per_gpu
    )
    elapsed_time = time.time() - start_time

    # Results are already saved incrementally, no need to save again

    successful = sum(1 for r in results if r is not None) if results else 0
    total = len(results) if results else 0
    logger.log_experiment_end(successful, total, elapsed_time)

    print(f"\n实验历史已保存到: {args.history_dir}")
    print(f"结果汇总已保存到: {args.output}")

    # Final summary
    check_and_display_results(args.output, args.history_dir)


if __name__ == '__main__':
    main()

