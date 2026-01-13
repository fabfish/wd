"""
Automated experiment runner for the three experiment sets with multi-GPU support.
Saves results to results.csv.
"""
import argparse
import csv
import os
from itertools import product

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import resnet18
from utils import set_seed, train_model
from gpu_scheduler import GPUScheduler, parse_gpu_ids


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


def run_single_experiment_worker(method, batch_size, lr, wd, momentum, epochs, seed, use_amp):
    """
    Worker function for running a single experiment.
    This function is called by the GPU scheduler in a subprocess.
    Device is automatically detected (uses CUDA_VISIBLE_DEVICES set by scheduler).
    """
    # Enable cudnn benchmark for performance
    torch.backends.cudnn.benchmark = True

    # Set seed for reproducibility
    set_seed(seed)

    # Device is automatically set by CUDA_VISIBLE_DEVICES
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Reduce num_workers to avoid too many processes
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


def experiment_set_1(gpu_ids, epochs=100, seed=42, use_amp=True):
    """
    Experiment Set 1: Optimal LR Ordering
    Compare SGD vs. SGD+WD vs. SGDM+WD
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET 1: Optimal LR Ordering")
    print("="*80)

    batch_size = 128
    lr_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    conditions = [
        ('SGD', 0, 0),           # (method, momentum, wd)
        ('SGD+WD', 0, 5e-4),
        ('SGDM+WD', 0.9, 5e-4),
    ]

    # Prepare all tasks
    tasks = []
    for (method, momentum, wd), lr in product(conditions, lr_values):
        task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp)
        tasks.append(task)

    total_runs = len(tasks)
    print(f"Total experiments: {total_runs}")
    print(f"Using GPUs: {gpu_ids if gpu_ids else 'CPU'}")

    # Run tasks in parallel using GPU scheduler
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)

    return results


def experiment_set_2(gpu_ids, epochs=100, seed=42, use_amp=True):
    """
    Experiment Set 2: Eta-Lambda Interaction (Heatmap)
    Verify inverse relationship between LR and WD
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET 2: Eta-Lambda Interaction (Heatmap)")
    print("="*80)

    batch_size = 128
    momentum = 0.9
    method = 'SGDM'

    lr_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    wd_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    # Prepare all tasks
    tasks = []
    for lr, wd in product(lr_values, wd_values):
        task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp)
        tasks.append(task)

    total_runs = len(tasks)
    print(f"Total experiments: {total_runs}")
    print(f"Using GPUs: {gpu_ids if gpu_ids else 'CPU'}")

    # Run tasks in parallel using GPU scheduler
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)

    return results


def experiment_set_3(gpu_ids, epochs=100, seed=42, use_amp=True):
    """
    Experiment Set 3: Batch Size Scaling
    Check relation between Batch Size and optimal WD
    Scale LR linearly with batch size: lr = 0.1 * (B / 128)
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET 3: Batch Size Scaling")
    print("="*80)

    momentum = 0.9
    method = 'SGDM'
    base_lr = 0.1
    base_batch_size = 128

    batch_sizes = [64, 128, 256, 512]
    wd_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

    # Prepare all tasks
    tasks = []
    for batch_size, wd in product(batch_sizes, wd_values):
        # Scale LR linearly with batch size
        lr = base_lr * (batch_size / base_batch_size)
        task = (method, batch_size, lr, wd, momentum, epochs, seed, use_amp)
        tasks.append(task)

    total_runs = len(tasks)
    print(f"Total experiments: {total_runs}")
    print(f"Using GPUs: {gpu_ids if gpu_ids else 'CPU'}")

    # Run tasks in parallel using GPU scheduler
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)

    return results


def save_results(results, output_file='results.csv'):
    """Save results to CSV file"""
    if not results:
        print("No results to save!")
        return

    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc']

    file_exists = os.path.exists(output_file)

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:
            if result is not None:  # Skip failed tasks
                writer.writerow(result)

    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run automated experiments with multi-GPU support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all available GPUs
  python run_experiments.py --experiment 1 --gpus all

  # Run on specific GPUs
  python run_experiments.py --experiment 2 --gpus 0,1,2,3

  # Run on GPU range
  python run_experiments.py --experiment 3 --gpus 0-3

  # Run on single GPU
  python run_experiments.py --experiment 1 --gpus 0

  # Run on CPU (no GPUs)
  python run_experiments.py --experiment 1
        """
    )
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], required=True,
                        help='Which experiment set to run (1, 2, or 3)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs to use (e.g., "0,1,2" or "0-3" or "all"). If not specified, uses CPU.')
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus == 'all':
        gpu_ids = None  # Will use all available GPUs
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []  # CPU mode

    # Print configuration
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Experiment Set: {args.experiment}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Mixed Precision: {args.use_amp}")
    print(f"Output File: {args.output}")
    print(f"GPU Configuration: {gpu_ids if gpu_ids else 'CPU'}")
    print("="*80)

    # Run selected experiment
    if args.experiment == 1:
        results = experiment_set_1(gpu_ids, args.epochs, args.seed, args.use_amp)
    elif args.experiment == 2:
        results = experiment_set_2(gpu_ids, args.epochs, args.seed, args.use_amp)
    elif args.experiment == 3:
        results = experiment_set_3(gpu_ids, args.epochs, args.seed, args.use_amp)

    # Save results
    save_results(results, args.output)

    # Count successful results
    successful_results = sum(1 for r in results if r is not None)

    print(f"\n{'='*80}")
    print(f"Experiment Set {args.experiment} completed!")
    print(f"Successful runs: {successful_results}/{len(results)}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
