"""
Automated experiment runner for the three experiment sets.
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


def get_cifar100_loaders(batch_size=128, num_workers=4):
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


def run_single_experiment(method, batch_size, lr, wd, momentum, epochs, seed, device, use_amp=True):
    """Run a single experiment configuration"""
    set_seed(seed)

    train_loader, test_loader = get_cifar100_loaders(batch_size, num_workers=4)
    model = resnet18(num_classes=100).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*80}")
    print(f"Running: {method} | BS={batch_size} | LR={lr} | WD={wd} | Mom={momentum}")
    print(f"{'='*80}")

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


def experiment_set_1(device, epochs=100, seed=42, use_amp=True):
    """
    Experiment Set 1: Optimal LR Ordering
    Compare SGD vs. SGD+WD vs. SGDM+WD
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET 1: Optimal LR Ordering")
    print("="*80)

    results = []
    batch_size = 128
    lr_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    conditions = [
        ('SGD', 0, 0),           # (method, momentum, wd)
        ('SGD+WD', 0, 5e-4),
        ('SGDM+WD', 0.9, 5e-4),
    ]

    total_runs = len(conditions) * len(lr_values)
    current_run = 0

    for (method, momentum, wd), lr in product(conditions, lr_values):
        current_run += 1
        print(f"\n>>> Run {current_run}/{total_runs}")

        result = run_single_experiment(
            method=method,
            batch_size=batch_size,
            lr=lr,
            wd=wd,
            momentum=momentum,
            epochs=epochs,
            seed=seed,
            device=device,
            use_amp=use_amp
        )
        results.append(result)

    return results


def experiment_set_2(device, epochs=100, seed=42, use_amp=True):
    """
    Experiment Set 2: Eta-Lambda Interaction (Heatmap)
    Verify inverse relationship between LR and WD
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET 2: Eta-Lambda Interaction (Heatmap)")
    print("="*80)

    results = []
    batch_size = 128
    momentum = 0.9
    method = 'SGDM'

    lr_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    wd_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    total_runs = len(lr_values) * len(wd_values)
    current_run = 0

    for lr, wd in product(lr_values, wd_values):
        current_run += 1
        print(f"\n>>> Run {current_run}/{total_runs}")

        result = run_single_experiment(
            method=method,
            batch_size=batch_size,
            lr=lr,
            wd=wd,
            momentum=momentum,
            epochs=epochs,
            seed=seed,
            device=device,
            use_amp=use_amp
        )
        results.append(result)

    return results


def experiment_set_3(device, epochs=100, seed=42, use_amp=True):
    """
    Experiment Set 3: Batch Size Scaling
    Check relation between Batch Size and optimal WD
    Scale LR linearly with batch size: lr = 0.1 * (B / 128)
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET 3: Batch Size Scaling")
    print("="*80)

    results = []
    momentum = 0.9
    method = 'SGDM'
    base_lr = 0.1
    base_batch_size = 128

    batch_sizes = [64, 128, 256, 512]
    wd_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

    total_runs = len(batch_sizes) * len(wd_values)
    current_run = 0

    for batch_size, wd in product(batch_sizes, wd_values):
        current_run += 1
        print(f"\n>>> Run {current_run}/{total_runs}")

        # Scale LR linearly with batch size
        lr = base_lr * (batch_size / base_batch_size)

        result = run_single_experiment(
            method=method,
            batch_size=batch_size,
            lr=lr,
            wd=wd,
            momentum=momentum,
            epochs=epochs,
            seed=seed,
            device=device,
            use_amp=use_amp
        )
        results.append(result)

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
            writer.writerow(result)

    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run automated experiments')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], required=True,
                        help='Which experiment set to run (1, 2, or 3)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision')
    args = parser.parse_args()

    # Enable cudnn benchmark
    torch.backends.cudnn.benchmark = True

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run selected experiment
    if args.experiment == 1:
        results = experiment_set_1(device, args.epochs, args.seed, args.use_amp)
    elif args.experiment == 2:
        results = experiment_set_2(device, args.epochs, args.seed, args.use_amp)
    elif args.experiment == 3:
        results = experiment_set_3(device, args.epochs, args.seed, args.use_amp)

    # Save results
    save_results(results, args.output)

    print(f"\n{'='*80}")
    print(f"Experiment Set {args.experiment} completed!")
    print(f"Total runs: {len(results)}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
