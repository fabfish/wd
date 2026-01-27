"""
Main entry point for running individual experiments with argparse.
"""
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from wd_core.models import resnet18
from wd_core.utils import set_seed, train_model


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


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 ResNet-18 Experiments')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Enable cudnn benchmark for performance
    torch.backends.cudnn.benchmark = True

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading CIFAR-100 with batch size {args.batch_size}...")
    train_loader, test_loader = get_cifar100_loaders(args.batch_size, args.num_workers)

    # Create model
    print("Creating ResNet-18 model...")
    model = resnet18(num_classes=100).to(device)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Print experiment configuration
    print("\n" + "="*60)
    print("Experiment Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.wd}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")
    print(f"  Mixed Precision: {args.use_amp}")
    print("="*60 + "\n")

    # Train
    best_test_acc, final_test_acc, final_train_loss = train_model(
        model, train_loader, test_loader, optimizer, scheduler,
        device, epochs=args.epochs, use_amp=args.use_amp
    )

    # Print final results
    print("\n" + "="*60)
    print("Final Results:")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"  Final Train Loss: {final_train_loss:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
