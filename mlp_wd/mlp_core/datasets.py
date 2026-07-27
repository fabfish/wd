"""CIFAR-10 / MNIST loaders with plain normalization (no augmentation).

We deliberately skip RandomCrop / RandomFlip etc. so that the loss-vs-(eta*lambda)
spoon shape reflects the optimizer dynamics, not data augmentation noise.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_DEFAULT_DATA_ROOT = Path("./data")


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def _make_loaders(train_dataset, test_dataset, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, test_loader


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str | Path = _DEFAULT_DATA_ROOT,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    root = str(data_root)
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return _make_loaders(train_dataset, test_dataset, batch_size, num_workers)


def get_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str | Path = _DEFAULT_DATA_ROOT,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    root = str(data_root)
    # Prefer local files (download=True races under multi-process spawn and
    # needs outbound HTTPS). Fall back to download only if raw/ is missing.
    raw_ok = (Path(root) / "MNIST" / "raw" / "train-images-idx3-ubyte").exists() \
        or (Path(root) / "MNIST" / "raw" / "train-images-idx3-ubyte.gz").exists()
    download = not raw_ok
    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    return _make_loaders(train_dataset, test_dataset, batch_size, num_workers)


def get_loaders(
    dataset: str,
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str | Path = _DEFAULT_DATA_ROOT,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return get_cifar10_loaders(batch_size, num_workers, data_root)
    if dataset == "mnist":
        return get_mnist_loaders(batch_size, num_workers, data_root)
    raise ValueError(f"Unknown dataset: {dataset!r}")
