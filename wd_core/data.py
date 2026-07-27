"""
Dataset helpers shared by the experiment runners.

Kept separate from the runners so that every experiment sees exactly the same
preprocessing, and so that the leave-one-out variant needed by the uniform
stability probe lives next to the standard loaders.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _cifar100_transforms(augment=True):
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    return transform_train, transform_test


def get_cifar100_loaders(batch_size=128, num_workers=4, data_dir='./data',
                         augment=True, download=True, drop_last=False):
    """Standard CIFAR-100 train/test loaders used by every vision experiment."""
    transform_train, transform_test = _cifar100_transforms(augment)

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=download, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=max(batch_size, 256), shuffle=False,
        num_workers=max(num_workers // 2, 1), pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def get_cifar100_neighbour_loaders(batch_size=128, num_workers=4, data_dir='./data',
                                   subset_size=None, replace_index=0, replace_with=None,
                                   seed=42, augment=True, download=True):
    """
    Build a pair of training sets S and S' that differ in exactly one example.

    This is the sampling model behind uniform stability: S' is obtained from S by
    *replacing* (not deleting) one example, so both sets have the same size and
    the same shuffling order under a fixed seed. That makes the parameter
    divergence ||theta_t - theta'_t|| attributable to the single swapped example.

    Returns:
        (loader_S, loader_S_prime, test_loader)
    """
    transform_train, transform_test = _cifar100_transforms(augment)

    full_train = datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=download, transform=transform_test
    )

    n_full = len(full_train)
    rng = np.random.RandomState(seed)

    if subset_size is None or subset_size >= n_full:
        indices = np.arange(n_full)
    else:
        indices = rng.choice(n_full, size=subset_size, replace=False)

    indices_s = np.array(indices, copy=True)
    indices_sp = np.array(indices, copy=True)

    if replace_with is None:
        # Pick a replacement that is not already in the subset.
        pool = np.setdiff1d(np.arange(n_full), indices_s, assume_unique=False)
        if len(pool) == 0:
            raise ValueError("No spare example available to build the neighbouring set")
        replace_with = int(rng.choice(pool))
    indices_sp[replace_index] = replace_with

    def _loader(idx):
        # Each loader gets its own generator seeded identically, so both runs
        # traverse the same positions of the index array in the same order.
        gen = torch.Generator()
        gen.manual_seed(seed)
        return DataLoader(
            Subset(full_train, idx.tolist()), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, generator=gen,
            persistent_workers=num_workers > 0,
        )

    test_loader = DataLoader(
        test_dataset, batch_size=max(batch_size, 256), shuffle=False,
        num_workers=max(num_workers // 2, 1), pin_memory=True,
    )
    return _loader(indices_s), _loader(indices_sp), test_loader
