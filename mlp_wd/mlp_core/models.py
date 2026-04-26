"""Configurable plain MLP for CIFAR-10 / MNIST.

No BatchNorm, no Dropout, no skip connections - we want a clean SGD/SGDM signal
so the eta x lambda scaling argument is not muddied by other regularizers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain ReLU MLP.

    Args:
        in_features: size of flattened input (CIFAR-10: 3072, MNIST: 784).
        hidden_dim: width of every hidden layer.
        num_layers: total number of Linear layers (>= 2). num_layers=3 means
            input -> hidden -> hidden -> output (2 hidden layers, ReLU between).
        num_classes: output classes.
    """

    def __init__(
        self,
        in_features: int = 3072,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        dims = [in_features] + [hidden_dim] * (num_layers - 1) + [num_classes]
        layers: list[nn.Module] = [nn.Flatten()]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp_for_dataset(
    dataset: str,
    *,
    hidden_dim: int = 512,
    num_layers: int = 3,
) -> MLP:
    """Convenience factory keyed by dataset name."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        return MLP(in_features=3 * 32 * 32, hidden_dim=hidden_dim,
                   num_layers=num_layers, num_classes=10)
    if dataset == "mnist":
        return MLP(in_features=28 * 28, hidden_dim=hidden_dim,
                   num_layers=num_layers, num_classes=10)
    raise ValueError(f"Unknown dataset: {dataset!r}")
