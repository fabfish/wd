#!/usr/bin/env python3
"""
MLP + MNIST: Reproducing Three Core WD Discoveries
====================================================

Reproduces the three key experiments from fabfish/wd on a simple
2-layer MLP + MNIST setup, validating:

1. LR Ordering: η_SGD > η_SGD+WD > η_SGDM+WD
2. η-λ Inverse Law: λ_opt ∝ η^(-1)  (or accuracy depends on η×λ)
3. Batch Size Scaling: λ_opt ∝ B

Author: Claw
Date: 2026-04-24
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Logging ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/mnt/afs/visitor13/mnist_wd_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")


# ── Model ────────────────────────────────────────────────

class MLP(nn.Module):
    """2-layer MLP: 784 → 256 → 128 → 10."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data ─────────────────────────────────────────────────

_cached_loaders: Dict[int, Tuple[DataLoader, DataLoader]] = {}


def get_loaders(batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Get MNIST loaders (cached by batch_size)."""
    if batch_size in _cached_loaders:
        return _cached_loaders[batch_size]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    _cached_loaders[batch_size] = (train_loader, test_loader)
    return train_loader, test_loader


# ── Training / Evaluation ────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return test accuracy."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / total


def train_eval(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    lr: float,
    wd: float,
    momentum: float,
    epochs: int,
    device: torch.device,
) -> float:
    """Train and return final test accuracy."""
    torch.manual_seed(42)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    return evaluate(model, test_loader, device)


# ── Plotting ─────────────────────────────────────────────

def save_plot(fig: plt.Figure, name: str) -> None:
    """Save plot to results directory."""
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info(f"  Plot saved: {path}")
    plt.close(fig)


# ── Experiment 1: LR Ordering ────────────────────────────

EXP1_CONFIGS = [
    ("SGD", 0.0, 0.0),
    ("SGD+WD", 0.0, 5e-4),
    ("SGDM", 0.9, 0.0),
    ("SGDM+WD", 0.9, 5e-4),
]

EXP1_LRS = np.logspace(-3, 0, 15)  # [0.001, 0.0014, ..., 1.0]


def run_exp1(model: nn.Module, epochs: int = 10, bs: int = 128) -> Dict:
    """
    Experiment 1: Search optimal LR for each method.
    Verify η_SGD > η_SGD+WD > η_SGDM+WD.
    
    Plot: Two separate figures - one for no WD, one for with WD.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: LR Ordering (Stability Bound)")
    logger.info("=" * 60)

    train_loader, test_loader = get_loaders(bs)
    results = {}

    for name, momentum, wd in EXP1_CONFIGS:
        logger.info(f"\n  Method: {name} (momentum={momentum}, wd={wd})")
        accs = []

        for lr in EXP1_LRS:
            acc = train_eval(model, train_loader, test_loader, lr, wd, momentum, epochs, DEVICE)
            accs.append(float(acc))
            logger.info(f"    lr={lr:.4f} → acc={acc:.4f}")

        results[name] = {
            "lrs": [float(x) for x in EXP1_LRS],
            "accs": accs,
            "best_lr": float(EXP1_LRS[np.argmax(accs)]),
            "best_acc": float(max(accs)),
        }

    # Plot 1a: No WD (SGD vs SGDM)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for name in ["SGD", "SGDM"]:
        data = results[name]
        ax1.semilogx(data["lrs"], data["accs"], 'o-', label=name, linewidth=2)
        best_idx = np.argmax(data["accs"])
        ax1.plot(data["lrs"][best_idx], data["accs"][best_idx], '*', 
                markersize=15, color=ax1.lines[-1].get_color())
        ax1.axvline(x=data["best_lr"], color=ax1.lines[-1].get_color(), 
                   linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Learning Rate η', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Exp 1a: Without Weight Decay', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.8, 1.0])
    save_plot(fig1, "exp1a_no_wd.png")

    # Plot 1b: With WD (SGD+WD vs SGDM+WD)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for name in ["SGD+WD", "SGDM+WD"]:
        data = results[name]
        ax2.semilogx(data["lrs"], data["accs"], 'o-', label=name, linewidth=2)
        best_idx = np.argmax(data["accs"])
        ax2.plot(data["lrs"][best_idx], data["accs"][best_idx], '*', 
                markersize=15, color=ax2.lines[-1].get_color())
        ax2.axvline(x=data["best_lr"], color=ax2.lines[-1].get_color(), 
                   linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Learning Rate η', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Exp 1b: With Weight Decay (λ=5e-4)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.8, 1.0])
    save_plot(fig2, "exp1b_with_wd.png")

    # Plot 1c: All four together (overview)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    colors = {'SGD': '#1f77b4', 'SGD+WD': '#2ca02c', 
              'SGDM': '#ff7f0e', 'SGDM+WD': '#d62728'}
    for name, momentum, wd in EXP1_CONFIGS:
        data = results[name]
        ax3.semilogx(data["lrs"], data["accs"], 'o-', label=name, 
                    color=colors[name], linewidth=2)
    
    ax3.set_xlabel('Learning Rate η', fontsize=12)
    ax3.set_ylabel('Test Accuracy', fontsize=12)
    ax3.set_title('Exp 1: LR Ordering - All Methods', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.8, 1.0])
    save_plot(fig3, "exp1_all_methods.png")

    # Verify ordering
    best_lrs = {k: v["best_lr"] for k, v in results.items()}
    logger.info(f"\n  Best LR ordering: {best_lrs}")

    if best_lrs["SGD"] > best_lrs["SGD+WD"] > best_lrs["SGDM+WD"]:
        logger.info("  ✅ Theory verified: η_SGD > η_SGD+WD > η_SGDM+WD")
    else:
        logger.info("  ⚠️ Ordering not strictly matched (expected on larger models)")

    path = RESULTS_DIR / "exp1_lr_ordering.json"
    path.write_text(json.dumps(results, indent=2))
    logger.info(f"  Saved: {path}")
    return results


# ── Experiment 2: η-λ Interaction ────────────────────────

EXP2_LRS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
EXP2_WDS = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]


def run_exp2(model: nn.Module, epochs: int = 10, bs: int = 128) -> Dict:
    """
    Experiment 2: Grid search η × λ.
    Verify: (1) diagonal high-acc region, (2) λ ∝ η^(-1), (3) acc depends on ηλ.
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: η-λ Interaction (Inverse Law)")
    logger.info("=" * 60)

    train_loader, test_loader = get_loaders(bs)
    momentum = 0.9

    # Grid search
    grid = np.zeros((len(EXP2_WDS), len(EXP2_LRS)))
    for i, wd in enumerate(EXP2_WDS):
        for j, lr in enumerate(EXP2_LRS):
            acc = train_eval(model, train_loader, test_loader, lr, wd, momentum, epochs, DEVICE)
            grid[i, j] = acc
            logger.info(f"  lr={lr:.2f}, wd={wd:.0e} → acc={acc:.4f}")

    # Find optimal points for each LR
    optimal_pairs = []
    for j, lr in enumerate(EXP2_LRS):
        best_i = int(np.argmax(grid[:, j]))
        optimal_pairs.append((lr, EXP2_WDS[best_i], grid[best_i, j]))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(EXP2_LRS)))
    ax.set_xticklabels([f'{x:.2f}' for x in EXP2_LRS])
    ax.set_yticks(range(len(EXP2_WDS)))
    ax.set_yticklabels([f'{x:.0e}' for x in EXP2_WDS])
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Weight Decay λ', fontsize=12)
    ax.set_title('Exp 2: Test Accuracy Heatmap (η × λ)', fontsize=14, fontweight='bold')
    
    # Mark optimal points
    for j, (lr, wd, acc) in enumerate(optimal_pairs):
        i = EXP2_WDS.index(wd)
        ax.plot(j, i, 'r*', markersize=15)
    
    plt.colorbar(im, ax=ax, label='Test Accuracy')
    save_plot(fig, "exp2_heatmap.png")

    # Fit λ = a * η^(-b)
    lrs_opt = np.array([p[0] for p in optimal_pairs])
    wds_opt = np.array([p[1] for p in optimal_pairs])

    def power_law(eta, a, b):
        return a * np.power(eta, -b)

    try:
        (a, b), _ = curve_fit(power_law, lrs_opt, wds_opt, p0=[1e-5, 1.0])
        logger.info(f"\n  Fitted: λ = {a:.2e} × η^(-{b:.3f})")
        if 0.8 < b < 1.2:
            logger.info("  ✅ b ≈ 1, inverse law verified!")
        else:
            logger.info(f"  ⚠️ b={b:.3f}, deviation from theory (b=1)")
    except Exception as e:
        logger.info(f"  Fit failed: {e}")
        a, b = None, None

    # Plot ηλ product
    eta_lambda_products = [p[0] * p[1] for p in optimal_pairs]
    eta_lambda_accs = [p[2] for p in optimal_pairs]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(eta_lambda_products, eta_lambda_accs, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('η × λ (Effective Regularization)', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Exp 2: Accuracy vs ηλ Product', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    save_plot(fig2, "exp2_eta_lambda_product.png")

    cv = np.std(eta_lambda_products) / np.mean(eta_lambda_products)
    logger.info(f"\n  η×λ products: {eta_lambda_products}")
    logger.info(f"  Coefficient of variation: {cv:.3f}")
    if cv < 0.5:
        logger.info("  ✅ η×λ relatively constant, product rule verified!")
    else:
        logger.info("  ⚠️ η×λ varies significantly")

    results = {
        "lrs": [float(x) for x in EXP2_LRS],
        "wds": [float(x) for x in EXP2_WDS],
        "grid": grid.tolist(),
        "optimal_pairs": [(float(lr), float(wd), float(acc)) for lr, wd, acc in optimal_pairs],
        "fit": {"a": float(a) if a else None, "b": float(b) if b else None},
        "eta_lambda_cv": float(cv),
    }

    path = RESULTS_DIR / "exp2_eta_lambda_interaction.json"
    path.write_text(json.dumps(results, indent=2))
    logger.info(f"  Saved: {path}")
    return results


# ── Experiment 3: Batch Size Scaling ────────────────────

EXP3_BATCH_SIZES = [32, 64, 128, 256, 512]
EXP3_WDS = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def run_exp3(model: nn.Module, epochs: int = 10) -> Dict:
    """
    Experiment 3: Vary batch size, verify λ_opt ∝ B.
    LR scaled linearly: η = 0.1 × (B / 128).
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3: Batch Size Scaling Law")
    logger.info("=" * 60)

    momentum = 0.9
    results = {}

    for bs in EXP3_BATCH_SIZES:
        lr = 0.1 * (bs / 128)
        logger.info(f"\n  Batch Size={bs}, lr={lr:.4f}")

        train_loader, test_loader = get_loaders(bs)
        accs = []

        for wd in EXP3_WDS:
            acc = train_eval(model, train_loader, test_loader, lr, wd, momentum, epochs, DEVICE)
            accs.append(float(acc))
            logger.info(f"    wd={wd:.0e} → acc={acc:.4f}")

        best_idx = int(np.argmax(accs))
        best_wd = EXP3_WDS[best_idx]
        best_acc = accs[best_idx]
        logger.info(f"  Best: wd={best_wd:.0e}, acc={best_acc:.4f}")

        results[f"BS_{bs}"] = {
            "batch_size": bs,
            "lr": float(lr),
            "wds": [float(x) for x in EXP3_WDS],
            "accs": accs,
            "best_wd": float(best_wd),
            "best_acc": float(best_acc),
        }

    # Plot: WD curves for different batch sizes
    fig, ax = plt.subplots(figsize=(12, 7))
    for bs in EXP3_BATCH_SIZES:
        data = results[f"BS_{bs}"]
        ax.semilogx(data["wds"], data["accs"], 'o-', label=f'BS={bs}', linewidth=2)
    
    ax.set_xlabel('Weight Decay λ', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Exp 3: WD Curves for Different Batch Sizes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_plot(fig, "exp3_wd_curves.png")

    # Plot: Optimal WD vs Batch Size
    bs_list = np.array([results[f"BS_{bs}"]["batch_size"] for bs in EXP3_BATCH_SIZES])
    wd_opt_list = np.array([results[f"BS_{bs}"]["best_wd"] for bs in EXP3_BATCH_SIZES])
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(bs_list, wd_opt_list, 'o-', linewidth=2, markersize=10)
    
    # Linear fit
    try:
        k = np.mean(wd_opt_list / bs_list)
        predicted = k * bs_list
        r2 = 1 - np.sum((wd_opt_list - predicted) ** 2) / np.sum((wd_opt_list - np.mean(wd_opt_list)) ** 2)
        ax2.plot(bs_list, predicted, '--', label=f'Linear fit: λ={k:.2e}×B, R²={r2:.3f}', alpha=0.7)
        logger.info(f"\n  Linear fit: λ = {k:.2e} × B")
        logger.info(f"  R² = {r2:.3f}")
        if r2 > 0.7:
            logger.info("  ✅ Linear scaling verified!")
        else:
            logger.info("  ⚠️ Deviation from linear scaling")
    except Exception as e:
        logger.info(f"  Fit failed: {e}")
        k, r2 = None, None
    
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Optimal Weight Decay λ_opt', fontsize=12)
    ax2.set_title('Exp 3: λ_opt vs Batch Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    save_plot(fig2, "exp3_scaling_law.png")

    summary = {
        "per_bs": results,
        "scaling_fit": {"k": float(k) if k else None, "r2": float(r2) if r2 else None},
    }

    path = RESULTS_DIR / "exp3_batch_size_scaling.json"
    path.write_text(json.dumps(summary, indent=2))
    logger.info(f"  Saved: {path}")
    return summary


# ── Main ────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("MLP+MNIST: Three Core WD Discoveries")
    logger.info("=" * 60)

    model = MLP().to(DEVICE)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())}")

    run_exp1(model, epochs=10, bs=128)
    run_exp2(model, epochs=10, bs=128)
    run_exp3(model, epochs=10)

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
