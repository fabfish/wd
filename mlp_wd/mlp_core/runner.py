"""Shared single-experiment worker used by all run_* scripts.

A worker = one (method, batch_size, lr, wd, momentum, epochs, seed, dataset,
hidden_dim, num_layers, history_dir, run_tag) call. Returns a dict with the
exact CSV schema the analysis scripts expect.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .datasets import get_loaders
from .models import build_mlp_for_dataset
from .utils import save_history_json, set_seed, train_model_with_history


CSV_FIELDS = [
    "method", "dataset", "num_layers", "hidden_dim", "use_bn", "norm_output",
    "batch_size", "lr", "wd", "momentum", "epochs", "seed",
    "final_train_loss", "final_train_acc",
    "final_test_loss", "final_test_acc",
    "best_test_loss", "best_test_acc",
    "diverged", "epochs_run",
]


def run_single_experiment(
    method: str,
    batch_size: int,
    lr: float,
    wd: float,
    momentum: float,
    epochs: int,
    seed: int,
    dataset: str = "cifar10",
    hidden_dim: int = 512,
    num_layers: int = 3,
    use_bn: bool = False,
    norm_output: bool = False,
    history_dir: str | None = None,
    run_tag: str = "",
    log_every: int = 0,
    num_workers: int = 2,
    grad_clip: float | None = None,
) -> dict[str, Any]:
    """Train one (method, lr, wd, ...) configuration and return a CSV-ready dict."""
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(dataset, batch_size=batch_size, num_workers=num_workers)
    model = build_mlp_for_dataset(
        dataset, hidden_dim=hidden_dim, num_layers=num_layers,
        use_bn=use_bn, norm_output=norm_output,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(
        f"[{run_tag or method}] BS={batch_size} LR={lr} WD={wd} mom={momentum} "
        f"layers={num_layers} hidden={hidden_dim} bn={int(use_bn)} "
        f"normout={int(norm_output)} epochs={epochs} ds={dataset} grad_clip={grad_clip}",
        flush=True,
    )

    summary, history = train_model_with_history(
        model, train_loader, test_loader, optimizer, scheduler,
        device=device, epochs=epochs, log_every=log_every,
        grad_clip=grad_clip,
    )

    if history_dir:
        cfg = {
            "method": method, "dataset": dataset,
            "num_layers": num_layers, "hidden_dim": hidden_dim,
            "use_bn": int(use_bn), "norm_output": int(norm_output),
            "batch_size": batch_size, "lr": lr, "wd": wd,
            "momentum": momentum, "epochs": epochs, "seed": seed,
        }
        tag = run_tag or f"{method}_bs{batch_size}_lr{lr}_wd{wd}_m{momentum}"
        out = Path(history_dir) / f"{tag}.json"
        save_history_json(history, cfg, out)

    record = {
        "method": method,
        "dataset": dataset,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "use_bn": int(use_bn),
        "norm_output": int(norm_output),
        "batch_size": batch_size,
        "lr": lr,
        "wd": wd,
        "momentum": momentum,
        "epochs": epochs,
        "seed": seed,
        "final_train_loss": summary["final_train_loss"],
        "final_train_acc": summary["final_train_acc"],
        "final_test_loss": summary["final_test_loss"],
        "final_test_acc": summary["final_test_acc"],
        "best_test_loss": summary["best_test_loss"],
        "best_test_acc": summary["best_test_acc"],
        "diverged": int(bool(summary["diverged"])),
        "epochs_run": summary["epochs_run"],
    }
    return record


def get_run_key(record: dict[str, Any]) -> str:
    """Stable identifier for resume / dedup."""
    bn_flag = int(record.get("use_bn", 0) or 0)
    no_flag = int(record.get("norm_output", 0) or 0)
    return (
        f"m{record['method']}|ds{record['dataset']}|L{record['num_layers']}|H{record['hidden_dim']}|"
        f"BN{bn_flag}|NO{no_flag}|B{record['batch_size']}|lr{record['lr']}|wd{record['wd']}|"
        f"mom{record['momentum']}|E{record['epochs']}|s{record['seed']}"
    )


def get_task_key(method: str, dataset: str, num_layers: int, hidden_dim: int,
                  batch_size: int, lr: float, wd: float, momentum: float,
                  epochs: int, seed: int, use_bn: bool = False,
                  norm_output: bool = False) -> str:
    return get_run_key({
        "method": method, "dataset": dataset, "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "use_bn": int(use_bn), "norm_output": int(norm_output),
        "batch_size": batch_size, "lr": lr, "wd": wd,
        "momentum": momentum, "epochs": epochs, "seed": seed,
    })
