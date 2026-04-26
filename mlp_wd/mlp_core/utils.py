"""Training / evaluation primitives for the MLP weight-decay experiments.

Compared to the original wd_core/utils.py, this version:
  - records best AND final values for both test_acc and test_loss
  - returns the per-epoch history as part of the result dict (small enough to keep)
  - drops mixed precision (fp16 with a 3-layer MLP gives no speedup and risks NaNs
    at the unstable end of the eta x lambda grid that we explicitly want to observe)
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _train_one_epoch(model, loader, optimizer, scheduler, device,
                     grad_clip: float | None = None) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    diverged = False

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            # Bound the per-step update so unstable LRs degrade gracefully
            # instead of NaN-ing on the very first batch. Stable runs keep
            # ||g|| << grad_clip so this is a no-op for them.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += bs

    if scheduler is not None:
        scheduler.step()

    if diverged or total == 0:
        return float("nan"), float("nan")
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            return float("nan"), float("nan")
        bs = inputs.size(0)
        total_loss += loss.item() * bs
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += bs

    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, 100.0 * correct / total


def train_model_with_history(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    epochs: int = 30,
    log_every: int = 0,
    grad_clip: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train and return (summary_dict, per_epoch_history).

    summary_dict keys:
        final_train_loss, final_train_acc,
        final_test_loss,  final_test_acc,
        best_test_loss,   best_test_acc,
        diverged (bool), epochs_run
    """
    history: list[dict[str, Any]] = []
    best_test_acc = -math.inf
    best_test_loss = math.inf
    diverged = False
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip=grad_clip,
        )
        if not math.isfinite(train_loss):
            diverged = True
            history.append({
                "epoch": epoch,
                "train_loss": float("nan"), "train_acc": float("nan"),
                "test_loss": float("nan"), "test_acc": float("nan"),
            })
            epochs_run = epoch
            break

        test_loss, test_acc = _evaluate(model, test_loader, device)
        if not math.isfinite(test_loss):
            diverged = True
            history.append({
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": float("nan"), "test_acc": float("nan"),
            })
            epochs_run = epoch
            break

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss), "train_acc": float(train_acc),
            "test_loss": float(test_loss), "test_acc": float(test_acc),
        })
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        epochs_run = epoch

        if log_every and (epoch % log_every == 0 or epoch == epochs):
            print(
                f"  epoch {epoch:3d}/{epochs} | "
                f"train {train_loss:.4f}/{train_acc:5.2f}% | "
                f"test {test_loss:.4f}/{test_acc:5.2f}% | "
                f"best_acc {best_test_acc:5.2f}% best_loss {best_test_loss:.4f}",
                flush=True,
            )

    if diverged or not history:
        last = history[-1] if history else {
            "train_loss": float("nan"), "train_acc": float("nan"),
            "test_loss": float("nan"), "test_acc": float("nan"),
        }
        summary = {
            "final_train_loss": last["train_loss"],
            "final_train_acc": last["train_acc"],
            "final_test_loss": last["test_loss"],
            "final_test_acc": last["test_acc"],
            "best_test_loss": float("nan") if best_test_loss == math.inf else best_test_loss,
            "best_test_acc": float("nan") if best_test_acc == -math.inf else best_test_acc,
            "diverged": True,
            "epochs_run": epochs_run,
        }
        return summary, history

    last = history[-1]
    summary = {
        "final_train_loss": last["train_loss"],
        "final_train_acc": last["train_acc"],
        "final_test_loss": last["test_loss"],
        "final_test_acc": last["test_acc"],
        "best_test_loss": float(best_test_loss),
        "best_test_acc": float(best_test_acc),
        "diverged": False,
        "epochs_run": epochs_run,
    }
    return summary, history


def save_history_json(history: list[dict[str, Any]], config: dict[str, Any], path: str | Path) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"config": config, "history": history}, f, indent=2)
