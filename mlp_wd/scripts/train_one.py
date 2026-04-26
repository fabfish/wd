"""Single-run entry point. Useful for sanity checks before kicking off a grid.

Example:
    python -m mlp_wd.scripts.train_one \
        --dataset cifar10 --num_layers 3 --hidden_dim 512 \
        --batch_size 128 --lr 0.05 --wd 1e-3 --momentum 0.9 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python mlp_wd/scripts/train_one.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlp_wd.mlp_core.runner import run_single_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single MLP run.")
    parser.add_argument("--method", default="SGDM+WD")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--history_dir", type=str, default="mlp_wd/outputs/history/single")
    args = parser.parse_args()

    record = run_single_experiment(
        method=args.method,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        momentum=args.momentum,
        epochs=args.epochs,
        seed=args.seed,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        history_dir=args.history_dir,
        run_tag=f"single_{args.method}_lr{args.lr}_wd{args.wd}_m{args.momentum}",
        log_every=args.log_every,
        num_workers=args.num_workers,
    )
    print("\nSummary:")
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
