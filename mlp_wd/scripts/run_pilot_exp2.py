"""Pilot run for Exp2 (eta-lambda spoons).

A small 4x4 grid we use to confirm the spoon shape (descend -> minimum -> rise)
on CIFAR-10 / 3-layer MLP before committing to the full 6x6 grid.

Run after a single sanity check:
    python -m mlp_wd.scripts.train_one --lr 0.05 --wd 1e-3 --momentum 0.9

Then:
    python -m mlp_wd.scripts.run_pilot_exp2 --gpus 1,2,3 --workers_per_gpu 8
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlp_wd.mlp_core.gpu_scheduler import parse_gpu_ids
from mlp_wd.mlp_core.grid import run_grid


PILOT_LRS = [1e-2, 3e-2, 1e-1, 3e-1]
PILOT_WDS = [3e-4, 1e-3, 3e-3, 1e-2]


def build_pilot_grid(epochs: int, seed: int):
    rows = []
    for lr, wd in product(PILOT_LRS, PILOT_WDS):
        rows.append({
            "method": "SGDM+WD",
            "batch_size": 128,
            "lr": lr,
            "wd": wd,
            "momentum": 0.9,
            "epochs": epochs,
            "seed": seed,
            "run_tag": f"pilot_lr{lr}_wd{wd}",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=4)
    ap.add_argument("--loader_workers", type=int, default=0,
                    help="DataLoader num_workers per task (0 = in-process; recommended when workers_per_gpu>=2)")
    ap.add_argument("--output", type=str, default="mlp_wd/outputs/results/exp2_pilot.csv")
    ap.add_argument("--history_dir", type=str, default="mlp_wd/outputs/history/exp2_pilot")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    rows = build_pilot_grid(args.epochs, args.seed)

    print(f"[pilot] {len(rows)} runs over LRs={PILOT_LRS}, WDs={PILOT_WDS}, epochs={args.epochs}")
    run_grid(
        rows,
        output_file=args.output,
        history_dir=args.history_dir,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        log_every=args.log_every,
        loader_workers=args.loader_workers,
    )


if __name__ == "__main__":
    main()
