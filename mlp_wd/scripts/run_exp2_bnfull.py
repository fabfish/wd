"""Exp2 full grid for BN-MLP with BN on the OUTPUT layer too.

Hypothesis test: in run_exp2_bn the per-lambda minima still showed ~100x spread
on the eta*lambda axis. The likely cause is the final classifier Linear, which
is not protected by BN, so global rescales of W_out change loss directly and
the eta*lambda scaling holds only approximately. Adding BatchNorm1d AFTER the
final Linear (before CE) makes the loss scale-invariant in W_out as well, so
the eta*lambda scaling should hold for ALL Linear layers and the spoon minima
should align tightly.
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


LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
WDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def build_grid(epochs, seed, batch_size, momentum):
    rows = []
    for lr, wd in product(LRS, WDS):
        rows.append({
            "method": "SGDM+WD" if wd > 0 else "SGDM",
            "batch_size": batch_size, "lr": lr, "wd": wd,
            "momentum": momentum, "epochs": epochs, "seed": seed,
            "run_tag": f"exp2bnfull_lr{lr}_wd{wd}",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=1)
    ap.add_argument("--loader_workers", type=int, default=2)
    ap.add_argument("--output", type=str,
                    default="mlp_wd/outputs/results/exp2_bnfull.csv")
    ap.add_argument("--history_dir", type=str,
                    default="mlp_wd/outputs/history/exp2_bnfull")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    rows = build_grid(args.epochs, args.seed, args.batch_size, args.momentum)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    print(f"[exp2-bnfull] {len(rows)} runs | hidden={args.hidden_dim} "
          f"layers={args.num_layers} bn=1 normout=1 mom={args.momentum} "
          f"grad_clip={grad_clip} epochs={args.epochs}")
    run_grid(
        rows,
        output_file=args.output,
        history_dir=args.history_dir,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_bn=True,
        norm_output=True,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        log_every=args.log_every,
        loader_workers=args.loader_workers,
        grad_clip=grad_clip,
    )


if __name__ == "__main__":
    main()
