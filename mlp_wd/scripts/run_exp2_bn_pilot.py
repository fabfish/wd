"""Exp2 pilot: BN-MLP, mirrors the original ResNet eta x lambda grid.

Goal: verify that adding BatchNorm to the MLP brings us into the regime
where the eta x lambda scaling law holds, so per-lambda minima cluster at
the same eta*lambda value (the "aligned spoons" effect from the reference
ResNet figure).

Pilot grid: 5 LRs x 3 WDs = 15 cells. If alignment looks clean, the next
step is to run the full 5x7 grid for 100 epochs (run_exp2_bn.py).

Settings mirror the original ResNet Exp2 setup as closely as possible:
    optimizer: SGDM with momentum=0.9 (BN absorbs scale, so high momentum
               is fine; no need for the gentle mom=0.5 + clip trick)
    schedule:  CosineAnnealingLR over the run
    grad_clip: 1.0 (cheap insurance; never bites the stable runs)
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


# Pilot: identical lr range to the original ResNet Exp2, sparse wd.
PILOT_LRS = [0.01, 0.05, 0.1, 0.2, 0.3]
PILOT_WDS = [1e-4, 1e-3, 1e-2]


def build_grid(epochs, seed, batch_size, momentum):
    rows = []
    for lr, wd in product(PILOT_LRS, PILOT_WDS):
        rows.append({
            "method": "SGDM+WD" if wd > 0 else "SGDM",
            "batch_size": batch_size, "lr": lr, "wd": wd,
            "momentum": momentum, "epochs": epochs, "seed": seed,
            "run_tag": f"exp2bnp_lr{lr}_wd{wd}",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--use_bn", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="Light clip; pass <=0 to disable")
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=8)
    ap.add_argument("--loader_workers", type=int, default=0)
    ap.add_argument("--output", type=str,
                    default="mlp_wd/outputs/results/exp2_bn_pilot.csv")
    ap.add_argument("--history_dir", type=str,
                    default="mlp_wd/outputs/history/exp2_bn_pilot")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    rows = build_grid(args.epochs, args.seed, args.batch_size, args.momentum)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None
    use_bn = bool(args.use_bn)

    print(f"[exp2-bn-pilot] {len(rows)} runs | hidden={args.hidden_dim} layers={args.num_layers} "
          f"bn={int(use_bn)} mom={args.momentum} grad_clip={grad_clip} epochs={args.epochs}")
    run_grid(
        rows,
        output_file=args.output,
        history_dir=args.history_dir,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_bn=use_bn,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        log_every=args.log_every,
        loader_workers=args.loader_workers,
        grad_clip=grad_clip,
    )


if __name__ == "__main__":
    main()
