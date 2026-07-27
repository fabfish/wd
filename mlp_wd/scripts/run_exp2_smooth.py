"""Exp2 with gradient clipping -- smooth spoons.

Same eta x lambda grid as run_exp2_eta_lambda + run_exp2_extend, but with
grad_clip=10.0. Stable runs (||g|| << 10) are unaffected; unstable runs that
would otherwise NaN on the first batch are bounded so they survive 30 epochs
and produce a finite (just worse) test loss. The result is a gentle rising
tail on every spoon instead of a cliff to log K.

Run after the raw runs:
    python -m mlp_wd.scripts.run_exp2_smooth --gpus 1,2,3 --workers_per_gpu 8 \
        --loader_workers 0 --epochs 30 --grad_clip 10.0
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


# Original 6x6 grid.
BASE_LRS = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
BASE_WDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

# Extra etas for small lambdas (exact mirror of run_exp2_extend).
EXTRA_FOR_WD = {
    1e-4: [0.5, 0.7, 1.5, 2.0, 3.0, 10.0],
    3e-4: [0.5, 0.7, 1.5, 2.0, 3.0],
    1e-3: [0.2, 0.5, 0.7, 1.5, 2.0],
    3e-3: [0.2, 0.5, 0.7],
    1e-2: [0.2],
}


def build_grid(epochs, seed, batch_size, momentum):
    rows = []
    for lr, wd in product(BASE_LRS, BASE_WDS):
        rows.append({
            "method": "SGDM+WD" if wd > 0 else "SGDM",
            "batch_size": batch_size, "lr": lr, "wd": wd,
            "momentum": momentum, "epochs": epochs, "seed": seed,
            "run_tag": f"exp2s_lr{lr}_wd{wd}",
        })
    for wd, etas in EXTRA_FOR_WD.items():
        for lr in etas:
            rows.append({
                "method": "SGDM+WD" if wd > 0 else "SGDM",
                "batch_size": batch_size, "lr": lr, "wd": wd,
                "momentum": momentum, "epochs": epochs, "seed": seed,
                "run_tag": f"exp2s_lr{lr}_wd{wd}",
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=10.0,
                    help="grad-norm clip; pass <=0 to disable")
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=8)
    ap.add_argument("--loader_workers", type=int, default=0)
    ap.add_argument("--output", type=str,
                    default="mlp_wd/outputs/results/exp2_smooth.csv")
    ap.add_argument("--history_dir", type=str,
                    default="mlp_wd/outputs/history/exp2_smooth")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    rows = build_grid(args.epochs, args.seed, args.batch_size, args.momentum)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    print(f"[exp2-smooth] {len(rows)} runs | grad_clip={grad_clip}")
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
        grad_clip=grad_clip,
    )


if __name__ == "__main__":
    main()
