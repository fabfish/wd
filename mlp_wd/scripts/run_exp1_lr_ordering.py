"""Exp1 -- LR ordering, two paired ablations.

Group A (mom=0):  SGD (wd=0) vs SGD+WD (wd=lambda_a)  -- 8 + 8 runs
Group B (mom=0.9): SGDM (wd=0) vs SGDM+WD (wd=lambda_b) -- 8 + 8 runs

Each pair shares the same eta grid so the WD-induced LR shift is read off
directly. Default: 32 runs total.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlp_wd.mlp_core.gpu_scheduler import parse_gpu_ids
from mlp_wd.mlp_core.grid import run_grid


GROUP_A_LRS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]   # mom=0
GROUP_B_LRS = [0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]  # mom=0.9
DEFAULT_WD_A = 1e-3
DEFAULT_WD_B = 1e-3


def build_grid(epochs, seed, batch_size, group_a_lrs, group_b_lrs, wd_a, wd_b):
    rows = []
    for lr in group_a_lrs:
        rows.append({"method": "SGD", "batch_size": batch_size,
                     "lr": lr, "wd": 0.0, "momentum": 0.0,
                     "epochs": epochs, "seed": seed,
                     "run_tag": f"exp1A_SGD_lr{lr}"})
        rows.append({"method": "SGD+WD", "batch_size": batch_size,
                     "lr": lr, "wd": wd_a, "momentum": 0.0,
                     "epochs": epochs, "seed": seed,
                     "run_tag": f"exp1A_SGDWD_lr{lr}"})
    for lr in group_b_lrs:
        rows.append({"method": "SGDM", "batch_size": batch_size,
                     "lr": lr, "wd": 0.0, "momentum": 0.9,
                     "epochs": epochs, "seed": seed,
                     "run_tag": f"exp1B_SGDM_lr{lr}"})
        rows.append({"method": "SGDM+WD", "batch_size": batch_size,
                     "lr": lr, "wd": wd_b, "momentum": 0.9,
                     "epochs": epochs, "seed": seed,
                     "run_tag": f"exp1B_SGDMWD_lr{lr}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wd_a", type=float, default=DEFAULT_WD_A,
                    help="lambda used for Group A (SGD+WD), default 1e-3")
    ap.add_argument("--wd_b", type=float, default=DEFAULT_WD_B,
                    help="lambda used for Group B (SGDM+WD), default 1e-3")
    ap.add_argument("--lrs_a", type=str, default=None)
    ap.add_argument("--lrs_b", type=str, default=None)
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=8)
    ap.add_argument("--loader_workers", type=int, default=0,
                    help="DataLoader num_workers per task (0 = in-process; recommended when workers_per_gpu>=2)")
    ap.add_argument("--output", type=str, default="mlp_wd/outputs/results/exp1.csv")
    ap.add_argument("--history_dir", type=str, default="mlp_wd/outputs/history/exp1")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    lrs_a = [float(x) for x in args.lrs_a.split(",")] if args.lrs_a else GROUP_A_LRS
    lrs_b = [float(x) for x in args.lrs_b.split(",")] if args.lrs_b else GROUP_B_LRS
    rows = build_grid(args.epochs, args.seed, args.batch_size, lrs_a, lrs_b,
                      args.wd_a, args.wd_b)

    print(f"[exp1] {len(rows)} runs | groupA(mom=0) lrs={lrs_a}, wd_a={args.wd_a} | "
          f"groupB(mom=0.9) lrs={lrs_b}, wd_b={args.wd_b}")
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
