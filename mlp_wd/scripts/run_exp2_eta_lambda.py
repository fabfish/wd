"""Exp2 -- full eta x lambda grid (the headline 'spoons' figure).

Default grid: 6 x 6 = 36 runs, SGDM (mom=0.9), batch_size=128.
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


# Default grid covers ~3 decades on each axis so every lambda curve
# clearly descends, hits a minimum, then rises (instability tail).
DEFAULT_LRS = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
DEFAULT_WDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def build_grid(lrs, wds, epochs, seed, batch_size, momentum):
    rows = []
    for lr, wd in product(lrs, wds):
        rows.append({
            "method": "SGDM+WD" if wd > 0 else "SGDM",
            "batch_size": batch_size,
            "lr": lr,
            "wd": wd,
            "momentum": momentum,
            "epochs": epochs,
            "seed": seed,
            "run_tag": f"exp2_lr{lr}_wd{wd}",
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
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=8)
    ap.add_argument("--loader_workers", type=int, default=0,
                    help="DataLoader num_workers per task (0 = in-process; recommended when workers_per_gpu>=2)")
    ap.add_argument("--output", type=str, default="mlp_wd/outputs/results/exp2.csv")
    ap.add_argument("--history_dir", type=str, default="mlp_wd/outputs/history/exp2")
    ap.add_argument("--lrs", type=str, default=None,
                    help="Comma-separated LRs (default uses DEFAULT_LRS).")
    ap.add_argument("--wds", type=str, default=None,
                    help="Comma-separated WDs (default uses DEFAULT_WDS).")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else DEFAULT_LRS
    wds = [float(x) for x in args.wds.split(",")] if args.wds else DEFAULT_WDS
    rows = build_grid(lrs, wds, args.epochs, args.seed, args.batch_size, args.momentum)

    print(f"[exp2] {len(rows)} runs | LRs={lrs} | WDs={wds} | epochs={args.epochs}")
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
