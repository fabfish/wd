"""Exp3 -- batch-size scaling with linear LR rule.

For each (B, lambda) we set eta = base_lr * (B / base_bs) and train SGDM (mom=0.9).
Default: 5 batch sizes x 5 lambdas = 25 runs.
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


DEFAULT_BS = [32, 64, 128, 256, 512]
DEFAULT_WDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def build_grid(batch_sizes, wds, base_lr, base_bs, momentum, epochs, seed):
    rows = []
    for bs, wd in product(batch_sizes, wds):
        lr = base_lr * (bs / base_bs)
        rows.append({
            "method": "SGDM+WD" if wd > 0 else "SGDM",
            "batch_size": bs,
            "lr": lr,
            "wd": wd,
            "momentum": momentum,
            "epochs": epochs,
            "seed": seed,
            "run_tag": f"exp3_bs{bs}_wd{wd}_lr{lr:.4f}",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_lr", type=float, default=0.05,
                    help="LR at the reference batch size (default 0.05 at base_bs=128)")
    ap.add_argument("--base_bs", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch_sizes", type=str, default=None,
                    help=f"Comma-separated, default {DEFAULT_BS}")
    ap.add_argument("--wds", type=str, default=None,
                    help=f"Comma-separated, default {DEFAULT_WDS}")
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=8)
    ap.add_argument("--loader_workers", type=int, default=0,
                    help="DataLoader num_workers per task (0 = in-process; recommended when workers_per_gpu>=2)")
    ap.add_argument("--output", type=str, default="mlp_wd/outputs/results/exp3.csv")
    ap.add_argument("--history_dir", type=str, default="mlp_wd/outputs/history/exp3")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else DEFAULT_BS
    wds = [float(x) for x in args.wds.split(",")] if args.wds else DEFAULT_WDS
    rows = build_grid(batch_sizes, wds, args.base_lr, args.base_bs, args.momentum,
                      args.epochs, args.seed)

    print(f"[exp3] {len(rows)} runs | BS={batch_sizes}, WDs={wds} | "
          f"linear LR rule: lr = {args.base_lr} * (B / {args.base_bs})")
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
