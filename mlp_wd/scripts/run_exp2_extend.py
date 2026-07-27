"""Exp2 extension: fill in the rising tails of the small-lambda curves.

The default 6x6 grid skips from eta=0.3 to eta=1.0, which leaves a 3.3x gap
right where the spoon's rising tail lives. For small lambdas (1e-4, 3e-4, 1e-3)
the curves end before the eta*lambda ~ 1e-4 to 1e-3 region; we want to push
them through it.

Strategy: keep the same lambda grid, but for each small lambda add intermediate
etas in [0.2, 0.5, 0.7, 1.5, 2.0, 3.0, 10.0] that cover the gap and the unstable
shoulder. Writes incremental rows into the SAME exp2.csv used by run_exp2.

Run after run_exp2_eta_lambda.py:
    python -m mlp_wd.scripts.run_exp2_extend --gpus 1,2,3 --workers_per_gpu 8 \
        --loader_workers 0 --epochs 30
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


# Per-lambda extensions: each entry adds (lambda, [extra etas]).
# Designed so each row reaches across the eta*lambda ~ 1e-4 to 1e-3 region
# even for small lambdas, and so the unstable "rising tail" is sampled.
EXTENSIONS = {
    1e-4: [0.5, 0.7, 1.5, 2.0, 3.0, 10.0],
    3e-4: [0.5, 0.7, 1.5, 2.0, 3.0],
    1e-3: [0.2, 0.5, 0.7, 1.5, 2.0],
    3e-3: [0.2, 0.5, 0.7],
    1e-2: [0.2],
}


def build_grid(epochs, seed, batch_size, momentum):
    rows = []
    for wd, etas in EXTENSIONS.items():
        for lr in etas:
            rows.append({
                "method": "SGDM+WD" if wd > 0 else "SGDM",
                "batch_size": batch_size,
                "lr": lr,
                "wd": wd,
                "momentum": momentum,
                "epochs": epochs,
                "seed": seed,
                "run_tag": f"exp2x_lr{lr}_wd{wd}",
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
    ap.add_argument("--loader_workers", type=int, default=0)
    ap.add_argument("--output", type=str,
                    default="mlp_wd/outputs/results/exp2.csv",
                    help="appended to the same CSV that holds the original 6x6 grid")
    ap.add_argument("--history_dir", type=str,
                    default="mlp_wd/outputs/history/exp2")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    rows = build_grid(args.epochs, args.seed, args.batch_size, args.momentum)
    print(f"[exp2-extend] {len(rows)} new runs across {len(EXTENSIONS)} lambdas")
    for wd, etas in EXTENSIONS.items():
        eta_lams = [f"{eta * wd:.1e}" for eta in etas]
        print(f"  lambda={wd:g}: extra etas={etas}, eta*lambda={eta_lams}")

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
