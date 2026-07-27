"""Exp2 grid for BN-MLP with pure SGD (no momentum).

Matches the rebuttal-figure recipe (lambdas spanning 1e-4..5e-2, etas spanning
3e-4..3.0) but runs pure SGD on MLP+CIFAR-10 instead of SGDM on ResNet+CIFAR-100.

Why pure SGD? Momentum (1-beta)^-1 inflates the effective step size in a way
that interacts with weight decay's stationary norm. Removing momentum keeps
the dynamics cleanly proportional to (eta, lambda), which is the regime in
which the eta*lambda scaling argument makes the per-lambda spoon minima land
at the same eta*lambda.

We keep BN on the hidden layers (use_bn=1) so the body of the network is
scale-invariant. norm_output stays off because the rebuttal's ResNet has a
plain Linear head and still gets aligned minima.
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


# 9 LRs spanning 1e4x. Pure SGD without momentum can take eta up to ~3 with BN.
LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
# 7 WDs matching the reference figure (lambda from 1e-4 to 5e-2).
WDS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]


def build_grid(epochs, seed, batch_size, momentum):
    rows = []
    for lr, wd in product(LRS, WDS):
        rows.append({
            "method": "SGD+WD" if wd > 0 else "SGD",
            "batch_size": batch_size, "lr": lr, "wd": wd,
            "momentum": momentum, "epochs": epochs, "seed": seed,
            "run_tag": f"exp2sgd_lr{lr}_wd{wd}",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--use_bn", type=int, default=1)
    ap.add_argument("--norm_output", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="0.0 for pure SGD; rebuttal uses 0.9 with SGDM")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Pure SGD converges slower than SGDM, give it more.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=0.0,
                    help="<=0 to disable; with BN+SGD we don't need it.")
    ap.add_argument("--gpus", type=str, default="1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=2)
    ap.add_argument("--loader_workers", type=int, default=2)
    ap.add_argument("--output", type=str,
                    default="mlp_wd/outputs/results/exp2_sgd.csv")
    ap.add_argument("--history_dir", type=str,
                    default="mlp_wd/outputs/history/exp2_sgd")
    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    rows = build_grid(args.epochs, args.seed, args.batch_size, args.momentum)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None
    use_bn = bool(args.use_bn)
    norm_output = bool(args.norm_output)

    print(f"[exp2-sgd] {len(rows)} runs | hidden={args.hidden_dim} "
          f"layers={args.num_layers} bn={int(use_bn)} normout={int(norm_output)} "
          f"mom={args.momentum} grad_clip={grad_clip} epochs={args.epochs}")
    run_grid(
        rows,
        output_file=args.output,
        history_dir=args.history_dir,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_bn=use_bn,
        norm_output=norm_output,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        log_every=args.log_every,
        loader_workers=args.loader_workers,
        grad_clip=grad_clip,
    )


if __name__ == "__main__":
    main()
