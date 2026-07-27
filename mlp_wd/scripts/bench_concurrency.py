"""Quick benchmark: how many concurrent tasks can share one A100 efficiently?

Run on a clean GPU and compare wallclock for N=1, 2, 4, 8 concurrent workers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlp_wd.mlp_core.gpu_scheduler import GPUScheduler


def bench_task(method, lr, wd, momentum, epochs, seed, dataset, hidden_dim, num_layers, num_workers):
    from mlp_wd.mlp_core.runner import run_single_experiment
    t0 = time.time()
    rec = run_single_experiment(
        method=method, batch_size=128, lr=lr, wd=wd, momentum=momentum,
        epochs=epochs, seed=seed, dataset=dataset, hidden_dim=hidden_dim,
        num_layers=num_layers, history_dir=None,
        run_tag=f"bench_{lr}_{wd}", log_every=0, num_workers=num_workers,
    )
    return time.time() - t0, rec["best_test_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4,
                    help="number of concurrent worker processes on the GPU")
    ap.add_argument("--loader_workers", type=int, default=0)
    args = ap.parse_args()

    # Build N identical tasks (different wd so caches don't collide).
    wds = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0][: args.workers]
    tasks = [
        ("SGDM+WD", 0.05, wd, 0.9, args.epochs, 42,
         "cifar10", 512, 3, args.loader_workers)
        for wd in wds
    ]

    scheduler = GPUScheduler(
        gpu_ids=[args.gpu], workers_per_gpu=args.workers, verbose=True,
    )
    t0 = time.time()
    results = scheduler.run_tasks(tasks, bench_task)
    total = time.time() - t0
    per_task_walls = [r[0] for r in results if r is not None]
    accs = [r[1] for r in results if r is not None]
    print(f"\nBENCH GPU={args.gpu} workers={args.workers} loader_workers={args.loader_workers}")
    print(f"  total wall = {total:.1f}s for {args.workers} tasks of {args.epochs} epochs each")
    print(f"  per-task wall mean = {sum(per_task_walls)/len(per_task_walls):.1f}s "
          f"(min={min(per_task_walls):.1f}, max={max(per_task_walls):.1f})")
    print(f"  per-task per-epoch = {sum(per_task_walls)/len(per_task_walls)/args.epochs:.2f}s/epoch")
    print(f"  best_test_acc = {accs}")


if __name__ == "__main__":
    main()
