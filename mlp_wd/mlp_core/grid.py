"""Grid orchestration helper used by run_exp{1,2,3}.py and run_pilot_exp2.py.

Each grid task is a tuple compatible with `run_single_experiment(*args)`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from .gpu_scheduler import GPUScheduler, parse_gpu_ids
from .io import append_result, load_completed_keys
from .runner import get_task_key, run_single_experiment


def _worker_entry(method, batch_size, lr, wd, momentum, epochs, seed,
                  dataset, hidden_dim, num_layers, use_bn, history_dir,
                  run_tag, log_every, num_workers, grad_clip) -> dict[str, Any]:
    """Module-level worker so multiprocessing 'spawn' can pickle it."""
    return run_single_experiment(
        method=method,
        batch_size=batch_size,
        lr=lr,
        wd=wd,
        momentum=momentum,
        epochs=epochs,
        seed=seed,
        dataset=dataset,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_bn=use_bn,
        history_dir=history_dir,
        run_tag=run_tag,
        log_every=log_every,
        num_workers=num_workers,
        grad_clip=grad_clip,
    )


def build_tasks(
    grid_rows: Iterable[dict[str, Any]],
    *,
    dataset: str,
    hidden_dim: int,
    num_layers: int,
    use_bn: bool,
    history_dir: str | None,
    log_every: int = 0,
    loader_workers: int = 2,
    grad_clip: float | None = None,
) -> list[tuple]:
    """Convert a list of grid dicts into tuples ready for the scheduler."""
    tasks = []
    for row in grid_rows:
        method = row["method"]
        batch_size = int(row["batch_size"])
        lr = float(row["lr"])
        wd = float(row["wd"])
        momentum = float(row.get("momentum", 0.0))
        epochs = int(row["epochs"])
        seed = int(row.get("seed", 42))
        run_tag = row.get("run_tag", f"{method}_bs{batch_size}_lr{lr}_wd{wd}_m{momentum}")
        tasks.append((
            method, batch_size, lr, wd, momentum, epochs, seed,
            dataset, hidden_dim, num_layers, use_bn, history_dir, run_tag,
            log_every, loader_workers, grad_clip,
        ))
    return tasks


def filter_completed(grid_rows, output_file, dataset, hidden_dim, num_layers,
                     use_bn: bool = False):
    """Remove rows already present in output_file."""
    completed = load_completed_keys(output_file)
    if not completed:
        return list(grid_rows), 0
    kept = []
    skipped = 0
    for row in grid_rows:
        key = get_task_key(
            method=row["method"],
            dataset=dataset,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            batch_size=int(row["batch_size"]),
            lr=float(row["lr"]),
            wd=float(row["wd"]),
            momentum=float(row.get("momentum", 0.0)),
            epochs=int(row["epochs"]),
            seed=int(row.get("seed", 42)),
        )
        if key in completed:
            skipped += 1
        else:
            kept.append(row)
    return kept, skipped


def run_grid(
    grid_rows,
    *,
    output_file: str | Path,
    history_dir: str | Path | None,
    dataset: str,
    hidden_dim: int,
    num_layers: int,
    gpu_ids,
    use_bn: bool = False,
    workers_per_gpu: int = 8,
    log_every: int = 0,
    loader_workers: int = 2,
    grad_clip: float | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run a grid with resume + incremental CSV writes."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if history_dir:
        Path(history_dir).mkdir(parents=True, exist_ok=True)

    pending, skipped = filter_completed(
        grid_rows, output_file, dataset=dataset,
        hidden_dim=hidden_dim, num_layers=num_layers, use_bn=use_bn,
    )
    if verbose:
        print(f"[grid] total={len(list(grid_rows)) if not isinstance(grid_rows, list) else len(grid_rows)} "
              f"pending={len(pending)} skipped(resume)={skipped}", flush=True)

    if not pending:
        return []

    tasks = build_tasks(
        pending,
        dataset=dataset, hidden_dim=hidden_dim, num_layers=num_layers,
        use_bn=use_bn,
        history_dir=str(history_dir) if history_dir else None,
        log_every=log_every, loader_workers=loader_workers,
        grad_clip=grad_clip,
    )

    scheduler = GPUScheduler(
        gpu_ids=gpu_ids, workers_per_gpu=workers_per_gpu, verbose=verbose,
    )

    def _on_complete(record):
        if record is not None:
            append_result(record, output_file)
            if verbose:
                print(
                    f"[grid] saved -> {record['method']} BS={record['batch_size']} "
                    f"lr={record['lr']} wd={record['wd']} mom={record['momentum']} | "
                    f"final_test_loss={record['final_test_loss']:.4f} "
                    f"best_test_acc={record['best_test_acc']:.2f}%",
                    flush=True,
                )

    t0 = time.time()
    results = scheduler.run_tasks(tasks, _worker_entry, on_complete=_on_complete)
    if verbose:
        print(f"[grid] finished {len(results)} runs in {(time.time() - t0)/60:.2f} min", flush=True)
    return results
