import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from pathlib import Path

import torch

from core import ensure_parent
from train_cifar100 import run_training


def parse_gpu_ids(spec):
    if spec in (None, "", "cpu"):
        return []
    if spec == "all":
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def experiment_tasks(experiment, epochs, seed, model):
    if experiment == 1:
        tasks = []
        for lr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            tasks.append({"method": "SGD", "batch_size": 128, "lr": lr, "wd": 0.0, "momentum": 0.0})
        for wd in [1e-3, 2e-3, 5e-3, 1e-2]:
            for lr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
                tasks.append({"method": "SGD+WD", "batch_size": 128, "lr": lr, "wd": wd, "momentum": 0.0})
        for lr in [0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2]:
            tasks.append({"method": "SGDM+WD", "batch_size": 128, "lr": lr, "wd": 5e-4, "momentum": 0.9})
        return [dict(t, experiment=experiment, epochs=epochs, seed=seed, model=model) for t in tasks]

    if experiment == 2:
        tasks = []
        for lr in [0.01, 0.05, 0.1, 0.2, 0.3]:
            for wd in [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3]:
                tasks.append({"method": "SGDM", "batch_size": 128, "lr": lr, "wd": wd, "momentum": 0.9})
        return [dict(t, experiment=experiment, epochs=epochs, seed=seed, model=model) for t in tasks]

    if experiment == 3:
        tasks = []
        for batch_size in [64, 128, 256, 512]:
            lr = 0.1 * batch_size / 128
            for wd in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
                tasks.append({"method": "SGDM", "batch_size": batch_size, "lr": lr, "wd": wd, "momentum": 0.9})
        return [dict(t, experiment=experiment, epochs=epochs, seed=seed, model=model) for t in tasks]

    raise ValueError(f"Unknown experiment {experiment}")


def make_train_args(task, device, args):
    return SimpleNamespace(
        model=task["model"],
        batch_size=task["batch_size"],
        lr=task["lr"],
        wd=task["wd"],
        momentum=task["momentum"],
        epochs=task["epochs"],
        seed=task["seed"],
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        device=device,
        use_amp=args.use_amp,
        log_interval=args.log_interval,
        output=None,
    )


def run_task(task, device, args):
    result = run_training(make_train_args(task, device, args))
    result.update(
        {
            "experiment": task["experiment"],
            "method": task["method"],
        }
    )
    return result


def write_result(result, output):
    ensure_parent(output)
    fields = [
        "experiment",
        "method",
        "model",
        "batch_size",
        "lr",
        "wd",
        "momentum",
        "epochs",
        "seed",
        "best_test_acc",
        "final_test_acc",
        "final_train_loss",
    ]
    exists = Path(output).exists()
    with open(output, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: result.get(key) for key in fields})


def main():
    parser = argparse.ArgumentParser(description="Run anonymized grid experiments.")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["resnet18", "resnet50", "vgg16"], default="resnet18")
    parser.add_argument("--gpus", type=str, default="all")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="outputs/results/results.csv")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    tasks = experiment_tasks(args.experiment, args.epochs, args.seed, args.model)
    gpu_ids = parse_gpu_ids(args.gpus)
    devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids] or ["cpu"]
    max_workers = max(1, len(devices) * args.workers_per_gpu)
    print(f"Running {len(tasks)} tasks on devices={devices}, max_workers={max_workers}")

    if max_workers == 1:
        for idx, task in enumerate(tasks):
            result = run_task(task, devices[0], args)
            write_result(result, args.output)
            print(f"completed {idx + 1}/{len(tasks)}")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, task in enumerate(tasks):
            device = devices[idx % len(devices)]
            futures[executor.submit(run_task, task, device, args)] = idx

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            write_result(result, args.output)
            print(f"completed {completed}/{len(tasks)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
