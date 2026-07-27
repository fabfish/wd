"""
Exp2 gap-filler: run ONLY the (lr, wd) cells that are missing from the
seed=42 ResNet-18/CIFAR-100 SGDM grid used in
rebuttal/figures/response_to_reviewer_9i84.png.

Goal: every λ-curve has a point at every η in SUPP_LRS so the plot has
no truncated tails. Uses the same training recipe as run_rebuttal.py
(SGDM, mom=0.9, BS=128) so values are directly comparable.
"""
import argparse
import csv
import os
import sys
import time
from itertools import product
from pathlib import Path

import filelock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wd_core.gpu_scheduler import GPUScheduler, parse_gpu_ids  # noqa: E402
from wd_core.logger import get_logger  # noqa: E402

# Re-use the worker from run_rebuttal so the recipe is identical.
from rebuttal.run_rebuttal import run_single_experiment_worker  # noqa: E402


CSV_FIELDS = ['method', 'batch_size', 'lr', 'wd', 'momentum',
              'final_test_acc', 'final_train_loss', 'best_test_acc',
              'final_test_loss']


def append_one_row(output_file: Path, row: dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(output_file) + '.lock')
    with lock:
        write_header = not output_file.exists()
        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


# Same grid as rebuttal/generate_figures.py (EXT_WDS x SUPP_LRS = 9 x 18 = 162 cells)
EXT_WDS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
SUPP_LRS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 3.0, 5.0]

EXP2_INPUT_CSVS = [
    ROOT / 'outputs/results/results.csv',
    ROOT / 'rebuttal/results/results_resnet18_seed42_exp2_ext.csv',
    ROOT / 'rebuttal/results/results_resnet18_seed42_exp2_ext2.csv',
    ROOT / 'rebuttal/results/results_resnet18_exp2_supplement.csv',
]


def find_completed_cells(extra_csvs=()):
    """Scan all known seed=42 ResNet-18 CSVs and snap each row to the EXT grid."""
    have = set()
    paths = list(EXP2_INPUT_CSVS) + [Path(p) for p in extra_csvs]
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        sub = df[(df['method'].isin(['SGDM', 'SGDM+WD'])) & (df['batch_size'] == 128)]
        for _, r in sub.iterrows():
            lr_match = next((lr for lr in SUPP_LRS if np.isclose(r['lr'], lr)), None)
            wd_match = next((wd for wd in EXT_WDS if np.isclose(r['wd'], wd)), None)
            if lr_match is not None and wd_match is not None:
                have.add((lr_match, wd_match))
    return have


FOCUS_WDS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]  # excludes 2e-4 and 2e-2 (figure filters them out)


def build_missing_tasks(
    have, model_name, batch_size, momentum, epochs, seed, use_amp,
    eta_lambda_lo: float | None = None,
    eta_lambda_hi: float | None = None,
    wds_subset=None,
):
    tasks = []
    method = 'SGDM+WD'
    wds = wds_subset if wds_subset else EXT_WDS
    for wd, lr in product(wds, SUPP_LRS):
        if (lr, wd) in have:
            continue
        elam = lr * wd
        if eta_lambda_lo is not None and elam < eta_lambda_lo:
            continue
        if eta_lambda_hi is not None and elam > eta_lambda_hi:
            continue
        tasks.append((model_name, method, batch_size, lr, wd, momentum, epochs, seed, use_amp))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=100,
                    help='Match the rebuttal recipe (100). Lower for a faster preview.')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--use_amp', action='store_true', default=True)
    ap.add_argument('--gpus', default='all')
    ap.add_argument('--workers_per_gpu', type=int, default=2,
                    help='ResNet-18 fits comfortably; raise if GPU memory permits.')
    ap.add_argument('--output', type=str,
                    default='rebuttal/results/results_resnet18_seed42_exp2_fill.csv')
    ap.add_argument('--eta_lambda_lo', type=float, default=None,
                    help='Skip cells with lr*wd below this threshold')
    ap.add_argument('--eta_lambda_hi', type=float, default=None,
                    help='Skip cells with lr*wd above this threshold')
    ap.add_argument('--focus', action='store_true',
                    help='Shortcut for --eta_lambda_lo 1e-5 --eta_lambda_hi 1e-3 '
                         'and the 7 lambdas the figure actually plots (no 2e-4, 2e-2).')
    ap.add_argument('--focus_left', action='store_true',
                    help='Shortcut for --eta_lambda_lo 1e-7 --eta_lambda_hi 1e-5 '
                         'and the 7 figure lambdas — the left tail of the spoons.')
    args = ap.parse_args()
    if args.focus:
        args.eta_lambda_lo = args.eta_lambda_lo or 1e-5
        args.eta_lambda_hi = args.eta_lambda_hi or 1e-3
    if args.focus_left:
        args.eta_lambda_lo = args.eta_lambda_lo or 1e-7
        args.eta_lambda_hi = args.eta_lambda_hi or 1e-5

    logger = get_logger(f"exp2_fill_{args.model}_s{args.seed}")
    logger.info(f"Model={args.model} seed={args.seed} epochs={args.epochs} bs={args.batch_size}")

    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    have = find_completed_cells(extra_csvs=[output_path])
    logger.info(f"already have {len(have)}/{len(EXT_WDS) * len(SUPP_LRS)} cells "
                f"(includes any rows already in {output_path.name})")

    tasks = build_missing_tasks(
        have,
        model_name=args.model,
        batch_size=args.batch_size,
        momentum=args.momentum,
        epochs=args.epochs,
        seed=args.seed,
        use_amp=args.use_amp,
        eta_lambda_lo=args.eta_lambda_lo,
        eta_lambda_hi=args.eta_lambda_hi,
        wds_subset=FOCUS_WDS if (args.focus or args.focus_left) else None,
    )
    if args.focus or args.focus_left:
        tag = 'focus_left' if args.focus_left else 'focus'
        logger.info(f"--{tag}: eta*lambda in [{args.eta_lambda_lo:g}, {args.eta_lambda_hi:g}], "
                    f"lambdas={FOCUS_WDS}")
    logger.info(f"missing cells to run: {len(tasks)}")
    if not tasks:
        logger.info('Nothing to fill. Exiting.')
        return

    if args.gpus == 'all':
        gpu_ids = None
    else:
        gpu_ids = parse_gpu_ids(args.gpus)

    scheduler = GPUScheduler(
        gpu_ids=gpu_ids,
        verbose=True,
        workers_per_gpu=args.workers_per_gpu,
    )

    done_counter = {'n': 0}

    def on_complete(result):
        if result is None:
            return
        done_counter['n'] += 1
        append_one_row(output_path, result)
        logger.info(
            f"[{done_counter['n']}/{len(tasks)}] saved lr={result['lr']:g} "
            f"wd={result['wd']:g} -> test_loss={result['final_test_loss']:.4f} "
            f"test_acc={result['final_test_acc']:.2f}"
        )

    start = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=on_complete)
    elapsed = (time.time() - start) / 60
    ok = sum(1 for r in results if r)
    logger.info(f"finished {ok}/{len(results)} cells in {elapsed:.1f} min, results in {output_path}")


if __name__ == '__main__':
    main()
