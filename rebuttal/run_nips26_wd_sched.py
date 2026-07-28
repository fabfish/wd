"""
E8: fixed weight decay vs scheduled weight decay (AdamW/SGDW schedule shapes).

Learning rate is held constant. Only lambda is scheduled. Phases:
  --phase sgd   momentum=0
  --phase sgdm  momentum=0.9 (reuse existing const-LR fixed-lambda cells)

Usage:
  python rebuttal/run_nips26_wd_sched.py --phase sgd  --gpus 0,2,3 --dry_run
  python rebuttal/run_nips26_wd_sched.py --phase sgd  --gpus 0,2,3
  python rebuttal/run_nips26_wd_sched.py --phase sgdm --gpus 0,2,3
  python rebuttal/run_nips26_wd_sched.py --phase all  --gpus 0,2,3
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wd_core.data import get_cifar100_loaders  # noqa: E402
from wd_core.models import get_model  # noqa: E402
from wd_core.utils import set_seed, train_model_ext  # noqa: E402
from wd_core.gpu_scheduler import GPUScheduler, parse_gpu_ids  # noqa: E402
from wd_core.logger import get_logger  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
DEFAULT_CSV = RESULTS_DIR / 'nips26_runs.csv'
DATA_DIR = str(Path(__file__).resolve().parent.parent / 'data')

# Identity includes wd_sched so fixed vs cosine at the same lambda0 are distinct.
RUN_KEY = ['model', 'dataset', 'method', 'batch_size', 'lr', 'wd', 'momentum',
           'epochs', 'scheduler', 'seed', 'wd_sched']

CSV_FIELDS = RUN_KEY + [
    'exp', 'sum_lr', 'best_test_acc', 'final_test_acc', 'final_train_loss',
    'final_train_acc', 'final_test_loss', 'diverged', 'epochs_run', 'wall_time',
]

DIVERGENCE_LOSS_THRESHOLD = 2.0 * math.log(100.0)
DEFAULT_NUM_WORKERS = 5
EXP_NAME = 'e8_wd_sched'

WD_SCHEDULES = ['fixed', 'cosine', 'linear', 'step', 'cosine_restarts']
LAMBDA0_GRID = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

# SGDR-style restart params (AdamW paper): first cycle Te, then Te*Tmult.
RESTART_TE = 50
RESTART_TMULT = 2


# --------------------------------------------------------------------------
# schedule multipliers applied only to weight decay
# --------------------------------------------------------------------------

def wd_multiplier(wd_sched, epoch, epochs, te=RESTART_TE, tmult=RESTART_TMULT):
    """Return eta_t in [0, 1] for a 0-based epoch index."""
    T = float(epochs)
    t = float(epoch)
    if wd_sched == 'fixed':
        return 1.0
    if wd_sched == 'cosine':
        return 0.5 * (1.0 + math.cos(math.pi * t / T))
    if wd_sched == 'linear':
        return max(0.0, 1.0 - t / T)
    if wd_sched == 'step':
        frac = t / T
        if frac < 0.5:
            return 1.0
        if frac < 0.75:
            return 0.1
        return 0.01
    if wd_sched == 'cosine_restarts':
        # SGDR: walk through cycles of length Te, Te*Tmult, ...
        remaining = t
        Ti = float(te)
        while remaining >= Ti:
            remaining -= Ti
            Ti *= float(tmult)
        return 0.5 * (1.0 + math.cos(math.pi * remaining / Ti))
    raise ValueError(f'unknown wd_sched={wd_sched}')


def make_wd_schedule_fn(wd_sched, lambda0, epochs):
    def fn(epoch):
        return float(lambda0) * wd_multiplier(wd_sched, epoch, epochs)
    return fn


# --------------------------------------------------------------------------
# CSV helpers (migrate in wd_sched column if missing)
# --------------------------------------------------------------------------

def cfg_key(cfg):
    out = []
    for field in RUN_KEY:
        value = cfg[field]
        out.append(f"{float(value):.10g}" if isinstance(value, float) else str(value))
    return tuple(out)


def ensure_csv_schema(csv_path):
    """Add wd_sched column to an existing nips26_runs.csv if needed."""
    path = Path(csv_path)
    if not path.exists():
        return
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if 'wd_sched' in fieldnames:
            return
        rows = list(reader)
    # Old const-LR fixed-lambda cells become wd_sched=fixed for reuse matching.
    for row in rows:
        sched = str(row.get('scheduler', 'cosine'))
        row['wd_sched'] = 'fixed' if sched == 'const' else ''
    tmp = path.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})
    tmp.replace(path)


def load_done_keys(csv_path):
    ensure_csv_schema(csv_path)
    if not Path(csv_path).exists():
        return set()
    done = set()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                cfg = {k: row[k] for k in RUN_KEY}
                for k in ('lr', 'wd', 'momentum'):
                    cfg[k] = float(cfg[k])
                # Blank wd_sched (legacy cosine-LR rows) do not match e8 keys.
                if not str(cfg.get('wd_sched', '')).strip():
                    continue
                done.add(cfg_key(cfg))
            except (KeyError, ValueError):
                continue
    return done


def append_row(csv_path, row):
    ensure_csv_schema(csv_path)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(csv_path).exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_cfg(wd_sched, lambda0, momentum, lr=0.1, epochs=100, batch_size=128,
             model='resnet18', seed=42, dataset='cifar100'):
    if momentum > 0:
        method = 'SGDM+WD' if lambda0 > 0 else 'SGDM'
    else:
        method = 'SGD+WD' if lambda0 > 0 else 'SGD'
    return {
        'model': model, 'dataset': dataset, 'method': method,
        'batch_size': int(batch_size), 'lr': float(lr), 'wd': float(lambda0),
        'momentum': float(momentum), 'epochs': int(epochs),
        'scheduler': 'const',  # LR fixed
        'seed': int(seed), 'wd_sched': wd_sched, 'exp': EXP_NAME,
        'num_workers': DEFAULT_NUM_WORKERS,
    }


def build_cfgs(phase):
    """phase in {'sgd', 'sgdm', 'all'}."""
    cfgs = []
    phases = []
    if phase in ('sgd', 'all'):
        phases.append(0.0)
    if phase in ('sgdm', 'all'):
        phases.append(0.9)
    for momentum in phases:
        for wd_sched in WD_SCHEDULES:
            for lam0 in LAMBDA0_GRID:
                cfgs.append(make_cfg(wd_sched, lam0, momentum))
    return cfgs


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------

def run_one(cfg):
    torch.backends.cudnn.benchmark = True
    set_seed(cfg['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(
        batch_size=cfg['batch_size'],
        num_workers=cfg.get('num_workers', DEFAULT_NUM_WORKERS),
        data_dir=DATA_DIR,
    )
    model = get_model(cfg['model'], num_classes=100).to(device)

    # Initial weight_decay is lambda0; schedule overwrites every epoch.
    optimizer = optim.SGD(
        model.parameters(), lr=cfg['lr'],
        momentum=cfg['momentum'], weight_decay=cfg['wd'],
    )
    wd_fn = make_wd_schedule_fn(cfg['wd_sched'], cfg['wd'], cfg['epochs'])

    tag = (f"[{cfg['exp']}|{cfg['method']}|wd_sched={cfg['wd_sched']}|"
           f"T={cfg['epochs']}|lr={cfg['lr']:g}|lam0={cfg['wd']:g}] ")
    print(tag + 'start', flush=True)

    start = time.time()
    metrics = train_model_ext(
        model, train_loader, test_loader, optimizer, scheduler=None,
        device=device, epochs=cfg['epochs'], use_amp=True,
        log_interval=max(cfg['epochs'] // 4, 1),
        divergence_loss_threshold=DIVERGENCE_LOSS_THRESHOLD,
        divergence_check_epoch=3, tag=tag,
        wd_schedule_fn=wd_fn,
    )
    metrics['wall_time'] = round(time.time() - start, 1)

    row = {k: cfg[k] for k in RUN_KEY}
    row['exp'] = cfg['exp']
    row.update(metrics)
    return row


def run_grid(cfgs, gpu_ids, workers_per_gpu, csv_path, logger, label=''):
    done = load_done_keys(csv_path)
    pending, skipped, seen = [], 0, set()
    for cfg in cfgs:
        key = cfg_key(cfg)
        if key in done or key in seen:
            skipped += 1
            continue
        seen.add(key)
        pending.append(cfg)

    logger.info(f'{label}: {len(cfgs)} configs, {skipped} already done, '
                f'{len(pending)} to run')
    if not pending:
        return []

    total_epochs = sum(c['epochs'] for c in pending)
    logger.info(f'{label}: budget ~{total_epochs / 100:.1f} units')

    if getattr(run_grid, '_dry_run', False):
        for c in pending:
            logger.info(f"  pending: mom={c['momentum']} wd_sched={c['wd_sched']} "
                        f"lam0={c['wd']:g}")
        return []

    scheduler = GPUScheduler(
        gpu_ids=gpu_ids, verbose=True, workers_per_gpu=workers_per_gpu)
    collected = []

    def on_complete(row):
        if row is not None:
            append_row(csv_path, row)
            collected.append(row)
            logger.info(
                f"  [{len(collected)}/{len(pending)}] "
                f"mom={row['momentum']} wd_sched={row['wd_sched']} "
                f"lam0={row['wd']:g} -> best={row['best_test_acc']:.2f} "
                f"diverged={row['diverged']} ({row['wall_time']:.0f}s)")

    start = time.time()
    scheduler.run_tasks([(c,) for c in pending], run_one, on_complete=on_complete)
    logger.info(f'{label}: finished {len(collected)}/{len(pending)} in '
                f'{(time.time() - start) / 60:.1f} min')
    return collected


def main():
    parser = argparse.ArgumentParser(description='E8 scheduled weight-decay sweep')
    parser.add_argument('--phase', choices=['sgd', 'sgdm', 'all'], default='sgd')
    parser.add_argument('--gpus', type=str, default='0,2,3')
    parser.add_argument('--workers_per_gpu', type=int, default=2)
    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV))
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    logger = get_logger(f'nips26_{EXP_NAME}_{args.phase}')
    gpu_ids = parse_gpu_ids(args.gpus)
    cfgs = build_cfgs(args.phase)
    logger.info(f'phase={args.phase} gpus={gpu_ids} workers_per_gpu='
                f'{args.workers_per_gpu} n_cfgs={len(cfgs)}')

    run_grid._dry_run = args.dry_run
    ensure_csv_schema(args.csv)
    run_grid(cfgs, gpu_ids, args.workers_per_gpu, args.csv, logger,
             label=f'E8-{args.phase}')


if __name__ == '__main__':
    main()
