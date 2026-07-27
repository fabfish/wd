"""
NeurIPS 17464 rebuttal experiment runner.

Unlike the earlier runners, `epochs` and the learning-rate schedule are swept
dimensions rather than global flags. That is the whole point of this round of
experiments: every result collected so far was at 100 epochs, which makes the
paper's lambda ~ 1/(eta*T) law empirically indistinguishable from a rule of the
form eta*lambda = const.

All experiments append to a single CSV, so a configuration that appears in more
than one experiment is trained once and reused.

Usage:
    python rebuttal/run_nips26.py --exp e1_prelim --gpus 0-3
    python rebuttal/run_nips26.py --exp e3 --gpus 0-3 --workers_per_gpu 4
    python rebuttal/run_nips26.py --exp e1_prelim --dry_run
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wd_core.data import get_cifar100_loaders  # noqa: E402
from wd_core.models import get_model  # noqa: E402
from wd_core.utils import set_seed, train_model_ext  # noqa: E402
from wd_core.gpu_scheduler import GPUScheduler, parse_gpu_ids  # noqa: E402
from wd_core.logger import get_logger  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
DEFAULT_CSV = RESULTS_DIR / 'nips26_runs.csv'

# Identity of a training run. `exp` is deliberately excluded so that a cell
# shared by two experiments (for example an E1 grid point that also serves as an
# E4 oracle) is only ever trained once.
RUN_KEY = ['model', 'dataset', 'method', 'batch_size', 'lr', 'wd', 'momentum',
           'epochs', 'scheduler', 'seed']

CSV_FIELDS = RUN_KEY + [
    'exp', 'sum_lr', 'best_test_acc', 'final_test_acc', 'final_train_loss',
    'final_train_acc', 'final_test_loss', 'diverged', 'epochs_run', 'wall_time',
]

# CIFAR-100 starts at ln(100) = 4.6; twice that means the run is not coming back.
DIVERGENCE_LOSS_THRESHOLD = 2.0 * math.log(100.0)

DEFAULT_NUM_WORKERS = 5
DATA_DIR = str(Path(__file__).resolve().parent.parent / 'data')


# --------------------------------------------------------------------------
# run bookkeeping
# --------------------------------------------------------------------------

def make_cfg(exp, lr, wd, epochs, momentum=0.9, batch_size=128, model='resnet18',
             scheduler='cosine', seed=42, dataset='cifar100', method=None):
    if method is None:
        if momentum > 0:
            method = 'SGDM+WD' if wd > 0 else 'SGDM'
        else:
            method = 'SGD+WD' if wd > 0 else 'SGD'
    return {
        'model': model, 'dataset': dataset, 'method': method,
        'batch_size': int(batch_size), 'lr': float(lr), 'wd': float(wd),
        'momentum': float(momentum), 'epochs': int(epochs),
        'scheduler': scheduler, 'seed': int(seed), 'exp': exp,
        # Carried in the config rather than read from a module global, because
        # workers are spawned and would otherwise miss the command-line value.
        'num_workers': DEFAULT_NUM_WORKERS,
    }


def cfg_key(cfg):
    """Hashable identity of a run, with floats normalized so 5e-4 == 0.0005."""
    out = []
    for field in RUN_KEY:
        value = cfg[field]
        out.append(f"{float(value):.10g}" if isinstance(value, float) else str(value))
    return tuple(out)


def load_done_keys(csv_path):
    if not Path(csv_path).exists():
        return set()
    done = set()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                done.add(cfg_key({k: float(row[k]) if k in
                                  ('lr', 'wd', 'momentum') else row[k] for k in RUN_KEY}))
            except (KeyError, ValueError):
                continue
    return done


def load_legacy_done_keys():
    """
    Configurations already covered by the pre-rebuttal sweeps.

    The analysis merges those runs with the new ones, so retraining a grid point
    that already exists only costs GPU hours. Only exact matches on the full run
    key are skipped, and only for cosine schedules, which is what every legacy
    run used.
    """
    try:
        from analysis.nips26_lib import load_legacy
    except ImportError:
        return set()
    try:
        df = load_legacy()
    except Exception:
        return set()

    done = set()
    for _, r in df.iterrows():
        try:
            if str(r.get('scheduler', 'cosine')) != 'cosine':
                continue
            done.add(cfg_key({
                'model': str(r['model']), 'dataset': str(r['dataset']),
                'method': str(r['method']), 'batch_size': int(r['batch_size']),
                'lr': float(r['lr']), 'wd': float(r['wd']),
                'momentum': float(r['momentum']), 'epochs': int(r['epochs']),
                'scheduler': 'cosine', 'seed': int(r['seed']),
            }))
        except (KeyError, ValueError, TypeError):
            continue
    return done


def load_rows(csv_path):
    if not Path(csv_path).exists():
        return []
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def append_row(csv_path, row):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(csv_path).exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------

def run_one(cfg):
    """Train a single configuration. Runs inside a worker process."""
    torch.backends.cudnn.benchmark = True
    set_seed(cfg['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar100_loaders(
        batch_size=cfg['batch_size'],
        num_workers=cfg.get('num_workers', DEFAULT_NUM_WORKERS),
        data_dir=DATA_DIR,
    )
    model = get_model(cfg['model'], num_classes=100).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'],
                          momentum=cfg['momentum'], weight_decay=cfg['wd'])
    if cfg['scheduler'] == 'cosine':
        lr_sched = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    elif cfg['scheduler'] == 'const':
        lr_sched = None
    else:
        raise ValueError(f"unknown scheduler {cfg['scheduler']}")

    tag = (f"[{cfg['exp']}|{cfg['model']}|T={cfg['epochs']}|B={cfg['batch_size']}|"
           f"lr={cfg['lr']:g}|wd={cfg['wd']:g}|m={cfg['momentum']:g}|{cfg['scheduler']}] ")
    print(tag + "start")

    start = time.time()
    metrics = train_model_ext(
        model, train_loader, test_loader, optimizer, lr_sched, device,
        epochs=cfg['epochs'], use_amp=True,
        log_interval=max(cfg['epochs'] // 4, 1),
        divergence_loss_threshold=DIVERGENCE_LOSS_THRESHOLD,
        divergence_check_epoch=3, tag=tag,
    )
    metrics['wall_time'] = round(time.time() - start, 1)

    row = dict(cfg)
    row.update(metrics)
    return row


# --------------------------------------------------------------------------
# grid driver
# --------------------------------------------------------------------------

def run_grid(cfgs, gpu_ids, workers_per_gpu, csv_path, logger, label="",
             use_legacy=True):
    """Train every configuration that is not already in the CSV."""
    done = load_done_keys(csv_path)
    if use_legacy:
        legacy = load_legacy_done_keys()
        if legacy:
            logger.info(f"{label}: {len(legacy)} configurations already covered "
                        f"by the pre-rebuttal sweeps")
            done |= legacy

    pending, skipped, seen = [], 0, set()
    for cfg in cfgs:
        key = cfg_key(cfg)
        if key in done or key in seen:
            skipped += 1
            continue
        seen.add(key)
        pending.append(cfg)

    logger.info(f"{label}: {len(cfgs)} configs, {skipped} already done, {len(pending)} to run")
    if not pending:
        return []

    total_epochs = sum(c['epochs'] for c in pending)
    logger.info(f"{label}: budget ~{total_epochs / 100:.1f} units "
                f"(1 unit = one 100-epoch resnet18 run)")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True, workers_per_gpu=workers_per_gpu)
    collected = []

    def on_complete(row):
        if row is not None:
            append_row(csv_path, row)
            collected.append(row)
            logger.info(f"  [{len(collected)}/{len(pending)}] "
                        f"T={row['epochs']} lr={row['lr']:g} wd={row['wd']:g} "
                        f"-> best={row['best_test_acc']:.2f} "
                        f"diverged={row['diverged']} ({row['wall_time']:.0f}s)")

    start = time.time()
    scheduler.run_tasks([(c,) for c in pending], run_one, on_complete=on_complete)
    logger.info(f"{label}: finished {len(collected)}/{len(pending)} in "
                f"{(time.time() - start) / 60:.1f} min")
    return collected


# --------------------------------------------------------------------------
# experiment definitions
# --------------------------------------------------------------------------

# The lambda ladder is a subset of the grid already collected at T=100, so the
# existing runs line up exactly with the new columns.
E1_LAMBDAS = [1e-4, 5e-4, 1e-3, 5e-3, 2e-2]
E1_LAMBDAS_FULL = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]


def build_e1_prelim():
    """
    E1-prelim: does the optimal lambda move with the training length?

    Fixed eta = 0.1, B = 128, SGDM. T = 100 is re-run here rather than taken
    from the legacy CSVs so that the whole T-series is produced by one code
    path. Our law predicts slope -1 for log lambda* against log T; a rule of the
    form eta*lambda = const predicts slope 0.
    """
    cfgs = []
    for T in [25, 100, 200]:
        for wd in E1_LAMBDAS:
            cfgs.append(make_cfg('e1_prelim', lr=0.1, wd=wd, epochs=T))
    return cfgs


def build_e1_fine():
    """
    E1-fine: the headline T-series on the dense lambda ladder.

    E1-prelim's ladder steps by factors of up to five, which is too coarse to
    resolve a shift the theory puts at roughly a factor of eight between T = 25
    and T = 200: the interpolation error would be the same size as the signal.
    This refines the same eta = 0.1 series to steps of about 2.2x and adds
    T = 50, and is the run the reported slope should come from.
    """
    return [make_cfg('e1_fine', lr=0.1, wd=wd, epochs=T)
            for T in [25, 50, 100, 200] for wd in E1_LAMBDAS_FULL]


def build_e1_rescue():
    """
    E1-rescue: recover the 1/T signal that the headline eta=0.1 grid masked.

    Reading of the data so far:
      - at eta=0.1 the grid argmax sits at 1e-3 for every T (quantization + a
        possible equilibrium floor on eta*lambda);
      - soft (accuracy-weighted) lambda* already drifts down with T;
      - at eta=0.02 the interpolated slope is ~-0.6, much closer to -1.

    This block densifies the peak, goes to shorter T (pre-equilibrium), densifies
    the low-eta arm, and redoes constant-LR with a ladder that reaches below
    1e-4 (the previous const arm peaked on its left edge).
    """
    cfgs = []

    # A. dense peak at eta=0.1 — resolve the soft leftward drift
    dense_peak = [4e-4, 6e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3, 2e-3, 2.5e-3]
    for T in [25, 50, 100, 200]:
        for wd in dense_peak:
            cfgs.append(make_cfg('e1_rescue', lr=0.1, wd=wd, epochs=T))

    # B. short T — timescale should dominate before rotational equilibrium
    for T in [5, 10, 15]:
        for wd in [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]:
            cfgs.append(make_cfg('e1_rescue', lr=0.1, wd=wd, epochs=T))

    # C. dense low-eta arm — where T-dependence already shows
    for T in [25, 50, 100, 200]:
        for wd in [1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2]:
            cfgs.append(make_cfg('e1_rescue', lr=0.02, wd=wd, epochs=T))

    # D. constant-LR with a ladder that can sit below 1e-4
    for T in [25, 100]:
        for wd in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]:
            cfgs.append(make_cfg('e1_rescue', lr=0.1, wd=wd, epochs=T,
                                 scheduler='const'))

    return cfgs


def build_e1_full():
    """
    E1-full: fill in T = 50, add a second learning rate, a second seed, and a
    constant-LR arm.

    The eta = 0.02 arm distinguishes a dependence on eta*T from one on T alone:
    at five times smaller eta the optimum should sit at five times larger
    lambda. The constant-LR arm tests the C/sum_lr form directly, since cosine
    decay halves sum_t eta_t relative to a constant schedule at matched eta and
    T -- which is where the paper's stray factor of two comes from.
    """
    cfgs = []

    # main eta = 0.1 series, remaining T values and the wider lambda ladder
    for T in [25, 50, 100, 200]:
        for wd in E1_LAMBDAS_FULL:
            cfgs.append(make_cfg('e1_full', lr=0.1, wd=wd, epochs=T))

    # second learning rate: lambda ladder shifted up one decade
    for T in [25, 100, 200]:
        for wd in [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
            cfgs.append(make_cfg('e1_full', lr=0.02, wd=wd, epochs=T))

    # second seed on the headline series
    for T in [25, 100, 200]:
        for wd in E1_LAMBDAS:
            cfgs.append(make_cfg('e1_full', lr=0.1, wd=wd, epochs=T, seed=123))

    # constant-LR arm
    for T in [25, 100]:
        for wd in E1_LAMBDAS:
            cfgs.append(make_cfg('e1_full', lr=0.1, wd=wd, epochs=T, scheduler='const'))

    return cfgs


def build_e2b(product_levels=None, csv_path=DEFAULT_CSV):
    """
    E2b: walk along a line of constant eta*lambda.

    If only the product mattered -- the reading that makes weight decay a
    redundant knob -- accuracy would be flat along this line. The prediction is
    that it is not: small eta under-optimizes within the budget and large eta
    crosses the stability boundary eta <= 2/(2*lambda + L).
    """
    if product_levels is None:
        product_levels = infer_product_levels(csv_path)
    etas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    cfgs = []
    for product in product_levels:
        for eta in etas:
            cfgs.append(make_cfg('e2b', lr=eta, wd=product / eta, epochs=100))
    return cfgs


def infer_product_levels(csv_path):
    """Best eta*lambda from the existing T=100 heatmap, and three times that."""
    from analysis.nips26_lib import load_legacy_grid  # noqa: E402
    grid = load_legacy_grid()
    best = grid.loc[grid['best_test_acc'].idxmax()]
    p = float(best['lr']) * float(best['wd'])
    return [round(p, 12), round(3 * p, 12)]


E3_LAMBDAS = [0.0, 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
E3_LADDER = [0.05, 0.2, 0.8, 3.0, 10.0, 20.0]
E3_EPOCHS = 15
E3_ACC_FLOOR = 5.0


def is_stable(row):
    return int(row['diverged']) == 0 and float(row['best_test_acc']) >= E3_ACC_FLOOR


def build_e3_ladder():
    cfgs = []
    for momentum in [0.0, 0.9]:
        for wd in E3_LAMBDAS:
            for lr in E3_LADDER:
                cfgs.append(make_cfg('e3', lr=lr, wd=wd, epochs=E3_EPOCHS,
                                     momentum=momentum, scheduler='const'))
    return cfgs


def run_e3(gpu_ids, workers_per_gpu, csv_path, logger, rounds=4,
           num_workers=DEFAULT_NUM_WORKERS):
    """
    E3: locate the largest learning rate that still trains, as a function of
    lambda, and check it against the shape the theory predicts.

    The bound eta <= 2/(2*lambda + L) rearranges to 1/eta_max = lambda + L/2,
    i.e. a straight line in lambda whose intercept identifies the smoothness
    constant. Fitting that line and comparing the intercept against a measured
    top Hessian eigenvalue is the concrete version of "does any of this survive
    outside the convex setting".

    A constant schedule is used so that eta means one thing throughout the run,
    and divergent cells abort within a few epochs.
    """
    ladder = build_e3_ladder()
    for cfg in ladder:
        cfg['num_workers'] = num_workers
    run_grid(ladder, gpu_ids, workers_per_gpu, csv_path, logger,
             label="E3 coarse ladder")

    for round_idx in range(rounds):
        rows = [r for r in load_rows(csv_path)
                if r['exp'] == 'e3' and int(r['epochs']) == E3_EPOCHS]
        cfgs = []
        for momentum in [0.0, 0.9]:
            for wd in E3_LAMBDAS:
                relevant = [r for r in rows
                            if abs(float(r['momentum']) - momentum) < 1e-12
                            and abs(float(r['wd']) - wd) < 1e-12]
                if not relevant:
                    continue
                stable = sorted(float(r['lr']) for r in relevant if is_stable(r))
                unstable = sorted(float(r['lr']) for r in relevant if not is_stable(r))
                if not stable or not unstable:
                    continue  # boundary outside the ladder; nothing to refine
                lo = max(stable)
                hi = min((u for u in unstable if u > lo), default=None)
                if hi is None:
                    continue
                if hi / lo < 1.15:
                    continue  # bracket already tight enough
                mid = math.sqrt(lo * hi)  # bisect in log space
                cfg = make_cfg('e3', lr=round(mid, 6), wd=wd, epochs=E3_EPOCHS,
                               momentum=momentum, scheduler='const')
                cfg['num_workers'] = num_workers
                cfgs.append(cfg)
        if not cfgs:
            logger.info(f"E3: brackets converged after {round_idx} refinement rounds")
            break
        run_grid(cfgs, gpu_ids, workers_per_gpu, csv_path, logger,
                 label=f"E3 refinement {round_idx + 1}/{rounds}")


def build_e4(csv_path=DEFAULT_CSV):
    """
    E4: predict lambda with no tuning at all, then check the cost of being wrong.

    Five strategies are calibrated once on the reference configuration and then
    applied blind to held-out settings. Oracles come from grids that already
    exist (or from E1), so only the strategy cells need training.
    """
    from analysis.nips26_lib import build_transfer_plan  # noqa: E402
    plan = build_transfer_plan(csv_path)
    cfgs = []
    for item in plan:
        cfgs.append(make_cfg('e4', lr=item['lr'], wd=item['wd'], epochs=item['epochs'],
                             batch_size=item['batch_size'], model=item['model']))
    return cfgs


def build_e5b(csv_path=DEFAULT_CSV):
    """
    E5b: how flat is the optimum in the unknown constant C?

    If a factor of three error in C costs a fraction of a percent, then the rule
    is usable even though C is not predicted from first principles.
    """
    from analysis.nips26_lib import fit_reference_C, sum_lr as exact_sum_lr  # noqa: E402
    C = fit_reference_C(csv_path)
    cfgs = []
    for (lr, T, B) in [(0.1, 100, 128), (0.1, 25, 128)]:
        # Exact cosine budget (was previously approximated as eta*T*(N/B)/2,
        # which shifted planned factors {0.1,1/3,3,10} to ~{0.16,0.54,5,16}).
        S = float(exact_sum_lr(lr, T, B, scheduler='cosine'))
        for factor in [0.1, 1.0 / 3.0, 3.0, 10.0]:
            wd = C * factor / S
            cfgs.append(make_cfg('e5b', lr=lr, wd=float(f"{wd:.4g}"),
                                 epochs=T, batch_size=B))
    return cfgs


def build_e6b():
    """
    E6b: what momentum does to the optimum and to the generalization gap.

    The bounds carry a momentum factor, and the effective step is eta/(1-beta),
    so the predictions are that eta* and lambda* both scale with (1-beta). The
    lambda = 0 arm answers the reviewer's direct question of whether momentum
    helps generalization on its own. Training accuracy is recorded here, which
    the older CSVs do not contain, so the gap is actually measurable.
    """
    cfgs = []
    betas = [0.0, 0.5, 0.9, 0.99]
    for beta in betas:
        scale = 1.0 - beta
        base_etas = [0.02, 0.05, 0.1, 0.2]
        etas = [round(e * scale / 0.1, 5) for e in base_etas] if beta > 0 else base_etas
        for eta in etas:
            if eta <= 0:
                continue
            for wd in [0.0, round(1e-3 * scale / 0.1, 8)]:
                cfgs.append(make_cfg('e6b', lr=eta, wd=wd, epochs=100, momentum=beta))
    return cfgs


BUILDERS = {
    'e1_prelim': build_e1_prelim,
    'e1_fine': build_e1_fine,
    'e1_full': build_e1_full,
    'e1_rescue': build_e1_rescue,
    'e2b': build_e2b,
    'e4': build_e4,
    'e5b': build_e5b,
    'e6b': build_e6b,
}

# The E1 series measures how the optimum moves with the training length, so
# every point on it has to come from the same code path. The legacy runs used a
# loss scaler that was rebuilt every epoch, and folding them in would put a
# procedural difference on exactly the axis being measured.
NO_LEGACY_REUSE = {'e1_prelim', 'e1_fine', 'e1_full', 'e1_rescue'}


def main():
    parser = argparse.ArgumentParser(description='NeurIPS 17464 rebuttal experiments')
    parser.add_argument('--exp', required=True,
                        choices=sorted(list(BUILDERS.keys()) + ['e3']))
    parser.add_argument('--gpus', type=str, default='all')
    parser.add_argument('--workers_per_gpu', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='dataloader workers per training run')
    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV))
    parser.add_argument('--dry_run', action='store_true',
                        help='list the configurations and the budget, then exit')
    args = parser.parse_args()

    gpu_ids = None if args.gpus == 'all' else parse_gpu_ids(args.gpus)
    logger = get_logger(f"nips26_{args.exp}")
    logger.info(f"Experiment {args.exp} | GPUs {args.gpus} | "
                f"{args.workers_per_gpu} workers/GPU | CSV {args.csv}")

    def build(exp):
        cfgs = build_e3_ladder() if exp == 'e3' else BUILDERS[exp]()
        for cfg in cfgs:
            cfg['num_workers'] = args.num_workers
        return cfgs

    use_legacy = args.exp not in NO_LEGACY_REUSE

    if args.dry_run:
        cfgs = build(args.exp)
        done = load_done_keys(args.csv)
        if use_legacy:
            done |= load_legacy_done_keys()
        pending = [c for c in cfgs if cfg_key(c) not in done]
        print(json.dumps(pending, indent=2))
        print(f"\n{len(cfgs)} configs, {len(cfgs) - len(pending)} done, "
              f"{len(pending)} pending, "
              f"{sum(c['epochs'] for c in pending) / 100:.1f} units")
        return

    start = time.time()
    if args.exp == 'e3':
        run_e3(gpu_ids, args.workers_per_gpu, args.csv, logger,
               num_workers=args.num_workers)
    else:
        run_grid(build(args.exp), gpu_ids, args.workers_per_gpu,
                 args.csv, logger, label=args.exp.upper(), use_legacy=use_legacy)
    logger.info(f"{args.exp} total wall time {(time.time() - start) / 60:.1f} min")


if __name__ == '__main__':
    main()
