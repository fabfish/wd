"""
E8/E9: fixed weight decay vs scheduled weight decay (AdamW/SGDW schedule shapes).

Modes:
  --lr_mode const   LR fixed; only lambda is scheduled (original E8)
  --lr_mode joint   same multiplier m(t) applied to both lr and weight_decay
                    (AdamW/SGDW SetScheduleMultiplier); no CosineAnnealingLR
  --lr_mode cosine  CosineAnnealingLR on eta; lambda constant (E4-style baseline)
  --lr_mode cos_shape  cosine eta driven manually, lambda follows an arbitrary
                    shape m_lambda(t). Used by E9 so that the cumulative
                    contraction sum_t eta_t lambda_t is known analytically
                    before the run starts. Writes scheduler='cosine' to the CSV
                    because the learning-rate trajectory is bit-identical to
                    CosineAnnealingLR (both step once per epoch).

E9 answers reviewer xkCF's follow-up that the `joint` arm gives
eta_t*lambda_t = eta_0*lambda_0*m(t)^2 and therefore does *not* preserve the
coupling our rule is about. Two sweeps:

  --sweep iso      lambda_t = lambda_0 * eta_0/eta_t, i.e. eta_t*lambda_t held
                   constant (clipped in the cosine tail, see ISO_M_FLOOR)
  --sweep matched  every lambda shape is rescaled so that all methods share the
                   same cumulative contraction sum_t eta_t lambda_t

Phases:
  --phase sgd|sgdm|all
  --sweep joint|long|e4_baselines|const|iso|matched  (what grid to build)

Usage:
  python rebuttal/run_nips26_wd_sched.py --sweep joint --phase sgdm --gpus 1,2,3
  python rebuttal/run_nips26_wd_sched.py --sweep iso     --phase sgdm --gpus 1,2,3
  python rebuttal/run_nips26_wd_sched.py --sweep matched --phase sgdm --gpus 1,2,3
"""
import argparse
import csv
import math
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
DATA_DIR = str(Path(__file__).resolve().parent.parent / 'data')

RUN_KEY = ['model', 'dataset', 'method', 'batch_size', 'lr', 'wd', 'momentum',
           'epochs', 'scheduler', 'seed', 'wd_sched']

CSV_FIELDS = RUN_KEY + [
    'exp', 'sum_lr', 'best_test_acc', 'final_test_acc', 'final_train_loss',
    'final_train_acc', 'final_test_loss', 'diverged', 'epochs_run', 'wall_time',
]

DIVERGENCE_LOSS_THRESHOLD = 2.0 * math.log(100.0)
DEFAULT_NUM_WORKERS = 5

WD_SCHEDULES = ['fixed', 'cosine', 'linear', 'step', 'cosine_restarts']
LAMBDA0_GRID = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
LONG_SCHEDULES = ['fixed', 'step', 'cosine_restarts']
LONG_LAMBDAS = [5e-4, 1e-3, 2e-3]

# SGDR-style restart params (AdamW paper): first cycle Te, then Te*Tmult.
RESTART_TE = 50
RESTART_TMULT = 2

# E4-ours reference lambda (C / sum_lr at R18 B=128 eta=0.1 T=100 cosine).
E4_OURS_LAMBDA = 5.982e-4

CIFAR100_TRAIN_N = 50000

# --- E9 (xkCF follow-up) -------------------------------------------------
# iso-product arm: lambda_t = lambda_0 / m_cos(t) diverges as the cosine tail
# goes to zero, so the multiplier is capped at 1/ISO_M_FLOOR. The product
# eta_t*lambda_t is then exactly constant while eta_t >= ISO_M_FLOOR*eta_0 and
# decays like the learning rate afterwards. The cap is part of the reported
# protocol, not a silent fix.
ISO_M_FLOOR = 0.1

# matched-contraction arm: shapes compared at a common budget sum_t eta_t lambda_t.
E9_SHAPES = ['fixed', 'cosine', 'linear', 'step', 'iso_product']
E9_BUDGET_FACTORS = [1.0 / 3.0, 1.0, 3.0]


def wd_multiplier(wd_sched, epoch, epochs, te=RESTART_TE, tmult=RESTART_TMULT):
    """Return m(t) in [0, 1] for a 0-based epoch index."""
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
        remaining = t
        Ti = float(te)
        while remaining >= Ti:
            remaining -= Ti
            Ti *= float(tmult)
        return 0.5 * (1.0 + math.cos(math.pi * remaining / Ti))
    raise ValueError(f'unknown wd_sched={wd_sched}')


def make_multiplier_fns(wd_sched, lr0, lambda0, epochs):
    """Return (lr_fn, wd_fn) for joint or wd-only scheduling."""
    def wd_fn(epoch):
        return float(lambda0) * wd_multiplier(wd_sched, epoch, epochs)

    def lr_fn(epoch):
        return float(lr0) * wd_multiplier(wd_sched, epoch, epochs)

    return lr_fn, wd_fn


# --- E9 helpers ----------------------------------------------------------

def cosine_lr_multiplier(epoch, epochs):
    """CosineAnnealingLR's per-epoch multiplier, stepped once per epoch."""
    return 0.5 * (1.0 + math.cos(math.pi * float(epoch) / float(epochs)))


def wd_shape_multiplier(wd_sched, epoch, epochs, m_floor=ISO_M_FLOOR):
    """
    m_lambda(t) for the E9 shapes, on top of a cosine learning rate.

    'iso_product' is reviewer xkCF's suggestion lambda_t = lambda_0*eta_0/eta_t,
    which makes eta_t*lambda_t constant. The 1/m_cos blow-up in the tail is
    capped at 1/m_floor.
    """
    if wd_sched == 'iso_product':
        m = cosine_lr_multiplier(epoch, epochs)
        return 1.0 / max(m, float(m_floor))
    return wd_multiplier(wd_sched, epoch, epochs)


def make_cos_shape_fns(wd_sched, lr0, lambda0, epochs):
    """Cosine eta driven by hand, lambda following m_lambda(t)."""
    def lr_fn(epoch):
        return float(lr0) * cosine_lr_multiplier(epoch, epochs)

    def wd_fn(epoch):
        return float(lambda0) * wd_shape_multiplier(wd_sched, epoch, epochs)

    return lr_fn, wd_fn


def contraction_sum(lr0, lambda0, epochs, batch_size, wd_sched,
                    n=CIFAR100_TRAIN_N):
    """
    sum_t eta_t*lambda_t accumulated over optimizer steps, for a cosine eta.

    Computed the same way the training loop applies the two schedules (one value
    per epoch, steps_per_epoch steps each), so the budget used to pick lambda_0
    is the budget the run actually spends.
    """
    steps = math.ceil(float(n) / float(batch_size))
    total = 0.0
    for epoch in range(int(epochs)):
        eta = float(lr0) * cosine_lr_multiplier(epoch, epochs)
        lam = float(lambda0) * wd_shape_multiplier(wd_sched, epoch, epochs)
        total += eta * lam
    return total * steps


def solve_lambda0_for_budget(budget, lr0, epochs, batch_size, wd_sched,
                             n=CIFAR100_TRAIN_N, sig=4):
    """lambda_0 such that contraction_sum(...) == budget (linear in lambda_0)."""
    unit = contraction_sum(lr0, 1.0, epochs, batch_size, wd_sched, n=n)
    if unit <= 0:
        raise ValueError(f'degenerate shape {wd_sched}')
    return float(f'%.{sig}g' % (float(budget) / unit))


def e9_budget_anchor(lr0=0.1, epochs=100, batch_size=128, n=CIFAR100_TRAIN_N):
    """
    The reference contraction budget C = lambda_ref * sum_t eta_t.

    Anchored on E4_OURS_LAMBDA rather than refitted, so that the `fixed` shape
    at budget 1.0*C is *literally* the E4/E8 baseline run already in the CSV
    (same RUN_KEY) and is reused instead of retrained. `analysis.nips26_lib.
    reference_point()['C']` agrees with this to 0.3%.
    """
    return contraction_sum(lr0, E4_OURS_LAMBDA, epochs, batch_size, 'fixed', n=n)


def cfg_key(cfg):
    out = []
    for field in RUN_KEY:
        value = cfg[field]
        out.append(f"{float(value):.10g}" if isinstance(value, float) else str(value))
    return tuple(out)


def ensure_csv_schema(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if 'wd_sched' in fieldnames:
            return
        rows = list(reader)
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
                ws = str(cfg.get('wd_sched', '')).strip()
                # Legacy cosine-LR rows have blank wd_sched; treat as fixed WD.
                if not ws:
                    if str(cfg.get('scheduler', '')) == 'cosine':
                        cfg['wd_sched'] = 'fixed'
                    else:
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
             model='resnet18', seed=42, dataset='cifar100',
             lr_mode='const', exp=None):
    """
    lr_mode:
      const     -> scheduler='const'   (fixed LR; schedule only WD)
      joint     -> scheduler='joint'   (same m(t) on LR and WD)
      cosine    -> scheduler='cosine'  (CosineAnnealingLR; WD constant)
      cos_shape -> scheduler='cosine'  (cosine LR by hand; WD follows a shape)
    """
    if momentum > 0:
        method = 'SGDM+WD' if lambda0 > 0 else 'SGDM'
    else:
        method = 'SGD+WD' if lambda0 > 0 else 'SGD'
    if exp is None:
        if lr_mode == 'joint':
            exp = 'e8_joint'
        elif lr_mode == 'cosine':
            exp = 'e8_e4_baseline'
        elif lr_mode == 'cos_shape':
            exp = 'e9_cos_shape'
        else:
            exp = 'e8_wd_sched'
    # cos_shape is numerically the same learning-rate trajectory as
    # CosineAnnealingLR, so it shares the 'cosine' scheduler label. That is what
    # lets the matched-contraction `fixed` arm dedup against the existing
    # E4/E8 baseline row instead of retraining it.
    scheduler_col = {
        'const': 'const', 'joint': 'joint',
        'cosine': 'cosine', 'cos_shape': 'cosine',
    }.get(lr_mode, 'const')
    return {
        'model': model, 'dataset': dataset, 'method': method,
        'batch_size': int(batch_size), 'lr': float(lr), 'wd': float(lambda0),
        'momentum': float(momentum), 'epochs': int(epochs),
        'scheduler': scheduler_col,
        'seed': int(seed), 'wd_sched': wd_sched, 'exp': exp,
        'lr_mode': lr_mode,
        'num_workers': DEFAULT_NUM_WORKERS,
    }


def build_cfgs(sweep, phase):
    """
    sweep:
      const         original E8 (fixed LR, all 5 WD schedules)
      joint         joint multiplier, T=100, SGDM schedules × λ0 grid
      long          joint, T=200, fixed/step/restarts × thin λ0
      e4_baselines  cosine LR + fixed λ in {E4-ours, 5e-4} (+ λ0 grid optional)
      iso           E9a: cosine LR, λ_t = λ0·η0/η_t (constant η_t·λ_t) × λ0 grid
      matched       E9b: cosine LR, every shape rescaled to a common budget
                    sum_t η_t·λ_t ∈ {C/3, C, 3C}
    """
    momenta = []
    if phase in ('sgd', 'all'):
        momenta.append(0.0)
    if phase in ('sgdm', 'all'):
        momenta.append(0.9)

    cfgs = []
    if sweep == 'const':
        for momentum in momenta:
            for wd_sched in WD_SCHEDULES:
                for lam0 in LAMBDA0_GRID:
                    cfgs.append(make_cfg(wd_sched, lam0, momentum, lr_mode='const'))
    elif sweep == 'joint':
        for momentum in momenta:
            for wd_sched in WD_SCHEDULES:
                for lam0 in LAMBDA0_GRID:
                    cfgs.append(make_cfg(
                        wd_sched, lam0, momentum, epochs=100, lr_mode='joint',
                        exp='e8_joint'))
    elif sweep == 'long':
        for momentum in momenta:
            for wd_sched in LONG_SCHEDULES:
                for lam0 in LONG_LAMBDAS:
                    cfgs.append(make_cfg(
                        wd_sched, lam0, momentum, epochs=200, lr_mode='joint',
                        exp='e8_long'))
    elif sweep == 'e4_baselines':
        # Cosine LR, constant lambda: E4-ours and default, plus the λ0 grid for
        # a fair peak comparison under the same LR schedule.
        lams = sorted(set(LAMBDA0_GRID + [E4_OURS_LAMBDA, 5e-4]))
        for momentum in momenta:
            for lam0 in lams:
                cfgs.append(make_cfg(
                    'fixed', lam0, momentum, epochs=100, lr_mode='cosine',
                    exp='e8_e4_baseline'))
    elif sweep == 'iso':
        for momentum in momenta:
            for lam0 in LAMBDA0_GRID:
                cfgs.append(make_cfg(
                    'iso_product', lam0, momentum, epochs=100,
                    lr_mode='cos_shape', exp='e9_iso'))
    elif sweep == 'matched':
        anchor = e9_budget_anchor()
        for momentum in momenta:
            for factor in E9_BUDGET_FACTORS:
                budget = factor * anchor
                for wd_sched in E9_SHAPES:
                    lam0 = solve_lambda0_for_budget(
                        budget, 0.1, 100, 128, wd_sched)
                    cfgs.append(make_cfg(
                        wd_sched, lam0, momentum, epochs=100,
                        lr_mode='cos_shape', exp='e9_matched'))
    else:
        raise ValueError(f'unknown sweep={sweep}')
    return cfgs


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

    optimizer = optim.SGD(
        model.parameters(), lr=cfg['lr'],
        momentum=cfg['momentum'], weight_decay=cfg['wd'],
    )

    lr_mode = cfg.get('lr_mode', cfg['scheduler'])
    lr_fn = wd_fn = None
    lr_sched = None

    if lr_mode == 'joint':
        lr_fn, wd_fn = make_multiplier_fns(
            cfg['wd_sched'], cfg['lr'], cfg['wd'], cfg['epochs'])
    elif lr_mode == 'const':
        _, wd_fn = make_multiplier_fns(
            cfg['wd_sched'], cfg['lr'], cfg['wd'], cfg['epochs'])
    elif lr_mode == 'cos_shape':
        # Cosine eta by hand (scheduler=None, or it would anneal twice) so that
        # the contraction budget is known analytically before the run.
        lr_fn, wd_fn = make_cos_shape_fns(
            cfg['wd_sched'], cfg['lr'], cfg['wd'], cfg['epochs'])
    elif lr_mode == 'cosine':
        lr_sched = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        # constant lambda; still set once via wd_fn for epoch-0 consistency
        wd_fn = (lambda epoch, lam=cfg['wd']: float(lam))
    else:
        raise ValueError(f"unknown lr_mode={lr_mode}")

    tag = (f"[{cfg['exp']}|{cfg['method']}|{lr_mode}|wd_sched={cfg['wd_sched']}|"
           f"T={cfg['epochs']}|lr0={cfg['lr']:g}|lam0={cfg['wd']:g}] ")
    print(tag + 'start', flush=True)

    start = time.time()
    metrics = train_model_ext(
        model, train_loader, test_loader, optimizer, scheduler=lr_sched,
        device=device, epochs=cfg['epochs'], use_amp=True,
        log_interval=max(cfg['epochs'] // 4, 1),
        divergence_loss_threshold=DIVERGENCE_LOSS_THRESHOLD,
        divergence_check_epoch=3, tag=tag,
        wd_schedule_fn=wd_fn,
        lr_schedule_fn=lr_fn,
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
            logger.info(
                f"  pending: {c['scheduler']} mom={c['momentum']} "
                f"wd_sched={c['wd_sched']} lam0={c['wd']:g} T={c['epochs']}")
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
                f"{row['scheduler']} mom={row['momentum']} "
                f"wd_sched={row['wd_sched']} lam0={row['wd']:g} T={row['epochs']} "
                f"-> best={row['best_test_acc']:.2f} "
                f"diverged={row['diverged']} ({row['wall_time']:.0f}s)")

    start = time.time()
    scheduler.run_tasks([(c,) for c in pending], run_one, on_complete=on_complete)
    logger.info(f'{label}: finished {len(collected)}/{len(pending)} in '
                f'{(time.time() - start) / 60:.1f} min')
    return collected


def main():
    parser = argparse.ArgumentParser(description='E8/E9 scheduled weight-decay sweep')
    parser.add_argument('--sweep',
                        choices=['const', 'joint', 'long', 'e4_baselines',
                                 'iso', 'matched'],
                        default='const')
    parser.add_argument('--phase', choices=['sgd', 'sgdm', 'all'], default='sgdm')
    parser.add_argument('--gpus', type=str, default='1,2,3')
    parser.add_argument('--workers_per_gpu', type=int, default=2)
    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV))
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    logger = get_logger(f'nips26_e8_{args.sweep}_{args.phase}')
    gpu_ids = parse_gpu_ids(args.gpus)
    cfgs = build_cfgs(args.sweep, args.phase)
    logger.info(f'sweep={args.sweep} phase={args.phase} gpus={gpu_ids} '
                f'workers_per_gpu={args.workers_per_gpu} n_cfgs={len(cfgs)}')
    if args.sweep in ('iso', 'matched'):
        anchor = e9_budget_anchor()
        logger.info(f'E9 contraction anchor C = {anchor:.6g} '
                    f'(= {E4_OURS_LAMBDA:g} x sum_lr)')
        for c in cfgs:
            spent = contraction_sum(c['lr'], c['wd'], c['epochs'],
                                    c['batch_size'], c['wd_sched'])
            logger.info(f"  plan mom={c['momentum']} shape={c['wd_sched']:>14s} "
                        f"lam0={c['wd']:.4g} sum_eta_lambda={spent:.4g} "
                        f"({spent / anchor:.3f} C)")

    run_grid._dry_run = args.dry_run
    ensure_csv_schema(args.csv)
    run_grid(cfgs, gpu_ids, args.workers_per_gpu, args.csv, logger,
             label=f'E8-{args.sweep}-{args.phase}')


if __name__ == '__main__':
    main()
