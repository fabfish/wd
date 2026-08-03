"""
E10: held-out test of the constant C across MLP widths.

Reviewer xkCF (2026-08-03) asked for a test where C is estimated in one setting
and then used to predict lambda* in another. Width is the cleanest such knob: it
changes the parameter count by more than an order of magnitude while keeping the
data, the optimizer and the schedule identical.

Three phases, deliberately separated so the prediction is verifiably blind:

  ladder   fit C on hidden_dim in {128, 256, 512} (MNIST) / {256, 512} (CIFAR-10)
  predict  regress log C on log width, extrapolate, and WRITE THE PREDICTED
           lambda* TO DISK. Refuses to overwrite, and warns loudly if held-out
           oracle rows already exist.
  heldout  train the predicted lambda*, the three competing rules, and a full
           oracle grid at the larger widths, then report the gap to the oracle

Usage:
  PY=/home/yzy/.conda/envs/trace/bin/python
  $PY -m mlp_wd.scripts.run_e10_c_width --dataset mnist   --phases ladder
  $PY -m mlp_wd.scripts.run_e10_c_width --dataset mnist   --phases predict
  $PY -m mlp_wd.scripts.run_e10_c_width --dataset mnist   --phases heldout
  $PY -m mlp_wd.scripts.run_e10_c_width --dataset cifar10 --phases ladder,predict,heldout
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlp_wd.analysis.e10_c_width import (  # noqa: E402
    DATASET_N, RULES, C_by_width, fit_C_grid, fit_C_vs_width, geo,
    predict_C, predictions_path, read_predictions, rule_lambda,
    write_predictions,
)
from mlp_wd.mlp_core.gpu_scheduler import parse_gpu_ids  # noqa: E402
from mlp_wd.mlp_core.grid import run_grid  # noqa: E402
from mlp_wd.mlp_core.io import load_completed_keys  # noqa: E402
from mlp_wd.mlp_core.runner import get_task_key  # noqa: E402

TABLE_DIR = REPO_ROOT / 'rebuttal' / 'nips_rebuttal' / '_data'

# Per-dataset protocol, each matched to a set of runs we already have so that the
# widest ladder rung is free.
SETTINGS = {
    'mnist': dict(
        epochs=20, batch_size=128, num_layers=3, seed=42,
        lrs=[0.05, 0.1, 0.2],
        wds=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        momenta=[0.0, 0.9],
        ladder_widths=[128, 256, 512],
        heldout_widths=[1024, 2048],
        output='mlp_wd/outputs/results/e10_mnist.csv',
        history_dir='mlp_wd/outputs/history/e10_mnist',
        # h=512 rung comes from the E5c phase-B grid, same protocol.
        reuse=['mlp_wd/outputs/results/e5c_mnist.csv'],
    ),
    'cifar10': dict(
        epochs=30, batch_size=128, num_layers=3, seed=42,
        lrs=[0.03, 0.1, 0.3],
        wds=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        momenta=[0.9],
        ladder_widths=[256, 512],
        heldout_widths=[1024],
        output='mlp_wd/outputs/results/e10_cifar10.csv',
        history_dir='mlp_wd/outputs/history/e10_cifar10',
        # h=512 rung comes from exp2.csv, which covers this exact lr x wd grid.
        reuse=['mlp_wd/outputs/results/exp2.csv'],
    ),
}

REF_LR_FOR_PRODUCT = {'mnist': 0.1, 'cifar10': 0.1}


def method_name(momentum, wd):
    base = 'SGDM' if momentum > 0 else 'SGD'
    return f'{base}+WD' if wd > 0 else base


def make_row(*, momentum, lr, wd, cfg, hidden_dim, tag):
    return {
        'method': method_name(momentum, wd),
        'batch_size': cfg['batch_size'],
        'lr': float(lr),
        'wd': float(wd),
        'momentum': float(momentum),
        'epochs': cfg['epochs'],
        'seed': cfg['seed'],
        'hidden_dim': int(hidden_dim),
        'run_tag': tag,
    }


def probe_rows(cfg, widths, prefix):
    rows = []
    for h in widths:
        for mom in cfg['momenta']:
            for lr in cfg['lrs']:
                for wd in cfg['wds']:
                    rows.append(make_row(
                        momentum=mom, lr=lr, wd=wd, cfg=cfg, hidden_dim=h,
                        tag=f'{prefix}_h{h}_m{mom}_lr{lr}_wd{wd}'))
    return rows


# --------------------------------------------------------------------------
# reading results across our own CSV plus the reusable legacy ones
# --------------------------------------------------------------------------

def load_pool(cfg, dataset):
    """Our own results plus the legacy CSVs whose protocol matches exactly."""
    frames = []
    for path in [cfg['output']] + list(cfg['reuse']):
        full = REPO_ROOT / path
        if not full.exists():
            continue
        df = pd.read_csv(full)
        if df.empty:
            continue
        if 'dataset' in df.columns:
            df = df[df['dataset'] == dataset]
        for col, default in (('use_bn', 0), ('norm_output', 0)):
            if col not in df.columns:
                df[col] = default
        df = df[(df['use_bn'].astype(int) == 0) & (df['norm_output'].astype(int) == 0)]
        df['source'] = Path(path).name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool['seed'].astype(int) == int(cfg['seed'])]
    return pool


def external_keys(cfg, dataset):
    """Task keys already present in the reusable CSVs, so we do not re-run them."""
    keys = set()
    for path in cfg['reuse']:
        full = REPO_ROOT / path
        if full.exists():
            keys |= load_completed_keys(full)
    return keys


def drop_reusable(rows, cfg, dataset):
    done = external_keys(cfg, dataset)
    if not done:
        return rows, 0
    kept, skipped = [], 0
    for row in rows:
        key = get_task_key(
            method=row['method'], dataset=dataset,
            num_layers=cfg['num_layers'], hidden_dim=row['hidden_dim'],
            batch_size=row['batch_size'], lr=row['lr'], wd=row['wd'],
            momentum=row['momentum'], epochs=row['epochs'], seed=row['seed'],
        )
        if key in done:
            skipped += 1
        else:
            kept.append(row)
    return kept, skipped


def submit(rows, cfg, dataset, args, label):
    rows, reused = drop_reusable(rows, cfg, dataset)
    print(f'[e10:{label}] {len(rows)} to submit, {reused} reused from legacy CSVs')
    if args.dry_run:
        for r in rows[:8]:
            print('   ', r)
        if len(rows) > 8:
            print(f'    ... and {len(rows) - 8} more')
        return
    if not rows:
        return
    # hidden_dim varies within the phase, so group by width: run_grid takes one
    # hidden_dim per call (it is a model-construction argument, not a grid axis).
    for h in sorted({r['hidden_dim'] for r in rows}):
        sub = [dict(r) for r in rows if r['hidden_dim'] == h]
        for r in sub:
            r.pop('hidden_dim')
        print(f'[e10:{label}] hidden_dim={h}: {len(sub)} runs')
        run_grid(
            sub,
            output_file=REPO_ROOT / cfg['output'],
            history_dir=REPO_ROOT / cfg['history_dir'],
            dataset=dataset,
            hidden_dim=h,
            num_layers=cfg['num_layers'],
            gpu_ids=args.gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
            log_every=0,
            loader_workers=args.loader_workers,
        )


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------

def phase_ladder(cfg, dataset, args):
    rows = probe_rows(cfg, cfg['ladder_widths'], 'e10L')
    submit(rows, cfg, dataset, args, 'ladder')


def phase_predict(cfg, dataset, args):
    n = DATASET_N[dataset]
    pool = load_pool(cfg, dataset)
    opt = fit_C_grid(
        pool, lrs=cfg['lrs'], wds=cfg['wds'], epochs=cfg['epochs'],
        batch_size=cfg['batch_size'], n=n, num_layers=cfg['num_layers'],
        widths=cfg['ladder_widths'],
    )
    cw = C_by_width(opt)
    missing = set(cfg['ladder_widths']) - set(cw['hidden_dim'])
    if missing:
        raise RuntimeError(f'ladder incomplete, missing widths {sorted(missing)}')
    fits = fit_C_vs_width(cw)

    # Warn if the held-out grids already exist: then the "prediction" is not blind.
    held = pool[pool['hidden_dim'].astype(int).isin(cfg['heldout_widths'])]
    if not held.empty:
        print(f'[e10:predict] WARNING: {len(held)} rows already exist at the '
              f'held-out widths {cfg["heldout_widths"]}. The prediction below is '
              f'NOT blind; delete them or report this.')

    ref_lr = REF_LR_FOR_PRODUCT[dataset]
    payload = {
        'dataset': dataset,
        'written_at': dt.datetime.now().isoformat(timespec='seconds'),
        'protocol': {k: cfg[k] for k in
                     ('epochs', 'batch_size', 'num_layers', 'seed', 'lrs', 'wds',
                      'momenta', 'ladder_widths', 'heldout_widths')},
        'n_train': n,
        'blind': bool(held.empty),
        'per_momentum': {},
    }
    for mom in cfg['momenta']:
        fit = fits[float(mom)]
        ref = opt[(np.isclose(opt['momentum'], mom))
                  & (np.isclose(opt['lr'], ref_lr))
                  & (opt['hidden_dim'] == max(cfg['ladder_widths']))]
        product_ref = float(ref['lr'].iloc[0] * ref['wd_interp'].iloc[0]) \
            if not ref.empty else float('nan')
        entry = {
            'C_ladder': {str(w): c for w, c in zip(fit['widths'], fit['C'])},
            'slope': fit['slope'], 'intercept': fit['intercept'],
            'slope_ci': [fit['lo'], fit['hi']],
            'product_ref': product_ref,
            'C_pred': {}, 'lambda_pred': {},
        }
        for h in cfg['heldout_widths']:
            C_pred = predict_C(fit, h)
            entry['C_pred'][str(h)] = C_pred
            entry['lambda_pred'][str(h)] = {
                f'{lr:g}': {
                    rule: rule_lambda(rule, lr=lr, epochs=cfg['epochs'],
                                      batch_size=cfg['batch_size'], n=n,
                                      C_pred=C_pred, product_ref=product_ref)
                    for rule in RULES
                } for lr in cfg['lrs']
            }
        payload['per_momentum'][f'{mom:g}'] = entry

    path = predictions_path(dataset)
    if path.exists() and not args.overwrite_predictions:
        old = read_predictions(dataset)
        print(f'[e10:predict] {path.name} already exists (written '
              f'{old.get("written_at")}); keeping it. Pass '
              f'--overwrite_predictions to replace.')
    else:
        write_predictions(payload, dataset)
        print(f'[e10:predict] wrote {path}')

    opt.to_csv(TABLE_DIR / f'e10_ladder_optima_{dataset}.csv', index=False)
    cw.to_csv(TABLE_DIR / f'e10_C_by_width_{dataset}.csv', index=False)
    for mom, fit in fits.items():
        print(f'[e10:predict] mom={mom:g}: C {fit["C"]} at widths '
              f'{fit["widths"]}, slope {fit["slope"]:+.3f} '
              f'[{fit["lo"]:+.3f}, {fit["hi"]:+.3f}]')
        for h in cfg['heldout_widths']:
            print(f'    predicted C({h}) = {predict_C(fit, h):.4g}')


def phase_heldout(cfg, dataset, args):
    pred = read_predictions(dataset)
    rows = probe_rows(cfg, cfg['heldout_widths'], 'e10H')  # oracle grids
    seen = {(r['hidden_dim'], r['momentum'], r['lr'], float(f"{r['wd']:.6g}"))
            for r in rows}
    for mom_key, entry in pred['per_momentum'].items():
        mom = float(mom_key)
        for h_key, per_lr in entry['lambda_pred'].items():
            h = int(h_key)
            for lr_key, per_rule in per_lr.items():
                lr = float(lr_key)
                for rule, lam in per_rule.items():
                    lam = float(f'{float(lam):.6g}')
                    if not np.isfinite(lam) or lam <= 0:
                        continue
                    key = (h, mom, lr, lam)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(make_row(
                        momentum=mom, lr=lr, wd=lam, cfg=cfg, hidden_dim=h,
                        tag=f'e10H_{rule}_h{h}_m{mom:g}_lr{lr:g}_wd{lam:g}'))
    submit(rows, cfg, dataset, args, 'heldout')


PHASES = {'ladder': phase_ladder, 'predict': phase_predict,
          'heldout': phase_heldout}


def main():
    ap = argparse.ArgumentParser(description='E10 C-vs-width held-out test')
    ap.add_argument('--dataset', default='mnist', choices=sorted(SETTINGS))
    ap.add_argument('--phases', default='ladder,predict,heldout')
    ap.add_argument('--gpus', default='1,2,3')
    ap.add_argument('--workers_per_gpu', type=int, default=6)
    ap.add_argument('--loader_workers', type=int, default=0)
    ap.add_argument('--overwrite_predictions', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    args.gpu_ids = parse_gpu_ids(args.gpus) if args.gpus != 'all' else None
    cfg = SETTINGS[args.dataset]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    for name in [p.strip() for p in args.phases.split(',') if p.strip()]:
        if name not in PHASES:
            raise SystemExit(f'unknown phase {name}')
        print(f'\n===== E10 {args.dataset} phase={name} =====')
        PHASES[name](cfg, args.dataset, args)
    print('[e10] done')


if __name__ == '__main__':
    main()
