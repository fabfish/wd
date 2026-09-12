#!/usr/bin/env python
"""
E12 full ladder table: every (setting, phase, wd_sched) x budget rung, with
realized budget in setting-local C units (E11 e11_matched_table style).

Rows come from the E12 CSV (e12_fixed / e12_matched) plus the E11 CSV for
R18/C100 (read-only). R50 rows appear as they land.

Writes e12_multi_setting/tables/e12_full_ladder_table.md (+ .csv).
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
E12 = ROOT / 'e12_multi_setting'
CSV = E12 / 'results' / 'e12_runs.csv'
E11_CSV = ROOT / 'e11_raise_up' / 'results' / 'nips26_e11_runs.csv'
DATA_DIR = E12 / 'tables'

ETA0, T_EPOCHS, BATCH = 0.1, 100, 128
TRAIN_N = {'cifar100': 50000, 'cifar10': 50000, 'mnist': 60000}
ISO_M_FLOOR = 0.1
SHAPES = ['fixed', 'linear_up', 'linear', 'iso_product']
SHAPE_LABELS = {'fixed': 'fixed', 'linear_up': 'linear up',
                'linear': 'linear down', 'iso_product': 'iso up'}
PHASE_LABELS = {0.9: 'SGDM', 0.0: 'SGD'}


def _cos_m(t):
    return 0.5 * (1.0 + math.cos(math.pi * t / T_EPOCHS))


def _wd_m(sched, t):
    frac = t / T_EPOCHS
    if sched == 'fixed':
        return 1.0
    if sched == 'linear':
        return max(0.0, 1.0 - frac)
    if sched == 'linear_up':
        return min(1.0, frac)
    if sched == 'iso_product':
        return 1.0 / max(_cos_m(t), ISO_M_FLOOR)
    raise ValueError(sched)


def contraction_sum(sched, lam0, dataset):
    n = TRAIN_N[dataset]
    steps = math.ceil(n / BATCH)
    return steps * sum(ETA0 * _cos_m(t) * lam0 * _wd_m(sched, t)
                       for t in range(T_EPOCHS))


def load_and_label(csv_path, tag):
    df = pd.read_csv(csv_path)
    df['src'] = tag
    return df


def build_rows():
    frames = []
    if CSV.exists():
        frames.append(load_and_label(CSV, 'e12'))
    if E11_CSV.exists():
        frames.append(load_and_label(E11_CSV, 'e11'))
    df = pd.concat(frames, ignore_index=True)

    # normalize
    df = df[df['epochs'] == 100]
    df = df[np.isclose(df['lr'].astype(float), 0.1)]
    df = df[df['batch_size'] == 128]
    df = df[df['scheduler'] == 'cosine']
    df = df[df['wd'].astype(float) > 0]
    mom = df['momentum'].astype(float)
    df = df[(np.isclose(mom, 0.0)) | (np.isclose(mom, 0.9))]
    ws = df['wd_sched'].fillna('').astype(str).str.strip()
    df = df.assign(wd_sched_norm=ws.replace('', 'fixed'))
    df = df[df['wd_sched_norm'].isin(SHAPES)]

    # per-(setting, phase) anchor: fixed oracle seed 42, GRID runs only
    # (e4 predicted points like 9.62e-4 are excluded from the anchor pool so
    # 1C means "the best grid-searched const WD").
    anchors = {}
    fixed = df[df['wd_sched_norm'] == 'fixed']
    fixed = fixed[fixed['exp'] != 'e4']
    for (model, dataset, mom), g in fixed.groupby(['model', 'dataset', 'momentum']):
        g = g[g['seed'] == 42]
        if g.empty:
            continue
        peak = g.loc[g['best_test_acc'].idxmax()]
        anchors[(model, dataset, mom)] = float(peak['wd'])

    rows = []
    for (model, dataset, mom), g in df.groupby(['model', 'dataset', 'momentum']):
        if (model, dataset, mom) not in anchors:
            continue
        lam_ref = anchors[(model, dataset, mom)]
        for sched in SHAPES:
            h = g[g['wd_sched_norm'] == sched]
            if h.empty:
                continue
            # one row per (seed42) lambda rung
            for _, r in h[h['seed'] == 42].iterrows():
                lam = float(r['wd'])
                rows.append({
                    'setting': f'{model}/{dataset}',
                    'phase': PHASE_LABELS[mom],
                    'wd_sched': sched,
                    'lambda0': lam,
                    'budget_c': contraction_sum(sched, lam, dataset)
                                / contraction_sum('fixed', lam_ref, dataset),
                    'best_acc': float(r['best_test_acc']),
                    'diverged': int(r['diverged']),
                    'exp': r['exp'],
                    'src': r['src'],
                })
    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    tab = pd.DataFrame(rows)

    tab = pd.DataFrame(rows)
    tab = tab.sort_values(
        ['setting', 'phase', 'wd_sched', 'budget_c']).reset_index(drop=True)
    tab.to_csv(DATA_DIR / 'e12_full_ladder_table.csv', index=False)

    # markdown, E11 style: rung per row
    md = [
        'E12 full ladder: cosine LR, B=128, eta0=0.1, T=100, seed 42.',
        'Budget in setting-local C units (C = fixed-WD oracle lambda_ref of',
        'that setting/phase). R18/C100 rows come from the E11 CSV (read-only);',
        'R50/C100 rows appear as they land.',
        '',
        '| setting | phase | wd_sched | lambda0 | budget_C | best_acc | diverged | src |',
        '|---|---|---|---:|---:|---:|---:|---|',
    ]
    for _, r in tab.iterrows():
        md.append(
            f"| {r['setting']} | {r['phase']} | {SHAPE_LABELS[r['wd_sched']]} | "
            f"{r['lambda0']:.4g} | {r['budget_c']:.2f} | {r['best_acc']:.2f} | "
            f"{int(r['diverged'])} | {r['src']} |")
    out = DATA_DIR / 'e12_full_ladder_table.md'
    out.write_text('\n'.join(md) + '\n')
    print(f'wrote {out} ({len(tab)} rows)')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
