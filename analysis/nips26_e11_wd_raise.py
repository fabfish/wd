"""
Analyze E11: raise-up lambda schedules under a cosine learning rate.

Sections:
  1. SGDM peak comparison (raise-up shapes vs fixed-lambda control vs E9 iso).
  2. SGD (momentum=0) same comparison; the fixed control grid comes from the
     e4_baselines sgd phase + legacy e6b rows.
  3. Matched-budget arm (e11_matched): raise-up shapes rescaled to
     sum_t eta_t*lambda_t in {C/3, C, 3C} -- separates budget from shape.
  4. Multi-seed arm (e11_raise_ms + deduped grid rows): mean +- std over seeds
     for the per-shape peak configs and the fixed control.

Reads the E11-dedicated CSV (a superset copy of nips26_runs.csv; the original
file is frozen).

Writes:
  outputs/plots/nips26/e11_wd_raise_{sgdm,sgd}.png
  rebuttal/nips_rebuttal/_data/e11_wd_raise_peaks.csv
  rebuttal/nips_rebuttal/_data/e11_wd_raise_table.md
  rebuttal/nips_rebuttal/_data/e11_matched_peaks.csv
  rebuttal/nips_rebuttal/_data/e11_matched_table.md
  rebuttal/nips_rebuttal/_data/e11_multiseed.csv
  rebuttal/nips_rebuttal/_data/e11_multiseed_table.md
"""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'rebuttal' / 'results' / 'nips26_e11_runs.csv'
PLOT_DIR = ROOT / 'outputs' / 'plots' / 'nips26'
DATA_DIR = ROOT / 'rebuttal' / 'nips_rebuttal' / '_data'

RAISE_SCHEDULES = ['linear_up', 'cosine_up', 'step_up']
LAMBDA0 = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
# (wd_sched, lambda0) peak configs replicated across seeds in the raise_ms arm.
PEAK_CONFIGS = [('linear_up', 5e-3), ('cosine_up', 5e-3), ('step_up', 1e-2),
                ('fixed', 6e-4)]

# --- contraction-budget math -------------------------------------------------
# Pure-python copies of the schedule math in rebuttal/run_nips26_wd_sched.py
# (kept dependency-free so this script never imports torch).
ETA0, T_EPOCHS, BATCH = 0.1, 100, 128
CIFAR100_TRAIN_N = 50000
ISO_M_FLOOR = 0.1
E4_OURS_LAMBDA = 5.982e-4


def _cos_m(t):
    return 0.5 * (1.0 + math.cos(math.pi * t / T_EPOCHS))


def _wd_m(sched, t):
    frac = t / T_EPOCHS
    if sched == 'fixed':
        return 1.0
    if sched == 'cosine':
        return 0.5 * (1.0 + math.cos(math.pi * frac))
    if sched == 'linear':
        return max(0.0, 1.0 - frac)
    if sched == 'step':
        return 1.0 if frac < 0.5 else (0.1 if frac < 0.75 else 0.01)
    if sched == 'linear_up':
        return min(1.0, frac)
    if sched == 'cosine_up':
        return 0.5 * (1.0 - math.cos(math.pi * frac))
    if sched == 'step_up':
        return 0.01 if frac < 0.5 else (0.1 if frac < 0.75 else 1.0)
    if sched == 'iso_product':
        return 1.0 / max(_cos_m(t), ISO_M_FLOOR)
    raise ValueError(f'unknown wd_sched={sched}')


def realized_budget_c(sched, lam0):
    """sum_t eta_t*lambda_t over optimizer steps, in units of the E4 anchor C."""
    steps = math.ceil(CIFAR100_TRAIN_N / BATCH)
    total = sum(ETA0 * _cos_m(t) * lam0 * _wd_m(sched, t)
                for t in range(T_EPOCHS))
    anchor = sum(ETA0 * _cos_m(t) * E4_OURS_LAMBDA for t in range(T_EPOCHS))
    return total * steps / (anchor * steps)


def _match_lams(series, lams):
    mask = np.zeros(len(series), dtype=bool)
    vals = series.values
    for lam in lams:
        mask |= np.isclose(vals, lam, rtol=1e-3, atol=1e-12)
    return mask


def base_slice(df, momentum):
    """R18 / CIFAR-100 / B=128 / lr=0.1 / T=100 / cosine LR."""
    return df[
        (df['model'] == 'resnet18')
        & (df['batch_size'] == 128)
        & (df['epochs'] == 100)
        & np.isclose(df['lr'], 0.1)
        & (df['scheduler'] == 'cosine')
        & np.isclose(df['momentum'], momentum)
        & (df['wd'] > 0)
    ].copy()


def fixed_control(df, momentum, seed=None):
    """Cosine LR + fixed lambda rows (blank wd_sched counts as fixed)."""
    g = base_slice(df, momentum)
    ws = g['wd_sched'].fillna('').astype(str).str.strip()
    g = g[(ws == '') | (ws == 'fixed')]
    if seed is not None:
        g = g[g['seed'] == seed]
    return g


def peak_of(g, lams=None):
    if lams is not None:
        g = g[_match_lams(g['wd'], lams)]
    if g.empty:
        return None
    return g.loc[g['best_test_acc'].idxmax()]


def phase_section(df, momentum, label):
    """Peak table + curve plot for one momentum phase. Returns peak rows."""
    raise_df = base_slice(df, momentum)
    raise_df = raise_df[raise_df['wd_sched'].isin(RAISE_SCHEDULES)]
    # The grid curves are defined on seed 42; multi-seed replicas are reported
    # separately in the multiseed section, not pooled into the peak.
    raise_df = raise_df[raise_df['seed'] == 42]

    fixed42 = fixed_control(df, momentum, seed=42)
    fixed_all = fixed_control(df, momentum)
    iso = base_slice(df, momentum)
    iso = iso[iso['wd_sched'] == 'iso_product']

    fixed42_peak = peak_of(fixed42)
    fixed_all_peak = peak_of(fixed_all)
    ref_acc = float(fixed42_peak['best_test_acc']) if fixed42_peak is not None else np.nan

    rows = []

    def add(tag, wd_sched, peak, n, note=''):
        rows.append({
            'tag': tag, 'phase': label, 'wd_sched': wd_sched,
            'peak_acc': float(peak['best_test_acc']),
            'peak_lambda0': float(peak['wd']), 'n': int(n),
            'delta_vs_fixed42': float(peak['best_test_acc'] - ref_acc)
            if np.isfinite(ref_acc) else np.nan,
            'note': note,
        })

    if fixed42_peak is not None:
        add('fixed_seed42', 'fixed', fixed42_peak, len(fixed42),
            'oracle over available lambdas, seed 42 (control)')
    if fixed_all_peak is not None:
        add('fixed_allseeds', 'fixed', fixed_all_peak, len(fixed_all),
            'oracle over available lambdas, all seeds')
    for sched in RAISE_SCHEDULES:
        g = raise_df[raise_df['wd_sched'] == sched]
        g = g[_match_lams(g['wd'], LAMBDA0)]
        peak = peak_of(g)
        if peak is not None:
            add('e11_raise', sched, peak, len(g))
    iso_peak = peak_of(iso)
    if iso_peak is not None:
        add('e9_iso', 'iso_product', iso_peak, len(iso),
            'analytic raise-up lambda0*eta0/eta_t (reference)')

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    pts = []
    for lam in sorted(fixed42['wd'].unique()):
        hit = fixed42[np.isclose(fixed42['wd'], lam, rtol=1e-3, atol=1e-12)]
        if not hit.empty:
            pts.append((lam, float(hit['best_test_acc'].max())))
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, 's--', ms=5, color='gray', label='fixed (seed 42)')
    for sched in RAISE_SCHEDULES:
        g = raise_df[raise_df['wd_sched'] == sched]
        pts = []
        for lam in LAMBDA0:
            hit = g[np.isclose(g['wd'], lam, rtol=1e-3, atol=1e-12)]
            if not hit.empty:
                pts.append((lam, float(hit['best_test_acc'].max())))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, 'o-', ms=5, label=sched)
    if not iso.empty:
        pts = []
        for lam in LAMBDA0:
            hit = iso[np.isclose(iso['wd'], lam, rtol=1e-3, atol=1e-12)]
            if not hit.empty:
                pts.append((lam, float(hit['best_test_acc'].max())))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '^:', ms=5, label='iso_product (E9)')
    if np.isfinite(ref_acc):
        ax.axhline(ref_acc, color='gray', ls=':', lw=1,
                   label=f'fixed peak (seed 42) = {ref_acc:.2f}')
    ax.set_xscale('log')
    ax.set_xlabel(r'weight-decay scale $\lambda_0$ (end-of-run peak for raise-up)')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title(rf'E11: {label}, cosine LR — raise-up $\lambda$ schedules')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'e11_wd_raise_{label.lower()}.png', dpi=160)
    plt.close(fig)
    return rows


def matched_section(df):
    """Matched-budget rows (e11_matched + E9's e9_matched for context).

    Each row's realized budget is computed from its (wd_sched, lambda0), so
    any budget-factor ladder is labelled correctly.
    """
    g = df[df['exp'].isin(['e11_matched', 'e9_matched'])].copy()
    if g.empty:
        return []
    rows = []
    for mom, label in [(0.9, 'SGDM'), (0.0, 'SGD')]:
        sub = g[np.isclose(g['momentum'], mom)]
        for (exp, sched), hit in sub.groupby(['exp', 'wd_sched']):
            for _, r in hit.iterrows():
                rows.append({
                    'phase': label, 'exp': exp, 'wd_sched': sched,
                    'budget_c': realized_budget_c(sched, float(r['wd'])),
                    'lambda0': float(r['wd']),
                    'best_test_acc': float(r['best_test_acc']),
                    'seed': int(r['seed']),
                    'diverged': int(r['diverged']),
                })
        # The E9-matched fixed arm at 1.0C dedup'd against the e8_e4_baseline
        # row (lambda = 5.982e-4), so re-add that anchor point explicitly.
        anchor = df[
            (df['exp'] == 'e8_e4_baseline')
            & np.isclose(df['momentum'], mom)
            & np.isclose(df['wd'], E4_OURS_LAMBDA, rtol=1e-3)
            & (df['scheduler'] == 'cosine') & (df['epochs'] == 100)
            & np.isclose(df['lr'], 0.1) & (df['batch_size'] == 128)
            & (df['model'] == 'resnet18') & (df['seed'] == 42)
        ]
        for _, r in anchor.iterrows():
            rows.append({
                'phase': label, 'exp': 'e9_matched', 'wd_sched': 'fixed',
                'budget_c': realized_budget_c('fixed', float(r['wd'])),
                'lambda0': float(r['wd']),
                'best_test_acc': float(r['best_test_acc']),
                'seed': int(r['seed']),
                'diverged': int(r['diverged']),
            })
    return rows


def plot_budget_curves(mdf, label, out_path):
    """best acc vs realized contraction budget (C units), one line per shape."""
    sub = mdf[mdf['phase'] == label]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (exp, sched), hit in sub.groupby(['exp', 'wd_sched']):
        hit = hit.sort_values('budget_c')
        style = ('o-', None) if exp == 'e11_matched' else ('^:', 0.6)
        ax.plot(hit['budget_c'], hit['best_test_acc'], style[0], ms=5,
                alpha=style[1] if style[1] else 1.0, label=sched)
    ax.set_xscale('log')
    ax.set_xlabel(r'realized contraction budget $\sum_t \eta_t \lambda_t$ (units of C)')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title(rf'E11/E9 matched: {label}, cosine LR — budget vs shape')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def multiseed_section(df):
    """Mean +- std over seeds for the peak configs (SGDM phase)."""
    g = base_slice(df, 0.9)
    ws = g['wd_sched'].fillna('').astype(str).str.strip()
    g = g.assign(wd_sched_norm=ws.replace('', 'fixed'))
    rows = []
    for sched, lam in PEAK_CONFIGS:
        hit = g[(g['wd_sched_norm'] == sched)
                & np.isclose(g['wd'], lam, rtol=1e-3, atol=1e-12)]
        if hit.empty:
            continue
        accs = hit.groupby('seed')['best_test_acc'].max()
        rows.append({
            'wd_sched': sched, 'lambda0': lam,
            'mean_acc': float(accs.mean()),
            'std_acc': float(accs.std(ddof=0)) if len(accs) > 1 else 0.0,
            'n_seeds': int(len(accs)),
            'seeds': ','.join(str(int(s)) for s in sorted(accs.index)),
        })
    return rows


def main():
    df = pd.read_csv(CSV)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1/2. peak tables per phase ---
    all_rows = []
    all_rows.extend(phase_section(df, 0.9, 'SGDM'))
    all_rows.extend(phase_section(df, 0.0, 'SGD'))
    peaks = pd.DataFrame(all_rows)
    peaks.to_csv(DATA_DIR / 'e11_wd_raise_peaks.csv', index=False)

    md = [
        'E11: raise-up lambda under cosine LR (R18/CIFAR-100, B=128, '
        'eta0=0.1, T=100).',
        'Raise-up grid runs are seed 42; lambda0 is the end-of-run peak value.',
        '',
        '| phase | tag | wd_sched | peak_acc | peak_lambda0 | delta vs fixed(seed42) | n | note |',
        '|---|---|---|---:|---:|---:|---:|---|',
    ]
    for _, r in peaks.iterrows():
        md.append(
            f"| {r['phase']} | {r['tag']} | {r['wd_sched']} | {r['peak_acc']:.2f} | "
            f"{r['peak_lambda0']:g} | {r['delta_vs_fixed42']:+.2f} | "
            f"{int(r['n'])} | {r['note']} |"
        )
    (DATA_DIR / 'e11_wd_raise_table.md').write_text('\n'.join(md) + '\n')

    # --- 3. matched budgets ---
    matched_rows = matched_section(df)
    if matched_rows:
        mdf = pd.DataFrame(matched_rows).sort_values(
            ['phase', 'exp', 'wd_sched', 'budget_c'])
        mdf.to_csv(DATA_DIR / 'e11_matched_peaks.csv', index=False)
        md_m = [
            'E11 matched-budget arm: shapes rescaled to a common contraction',
            'budget sum_t eta_t*lambda_t (C = E4-ours reference). e9_matched',
            'rows (fixed/cosine/linear/step/iso_product) included as context.',
            '',
            '| phase | exp | wd_sched | budget | lambda0 | best_acc | diverged |',
            '|---|---|---|---:|---:|---:|---:|',
        ]
        for _, r in mdf.iterrows():
            md_m.append(
                f"| {r['phase']} | {r['exp']} | {r['wd_sched']} | "
                f"{r['budget_c']:.2f}C | {r['lambda0']:.4g} | "
                f"{r['best_test_acc']:.2f} | {int(r['diverged'])} |"
            )
        (DATA_DIR / 'e11_matched_table.md').write_text('\n'.join(md_m) + '\n')
        plot_budget_curves(mdf, 'SGDM', PLOT_DIR / 'e11_matched_budget_sgdm.png')
        plot_budget_curves(mdf, 'SGD', PLOT_DIR / 'e11_matched_budget_sgd.png')

    # --- 4. multi-seed ---
    ms_rows = multiseed_section(df)
    if ms_rows:
        sdf = pd.DataFrame(ms_rows)
        sdf.to_csv(DATA_DIR / 'e11_multiseed.csv', index=False)
        md_s = [
            'E11 multi-seed: SGDM peak configs replicated across seeds.',
            '',
            '| wd_sched | lambda0 | mean_acc | std | n_seeds | seeds |',
            '|---|---:|---:|---:|---:|---|',
        ]
        for _, r in sdf.iterrows():
            md_s.append(
                f"| {r['wd_sched']} | {r['lambda0']:g} | {r['mean_acc']:.2f} | "
                f"{r['std_acc']:.2f} | {int(r['n_seeds'])} | {r['seeds']} |"
            )
        (DATA_DIR / 'e11_multiseed_table.md').write_text('\n'.join(md_s) + '\n')

    print('\n'.join(md))
    if matched_rows:
        print()
        print((DATA_DIR / 'e11_matched_table.md').read_text())
    if ms_rows:
        print()
        print((DATA_DIR / 'e11_multiseed_table.md').read_text())
    print(f'wrote outputs under {DATA_DIR} and {PLOT_DIR}')


if __name__ == '__main__':
    main()
