"""
Analyze E12: cross-setting WD-schedule grid under a cosine learning rate.

For every (model, dataset) setting the pipeline produced:
  - e12_fixed   cosine LR + fixed lambda, phase-dependent grid (oracle sweep;
                also the const-shape budget curve since budget ~ lambda)
  - e12_matched shapes {linear_up, linear (down), iso_product} at a
                phase-dependent budget ladder in units of the SETTING-LOCAL
                C = lambda_ref * sum_t eta_t, where lambda_ref is that
                setting-phase's fixed-WD oracle (anchors.json).
  - e12_ms      multi-seed replicas of the peak configs.

R18/CIFAR-100 is not rerun: its column comes read-only from the E11 CSV
(e11_raise_up/results/nips26_e11_runs.csv), with anchors recomputed from the
fixed oracle per phase (SGDM 6e-4-ish, SGD 5e-3).

Outputs (all under e12_multi_setting/):
  figures/e12_setting_{model}_{dataset}_{phase}.png   budget-vs-acc per setting
  figures/e12_cstar_comparison.png                    C* per shape across settings
  tables/e12_setting_table.md                         per-setting peak table
  tables/e12_cstar_summary.csv/.md                    cross-setting C* summary
  tables/e12_multiseed.csv/.md                        multi-seed mean+-std

Reads e11 CSV read-only; writes only under e12_multi_setting/.
"""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
E12 = ROOT / 'e12_multi_setting'
CSV = E12 / 'results' / 'e12_runs.csv'
E11_CSV = ROOT / 'e11_raise_up' / 'results' / 'nips26_e11_runs.csv'
PLOT_DIR = E12 / 'figures'
DATA_DIR = E12 / 'tables'

# --- contraction-budget math ------------------------------------------------
# Pure-python copies of the schedule math in rebuttal/run_nips26_wd_sched.py.
ETA0, T_EPOCHS, BATCH = 0.1, 100, 128
TRAIN_N = {'cifar100': 50000, 'cifar10': 50000, 'mnist': 60000}
ISO_M_FLOOR = 0.1
# E12 shapes; 'linear' is the linear-DOWN shape (1 - t/T).
E12_SHAPES = ['fixed', 'linear_up', 'linear', 'iso_product']
SHAPE_LABELS = {'fixed': 'fixed (const)', 'linear_up': 'linear up',
                'linear': 'linear down', 'iso_product': 'iso up (product)'}
PHASE_LABELS = {0.9: 'SGDM', 0.0: 'SGD'}
LADDER = {'sgdm': [1 / 3, 1, 1.5, 2, 2.5, 3, 4],
          'sgd': [1, 2, 3, 4, 6, 9, 15]}


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
    raise ValueError(f'unknown wd_sched={sched}')


def contraction_sum(sched, lam0, dataset, n_epochs=T_EPOCHS):
    """sum_t eta_t*lambda_t over optimizer steps (one value per epoch)."""
    n = TRAIN_N[dataset]
    steps = math.ceil(n / BATCH)
    total = sum(ETA0 * _cos_m(t) * lam0 * _wd_m(sched, t)
                for t in range(n_epochs))
    return total * steps


def realized_budget_c(sched, lam0, dataset, lam_ref):
    """Realized contraction budget in units of the setting-local C."""
    unit = contraction_sum('fixed', lam_ref, dataset)
    return contraction_sum(sched, lam0, dataset) / unit


def phase_of(momentum):
    return 'sgd' if float(momentum) == 0.0 else 'sgdm'


def slice_setting(df, model, dataset, momentum):
    return df[
        (df['model'] == model)
        & (df['dataset'] == dataset)
        & (df['batch_size'] == 128)
        & (df['epochs'] == 100)
        & np.isclose(df['lr'], 0.1)
        & (df['scheduler'] == 'cosine')
        & np.isclose(df['momentum'], momentum)
        & (df['wd'] > 0)
    ].copy()


def fixed_oracle(df, model, dataset, momentum):
    """Peak fixed-lambda row (seed 42, exp e12_fixed / e8_e4_baseline)."""
    g = slice_setting(df, model, dataset, momentum)
    ws = g['wd_sched'].fillna('').astype(str).str.strip()
    g = g[(ws == '') | (ws == 'fixed')]
    g = g[g['seed'] == 42]
    if g.empty:
        return None
    return g.loc[g['best_test_acc'].idxmax()]


def setting_anchor(df, model, dataset, momentum):
    """lambda_ref for the setting-local C (the fixed oracle, seed 42)."""
    peak = fixed_oracle(df, model, dataset, momentum)
    return float(peak['wd']) if peak is not None else None


def build_budget_rows(df, model, dataset, momentum, lam_ref):
    """One row per run with realized budget in local C units."""
    g = slice_setting(df, model, dataset, momentum)
    rows = []
    for _, r in g.iterrows():
        sched = str(r['wd_sched']).strip() or 'fixed'
        if sched not in E12_SHAPES:
            continue
        rows.append({
            'model': model, 'dataset': dataset, 'phase': PHASE_LABELS[momentum],
            'exp': r['exp'], 'wd_sched': sched, 'seed': int(r['seed']),
            'budget_c': realized_budget_c(sched, float(r['wd']), dataset,
                                          lam_ref),
            'lambda0': float(r['wd']),
            'best_test_acc': float(r['best_test_acc']),
            'diverged': int(r['diverged']),
        })
    return pd.DataFrame(rows)


def peak_by_shape(bdf, sched, exp_prefixes=None):
    """Peak acc row of one shape within the matched ladder (seed 42)."""
    g = bdf[(bdf['wd_sched'] == sched) & (bdf['seed'] == 42)]
    if exp_prefixes is not None:
        g = g[g['exp'].isin(exp_prefixes)]
    if g.empty:
        return None
    return g.loc[g['best_test_acc'].idxmax()]


def setting_section(df, model, dataset, momentum):
    """Full per-(setting, phase) analysis: peak table + budget figure."""
    lam_ref = setting_anchor(df, model, dataset, momentum)
    if lam_ref is None:
        print(f'[skip] {model}/{dataset}/{PHASE_LABELS[momentum]}: no fixed rows')
        return None, None
    bdf = build_budget_rows(df, model, dataset, momentum, lam_ref)
    if bdf.empty:
        return None, None

    label = f'{model}/{dataset}/{PHASE_LABELS[momentum]}'
    fixed_peak = peak_by_shape(bdf, 'fixed')
    rows = []
    for sched in E12_SHAPES:
        if sched == 'fixed':
            peak = fixed_peak
        else:
            # e12 rows come from this CSV; R18 rows come from the E11 CSV
            # (e11_matched / e9_matched), read-only.
            peak = peak_by_shape(
                bdf, sched, ['e12_matched', 'e11_matched', 'e9_matched'])
        if peak is None:
            continue
        rows.append({
            'setting': label, 'wd_sched': sched, 'phase': PHASE_LABELS[momentum],
            'peak_acc': float(peak['best_test_acc']),
            'peak_budget_c': float(peak['budget_c']),
            'peak_lambda0': float(peak['lambda0']),
            'n_runs': int((bdf['wd_sched'] == sched).sum()),
        })

    # --- per-setting figure: budget vs acc, one line per shape ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for sched in E12_SHAPES:
        sub = bdf[(bdf['wd_sched'] == sched) & (bdf['seed'] == 42)]
        if sub.empty:
            continue
        sub = sub.sort_values('budget_c')
        style = {'fixed': 's--', 'linear_up': 'o-', 'linear': 'v-',
                 'iso_product': '^:'}[sched]
        ax.plot(sub['budget_c'], sub['best_test_acc'], style, ms=5,
                label=SHAPE_LABELS[sched])
    ax.set_xscale('log')
    ax.set_xlabel(r'realized contraction budget $\sum_t \eta_t \lambda_t$ (setting-local C)')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title(f'E12: {label}, cosine LR — WD schedule vs budget')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f'e12_setting_{model}_{dataset}_{PHASE_LABELS[momentum].lower()}.png'
    fig.savefig(PLOT_DIR / fname, dpi=160)
    plt.close(fig)
    return rows, bdf


def cstar_summary(setting_rows):
    """Cross-setting table: optimal budget C* per shape, SGDM/SGD ratio."""
    df = pd.DataFrame(setting_rows)
    if df.empty:
        return df
    piv = df.pivot_table(index=['setting', 'phase'], columns='wd_sched',
                         values='peak_budget_c')
    acc = df.pivot_table(index=['setting', 'phase'], columns='wd_sched',
                         values='peak_acc')
    out = piv.join(acc, lsuffix='_C', rsuffix='_acc')
    # SGDM/SGD ratio of the linear_up optimum per setting (momentum story).
    lu = df[df['wd_sched'] == 'linear_up'].set_index(['setting', 'phase'])
    ratios = []
    for setting, g in lu.groupby('setting'):
        if 'SGDM' in g.index and 'SGD' in g.index:
            ratios.append({
                'setting': setting,
                'cstar_sgdm': float(g.loc['SGDM', 'peak_budget_c']),
                'cstar_sgd': float(g.loc['SGD', 'peak_budget_c']),
                'ratio_sgd_over_sgdm': float(
                    g.loc['SGD', 'peak_budget_c'] / g.loc['SGDM', 'peak_budget_c']),
            })
    return out, pd.DataFrame(ratios)


def multiseed_section(df):
    """Mean +- std over seeds for e12_ms rows (and R18's e11_raise_ms)."""
    g = df[df['exp'].isin(['e12_ms', 'e11_raise_ms'])].copy()
    if g.empty:
        return None
    rows = []
    for (model, dataset, momentum, sched), hit in g.groupby(
            ['model', 'dataset', 'momentum', 'wd_sched']):
        lam = float(hit['wd'].mode().iloc[0])
        accs = hit.groupby('seed')['best_test_acc'].max()
        rows.append({
            'setting': f'{model}/{dataset}/{PHASE_LABELS[momentum]}',
            'wd_sched': sched, 'lambda0': lam,
            'mean_acc': float(accs.mean()),
            'std_acc': float(accs.std(ddof=0)) if len(accs) > 1 else 0.0,
            'n_seeds': int(len(accs)),
            'seeds': ','.join(str(int(s)) for s in sorted(accs.index)),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    e11 = pd.read_csv(E11_CSV) if E11_CSV.exists() else pd.DataFrame()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Settings present in the E12 CSV (skip resnet18/cifar100, which is E11).
    e12_pairs = sorted(
        set(zip(df['model'], df['dataset'])))
    e12_pairs = [(m, d) for (m, d) in e12_pairs
                 if not (m == 'resnet18' and d == 'cifar100')]
    # R18/C100 column comes from the E11 CSV, read-only.
    r18 = ('resnet18', 'cifar100')

    all_rows = []
    for (model, dataset) in e12_pairs:
        for mom in (0.9, 0.0):
            rows, _ = setting_section(df, model, dataset, mom)
            if rows:
                all_rows.extend(rows)
    for mom in (0.9, 0.0):
        rows, _ = setting_section(e11, r18[0], r18[1], mom)
        if rows:
            all_rows.extend(rows)

    if all_rows:
        sdf = pd.DataFrame(all_rows)
        sdf.to_csv(DATA_DIR / 'e12_setting_table.csv', index=False)
        md = [
            'E12 per-setting peaks: cosine LR, B=128, eta0=0.1, T=100.',
            'Budget in setting-local C units (lambda_ref = fixed oracle of',
            'that setting/phase, seed 42). n_runs counts seed-42 rows.',
            '',
            '| setting | phase | wd_sched | peak_acc | peak_budget_C | peak_lambda0 | n_runs |',
            '|---|---|---|---:|---:|---:|---:|',
        ]
        for _, r in sdf.iterrows():
            md.append(
                f"| {r['setting']} | {r['phase']} | {r['wd_sched']} | "
                f"{r['peak_acc']:.2f} | {r['peak_budget_c']:.2f}C | "
                f"{r['peak_lambda0']:.4g} | {int(r['n_runs'])} |")
        (DATA_DIR / 'e12_setting_table.md').write_text('\n'.join(md) + '\n')

        pivot, ratios = cstar_summary(sdf)
        pivot.to_csv(DATA_DIR / 'e12_cstar_summary.csv')
        md_c = [
            'E12 cross-setting summary: optimal budget C* per shape',
            '(columns "wd_sched_C" are peak budgets in local C units;',
            '"wd_sched_acc" the corresponding accuracy).',
            '',
        ]
        md_c.append(pivot.round(2).to_markdown() + '\n')
        if not ratios.empty:
            ratios.to_csv(DATA_DIR / 'e12_cstar_ratios.csv', index=False)
            md_c.append('SGDM vs SGD optimal budget for linear_up:')
            md_c.append('')
            md_c.append(ratios.round(2).to_markdown() + '\n')
        (DATA_DIR / 'e12_cstar_summary.md').write_text('\n'.join(md_c) + '\n')

    frames = [f for f in (df, e11) if not f.empty]
    ms = multiseed_section(pd.concat(frames, ignore_index=True)) if frames else None
    if ms is not None and not ms.empty:
        ms.to_csv(DATA_DIR / 'e12_multiseed.csv', index=False)
        md_s = [
            'E12 multi-seed: peak configs replicated across seeds.',
            '',
            ms.to_markdown(index=False),
        ]
        (DATA_DIR / 'e12_multiseed.md').write_text('\n'.join(md_s) + '\n')

    print(f'wrote outputs under {DATA_DIR} and {PLOT_DIR}')
    if all_rows:
        print('\n'.join(md))


if __name__ == '__main__':
    main()
