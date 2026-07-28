"""
Analyze E8: fixed vs scheduled weight decay under constant learning rate.

Writes:
  outputs/plots/nips26/e8_wd_sched_sgd.png
  outputs/plots/nips26/e8_wd_sched_sgdm.png
  rebuttal/nips_rebuttal/_data/e8_wd_sched_peaks.csv
  rebuttal/nips_rebuttal/_data/e8_wd_sched_table.md
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'rebuttal' / 'results' / 'nips26_runs.csv'
PLOT_DIR = ROOT / 'outputs' / 'plots' / 'nips26'
DATA_DIR = ROOT / 'rebuttal' / 'nips_rebuttal' / '_data'

SCHEDULES = ['fixed', 'cosine', 'linear', 'step', 'cosine_restarts']
LAMBDA0 = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]


def load_e8(csv_path=CSV):
    df = pd.read_csv(csv_path)
    if 'wd_sched' not in df.columns:
        raise RuntimeError('CSV missing wd_sched; run E8 runner first')
    m = (
        (df['model'] == 'resnet18')
        & (df['batch_size'] == 128)
        & (df['epochs'] == 100)
        & np.isclose(df['lr'], 0.1)
        & (df['scheduler'] == 'const')
        & (df['seed'] == 42)
        & (df['wd_sched'].isin(SCHEDULES))
        & (df['wd'] > 0)
    )
    out = df[m].copy()
    # Prefer e8 rows when duplicates exist; else keep highest acc.
    out = (out.sort_values(['best_test_acc'], ascending=False)
              .drop_duplicates(
                  subset=['momentum', 'wd_sched', 'wd'], keep='first'))
    return out


def peak_table(df):
    rows = []
    for momentum, name in [(0.0, 'SGD'), (0.9, 'SGDM')]:
        sub = df[np.isclose(df['momentum'], momentum)]
        for sched in SCHEDULES:
            g = sub[sub['wd_sched'] == sched]
            if g.empty:
                continue
            # restrict to the planned lambda0 grid (tolerance)
            mask = np.zeros(len(g), dtype=bool)
            for lam in LAMBDA0:
                mask |= np.isclose(g['wd'].values, lam, rtol=1e-3, atol=0)
            g = g[mask]
            if g.empty:
                continue
            best = g.loc[g['best_test_acc'].idxmax()]
            fixed = sub[sub['wd_sched'] == 'fixed']
            fmask = np.zeros(len(fixed), dtype=bool)
            for lam in LAMBDA0:
                fmask |= np.isclose(fixed['wd'].values, lam, rtol=1e-3, atol=0)
            fixed_peak = fixed[fmask]['best_test_acc'].max() if fmask.any() else np.nan
            rows.append({
                'optimizer': name,
                'momentum': momentum,
                'wd_sched': sched,
                'peak_acc': float(best['best_test_acc']),
                'peak_lambda0': float(best['wd']),
                'n': int(len(g)),
                'delta_vs_fixed': float(best['best_test_acc'] - fixed_peak)
                if np.isfinite(fixed_peak) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_phase(df, momentum, title, out_path):
    sub = df[np.isclose(df['momentum'], momentum)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for sched in SCHEDULES:
        g = sub[sub['wd_sched'] == sched].copy()
        if g.empty:
            continue
        rows = []
        for lam in LAMBDA0:
            hit = g[np.isclose(g['wd'], lam, rtol=1e-3, atol=0)]
            if not hit.empty:
                rows.append((lam, float(hit['best_test_acc'].max())))
        if not rows:
            continue
        xs, ys = zip(*rows)
        ax.plot(xs, ys, 'o-', ms=5, label=sched)
    ax.set_xscale('log')
    ax.set_xlabel(r'initial weight decay $\lambda_0$')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def to_markdown(peaks):
    lines = [
        'Peak best-test accuracy under constant LR (η=0.1, T=100, B=128, R18/CIFAR-100).',
        '',
        '| optimizer | wd_sched | peak_acc | peak_λ0 | Δ vs fixed | n |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for _, r in peaks.iterrows():
        lines.append(
            f"| {r['optimizer']} | {r['wd_sched']} | {r['peak_acc']:.2f} | "
            f"{r['peak_lambda0']:g} | {r['delta_vs_fixed']:+.2f} | {int(r['n'])} |"
        )
    return '\n'.join(lines) + '\n'


def main():
    df = load_e8()
    peaks = peak_table(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(DATA_DIR / 'e8_wd_sched_peaks.csv', index=False)
    md = to_markdown(peaks)
    (DATA_DIR / 'e8_wd_sched_table.md').write_text(md)

    plot_phase(
        df, 0.0,
        r'E8a: SGD (mom=0), fixed LR — $\lambda$ schedules',
        PLOT_DIR / 'e8_wd_sched_sgd.png',
    )
    plot_phase(
        df, 0.9,
        r'E8b: SGDM (mom=0.9), fixed LR — $\lambda$ schedules',
        PLOT_DIR / 'e8_wd_sched_sgdm.png',
    )
    print(md)
    print(f'wrote {DATA_DIR / "e8_wd_sched_peaks.csv"}')
    print(f'wrote {PLOT_DIR / "e8_wd_sched_sgd.png"}')
    print(f'wrote {PLOT_DIR / "e8_wd_sched_sgdm.png"}')


if __name__ == '__main__':
    main()
