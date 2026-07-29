"""
Analyze E8: const-LR schedules, joint multiplier, long-T restarts, vs E4 baselines.

Writes:
  outputs/plots/nips26/e8_wd_sched_{sgd,sgdm}.png
  outputs/plots/nips26/e8_joint_sgdm.png
  outputs/plots/nips26/e8_long_sgdm.png
  rebuttal/nips_rebuttal/_data/e8_wd_sched_peaks.csv
  rebuttal/nips_rebuttal/_data/e8_wd_sched_table.md
  rebuttal/nips_rebuttal/_data/e8_followup_peaks.csv
  rebuttal/nips_rebuttal/_data/e8_followup_table.md
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
LONG_SCHEDULES = ['fixed', 'step', 'cosine_restarts']
LONG_LAMBDAS = [5e-4, 1e-3, 2e-3]
E4_OURS = 5.982e-4


def _match_lams(series, lams):
    mask = np.zeros(len(series), dtype=bool)
    vals = series.values
    for lam in lams:
        mask |= np.isclose(vals, lam, rtol=1e-3, atol=1e-12)
    return mask


def load_slice(df, scheduler, epochs=100, momentum=None, schedules=None):
    m = (
        (df['model'] == 'resnet18')
        & (df['batch_size'] == 128)
        & (df['epochs'] == epochs)
        & np.isclose(df['lr'], 0.1)
        & (df['scheduler'] == scheduler)
        & (df['seed'] == 42)
        & (df['wd'] > 0)
    )
    out = df[m].copy()
    if 'wd_sched' in out.columns:
        ws = out['wd_sched'].fillna('').astype(str).str.strip()
        # Legacy cosine-LR rows have blank wd_sched (= fixed lambda).
        if scheduler == 'cosine':
            out.loc[ws == '', 'wd_sched'] = 'fixed'
            ws = out['wd_sched'].fillna('').astype(str).str.strip()
        out = out[ws.str.len() > 0]
    if momentum is not None:
        out = out[np.isclose(out['momentum'], momentum)]
    if schedules is not None:
        out = out[out['wd_sched'].isin(schedules)]
    keys = ['momentum', 'wd_sched', 'wd', 'scheduler', 'epochs']
    out = (out.sort_values(['best_test_acc'], ascending=False)
              .drop_duplicates(subset=keys, keep='first'))
    return out


def peak_rows(df, schedules, lams, optimizer_name, tag):
    rows = []
    for sched in schedules:
        g = df[df['wd_sched'] == sched]
        g = g[_match_lams(g['wd'], lams)]
        if g.empty:
            continue
        best = g.loc[g['best_test_acc'].idxmax()]
        fixed = df[df['wd_sched'] == 'fixed']
        fixed = fixed[_match_lams(fixed['wd'], lams)]
        fixed_peak = fixed['best_test_acc'].max() if not fixed.empty else np.nan
        rows.append({
            'tag': tag,
            'optimizer': optimizer_name,
            'wd_sched': sched,
            'peak_acc': float(best['best_test_acc']),
            'peak_lambda0': float(best['wd']),
            'n': int(len(g)),
            'delta_vs_fixed': float(best['best_test_acc'] - fixed_peak)
            if np.isfinite(fixed_peak) else np.nan,
        })
    return rows


def plot_curves(df, schedules, lams, title, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for sched in schedules:
        g = df[df['wd_sched'] == sched]
        pts = []
        for lam in lams:
            hit = g[np.isclose(g['wd'], lam, rtol=1e-3, atol=1e-12)]
            if not hit.empty:
                pts.append((lam, float(hit['best_test_acc'].max())))
        if pts:
            xs, ys = zip(*pts)
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


def vs_e4_table(joint_df, cosine_df):
    """Compare E4 constant-lambda baselines vs joint schedule peaks."""
    lines = [
        'E8 follow-up: unified η₀=0.1, T=100, SGDM. Joint = same m(t) on η and λ.',
        '',
        '| method | best_acc | λ₀ / note |',
        '|---|---:|---|',
    ]
    rows = []

    def add(name, acc, note):
        lines.append(f'| {name} | {acc:.2f} | {note} |')
        rows.append({'method': name, 'best_acc': acc, 'note': note})

    if not cosine_df.empty:
        for lam, label in [(E4_OURS, 'E4-ours C/∑η'), (5e-4, 'default 5e-4')]:
            hit = cosine_df[np.isclose(cosine_df['wd'], lam, rtol=1e-3, atol=1e-12)]
            if not hit.empty:
                add(f'cosine LR + fixed λ ({label})',
                    float(hit['best_test_acc'].max()), f'{lam:g}')
        # peak over λ0 grid under cosine LR + fixed λ
        grid = cosine_df[_match_lams(cosine_df['wd'], LAMBDA0)]
        if not grid.empty:
            best = grid.loc[grid['best_test_acc'].idxmax()]
            add('cosine LR + fixed λ (oracle over λ₀ grid)',
                float(best['best_test_acc']), f"λ₀={best['wd']:g}")

    if not joint_df.empty:
        for sched in SCHEDULES:
            g = joint_df[joint_df['wd_sched'] == sched]
            g = g[_match_lams(g['wd'], LAMBDA0)]
            if g.empty:
                continue
            best = g.loc[g['best_test_acc'].idxmax()]
            add(f'joint m(t) = {sched} (peak over λ₀)',
                float(best['best_test_acc']), f"λ₀={best['wd']:g}")

    return '\n'.join(lines) + '\n', pd.DataFrame(rows)


def main():
    df = pd.read_csv(CSV)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- original const-LR E8 ---
    const_all = load_slice(df, 'const', epochs=100, schedules=SCHEDULES)
    const_peaks = []
    for mom, name in [(0.0, 'SGD'), (0.9, 'SGDM')]:
        sub = const_all[np.isclose(const_all['momentum'], mom)]
        const_peaks.extend(peak_rows(sub, SCHEDULES, LAMBDA0, name, 'const'))
        plot_curves(
            sub, SCHEDULES, LAMBDA0,
            rf'E8a/b: {name}, fixed LR — $\lambda$ schedules',
            PLOT_DIR / (f'e8_wd_sched_{"sgd" if mom == 0 else "sgdm"}.png'),
        )
    const_peaks_df = pd.DataFrame(const_peaks)
    const_peaks_df.to_csv(DATA_DIR / 'e8_wd_sched_peaks.csv', index=False)
    md_const = [
        'Peak best-test accuracy under constant LR (η=0.1, T=100, B=128, R18/CIFAR-100).',
        '',
        '| optimizer | wd_sched | peak_acc | peak_λ0 | Δ vs fixed | n |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for _, r in const_peaks_df.iterrows():
        md_const.append(
            f"| {r['optimizer']} | {r['wd_sched']} | {r['peak_acc']:.2f} | "
            f"{r['peak_lambda0']:g} | {r['delta_vs_fixed']:+.2f} | {int(r['n'])} |"
        )
    (DATA_DIR / 'e8_wd_sched_table.md').write_text('\n'.join(md_const) + '\n')

    # --- joint follow-up ---
    joint = load_slice(df, 'joint', epochs=100, momentum=0.9, schedules=SCHEDULES)
    long = load_slice(df, 'joint', epochs=200, momentum=0.9, schedules=LONG_SCHEDULES)
    cosine = load_slice(df, 'cosine', epochs=100, momentum=0.9, schedules=['fixed'])

    follow_rows = []
    follow_rows.extend(peak_rows(joint, SCHEDULES, LAMBDA0, 'SGDM', 'joint_T100'))
    follow_rows.extend(peak_rows(long, LONG_SCHEDULES, LONG_LAMBDAS, 'SGDM', 'joint_T200'))
    follow_df = pd.DataFrame(follow_rows)
    follow_df.to_csv(DATA_DIR / 'e8_followup_peaks.csv', index=False)

    if not joint.empty:
        plot_curves(
            joint, SCHEDULES, LAMBDA0,
            r'E8 follow-up: SGDM joint $m(t)$ on $\eta$ and $\lambda$ (T=100)',
            PLOT_DIR / 'e8_joint_sgdm.png',
        )
    if not long.empty:
        plot_curves(
            long, LONG_SCHEDULES, LONG_LAMBDAS,
            r'E8 follow-up: SGDM joint, T=200 (Te=50, Tmult=2)',
            PLOT_DIR / 'e8_long_sgdm.png',
        )

    vs_md, vs_df = vs_e4_table(joint, cosine)
    vs_df.to_csv(DATA_DIR / 'e8_followup_vs_e4.csv', index=False)

    follow_md = [
        '# E8 follow-up peaks',
        '',
        '| tag | wd_sched | peak_acc | peak_λ0 | Δ vs fixed | n |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for _, r in follow_df.iterrows():
        follow_md.append(
            f"| {r['tag']} | {r['wd_sched']} | {r['peak_acc']:.2f} | "
            f"{r['peak_lambda0']:g} | {r['delta_vs_fixed']:+.2f} | {int(r['n'])} |"
        )
    follow_md += ['', '## vs E4 constant λ (same η₀, T, SGDM)', '', vs_md]
    (DATA_DIR / 'e8_followup_table.md').write_text('\n'.join(follow_md))

    print('\n'.join(md_const))
    print()
    print('\n'.join(follow_md))
    print(f'wrote {DATA_DIR / "e8_followup_table.md"}')
    print(f'wrote {PLOT_DIR / "e8_joint_sgdm.png"} (if data)')
    print(f'wrote {PLOT_DIR / "e8_long_sgdm.png"} (if data)')


if __name__ == '__main__':
    main()
