"""
Analyze E9 (reviewer xkCF follow-up, 2026-08-03).

Two questions, both raised by the reviewer about the E8 `joint` arm:

  1. The joint multiplier gives eta_t*lambda_t = eta_0*lambda_0*m(t)^2, so it
     does *not* preserve the coupling. What happens with a schedule that does,
     lambda_t = lambda_0*eta_0/eta_t?
  2. Comparing schedule shapes at a common lambda_0 is not a controlled
     comparison. What if every shape is matched on cumulative contraction
     sum_t eta_t*lambda_t instead?

Writes:
  rebuttal/nips_rebuttal/_data/e9_iso_matched.csv
  rebuttal/nips_rebuttal/_data/e9_table.md
  rebuttal/nips_rebuttal/_data/e9_tokens.md
  outputs/plots/nips26/e9_iso_matched.png
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'rebuttal' / 'results' / 'nips26_runs.csv'
DATA_DIR = ROOT / 'rebuttal' / 'nips_rebuttal' / '_data'
PLOT_DIR = ROOT / 'outputs' / 'plots' / 'nips26'


def _load_runner():
    """The E9 schedule math lives in the runner; import it rather than copy it."""
    path = ROOT / 'rebuttal' / 'run_nips26_wd_sched.py'
    spec = importlib.util.spec_from_file_location('e9_runner', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_runner()

# Reference points for the same eta_0 = 0.1, T = 100, SGDM, B = 128protocol.
# All are peaks already reported in _data/e8_followup_table.md.
REFERENCES = [
    ('cosine LR + constant lambda (oracle over lambda_0 grid)', 77.28),
    ('cosine LR + constant lambda (ours, C/sum_eta)', 76.72),
    ('cosine LR + constant lambda (default 5e-4)', 76.73),
    ('joint m(t) on eta and lambda (best shape: cosine)', 76.42),
    ('constant LR + scheduled lambda (best shape: step)', 73.10),
    ('constant LR + constant lambda', 66.67),
]


def lookup(df, *, wd, wd_sched, momentum=0.9, epochs=100, lr=0.1,
           batch_size=128, model='resnet18', seed=42):
    """
    Find the run for one E9 cell.

    E9 writes scheduler='cosine' because the learning-rate trajectory is
    identical to CosineAnnealingLR. Legacy cosine rows have an empty wd_sched and
    are constant-lambda, i.e. wd_sched == 'fixed'.
    """
    ws = df['wd_sched'].fillna('').astype(str).str.strip()
    ws = ws.where(ws != '', 'fixed')
    m = ((df['model'] == model) & (df['batch_size'] == batch_size)
         & (df['epochs'] == epochs) & np.isclose(df['lr'], lr)
         & np.isclose(df['momentum'], momentum) & (df['seed'] == seed)
         & (df['scheduler'] == 'cosine')
         & np.isclose(df['wd'], wd, rtol=1e-4, atol=1e-12)
         & (ws == wd_sched))
    hit = df[m]
    if hit.empty:
        return None
    return hit.loc[hit['best_test_acc'].idxmax()]


def build_table(df):
    anchor = R.e9_budget_anchor()
    rows = []

    # --- matched contraction: every shape at the same sum_t eta_t*lambda_t ---
    for factor in R.E9_BUDGET_FACTORS:
        budget = factor * anchor
        for shape in R.E9_SHAPES:
            lam0 = R.solve_lambda0_for_budget(budget, 0.1, 100, 128, shape)
            hit = lookup(df, wd=lam0, wd_sched=shape)
            spent = R.contraction_sum(0.1, lam0, 100, 128, shape)
            rows.append({
                'arm': 'matched', 'budget_factor': factor,
                'shape': shape, 'lambda0': lam0,
                'contraction': spent, 'contraction_over_C': spent / anchor,
                'best_test_acc': float(hit['best_test_acc']) if hit is not None else np.nan,
                'final_test_acc': float(hit['final_test_acc']) if hit is not None else np.nan,
                'train_acc': float(hit['final_train_acc']) if hit is not None else np.nan,
                'diverged': int(hit['diverged']) if hit is not None else -1,
                'exp': str(hit['exp']) if hit is not None else '',
            })

    # --- iso-product arm on the standard lambda_0 ladder ---
    for lam0 in R.LAMBDA0_GRID:
        hit = lookup(df, wd=lam0, wd_sched='iso_product')
        spent = R.contraction_sum(0.1, lam0, 100, 128, 'iso_product')
        rows.append({
            'arm': 'iso_grid', 'budget_factor': np.nan,
            'shape': 'iso_product', 'lambda0': lam0,
            'contraction': spent, 'contraction_over_C': spent / anchor,
            'best_test_acc': float(hit['best_test_acc']) if hit is not None else np.nan,
            'final_test_acc': float(hit['final_test_acc']) if hit is not None else np.nan,
            'train_acc': float(hit['final_train_acc']) if hit is not None else np.nan,
            'diverged': int(hit['diverged']) if hit is not None else -1,
            'exp': str(hit['exp']) if hit is not None else '',
        })

    # --- the joint arm the reviewer objected to, for the same shapes ---
    for shape in ['cosine', 'linear', 'step']:
        for lam0 in R.LAMBDA0_GRID:
            m = ((df['model'] == 'resnet18') & (df['scheduler'] == 'joint')
                 & (df['epochs'] == 100) & np.isclose(df['lr'], 0.1)
                 & np.isclose(df['momentum'], 0.9)
                 & (df['wd_sched'].astype(str) == shape)
                 & np.isclose(df['wd'], lam0, rtol=1e-4, atol=1e-12))
            hit = df[m]
            if hit.empty:
                continue
            hit = hit.loc[hit['best_test_acc'].idxmax()]
            # joint: eta_t = eta_0 m(t) and lambda_t = lambda_0 m(t)
            steps = 391
            total = sum(
                0.1 * R.wd_multiplier(shape, e, 100)
                * lam0 * R.wd_multiplier(shape, e, 100) for e in range(100))
            rows.append({
                'arm': 'joint', 'budget_factor': np.nan,
                'shape': shape, 'lambda0': lam0,
                'contraction': total * steps,
                'contraction_over_C': total * steps / anchor,
                'best_test_acc': float(hit['best_test_acc']),
                'final_test_acc': float(hit['final_test_acc']),
                'train_acc': float(hit['final_train_acc']),
                'diverged': int(hit['diverged']), 'exp': str(hit['exp']),
            })

    return pd.DataFrame(rows), anchor


def plot(tab, anchor):
    matched = tab[tab['arm'] == 'matched']
    iso = tab[tab['arm'] == 'iso_grid']
    joint = tab[tab['arm'] == 'joint']

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    for shape, g in matched.groupby('shape'):
        g = g.sort_values('budget_factor')
        ax.plot(g['budget_factor'], g['best_test_acc'], 'o-', ms=6, label=shape)
    ax.axhline(77.28, color='k', ls='--', lw=1.2,
               label='cosine LR + const $\\lambda$ (oracle)')
    ax.set_xscale('log')
    ax.set_xlabel(r'contraction budget $\sum_t \eta_t\lambda_t$  (units of $C$)')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title('(a) matched cumulative contraction:\nbudget dominates, shape does not')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, g, style in [
        ('iso-product ($\\eta_t\\lambda_t$ const)', iso, 'o-'),
        ('joint (cosine) — $\\eta_t\\lambda_t\\propto m(t)^2$',
         joint[joint['shape'] == 'cosine'], 's--'),
    ]:
        g = g.sort_values('contraction_over_C')
        if g.empty:
            continue
        ax.plot(g['contraction_over_C'], g['best_test_acc'], style, ms=6, label=label)
    g = matched[matched['shape'] == 'fixed'].sort_values('contraction_over_C')
    ax.plot(g['contraction_over_C'], g['best_test_acc'], '^-', ms=6,
            label='constant $\\lambda$')
    ax.axhline(77.28, color='k', ls='--', lw=1.2)
    ax.axvline(1.0, color='r', ls=':', lw=1.2, label=r'our rule: budget $=C$')
    ax.set_xscale('log')
    ax.set_xlabel(r'realized $\sum_t \eta_t\lambda_t$  (units of $C$)')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title('(b) everything collapses onto the contraction budget')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / 'e9_iso_matched.png', dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(CSV)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tab, anchor = build_table(df)
    tab.to_csv(DATA_DIR / 'e9_iso_matched.csv', index=False)

    matched = tab[tab['arm'] == 'matched']
    iso = tab[tab['arm'] == 'iso_grid']
    have = matched['best_test_acc'].notna()

    md = [
        '# E9: schedules that preserve the coupling, and a matched-contraction '
        'comparison',
        '',
        'ResNet-18 / CIFAR-100, `B = 128`, `eta_0 = 0.1`, `T = 100`, SGDM '
        '(`beta = 0.9`), seed 42, cosine learning rate throughout — so these are '
        'directly comparable to the main protocol rather than to the constant-LR '
        'E8 arms.',
        '',
        f'Contraction budget unit: `C = {anchor:.4g}`, i.e. '
        '`lambda_ref * sum_t eta_t` at the reference setting '
        '(`lambda_ref = 5.982e-4`, the value our rule already predicts there).',
        '',
        '## (a) Matched cumulative contraction',
        '',
        'Every shape `m_lambda(t)` is rescaled so all methods spend the same '
        '`sum_t eta_t*lambda_t`. `iso_product` is the reviewer\'s '
        '`lambda_t = lambda_0*eta_0/eta_t` (cap: the multiplier is limited to '
        f'{1 / R.ISO_M_FLOOR:g}x, i.e. eta is floored at eta_0/'
        f'{1 / R.ISO_M_FLOOR:g} inside the lambda formula).',
        '',
        '| budget | shape | lambda_0 | realized sum eta*lambda | best acc | train acc |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for _, r in matched.sort_values(['budget_factor', 'shape']).iterrows():
        acc = f"{r['best_test_acc']:.2f}" if np.isfinite(r['best_test_acc']) else 'PENDING'
        tr = f"{r['train_acc']:.1f}" if np.isfinite(r['train_acc']) else '-'
        md.append(
            f"| {r['budget_factor']:.3g} C | {r['shape']} | {r['lambda0']:.4g} | "
            f"{r['contraction']:.3g} ({r['contraction_over_C']:.2f} C) | "
            f"{acc} | {tr} |")

    if have.any():
        by_budget = matched[have].groupby('budget_factor')['best_test_acc']
        md += ['', 'Spread across shapes at fixed budget:', '']
        for factor, g in by_budget:
            md.append(f'- budget {factor:.3g} C: '
                      f'{g.min():.2f} to {g.max():.2f} '
                      f'(spread {g.max() - g.min():.2f} pp, n={len(g)})')
        md += ['',
               'Spread across budgets at fixed shape:', '']
        for shape, g in matched[have].groupby('shape')['best_test_acc']:
            md.append(f'- {shape}: {g.min():.2f} to {g.max():.2f} '
                      f'(spread {g.max() - g.min():.2f} pp, n={len(g)})')

    md += [
        '',
        '## (b) Iso-product arm on the standard lambda_0 ladder',
        '',
        '| lambda_0 | realized sum eta*lambda | best acc | train acc | diverged |',
        '|---:|---:|---:|---:|---:|',
    ]
    for _, r in iso.sort_values('lambda0').iterrows():
        acc = f"{r['best_test_acc']:.2f}" if np.isfinite(r['best_test_acc']) else 'PENDING'
        tr = f"{r['train_acc']:.1f}" if np.isfinite(r['train_acc']) else '-'
        md.append(f"| {r['lambda0']:.4g} | {r['contraction']:.3g} "
                  f"({r['contraction_over_C']:.2f} C) | {acc} | {tr} | "
                  f"{int(r['diverged'])} |")

    md += ['', '## Reference points at the same eta_0, T, optimizer', '',
           '| method | best acc |', '|---|---:|']
    for name, acc in REFERENCES:
        md.append(f'| {name} | {acc:.2f} |')

    md += ['', 'Figure: `outputs/plots/nips26/e9_iso_matched.png`.', '']
    (DATA_DIR / 'e9_table.md').write_text('\n'.join(md))

    tokens = ['# E9 resolved tokens', '']
    if have.any():
        best = matched.loc[matched[have]['best_test_acc'].idxmax()]
        iso_have = iso['best_test_acc'].notna()
        spreads = matched[have].groupby('budget_factor')['best_test_acc'].agg(
            lambda s: s.max() - s.min())
        tokens += [
            f'- `[[E9-C]]` = {anchor:.4g}',
            f'- `[[E9-MATCHED-BEST]]` = {best["best_test_acc"]:.2f} '
            f'({best["shape"]} at {best["budget_factor"]:.3g} C)',
            f'- `[[E9-SHAPE-SPREAD]]` = '
            + ', '.join(f'{v:.2f}' for v in spreads) + ' pp at budgets '
            + ', '.join(f'{k:.3g}C' for k in spreads.index),
            f'- `[[E9-BUDGET-SPREAD]]` = '
            f'{matched[have]["best_test_acc"].max() - matched[have]["best_test_acc"].min():.2f} pp',
        ]
        if iso_have.any():
            tokens.append(
                f'- `[[E9-ISO-BEST]]` = {iso[iso_have]["best_test_acc"].max():.2f} '
                f'at lambda_0 = '
                f'{iso.loc[iso[iso_have]["best_test_acc"].idxmax(), "lambda0"]:.4g}')
    else:
        tokens.append('PENDING: no E9 runs found yet')
    (DATA_DIR / 'e9_tokens.md').write_text('\n'.join(tokens) + '\n')

    plot(tab, anchor)

    print('\n'.join(md))
    print()
    print('\n'.join(tokens))
    n_missing = int((~have).sum()) + int(iso['best_test_acc'].isna().sum())
    print(f'\nmissing cells: {n_missing}')
    print(f'wrote {DATA_DIR / "e9_table.md"}')


if __name__ == '__main__':
    main()
