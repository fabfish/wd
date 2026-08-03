"""
Report E10: does a C calibrated on small MLPs predict lambda* on larger ones?

Reads the ladder + held-out runs produced by
`mlp_wd/scripts/run_e10_c_width.py` and the blind predictions recorded in
`_data/e10_predictions_<dataset>.json`, then writes:

  _data/e10_C_by_width_<ds>.csv        C fitted per (momentum, width)
  _data/e10_ladder_optima_<ds>.csvthe per-cell lambda* behind it
  _data/e10_heldout_<ds>.csv           per-cell gap of every rule to the oracle
  _data/e10_heldout_table_<ds>.md      the summary table for the response
  _data/e10_tokens.md                  resolved tokens (both datasets)
  outputs/plots/nips26/e10_c_width_<ds>.png

Usage:
  python -m mlp_wd.analysis.report_e10_c_width --dataset mnist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.nips26_lib import sum_lr  # noqa: E402
from mlp_wd.analysis.e10_c_width import (  # noqa: E402
    DATASET_N, PLOT_DIR, RULES, TABLE_DIR, C_by_width, fit_C_grid,
    fit_C_vs_width, geo, predict_C, read_predictions,
)
from mlp_wd.scripts.run_e10_c_width import SETTINGS, load_pool  # noqa: E402

RULE_LABEL = {
    'ours': 'ours: C_pred(h)/sum_t eta_t',
    'default': 'fixed default 5e-4',
    'wang': '1/(eta*T)',
    'kosson':'constant eta*lambda',
}


def find_run(pool, *, hidden_dim, momentum, lr, wd, cfg):
    m = ((pool['hidden_dim'].astype(int) == int(hidden_dim))
         & np.isclose(pool['momentum'].astype(float), momentum)
         & np.isclose(pool['lr'].astype(float), lr)
         & np.isclose(pool['wd'].astype(float), wd, rtol=1e-4, atol=1e-15)
         & (pool['epochs'].astype(int) == cfg['epochs'])
         & (pool['batch_size'].astype(int) == cfg['batch_size'])
         & (pool['num_layers'].astype(int) == cfg['num_layers']))
    hit = pool[m].copy()
    if hit.empty:
        return None
    # A diverged run can carry a NaN loss. Keep it (it is a real outcome for a
    # rule that picked that lambda) but never let it win the argmin.
    hit['_loss'] = pd.to_numeric(hit['best_test_loss'], errors='coerce')
    if hit['_loss'].notna().any():
        return hit.loc[hit['_loss'].idxmin()]
    return hit.iloc[0]


def heldout_table(pool, pred, cfg, dataset):
    """Per (width, momentum, eta): oracle over the grid vs each zero-tuning rule."""
    n = DATASET_N[dataset]
    rows = []
    for mom_key, entry in pred['per_momentum'].items():
        mom = float(mom_key)
        for h_key, per_lr in entry['lambda_pred'].items():
            h = int(h_key)
            C_pred = float(entry['C_pred'][h_key])
            for lr_key in per_lr:
                lr = float(lr_key)
                # The oracle is the best weight decay anyone measured in this
                # cell: the 5-point ladder plus every rule's own lambda. Using
                # only the ladder would let an off-ladder rule "beat" the oracle,
                # which just means the ladder is coarse, not that the rule is
                # better than tuning.
                candidates = list(cfg['wds']) + [
                    float(f"{float(per_lr[lr_key][rule]):.6g}") for rule in RULES]
                grid = []
                for wd in sorted(set(candidates)):
                    if not np.isfinite(wd) or wd <= 0:
                        continue
                    hit = find_run(pool, hidden_dim=h, momentum=mom, lr=lr,
                                   wd=wd, cfg=cfg)
                    if hit is None:
                        continue
                    loss = float(pd.to_numeric(hit['best_test_loss'],
                                               errors='coerce'))
                    if not np.isfinite(loss):
                        continue
                    grid.append((float(wd), loss, float(hit['best_test_acc'])))
                if len(grid) < 3:
                    continue
                g = pd.DataFrame(grid, columns=['wd', 'loss', 'acc'])
                oracle_loss = float(g['loss'].min())
                oracle_acc = float(g['acc'].max())
                wd_oracle = float(g.loc[g['loss'].idxmin(), 'wd'])
                S = float(sum_lr(lr, cfg['epochs'], cfg['batch_size'],
                                 'cosine', n=n))
                for rule in RULES:
                    lam = float(f"{float(per_lr[lr_key][rule]):.6g}")
                    hit = find_run(pool, hidden_dim=h, momentum=mom, lr=lr,
                                   wd=lam, cfg=cfg)
                    if hit is None:
                        continue
                    loss = float(pd.to_numeric(hit['best_test_loss'],
                                               errors='coerce'))
                    acc = float(hit['best_test_acc'])
                    rows.append({
                        'hidden_dim': h, 'momentum': mom, 'lr': lr,
                        'rule': rule, 'lambda_rule': lam,
                        'lambda_oracle': wd_oracle,
                        'ratio': lam / wd_oracle if wd_oracle > 0 else np.nan,
                        'loss': loss, 'acc': acc,
                        'oracle_loss': oracle_loss, 'oracle_acc': oracle_acc,
                        'gap_loss': loss - oracle_loss,
                        'gap_acc': oracle_acc - acc,
                        'diverged': int(hit.get('diverged', 0) or 0),
                        'n_grid': int(len(g)), 'C_pred': C_pred, 'sum_lr': S,
                    })
    return pd.DataFrame(rows)


def plot(cw, fits, pred, held, cfg, dataset):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    for mom, g in cw.groupby('momentum'):
        g = g.sort_values('hidden_dim')
        ax.plot(g['hidden_dim'], g['C'], 'o-', ms=7,
                label=f'ladder fit, momentum {mom:g}')
        fit = fits[float(mom)]
        hs = np.array(sorted(set(list(g['hidden_dim'])
                                 + list(cfg['heldout_widths']))), dtype=float)
        ax.plot(hs, [predict_C(fit, h) for h in hs], '--', lw=1.2, alpha=0.8,
                label=f'  slope {fit["slope"]:+.2f}')
    for h in cfg['heldout_widths']:
        ax.axvline(h, color='r', ls=':', lw=1.0)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('MLP width (hidden units)')
    ax.set_ylabel(r'$C = \lambda^* \sum_t \eta_t$')
    ax.set_title('(a) C across the width ladder\n(dotted: held-out widths)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)

    ax = axes[1]
    if not held.empty:
        for rule, g in held.groupby('rule'):
            ax.plot(g['lambda_oracle'], g['lambda_rule'], 'o', ms=7,
                    alpha=0.8, label=RULE_LABEL[rule])
        lim = [held[['lambda_rule', 'lambda_oracle']].values.min() * 0.6,
               held[['lambda_rule', 'lambda_oracle']].values.max() * 1.6]
        ax.plot(lim, lim, 'k--', lw=1.0, label='perfect prediction')
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda^*$ from the held-out oracle grid')
    ax.set_ylabel(r'$\lambda$ predicted with zero tuning')
    ax.set_title('(b) blind prediction vs oracle\nat the held-out widths')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)

    ax = axes[2]
    if not held.empty:
        agg = held.groupby('rule')['gap_acc'].agg(['mean', 'max'])
        agg = agg.reindex([r for r in RULES if r in agg.index])
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg['mean'], 0.4, label='mean gap')
        ax.bar(x + 0.2, agg['max'], 0.4, label='worst gap')
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=15)
    ax.set_ylabel('test-accuracy gap to oracle (pp)')
    ax.set_title('(c) cost of not tuning\n(lower is better)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f'e10_c_width_{dataset}.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description='E10 report')
    ap.add_argument('--dataset', default='mnist', choices=sorted(SETTINGS))
    args = ap.parse_args()
    dataset = args.dataset
    cfg = SETTINGS[dataset]
    n = DATASET_N[dataset]

    pool = load_pool(cfg, dataset)
    if pool.empty:
        raise SystemExit('no runs found yet')

    opt = fit_C_grid(pool, lrs=cfg['lrs'], wds=cfg['wds'],
                     epochs=cfg['epochs'], batch_size=cfg['batch_size'], n=n,
                     num_layers=cfg['num_layers'], widths=cfg['ladder_widths'])
    cw = C_by_width(opt)
    fits = fit_C_vs_width(cw)
    opt.to_csv(TABLE_DIR / f'e10_ladder_optima_{dataset}.csv', index=False)
    cw.to_csv(TABLE_DIR / f'e10_C_by_width_{dataset}.csv', index=False)

    pred = read_predictions(dataset)
    held = heldout_table(pool, pred, cfg, dataset)
    held.to_csv(TABLE_DIR / f'e10_heldout_{dataset}.csv', index=False)

    # C measured directly at the held-out widths, for the honest comparison
    # against what the ladder extrapolated.
    try:
        opt_h = fit_C_grid(pool, lrs=cfg['lrs'], wds=cfg['wds'],
                           epochs=cfg['epochs'], batch_size=cfg['batch_size'],
                           n=n, num_layers=cfg['num_layers'],
                           widths=cfg['heldout_widths'])
        cw_h = C_by_width(opt_h)
    except RuntimeError:
        cw_h = pd.DataFrame(columns=['momentum', 'hidden_dim', 'C'])

    md = [
        f'# E10 ({dataset}): held-out test of the constant C across MLP widths',
        '',
        f'{cfg["num_layers"]}-layer ReLU MLP, no normalization, '
        f'{dataset.upper()}, `B = {cfg["batch_size"]}`, `T = {cfg["epochs"]}` '
        f'epochs, cosine learning rate, seed {cfg["seed"]}; momenta '
        f'{cfg["momenta"]}, learning rates {cfg["lrs"]}, weight-decay ladder '
        f'{cfg["wds"]}.',
        '',
        f'Calibration widths **{cfg["ladder_widths"]}**, held out '
        f'**{cfg["heldout_widths"]}**. Predictions were written to '
        f'`e10_predictions_{dataset}.json` at '
        f'{pred.get("written_at")} '
        f'({"blind" if pred.get("blind") else "NOT BLIND - held-out rows existed"}).',
        '',
        '## (a) C across the calibration ladder',
        '',
        '| momentum | width | C | n eta cells |',
        '|---:|---:|---:|---:|',
    ]
    for _, r in cw.iterrows():
        md.append(f"| {r['momentum']:g} | {int(r['hidden_dim'])} | "
                  f"{r['C']:.3f} | {int(r['n_lr'])} |")
    md += ['', 'Regression of `log C` on `log width`:', '']
    for mom, fit in sorted(fits.items()):
        md.append(f'- momentum {mom:g}: slope **{fit["slope"]:+.3f}** '
                  f'[{fit["lo"]:+.3f}, {fit["hi"]:+.3f}] over widths '
                  f'{fit["widths"]}; C = '
                  + ', '.join(f'{c:.3f}' for c in fit['C']))
    md += ['', 'Extrapolated to the held-out widths:', '',
           '| momentum | width | C predicted | C measured | ratio |',
           '|---:|---:|---:|---:|---:|']
    for mom, fit in sorted(fits.items()):
        for h in cfg['heldout_widths']:
            cp = predict_C(fit, h)
            hit = cw_h[(np.isclose(cw_h['momentum'], mom))
                       & (cw_h['hidden_dim'] == h)] if not cw_h.empty else cw_h
            cm = float(hit['C'].iloc[0]) if len(hit) else np.nan
            md.append(f"| {mom:g} | {h} | {cp:.3f} | "
                      + (f'{cm:.3f} | {cp / cm:.2f}x |' if np.isfinite(cm)
                         else 'PENDING | - |'))

    if held.empty:
        md += ['', '## (b) Held-out comparison', '', 'PENDING: no held-out runs yet.', '']
    else:
        agg = held.groupby('rule').agg(
            n=('gap_acc', 'size'),
            n_valid=('gap_acc', lambda s: int(np.isfinite(s).sum())),
            n_diverged=('diverged', 'sum'),
            gap_acc_mean=('gap_acc', 'mean'), gap_acc_max=('gap_acc', 'max'),
            gap_loss_mean=('gap_loss', 'mean'), gap_loss_max=('gap_loss', 'max'),
            ratio_geo=('ratio', geo),
        ).reindex([r for r in RULES if r in held['rule'].unique()])
        md += [
            '', '## (b) Zero-tuning rules at the held-out widths',
            '',
            'Every rule is applied blind. The oracle for each cell is the best '
            'weight decay measured anywhere in that cell — the '
            f'{len(cfg["wds"])}-point ladder plus every rule\'s own lambda — so '
            f'no rule can appear to beat tuning just because the ladder is '
            f'coarse. Tuning cost: {len(cfg["wds"])}+ training runs per '
            '(width, momentum, eta) cell for the oracle, one run for any rule, '
            'zero after a one-time calibration.',
            '',
            'Gaps are averaged over the cells that trained; diverged cells are '
            'counted separately rather than imputed.',
            '',
            '| rule | mean acc gap (pp) | worst acc gap (pp) | mean loss gap | '
            'worst loss gap | lambda_rule/lambda_oracle (geo) | cells | diverged |',
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for rule, r in agg.iterrows():
            md.append(
                f"| {RULE_LABEL[rule]} | {r['gap_acc_mean']:.2f} | "
                f"{r['gap_acc_max']:.2f} | {r['gap_loss_mean']:.4f} | "
                f"{r['gap_loss_max']:.4f} | {r['ratio_geo']:.2f}x | "
                f"{int(r['n_valid'])}/{int(r['n'])} | {int(r['n_diverged'])} |")
        md += ['', 'Per-cell detail:', '',
               '| width | momentum | eta | rule | lambda | lambda oracle | '
               'acc | oracle acc | acc gap |',
               '|---:|---:|---:|---|---:|---:|---:|---:|---:|']
        for _, r in held.sort_values(
                ['hidden_dim', 'momentum', 'lr', 'rule']).iterrows():
            acc = 'diverged' if not np.isfinite(r['acc']) else f"{r['acc']:.2f}"
            gap = '-' if not np.isfinite(r['gap_acc']) else f"{r['gap_acc']:+.2f}"
            md.append(
                f"| {int(r['hidden_dim'])} | {r['momentum']:g} | {r['lr']:g} | "
                f"{r['rule']} | {r['lambda_rule']:.4g} | "
                f"{r['lambda_oracle']:.4g} | {acc} | "
                f"{r['oracle_acc']:.2f} | {gap} |")

    fig_path = plot(cw, fits, pred, held, cfg, dataset)
    md += ['', f'Figure: `{fig_path.relative_to(REPO_ROOT)}`.', '']
    out_md = TABLE_DIR / f'e10_heldout_table_{dataset}.md'
    out_md.write_text('\n'.join(md))

    tokens = [f'# E10 tokens ({dataset})', '']
    for mom, fit in sorted(fits.items()):
        tokens.append(f'- `[[E10-SLOPE-M{mom:g}]]` = {fit["slope"]:+.2f} '
                      f'[{fit["lo"]:+.2f}, {fit["hi"]:+.2f}]')
    if not held.empty:
        agg = held.groupby('rule')['gap_acc'].agg(['mean', 'max'])
        for rule in RULES:
            if rule in agg.index:
                tokens.append(
                    f'- `[[E10-GAP-{rule.upper()}]]` = '
                    f'{agg.loc[rule, "mean"]:.2f} mean / '
                    f'{agg.loc[rule, "max"]:.2f} worst pp')
        tokens.append(
            f'- `[[E10-RATIO-OURS]]` = '
            f'{geo(held[held["rule"] == "ours"]["ratio"]):.2f}x')
    tokens.append(f'- `[[E10-FIG-{dataset.upper()}]]` = '
                  f'{fig_path.relative_to(REPO_ROOT)}')
    (TABLE_DIR / f'e10_tokens_{dataset}.md').write_text('\n'.join(tokens) + '\n')

    print('\n'.join(md))
    print()
    print('\n'.join(tokens))
    print(f'\nwrote {out_md}')


if __name__ == '__main__':
    main()
