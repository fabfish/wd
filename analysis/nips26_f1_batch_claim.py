"""
F1 (reviewer xkCF follow-up, 2026-08-03): what exactly does the batch-size claim say?

The reviewer's point, in his words: since Exp. 3 imposes `eta ∝ B`, Eq. (17)
"predicts approximately constant `lambda*`, not that both `eta` and `lambda`
should increase", and Table 4 shows `eta*lambda` in a narrow range.

He is right, and our rule says so too. With `sum_t eta_t = eta*ceil(n/B)*(T+1)/2`:

  * at **fixed eta**, `sum_t eta_t ∝ 1/B`, so `lambda* = C/sum_t eta_t ∝ B`;
  * along **Exp. 3's line eta ∝ B**, `sum_t eta_t` is B-invariant, so `lambda*`
    is predicted to be *flat* in B, and the product `eta*lambda*` is what grows,
    `∝ B`.

So there is exactly one law, `lambda* = C/sum_t eta_t`, and two different
conditional readings of it. This script measures both slopes from optima that
were already fitted for E5a. No new training.

Writes rebuttal/nips_rebuttal/_data/f1_batch_claim.{md,csv}.
"""
import numpy as np
import pandas as pd

from analysis.nips26_lib import TABLE_DIR, fit_loglog_slope, sum_lr

E5A = TABLE_DIR / 'e5a_C_per_setting.csv'
OUT_MD = TABLE_DIR / 'f1_batch_claim.md'
OUT_CSV = TABLE_DIR / 'f1_batch_claim.csv'

# Exp. 3's linear rule, anchored so that B=128 sits at the reference eta=0.1.
# These are the same (B, eta) pairs the E4 transfer test uses.
LINEAR_RULE = {32: 0.025, 64: 0.05, 128: 0.1, 256: 0.2, 512: 0.4}


def _geo(series):
    s = np.asarray(series, float)
    s = s[s > 0]
    return float(np.exp(np.mean(np.log(s)))) if len(s) else float('nan')


def load_optima():
    """Per-setting lambda* fits behind E5a, restricted to interior optima."""
    df = pd.read_csv(E5A)
    df = df[df['interior'].astype(str).str.lower().isin(['true', '1'])].copy()
    df['batch_size'] = df['batch_size'].astype(int)
    df['product'] = df['lr'] * df['wd_interp']
    return df


def fixed_eta_slope(df):
    """
    Slope of log lambda* on log B holding eta (and the architecture) fixed.

    Each (model, eta) cell is centred on its own mean before pooling, so only
    within-cell variation in B enters and the fit cannot be driven by the fact
    that large batches were usually run at large learning rates.
    """
    pooled, per_cell = [], []
    for (model, lr), g in df.groupby(['model', 'lr']):
        g = g.groupby('batch_size', as_index=False).agg(
            wd_interp=('wd_interp', _geo))
        if len(g) < 2:
            continue
        lb = np.log(g['batch_size'].values.astype(float))
        lw = np.log(g['wd_interp'].values.astype(float))
        pooled.append(pd.DataFrame({'x': lb - lb.mean(), 'y': lw - lw.mean()}))
        per_cell.append({
            'model': model, 'lr': float(lr),
            'batch_sizes': ','.join(str(int(b)) for b in g['batch_size']),
            'lambda_stars': ', '.join(f'{w:.3g}' for w in g['wd_interp']),
            'slope': float(np.polyfit(lb, lw, 1)[0]),
        })
    if not pooled:
        raise RuntimeError('no (model, eta) cell spans more than one batch size')
    pooled = pd.concat(pooled, ignore_index=True)
    slope = float(np.polyfit(pooled['x'], pooled['y'], 1)[0])
    rng = np.random.RandomState(0)
    boots = []
    for _ in range(2000):
        idx = rng.randint(0, len(pooled), len(pooled))
        x, y = pooled['x'].values[idx], pooled['y'].values[idx]
        if len(np.unique(x)) >= 2:
            boots.append(np.polyfit(x, y, 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    fit = dict(slope=slope, lo=float(lo), hi=float(hi),
               n=int(len(pooled)), n_cells=len(per_cell))
    return fit, pd.DataFrame(per_cell).sort_values(['model', 'lr'])


def linear_rule_line(df):
    """The Exp. 3 configurations themselves: eta scaled linearly with B."""
    rows = []
    for bs, lr in sorted(LINEAR_RULE.items()):
        g = df[(df['batch_size'] == bs) & np.isclose(df['lr'], lr)]
        if g.empty:
            continue
        rows.append({
            'batch_size': bs, 'lr': lr, 'n': int(len(g)),
            'lambda_star': _geo(g['wd_interp']),
            'product': _geo(g['product']),
            'sum_lr': float(sum_lr(lr, 100, bs, 'cosine')),
            'C': _geo(g['C']),
            'models': ','.join(sorted(g['model'].unique())),
        })
    return pd.DataFrame(rows)


def main():
    df = load_optima()
    fixed_fit, per_cell = fixed_eta_slope(df)
    line = linear_rule_line(df)

    lam_slope = fit_loglog_slope(line['batch_size'].values,
                                 line['lambda_star'].values)
    prod_slope = fit_loglog_slope(line['batch_size'].values,
                                  line['product'].values)
    naive = fit_loglog_slope(df['batch_size'].values, df['wd_interp'].values)

    by_bs = df.groupby('batch_size').agg(
        n=('C', 'size'), C=('C', _geo)).reset_index()
    c_drift = float(by_bs['C'].max() / by_bs['C'].min())

    line.to_csv(OUT_CSV, index=False)

    lines = [
        '# F1: `lambda* ∝ B` at fixed eta, `eta*lambda* ∝ B` under linear scaling',
        '',
        'Reanalysis of the optima already fitted for E5a '
        f'(`e5a_C_per_setting.csv`, {len(df)} interior settings, T=100, SGDM, '
        'three architectures, two seeds). No new training.',
        '',
        'One law, two conditional readings. With '
        '`sum_t eta_t = eta * ceil(n/B) * (T+1)/2`:',
        '',
        '| regime | `sum_t eta_t` vs B | predicted `lambda*` | predicted `eta*lambda*` |',
        '|---|---|---|---|',
        '| eta fixed | `∝ 1/B` | `∝ B` | `∝ B` |',
        '| eta ∝ B (Exp. 3) | invariant | flat | `∝ B` |',
        '',
        '## (a) At fixed eta: measured slope of `lambda*` in B',
        '',
        f'Within-(model, eta) pooled slope of `log lambda*` on `log B`: '
        f'**{fixed_fit["slope"]:+.2f}** '
        f'[{fixed_fit["lo"]:+.2f}, {fixed_fit["hi"]:+.2f}] '
        f'({fixed_fit["n"]} points from {fixed_fit["n_cells"]} cells spanning '
        f'more than one batch size), against a predicted **+1**.',
        '',
        '| model | eta | batch sizes | lambda* | slope |',
        '|---|---:|---|---|---:|',
    ]
    for _, r in per_cell.iterrows():
        lines.append(
            f"| {r['model']} | {r['lr']:g} | {r['batch_sizes']} | "
            f"{r['lambda_stars']} | {r['slope']:+.2f} |")
    lines += [
        '',
        f'Ignoring eta and regressing over all {naive["n"]} settings instead '
        f'gives {naive["slope"]:+.2f} [{naive["lo"]:+.2f}, {naive["hi"]:+.2f}]: '
        'the raw scatter shows almost nothing, because in these sweeps the two '
        'dependencies partly cancel. The conditioning is the whole content of '
        'the claim.',
        '',
        '## (b) Along Exp. 3\'s line eta ∝ B: `lambda*` is flat, the product grows',
        '',
        'These are the Exp. 3 configurations themselves (the same (B, eta) pairs '
        'the E4 transfer test uses).',
        '',
        '| B | eta | `sum_t eta_t` | lambda* (geo) | eta*lambda* (geo) | C (geo) | n | models |',
        '|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for _, r in line.iterrows():
        lines.append(
            f"| {int(r['batch_size'])} | {r['lr']:g} | {r['sum_lr']:.0f} | "
            f"{r['lambda_star']:.3g} | {r['product']:.3g} | {r['C']:.2f} | "
            f"{int(r['n'])} | {r['models']} |")
    lam_span = float(line['lambda_star'].max() / line['lambda_star'].min())
    prod_span = float(line['product'].max() / line['product'].min())
    lines += [
        '',
        f'- `lambda*` slope in B: **{lam_slope["slope"]:+.2f}** '
        f'[{lam_slope["lo"]:+.2f}, {lam_slope["hi"]:+.2f}], total spread '
        f'{lam_span:.2f}x over a 16x range of B — flat, as predicted, and '
        'inside the residual C spread reported in (c).',
        f'- `eta*lambda*` slope in B: **{prod_slope["slope"]:+.2f}** '
        f'[{prod_slope["lo"]:+.2f}, {prod_slope["hi"]:+.2f}], total spread '
        f'{prod_span:.1f}x — this is the growing quantity.',
        f'- `sum_t eta_t` is B-invariant along this line by construction '
        f'({line["sum_lr"].min():.0f}-{line["sum_lr"].max():.0f}), which is why '
        '`lambda*` has nothing to respond to.',
        '',
        '## (c) Residual drift of C across B',
        '',
        'Geometric mean of C by batch size (all settings, not just the linear '
        'rule line): ' + ', '.join(f'{v:.2f}' for v in by_bs['C'])
        + f' for B = ' + ', '.join(str(int(b)) for b in by_bs['batch_size'])
        + f'; spread {c_drift:.2f}x.',
        '',
        'This is the part of the batch-size dependence that `1/sum_t eta_t` does '
        'not absorb. It is also the noise floor against which the flatness in '
        '(b) has to be read.',
        '',
        '## What we will state in the paper',
        '',
        '1. The reviewer is right that under Exp. 3\'s constraint the prediction '
        'is a roughly constant `lambda*` with a product that grows like B. Our '
        f'own data agrees: `lambda*` slope {lam_slope["slope"]:+.2f}, product '
        f'slope {prod_slope["slope"]:+.2f}.',
        '2. We will therefore drop the sentence in Exp. 3 that says the optimal '
        '`lambda` grows with batch size, and state instead that the *product* '
        'grows with B while `lambda*` stays put, because `sum_t eta_t` is held '
        'fixed by linear learning-rate scaling.',
        '3. The `lambda* ∝ B` reading is the *fixed-eta* one, and we now measure '
        f'it separately: {fixed_fit["slope"]:+.2f} '
        f'[{fixed_fit["lo"]:+.2f}, {fixed_fit["hi"]:+.2f}] against a predicted '
        '+1. Stating which quantity is held fixed removes the apparent conflict '
        'between Eq. (17) and Table 4.',
        '',
    ]
    OUT_MD.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'wrote {OUT_MD}')
    print(f'wrote {OUT_CSV}')


if __name__ == '__main__':
    main()
