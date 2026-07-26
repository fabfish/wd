"""
Resolve the placeholder tokens used in the rebuttal drafts.

Every number quoted in rebuttal/nips_rebuttal/*/response.md that is not already
literal comes from here. Run this after each wave; anything without data yet is
reported as PENDING rather than guessed.

    python -m analysis.nips26_report
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from analysis.nips26_lib import (  # noqa: E402
    PLOT_DIR, TABLE_DIR, ensure_dirs, load_all, sum_lr, optimal_wd,
    fit_loglog_slope, _parabola_peak, predict_wd, reference_point,
    STRATEGIES, TRANSFER_CONFIGS, steps_per_epoch,
)

PENDING = 'PENDING'


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return PENDING
    return f'{x:.{nd}g}'


def _series_optima(df, group_col, acc_col='best_test_acc'):
    """lambda* per value of group_col, with an interior-optimum flag."""
    out = optimal_wd(df, [group_col], acc_col=acc_col, min_points=3)
    return out.sort_values(group_col) if not out.empty else out


# --------------------------------------------------------------------------
# E1: does the optimum move with the training length?
# --------------------------------------------------------------------------

def e1(df, tokens, notes, min_ladder=6):
    """
    Prefer the dense ladder (e1_fine / e1_full). The coarse e1_prelim ladder
    steps by up to 5x and cannot resolve the predicted ~8x shift in lambda*
    between T=25 and T=200; quoting its slope would be misleading.
    """
    base = df[(df.model == 'resnet18') & (df.batch_size == 128)
              & (df.momentum == 0.9) & (df.wd > 0) & (df.seed == 42)
              & (df.scheduler == 'cosine')]

    main = base[np.isclose(base.lr, 0.1)].copy()
    # Prefer dense-ladder runs when available; fall back to everything only if
    # the dense series is empty (should not happen once the queue advances).
    if 'exp' in main.columns:
        dense = main[main['exp'].isin(['e1_fine', 'e1_full'])]
        if not dense.empty:
            main = dense
        else:
            notes.append("E1: only the coarse e1_prelim ladder is available; "
                         "tokens left PENDING until e1_fine finishes")
            # Still report qualitative notes, but do not fill headline tokens.
            opt_coarse = _series_optima(main, 'epochs')
            if not opt_coarse.empty:
                notes.append(
                    "E1 (coarse, do not quote): "
                    + ", ".join(
                        f"T={int(r.epochs)} wd_interp={r.wd_interp:.2e} "
                        f"(argmax={r.wd_argmax:g}, interior={r.interior})"
                        for r in opt_coarse.itertuples()))
            return None

    # Require each T to have a dense enough ladder before fitting.
    counts = main.groupby('epochs')['wd'].nunique()
    usable_T = counts[counts >= min_ladder].index
    main = main[main['epochs'].isin(usable_T)]
    opt = _series_optima(main, 'epochs')
    if opt.empty or len(opt) < 2:
        notes.append(f"E1: need >=2 training lengths with >= {min_ladder} "
                     f"lambda values each; have "
                     f"{list(zip(counts.index.astype(int), counts.values))}")
        return None

    fit = fit_loglog_slope(opt['epochs'], opt['wd_interp'])
    tokens['E1-T-SLOPE'] = _fmt(fit['slope'], 3)
    tokens['E1-T-CI'] = f"[{fit['lo']:.2f}, {fit['hi']:.2f}]" if np.isfinite(fit['lo']) else PENDING
    for T in (25, 100, 200):
        row = opt[opt['epochs'] == T]
        tokens[f'E1-T-LAMBDA-{T}'] = _fmt(float(row['wd_interp'].iloc[0]), 3) \
            if not row.empty else PENDING

    if len(opt) >= 2:
        lo, hi = opt.iloc[0], opt.iloc[-1]
        drift = (lo['wd_interp'] * 0.1) / (hi['wd_interp'] * 0.1)
        tokens['E1-PRODUCT-DRIFT'] = (
            f"{drift:.2f} (from T={int(lo['epochs'])} to T={int(hi['epochs'])})")

    # second learning rate arm
    low = base[np.isclose(base.lr, 0.02)]
    opt_low = _series_optima(low, 'epochs')
    if len(opt_low) >= 2:
        fit_low = fit_loglog_slope(opt_low['epochs'], opt_low['wd_interp'])
        tokens['E1-LOWLR-SLOPE'] = (
            f"{fit_low['slope']:.2f} [{fit_low['lo']:.2f}, {fit_low['hi']:.2f}]")
        both = pd.concat([opt.assign(lr=0.1), opt_low.assign(lr=0.02)])
        both['S'] = sum_lr(both['lr'].values, both['epochs'].values, 128, 'cosine')
        both['C'] = both['wd_interp'] * both['S']
        spread = float(np.exp(np.std(np.log(both['C']), ddof=1)))
        tokens['E1-ETAT-COLLAPSE'] = f"x/{spread:.2f} over {len(both)} points"

    # constant-LR arm
    const = df[(df.model == 'resnet18') & (df.batch_size == 128)
               & (df.momentum == 0.9) & (df.wd > 0) & (df.seed == 42)
               & (df.scheduler == 'const') & np.isclose(df.lr, 0.1)]
    opt_const = _series_optima(const, 'epochs')
    if not opt_const.empty:
        ratios = []
        for _, r in opt_const.iterrows():
            match = opt[opt['epochs'] == r['epochs']]
            if not match.empty:
                ratios.append(r['wd_interp'] / float(match['wd_interp'].iloc[0]))
        if ratios:
            tokens['E1-SCHED-RATIO'] = (
                f"{np.mean(ratios):.2f} (prediction 0.5, n={len(ratios)})")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    for T, g in main.groupby('epochs'):
        g = g.groupby('wd', as_index=False)['best_test_acc'].max().sort_values('wd')
        ax.plot(g['wd'], g['best_test_acc'], 'o-', ms=5, label=f'T = {int(T)} epochs')
    ax.set_xscale('log')
    ax.set_xlabel(r'weight decay $\lambda$')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title(r'(a) The optimum in $\lambda$ shifts with training length'
                 '\n' r'ResNet-18 / CIFAR-100, $\eta=0.1$, $B=128$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(opt['epochs'], opt['wd_interp'], 'o', ms=10, label=r'measured $\lambda^*$')
    xs = np.array([opt['epochs'].min(), opt['epochs'].max()], dtype=float)
    ax.plot(xs, np.exp(fit['intercept']) * xs ** fit['slope'], 'r-', lw=2.2,
            label=f"fit slope {fit['slope']:.2f} [{fit['lo']:.2f}, {fit['hi']:.2f}]")
    anchor = float(opt['wd_interp'].iloc[0]) * float(opt['epochs'].iloc[0])
    ax.plot(xs, anchor / xs, 'k--', lw=1.6, label=r'ours: slope $-1$')
    ax.plot(xs, [float(opt['wd_interp'].iloc[0])] * 2, ls=':', color='purple', lw=1.8,
            label=r'rotational equilibrium: slope $0$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('training length T (epochs)')
    ax.set_ylabel(r'optimal weight decay $\lambda^*$')
    ax.set_title(r'(b) $\lambda^*$ against $T$: the discriminating test')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(PLOT_DIR / f'e1_training_length.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    opt.to_csv(TABLE_DIR / 'e1_optima.csv', index=False)
    notes.append(
        f"E1: {len(opt)} training lengths "
        f"({', '.join(str(int(t)) for t in opt['epochs'])}), "
        f"slope {fit['slope']:.3f} [{fit['lo']:.2f}, {fit['hi']:.2f}]. "
        f"Interior optima: {opt['interior'].sum()}/{len(opt)}.")
    if not opt['interior'].all():
        notes.append("  WARNING: some optima sit on the edge of the lambda ladder; "
                     "widen the grid before quoting the slope.")
    return opt


# --------------------------------------------------------------------------
# E2b: is accuracy flat along a line of constant eta*lambda?
# --------------------------------------------------------------------------

def e2b(df, tokens, notes):
    d = df[(df.exp == 'e2b')] if 'exp' in df.columns else pd.DataFrame()
    if d.empty:
        notes.append("E2b: no runs yet")
        return None
    d = d.copy()
    d['product'] = (d['lr'] * d['wd']).round(12)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    summary = []
    for product, g in d.groupby('product'):
        g = g.sort_values('lr')
        peak = float(g['best_test_acc'].max())
        drop_lo = peak - float(g['best_test_acc'].iloc[0])
        drop_hi = peak - float(g['best_test_acc'].iloc[-1])
        within = g[g['best_test_acc'] >= peak - 1.0]
        span = (float(within['lr'].max() / within['lr'].min())
                if len(within) > 1 else 1.0)
        summary.append(dict(product=product, peak=peak, drop_low_eta=drop_lo,
                            drop_high_eta=drop_hi, span_within_1pp=span))
        ax.plot(g['lr'], g['best_test_acc'], 'o-', ms=6,
                label=rf'$\eta\lambda$ = {product:.2g}')
    ax.set_xscale('log')
    ax.set_xlabel(r'learning rate $\eta$ (with $\lambda$ set to hold $\eta\lambda$ fixed)')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title('Accuracy along a line of constant $\\eta\\lambda$\n'
                 'flat would mean only the product matters')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(PLOT_DIR / f'e2b_isoproduct.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    s = pd.DataFrame(summary)
    s.to_csv(TABLE_DIR / 'e2b_isoproduct.csv', index=False)
    worst = float(max(s['drop_low_eta'].max(), s['drop_high_eta'].max()))
    tokens['E2B-ISO-DROP'] = f"{worst:.1f}"
    tokens['E2B-ISO-RANGE'] = f"a factor of {s['span_within_1pp'].max():.0f} in eta"
    notes.append(f"E2b: {len(d)} runs over {len(s)} product levels, "
                 f"largest end-to-peak drop {worst:.2f} points")
    return s


# --------------------------------------------------------------------------
# E3: the learning-rate ceiling
# --------------------------------------------------------------------------

def e3(df, tokens, notes, acc_floor=5.0):
    d = df[(df.exp == 'e3')] if 'exp' in df.columns else pd.DataFrame()
    if d.empty:
        notes.append("E3: no runs yet")
        return None
    d = d.copy()
    d['stable'] = (d['diverged'].astype(int) == 0) & (d['best_test_acc'] >= acc_floor)

    rows = []
    for (momentum, wd), g in d.groupby(['momentum', 'wd']):
        stable = g[g['stable']]['lr']
        unstable = g[~g['stable']]['lr']
        if stable.empty or unstable.empty:
            continue
        lo = float(stable.max())
        higher = unstable[unstable > lo]
        if higher.empty:
            continue
        hi = float(higher.min())
        rows.append(dict(momentum=momentum, wd=wd, eta_max=np.sqrt(lo * hi),
                         bracket_lo=lo, bracket_hi=hi, tightness=hi / lo))
    if not rows:
        notes.append("E3: no bracketed thresholds yet")
        return None
    b = pd.DataFrame(rows).sort_values(['momentum', 'wd'])
    b.to_csv(TABLE_DIR / 'e3_boundary.csv', index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for momentum, g in b.groupby('momentum'):
        g = g.sort_values('wd')
        ax.errorbar(g['wd'], 1.0 / g['eta_max'],
                    yerr=[1 / g['eta_max'] - 1 / g['bracket_hi'],
                          1 / g['bracket_lo'] - 1 / g['eta_max']],
                    fmt='o', ms=7, capsize=3, label=rf'$\beta$ = {momentum:g}')
        if len(g) >= 3:
            slope, intercept = np.polyfit(g['wd'], 1.0 / g['eta_max'], 1)
            xs = np.linspace(0, g['wd'].max(), 50)
            ax.plot(xs, intercept + slope * xs, lw=1.8,
                    label=rf'  fit: $1/\eta_{{max}}$ = {slope:.2f}$\lambda$ + {intercept:.2f}')
            if np.isclose(momentum, 0.0):
                tokens['E3-SLOPE'] = f"{slope:.2f}"
                tokens['E3-INTERCEPT'] = f"{intercept:.2f} (implies L = {2 * intercept:.1f})"
    ax.set_xlabel(r'weight decay $\lambda$')
    ax.set_ylabel(r'$1/\eta_{max}$')
    ax.set_title('Divergence boundary against weight decay\n'
                 r'theory: $1/\eta_{max} = \lambda + L/2$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(PLOT_DIR / f'e3_boundary.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    zero = b[np.isclose(b['momentum'], 0.0)]
    nine = b[np.isclose(b['momentum'], 0.9)]
    if not zero.empty and not nine.empty:
        merged = zero.merge(nine, on='wd', suffixes=('_0', '_9'))
        if not merged.empty:
            ratio = float(np.exp(np.mean(np.log(merged['eta_max_9'] / merged['eta_max_0']))))
            tokens['E3-MOM-RATIO'] = f"{ratio:.2f}"
    notes.append(f"E3: thresholds bracketed for {len(b)} (beta, lambda) pairs; "
                 f"median bracket tightness {b['tightness'].median():.2f}x")
    return b


# --------------------------------------------------------------------------
# E4: zero-tuning transfer against the alternatives
# --------------------------------------------------------------------------

def e4(df, tokens, notes, min_oracle_points=5):
    try:
        ref = reference_point()
    except RuntimeError as exc:
        notes.append(f"E4: {exc}")
        return None

    records, missing, thin = [], 0, []
    for cfg in TRANSFER_CONFIGS:
        pool = df[(df.model == cfg['model'])
                  & (df.batch_size == cfg['batch_size'])
                  & np.isclose(df.lr, cfg['lr'])
                  & (df.epochs == cfg['epochs'])
                  & np.isclose(df.momentum, 0.9)
                  & (df.scheduler == 'cosine')]
        n_oracle = int(pool['wd'].nunique()) if not pool.empty else 0
        # A "gap to oracle" computed against two or three weight decays is not a
        # gap to an oracle, so such a setting is reported as incomplete instead
        # of being averaged in.
        if n_oracle < min_oracle_points:
            thin.append(f"{cfg['name']} ({n_oracle} lambda values)")
            missing += len(STRATEGIES)
            continue
        oracle = float(pool['best_test_acc'].max())
        for strategy in STRATEGIES:
            wd = predict_wd(strategy, cfg['lr'], cfg['epochs'], cfg['batch_size'],
                            ref['C'], ref)
            hit = pool[np.isclose(pool['wd'], wd, rtol=0.05, atol=1e-12)]
            if hit.empty:
                missing += 1
                continue
            acc = float(hit['best_test_acc'].max())
            records.append(dict(config=cfg['name'], strategy=strategy,
                                wd=wd, acc=acc, oracle=oracle,
                                gap=oracle - acc, oracle_runs=n_oracle))
    if thin:
        notes.append("E4: oracle grid too thin in " + "; ".join(thin))
    if not records:
        notes.append("E4: no transfer cells available yet")
        return None

    t = pd.DataFrame(records)
    complete = [c for c, g in t.groupby('config')
                if g['strategy'].nunique() == len(STRATEGIES)]
    if len(complete) < len(TRANSFER_CONFIGS):
        notes.append(f"E4: {len(complete)}/{len(TRANSFER_CONFIGS)} settings have "
                     f"all {len(STRATEGIES)} strategies; means below cover only those")
    t = t[t['config'].isin(complete)]
    if t.empty:
        notes.append("E4: no setting has every strategy yet, nothing quotable")
        return None
    t.to_csv(TABLE_DIR / 'e4_transfer.csv', index=False)
    pivot = t.pivot_table(index='config', columns='strategy', values='gap')
    pivot = pivot.reindex(columns=[s for s in STRATEGIES if s in pivot.columns])
    (TABLE_DIR / 'e4_transfer_table.md').write_text(
        "Accuracy gap to the per-setting oracle, in percentage points "
        "(lower is better)\n\n" + pivot.round(2).to_markdown())
    tokens['E4-TABLE'] = "see _data/e4_transfer_table.md"

    for strategy in STRATEGIES:
        g = t[t['strategy'] == strategy]
        if g.empty:
            continue
        name = {'ours': 'OURS', 'default': 'DEFAULT', 'kosson': 'KOSSON',
                'wang': 'WANG', 'zero': 'ZERO'}[strategy]
        tokens[f'E4-{name}-MEAN'] = f"{g['gap'].mean():.2f}"
        if strategy == 'ours':
            tokens['E4-OURS-WORST'] = f"{g['gap'].max():.2f}"

    notes.append(f"E4: {len(t)} of {len(TRANSFER_CONFIGS) * len(STRATEGIES)} "
                 f"cells resolved ({missing} still missing)")
    return t


# --------------------------------------------------------------------------
# E5b: how much does a wrong constant cost?
# --------------------------------------------------------------------------

def e5b(df, tokens, notes):
    d = df[(df.exp == 'e5b')] if 'exp' in df.columns else pd.DataFrame()
    if d.empty:
        notes.append("E5b: no runs yet")
        return None
    ref = reference_point()
    rows = []
    for (lr, T, B), g in d.groupby(['lr', 'epochs', 'batch_size']):
        pool = df[(df.model == 'resnet18') & (df.batch_size == B)
                  & np.isclose(df.lr, lr) & (df.epochs == T)
                  & np.isclose(df.momentum, 0.9) & (df.wd > 0)
                  & (df.scheduler == 'cosine')]
        if pool.empty:
            continue
        best = float(pool['best_test_acc'].max())
        S = float(sum_lr(lr, T, B, 'cosine'))
        for _, r in g.iterrows():
            rows.append(dict(lr=lr, epochs=T, batch_size=B,
                             factor=float(r['wd']) * S / ref['C'],
                             acc=float(r['best_test_acc']), loss=best - float(r['best_test_acc'])))
    if not rows:
        notes.append("E5b: runs present but no reference optimum to compare against")
        return None
    s = pd.DataFrame(rows)
    s.to_csv(TABLE_DIR / 'e5b_sensitivity.csv', index=False)

    for target, token in ((3.0, 'E5B-3X'), (10.0, 'E5B-10X')):
        near = s[np.isclose(s['factor'], target, rtol=0.35)
                 | np.isclose(s['factor'], 1.0 / target, rtol=0.35)]
        if not near.empty:
            tokens[token] = f"{near['loss'].max():.2f}"

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for (lr, T, B), g in s.groupby(['lr', 'epochs', 'batch_size']):
        g = g.sort_values('factor')
        ax.plot(g['factor'], g['loss'], 'o-', ms=7,
                label=rf'$\eta$={lr:g}, T={int(T)}, B={int(B)}')
    ax.axvline(1.0, color='k', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('factor by which C is wrong')
    ax.set_ylabel('accuracy given up (pp)')
    ax.set_title('Cost of mis-specifying the constant')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(PLOT_DIR / f'e5b_sensitivity.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    notes.append(f"E5b: {len(s)} mis-specified runs analysed")
    return s


# --------------------------------------------------------------------------
# E6b: momentum, the optimum, and the generalization gap
# --------------------------------------------------------------------------

def e6b(df, tokens, notes):
    d = df[(df.exp == 'e6b')] if 'exp' in df.columns else pd.DataFrame()
    if d.empty:
        notes.append("E6b: no runs yet")
        return None
    d = d.copy()
    d['gap'] = d['final_train_acc'] - d['final_test_acc']

    coupled = d[d['wd'] > 0]
    opt = optimal_wd(coupled, ['momentum'], min_points=2)
    if len(opt) >= 3:
        opt['one_minus_beta'] = 1.0 - opt['momentum']
        fit = fit_loglog_slope(opt['one_minus_beta'], opt['wd_interp'])
        tokens['E6B-LAMBDA-SLOPE'] = (
            f"{fit['slope']:.2f} [{fit['lo']:.2f}, {fit['hi']:.2f}]")

    for beta, label in ((0.0, 'SGD'), (0.9, 'SGDM')):
        g = coupled[np.isclose(coupled['momentum'], beta)]
        if g.empty:
            continue
        best = g.loc[g['best_test_acc'].idxmax()]
        tokens[f'E6B-GAP-{label}'] = f"{best['gap']:.1f} points"

    d.to_csv(TABLE_DIR / 'e6b_momentum.csv', index=False)
    notes.append(f"E6b: {len(d)} runs")
    return d


def main():
    ensure_dirs()
    df = load_all()
    if 'exp' not in df.columns:
        df['exp'] = ''
    df['exp'] = df['exp'].fillna('')

    tokens, notes = {}, []
    for fn in (e1, e2b, e3, e4, e5b, e6b):
        try:
            fn(df, tokens, notes)
        except Exception as exc:  # keep one failing section from blocking the rest
            notes.append(f"{fn.__name__}: FAILED with {type(exc).__name__}: {exc}")

    all_tokens = [
        'E1-T-SLOPE', 'E1-T-CI', 'E1-T-LAMBDA-25', 'E1-T-LAMBDA-100',
        'E1-T-LAMBDA-200', 'E1-PRODUCT-DRIFT', 'E1-LOWLR-SLOPE',
        'E1-ETAT-COLLAPSE', 'E1-SCHED-RATIO', 'E2B-ISO-DROP', 'E2B-ISO-RANGE',
        'E3-SLOPE', 'E3-INTERCEPT', 'E3-LMAX', 'E3-MOM-RATIO', 'E4-TABLE',
        'E4-OURS-MEAN', 'E4-OURS-WORST', 'E4-DEFAULT-MEAN', 'E4-KOSSON-MEAN',
        'E4-WANG-MEAN', 'E4-ZERO-MEAN', 'E5B-3X', 'E5B-10X',
        'E6B-LAMBDA-SLOPE', 'E6B-GAP-SGD', 'E6B-GAP-SGDM',
        'E7-DIVERGENCE-RATIO', 'E7-PLATEAU', 'E7-BN',
    ]

    lines = ["# Resolved placeholder values", "",
             f"From {len(df)} runs ({int((df['exp'] != '').sum())} from this round).", ""]
    ready = 0
    for token in all_tokens:
        value = tokens.get(token, PENDING)
        if value != PENDING:
            ready += 1
        lines.append(f"- `[[{token}]]` = {value}")
    lines += ["", f"{ready} of {len(all_tokens)} resolved.", "", "## Notes", ""]
    lines += [f"- {n}" for n in notes]

    out = TABLE_DIR / 'resolved_tokens.md'
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
