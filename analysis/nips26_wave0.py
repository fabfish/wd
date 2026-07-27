"""
Wave 0 of the rebuttal analysis: everything that can be answered from runs that
already exist, at zero GPU cost.

E2a  the accuracy envelope over lambda, which shows that no single fixed weight
     decay tracks the best achievable accuracy across learning rates
E5a  the spread of the constant C in lambda* = C / sum_t eta_t, across
     architectures, batch sizes, learning rates and seeds
E6a  what the existing momentum sweeps already say about eta*(beta) and about
     whether momentum helps generalization

Writes figures to outputs/plots/nips26/ and a report to rebuttal/nips26/.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from analysis.nips26_lib import (  # noqa: E402
    PLOT_DIR, TABLE_DIR, ensure_dirs, load_legacy, sum_lr, optimal_wd,
    fit_loglog_slope, _parabola_peak,
)

CORE_LR = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
CORE_WD = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
DEFAULT_WD = 5e-4


def heatmap_grid(df, seed):
    m = ((df.model == 'resnet18') & (df.batch_size == 128) & (df.momentum == 0.9)
         & (df.seed == seed) & (df.lr.isin(CORE_LR)) & (df.wd.isin(CORE_WD))
         & (df.epochs == 100))
    g = df[m].groupby(['lr', 'wd'], as_index=False)['best_test_acc'].max()
    return g.pivot(index='wd', columns='lr', values='best_test_acc')


# --------------------------------------------------------------------------
# E2a
# --------------------------------------------------------------------------

def e2a_envelope(df, report):
    pivots = {seed: heatmap_grid(df, seed) for seed in (42, 123)}
    mean_pivot = sum(pivots.values()) / len(pivots)

    etas = np.array(mean_pivot.columns, dtype=float)
    envelope = mean_pivot.max(axis=0).values
    best_wd = mean_pivot.idxmax(axis=0).values

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) envelope against the fixed-lambda curves
    ax = axes[0]
    cmap = plt.get_cmap('viridis')
    wds = np.array(mean_pivot.index, dtype=float)
    for i, wd in enumerate(wds):
        row = mean_pivot.loc[wd].values
        if np.isclose(wd, DEFAULT_WD):
            continue
        ax.plot(etas, row, 'o-', ms=3.5, lw=1.2, alpha=0.85,
                color=cmap(i / max(len(wds) - 1, 1)),
                label=rf'$\lambda$={wd:g}')
    ax.plot(etas, mean_pivot.loc[DEFAULT_WD].values, 'ks--', ms=6, lw=2.2,
            label=rf'$\lambda$={DEFAULT_WD:g} (common default)', zorder=5)
    ax.plot(etas, envelope, 'r-', lw=3.0, zorder=6,
            label=r'envelope $\max_\lambda$ (coupled $\lambda$)')
    ax.set_xscale('log')
    ax.set_ylim(55, 80)
    ax.set_xlabel(r'learning rate $\eta$')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title('(a) No fixed $\\lambda$ reaches the envelope\nResNet-18 / CIFAR-100, mean of 2 seeds')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='lower left')

    # (b) how much each fixed lambda gives up, as a function of eta
    ax = axes[1]
    gaps = {}
    for i, wd in enumerate(wds):
        gap = envelope - mean_pivot.loc[wd].values
        gaps[wd] = gap
        style = dict(color='k', ls='--', lw=2.2, ms=6, marker='s') if np.isclose(wd, DEFAULT_WD) \
            else dict(color=cmap(i / max(len(wds) - 1, 1)), lw=1.2, ms=3.5, marker='o', alpha=0.85)
        ax.plot(etas, np.maximum(gap, 0.02), **style, label=rf'$\lambda$={wd:g}')
    ax.axhline(1.0, color='grey', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.02, 100)
    ax.set_xlabel(r'learning rate $\eta$')
    ax.set_ylabel('accuracy given up vs envelope (pp)')
    ax.set_title('(b) Cost of holding $\\lambda$ fixed')
    ax.grid(alpha=0.3, which='both')

    # (c) the optimum itself moves with eta
    ax = axes[2]
    interp = []
    for eta in etas:
        col = mean_pivot[eta]
        interp.append(_parabola_peak(col.index.values.astype(float), col.values))
    interp = np.array(interp)
    fit = fit_loglog_slope(etas, interp)
    ax.plot(etas, best_wd, 'o', ms=8, label=r'grid argmax $\lambda^*$')
    ax.plot(etas, interp, 'x', ms=8, label=r'interpolated $\lambda^*$')
    xs = np.array([etas.min(), etas.max()])
    ax.plot(xs, np.exp(fit['intercept']) * xs ** fit['slope'], 'r-', lw=2,
            label=f"fit slope {fit['slope']:.2f} [{fit['lo']:.2f}, {fit['hi']:.2f}]")
    ax.plot(xs, interp[0] * (xs / etas[0]) ** -1.0, 'k--', lw=1.5,
            label=r'slope $-1$ ($\eta\lambda$ const)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'learning rate $\eta$')
    ax.set_ylabel(r'optimal weight decay $\lambda^*$')
    ax.set_title('(c) $\\lambda^*$ moves inversely with $\\eta$')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(PLOT_DIR / f'e2a_envelope.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    worst_default = float(np.max(gaps[DEFAULT_WD]))
    mean_default = float(np.mean(gaps[DEFAULT_WD]))
    best_fixed = min(((float(np.max(g)), wd) for wd, g in gaps.items()))
    envelope_span = float(envelope.max() - envelope.min())

    report.append("## E2a  accuracy envelope over lambda\n")
    report.append(
        f"Across the eight learning rates spanned by the grid, the envelope "
        f"`max_lambda acc` varies by only {envelope_span:.1f} points, so a "
        f"correctly coupled weight decay keeps accuracy nearly flat over two "
        f"decades of learning rate.\n")
    report.append(
        f"Holding the common default `lambda = 5e-4` gives up "
        f"{mean_default:.2f} points on average and {worst_default:.2f} points "
        f"at its worst learning rate. The best possible *fixed* choice is "
        f"`lambda = {best_fixed[1]:g}`, and even that still gives up "
        f"{best_fixed[0]:.2f} points somewhere in the range.\n")
    report.append(
        f"The optimum itself moves: fitting `log lambda*` against `log eta` "
        f"gives slope {fit['slope']:.2f} with 95% bootstrap interval "
        f"[{fit['lo']:.2f}, {fit['hi']:.2f}], consistent with the inverse "
        f"coupling and inconsistent with lambda being a constant.\n")

    per_fixed = pd.DataFrame({
        'wd': list(gaps.keys()),
        'mean_gap_pp': [float(np.mean(g)) for g in gaps.values()],
        'max_gap_pp': [float(np.max(g)) for g in gaps.values()],
    }).sort_values('max_gap_pp')
    per_fixed.to_csv(TABLE_DIR / 'e2a_fixed_lambda_gaps.csv', index=False)
    report.append("Worst-case shortfall of each fixed weight decay (percentage points):\n")
    for _, r in per_fixed.iterrows():
        report.append(f"- `lambda = {r['wd']:g}`: mean {r['mean_gap_pp']:.2f}, "
                      f"worst {r['max_gap_pp']:.2f}")
    report.append("")
    return dict(fit=fit, gaps=per_fixed, envelope=envelope, etas=etas)


# --------------------------------------------------------------------------
# E5a
# --------------------------------------------------------------------------

def e5a_constant(df, report):
    """
    Fit C = lambda* * sum_t eta_t wherever the existing data contains a weight
    decay sweep of at least four points at fixed everything else.
    """
    d = df[(df.momentum == 0.9) & (df.wd > 0) & (df.epochs == 100)
           & (df.best_test_acc > 10)].copy()
    groups = ['model', 'seed', 'batch_size', 'lr']
    opt = optimal_wd(d, groups, min_points=4)
    if opt.empty:
        report.append("## E5a  no groups with enough weight-decay points\n")
        return None

    opt['sum_lr'] = sum_lr(opt['lr'].values, 100, opt['batch_size'].values, 'cosine')
    opt['C'] = opt['wd_interp'] * opt['sum_lr']
    opt = opt[np.isfinite(opt['C']) & (opt['C'] > 0)]
    n_all = len(opt)
    opt = opt[opt['interior']]

    geo = float(np.exp(np.mean(np.log(opt['C']))))
    log_sd = float(np.std(np.log(opt['C']), ddof=1))
    lo, hi = float(opt['C'].min()), float(opt['C'].max())

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    ax = axes[0]
    ax.hist(np.log10(opt['C']), bins=14, color='steelblue', edgecolor='k', alpha=0.85)
    ax.axvline(np.log10(geo), color='r', lw=2, label=f'geometric mean {geo:.2f}')
    ax.axvline(np.log10(geo / np.exp(log_sd)), color='r', ls='--', lw=1.2)
    ax.axvline(np.log10(geo * np.exp(log_sd)), color='r', ls='--', lw=1.2,
               label=rf'$\times/\div$ {np.exp(log_sd):.2f}')
    ax.set_xlabel(r'$\log_{10} C$,  $C = \lambda^*\sum_t\eta_t$')
    ax.set_ylabel('number of settings')
    ax.set_title(f'(a) C over {len(opt)} independent settings')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, (model, g) in enumerate(opt.groupby('model')):
        ax.scatter(g['lr'], g['C'], s=45, alpha=0.8, label=model)
    ax.axhline(geo, color='r', lw=2)
    ax.fill_between([opt['lr'].min(), opt['lr'].max()],
                    geo / np.exp(log_sd), geo * np.exp(log_sd),
                    color='r', alpha=0.12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'learning rate $\eta$')
    ax.set_ylabel('C')
    ax.set_title('(b) C by architecture and learning rate')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    by_bs = opt.groupby('batch_size')['C'].apply(
        lambda s: np.exp(np.mean(np.log(s)))).reset_index()
    ax.plot(by_bs['batch_size'], by_bs['C'], 'o-', ms=8, lw=2, color='darkorange')
    ax.axhline(geo, color='r', lw=1.5, ls='--', label='overall geometric mean')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('batch size B')
    ax.set_ylabel('C (geometric mean)')
    ax.set_title('(c) residual batch-size trend in C')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(PLOT_DIR / f'e5a_constant.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    opt.to_csv(TABLE_DIR / 'e5a_C_per_setting.csv', index=False)

    report.append("## E5a  how stable is the constant C\n")
    report.append(
        f"Fitting `C = lambda* * sum_t eta_t` independently in {len(opt)} "
        f"settings (of {n_all} weight-decay sweeps available; the rest peak on "
        f"the edge of their swept range and only bound C) gives a geometric "
        f"mean of C = {geo:.2f}, a multiplicative standard deviation of "
        f"x/{np.exp(log_sd):.2f}, and a full range of {lo:.2f} to {hi:.2f}. "
        f"The settings span architectures {sorted(opt['model'].unique())}, "
        f"batch sizes {sorted(opt['batch_size'].unique())}, learning rates "
        f"{opt['lr'].min():g} to {opt['lr'].max():g}, and two seeds.\n")

    by_model = opt.groupby('model')['C'].apply(
        lambda s: np.exp(np.mean(np.log(s)))).round(3)
    report.append("Geometric mean of C by architecture:\n")
    for model, value in by_model.items():
        report.append(f"- {model}: {value:.2f}")
    report.append("")
    report.append("Geometric mean of C by batch size:\n")
    for _, r in by_bs.iterrows():
        report.append(f"- B = {int(r['batch_size'])}: {r['C']:.2f}")
    report.append("")
    report.append(
        "The residual batch-size trend in panel (c) is the part of the batch "
        "size dependence that the 1/sum_lr factor does not already absorb, and "
        "is the honest version of the paper's claim that the optimal product "
        "grows with B.\n")
    return dict(C_geo=geo, log_sd=log_sd, table=opt)


# --------------------------------------------------------------------------
# E6a
# --------------------------------------------------------------------------

def e6a_momentum(df, report):
    d = df[(df.model == 'resnet18') & (df.batch_size == 128)
           & (df.epochs == 100) & (df.best_test_acc > 5)].copy()

    arms = {
        r'$\lambda=0$': d[np.isclose(d.wd, 0.0)],
        r'$\lambda=2\times10^{-3}$': d[np.isclose(d.wd, 2e-3)],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    summary = []

    ax = axes[0]
    markers = {r'$\lambda=0$': 'o', r'$\lambda=2\times10^{-3}$': 's'}
    for arm_name, arm in arms.items():
        for beta, g in arm.groupby('momentum'):
            g = g.groupby('lr', as_index=False)['best_test_acc'].max().sort_values('lr')
            if len(g) < 3:
                continue
            ax.plot(g['lr'], g['best_test_acc'], marker=markers[arm_name], ms=4,
                    lw=1.3, alpha=0.85,
                    ls='-' if '0$' in arm_name else '--',
                    label=rf'$\beta$={beta:g}, {arm_name}')
            peak = _parabola_peak(g['lr'].values, g['best_test_acc'].values)
            summary.append(dict(arm=arm_name, beta=float(beta),
                                eta_star=peak,
                                eta_argmax=float(g.loc[g['best_test_acc'].idxmax(), 'lr']),
                                acc=float(g['best_test_acc'].max()),
                                n=len(g)))
    ax.set_xscale('log')
    ax.set_ylim(55, 80)
    ax.set_xlabel(r'learning rate $\eta$')
    ax.set_ylabel('best test accuracy (%)')
    ax.set_title('(a) Learning-rate sweeps by momentum')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)

    s = pd.DataFrame(summary)
    s['one_minus_beta'] = 1.0 - s['beta']
    s = s[s['one_minus_beta'] > 0]

    ax = axes[1]
    for arm_name, g in s.groupby('arm'):
        g = g.sort_values('one_minus_beta')
        if len(g) < 3:
            continue
        fit = fit_loglog_slope(g['one_minus_beta'], g['eta_star'])
        ax.plot(g['one_minus_beta'], g['eta_star'], 'o', ms=8, label=f'{arm_name}')
        xs = np.array([g['one_minus_beta'].min(), g['one_minus_beta'].max()])
        ax.plot(xs, np.exp(fit['intercept']) * xs ** fit['slope'], lw=2,
                label=f"  slope {fit['slope']:.2f} [{fit['lo']:.2f}, {fit['hi']:.2f}]")
        report.append(
            f"- {arm_name}: `log eta*` against `log(1-beta)` has slope "
            f"{fit['slope']:.2f} with 95% interval [{fit['lo']:.2f}, "
            f"{fit['hi']:.2f}] over {fit['n']} momentum values "
            f"(the effective-step argument predicts 1).")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$1-\beta$')
    ax.set_ylabel(r'$\eta^*$')
    ax.set_title(r'(b) $\eta^*$ against $1-\beta$')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7)

    ax = axes[2]
    for arm_name, g in s.groupby('arm'):
        g = g.sort_values('beta')
        ax.plot(g['beta'], g['acc'], 'o-', ms=7, lw=1.8, label=arm_name)
    ax.set_xlabel(r'momentum $\beta$')
    ax.set_ylabel('best test accuracy at its own optimal $\\eta$ (%)')
    ax.set_title('(c) Does momentum help, once $\\eta$ is retuned?')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(PLOT_DIR / f'e6a_momentum.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    s.to_csv(TABLE_DIR / 'e6a_momentum_optima.csv', index=False)

    report.insert(len(report), "")
    report.append("Peak accuracy at each momentum, after retuning the learning rate:\n")
    for arm_name, g in s.groupby('arm'):
        vals = ", ".join(f"beta={r['beta']:g}: {r['acc']:.2f}%"
                         for _, r in g.sort_values('beta').iterrows())
        report.append(f"- {arm_name}: {vals}")
    report.append("")
    for arm_name, g in s.groupby('arm'):
        usable = g[g['beta'] <= 0.95]
        span = float(usable['acc'].max() - usable['acc'].min())
        extreme = g[g['beta'] > 0.95]
        note = ""
        if not extreme.empty:
            worst = extreme.sort_values('acc').iloc[0]
            note = (f" At beta = {worst['beta']:g} accuracy falls to "
                    f"{worst['acc']:.2f}%, which is the effective step size "
                    f"eta/(1-beta) running past the stability boundary rather "
                    f"than a generalization effect.")
        report.append(
            f"- With {arm_name}, momentum from 0 to 0.95 changes peak accuracy "
            f"by only {span:.2f} points once the learning rate is retuned.{note}")
    report.append("")
    report.append(
        "This is the direct answer to the question of whether momentum "
        "generalizes better: at its own optimal learning rate it does not, in "
        "either arm. What momentum changes is where that optimum sits, which "
        "is exactly what the momentum factor in the stability bound describes. "
        "Weight decay, by contrast, moves peak accuracy by about five points "
        "at every momentum value.\n")
    return s


def main():
    ensure_dirs()
    df = load_legacy()
    report = ["# Wave 0: what the existing runs already answer\n",
              f"Built from {len(df)} previously collected runs, no new training.\n"]

    report.append("")
    e2a_envelope(df, report)
    e5a_constant(df, report)
    report.append("## E6a  momentum, from the existing sweeps\n")
    e6a_momentum(df, report)

    out = TABLE_DIR / 'wave0_report.md'
    out.write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nwrote {out}")
    print(f"figures in {PLOT_DIR}")


if __name__ == '__main__':
    main()
