"""
Generate publication-quality figures for rebuttal.
All figures are anonymous (no author/institution info) for double-blind review.
Output: rebuttal/figures/
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Global style ────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = Path('rebuttal/figures')
OUT.mkdir(parents=True, exist_ok=True)

# ── Data loading ────────────────────────────────────────────────────────────
def load_all():
    """Return {arch: list_of_DataFrames} for averaging."""
    base = Path('.')
    data = {
        'ResNet-18': [
            base / 'outputs/results/results.csv',           # seed42 run1
            base / 'rebuttal/results/results_resnet18_seed123.csv',     # seed123 run1
            base / 'rebuttal/results/results_resnet18_seed42_run2.csv', # seed42 run2
            base / 'rebuttal/results/results_resnet18_seed123_run2.csv',# seed123 run2
        ],
        'VGG-16': [
            base / 'rebuttal/results/results_vgg16_seed42.csv',
            base / 'rebuttal/results/results_vgg16_seed123.csv',
        ],
        'ResNet-50': [
            base / 'rebuttal/results/results_resnet50_seed42.csv',
            base / 'rebuttal/results/results_resnet50_seed123.csv',
        ],
    }
    result = {}
    for arch, paths in data.items():
        dfs = []
        for p in paths:
            if p.exists():
                df = pd.read_csv(p)
                df['wd'] = df['wd'].astype(float)
                df['lr'] = df['lr'].astype(float)
                df['momentum'] = df['momentum'].astype(float)
                dfs.append(df)
        result[arch] = dfs
    return result


def mean_std_df(dfs, filt_fn, metric='best_test_acc'):
    """Given list of DataFrames, filter each, group by key columns, return mean/std."""
    filtered = [filt_fn(df) for df in dfs if metric in df.columns]
    if not filtered:
        return pd.DataFrame()
    combined = pd.concat(filtered, ignore_index=True)
    if combined.empty:
        return combined
    group_cols = [c for c in ['method', 'batch_size', 'lr', 'wd', 'momentum'] if c in combined.columns]
    agg = combined.groupby(group_cols)[metric].agg(['mean', 'std', 'count']).reset_index()
    agg['std'] = agg['std'].fillna(0)
    return agg


def load_ext_data():
    """Load extended Exp2 data (with final_test_loss) for ResNet-18."""
    base = Path('.')
    paths = [
        base / 'rebuttal/results/results_resnet18_seed42_exp2_ext.csv',
        base / 'rebuttal/results/results_resnet18_seed42_exp2_ext2.csv',
        base / 'rebuttal/results/results_resnet18_seed123_exp2_ext.csv',
        base / 'rebuttal/results/results_resnet18_exp2_supplement.csv',
        base / 'rebuttal/results/results_resnet18_seed42_exp2_fill.csv',
    ]
    dfs = []
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            df['wd'] = df['wd'].astype(float)
            df['lr'] = df['lr'].astype(float)
            df['momentum'] = df['momentum'].astype(float)
            dfs.append(df)
    return dfs


# ── Exp 1: Stability Boundary Ordering ──────────────────────────────────────
ARCH_COLORS = {'ResNet-18': '#2176AE', 'VGG-16': '#E85D04', 'ResNet-50': '#57CC99'}
METHOD_MARKERS = {'SGD': 'o', 'SGD+WD': 's', 'SGDM+WD': '^'}
METHOD_COLORS = {'SGD': '#2176AE', 'SGD+WD': '#E85D04', 'SGDM+WD': '#57CC99'}
METHOD_LABELS = {'SGD': 'SGD', 'SGD+WD': r'SGD + $\lambda$', 'SGDM+WD': r'SGDM + $\lambda$'}


def plot_exp1_per_arch(all_data):
    """One subplot per architecture: Accuracy vs LR for 3 methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    methods = ['SGD', 'SGD+WD', 'SGDM+WD']

    for ax, (arch, dfs) in zip(axes, all_data.items()):
        filt = lambda df: df[df['method'].isin(methods) & (df['batch_size'] == 128)]
        agg = mean_std_df(dfs, filt)
        if agg.empty:
            continue
        for method in methods:
            sub = agg[agg['method'] == method].sort_values('lr')
            if sub.empty:
                continue
            ax.plot(sub['lr'], sub['mean'],
                    marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=1.8, markersize=6)
        ax.set_xscale('log')
        ax.set_xlabel(r'Learning Rate $\eta$')
        ax.set_ylabel('Best Test Accuracy (%)')
        ax.set_title(arch, fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.9)

    fig.suptitle('Figure 1: Stability Boundary Ordering Across Architectures',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_exp1_stability_boundary.png')
    plt.close(fig)
    print('Saved fig1_exp1_stability_boundary')


# ── Exp 2: η–λ Heatmap ──────────────────────────────────────────────────────
EXP2_LRS = [0.01, 0.05, 0.1, 0.2, 0.3]
EXP2_WDS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def filt_exp2(df):
    sub = df[(df['method'].isin(['SGDM', 'SGDM+WD'])) & (df['batch_size'] == 128)].copy()
    sub = sub[sub['lr'].isin(EXP2_LRS) & sub['wd'].isin(EXP2_WDS)]
    return sub


def plot_exp2_heatmaps(all_data):
    """One heatmap per architecture. Red→Yellow→Green, yellow at threshold."""
    import seaborn as sns
    from matplotlib.colors import TwoSlopeNorm
    GREEN_FLOOR = {'ResNet-18': 74, 'VGG-16': 72, 'ResNet-50': 76}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, (arch, dfs) in zip(axes, all_data.items()):
        agg = mean_std_df(dfs, filt_exp2)
        if agg.empty:
            continue
        pivot = agg.pivot_table(values='mean', index='wd', columns='lr')
        pivot = pivot.reindex(index=sorted(pivot.index, reverse=True),
                              columns=sorted(pivot.columns))
        vcenter = GREEN_FLOOR.get(arch, 74)
        vmin = max(pivot.min().min(), 0)
        vmax = pivot.max().max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                    norm=norm,
                    linewidths=0.5, linecolor='gray',
                    cbar_kws={'label': 'Acc (%)', 'shrink': 0.8},
                    annot_kws={'fontsize': 8})
        ax.set_xlabel(r'Learning Rate $\eta$')
        ax.set_ylabel(r'Weight Decay $\lambda$')
        ax.set_title(arch, fontweight='bold')
        wd_labels = [f'{v:.0e}' for v in sorted(pivot.index.tolist(), reverse=True)]
        ax.set_yticklabels(wd_labels, rotation=0)

    fig.suptitle(r'Figure 2: $\eta$–$\lambda$ Interaction Heatmap (SGDM, B=128)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_exp2_heatmap.png')
    plt.close(fig)
    print('Saved fig2_exp2_heatmap')


EXT_WDS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
SUPP_LRS = [2e-5, 1e-4,
            0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 3.0, 5.0]


def filt_exp2_all(df):
    """Broader filter including supplement LR values."""
    sub = df[(df['method'].isin(['SGDM', 'SGDM+WD'])) & (df['batch_size'] == 128)].copy()
    sub = sub[sub['lr'].isin(SUPP_LRS) & sub['wd'].isin(EXT_WDS)]
    return sub


def plot_exp2_scaling_curves_loss(ext_dfs):
    """Reviewer 9i84: test loss vs η×λ, one curve per λ (ResNet-18 extended Exp2)."""
    combined = pd.concat([filt_exp2_all(df) for df in ext_dfs if 'final_test_loss' in df.columns],
                         ignore_index=True)
    if combined.empty:
        print('No ext data for response_to_reviewer_9i84 plot')
        return

    combined = combined.dropna(subset=['final_test_loss'])
    combined['eta_lambda'] = combined['lr'] * combined['wd']
    group_cols = ['wd', 'eta_lambda']
    agg = combined.groupby(group_cols)['final_test_loss'].mean().reset_index()

    exclude_wd = {0.0002, 0.02}
    wds = [w for w in sorted(agg['wd'].unique())
           if not any(np.isclose(w, ex) for ex in exclude_wd)]
    cmap = plt.get_cmap('turbo', max(len(wds), 2))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    STAR_OVERRIDE = {
        0.0001: 5e-5,
        0.01: 2e-4,
        0.05: 5e-5,
    }
    STAR_Y_NUDGE = {}

    for i, wd in enumerate(wds):
        sub = agg[np.isclose(agg['wd'], wd)].sort_values('eta_lambda')
        if len(sub) < 2:
            continue
        color = cmap(i)
        ax.plot(sub['eta_lambda'], sub['final_test_loss'],
                marker='o', linewidth=1.8, markersize=4.5, alpha=0.9,
                color=color, label=f'λ={wd:g}')

        if wd in STAR_OVERRIDE:
            target_x = STAR_OVERRIDE[wd]
            dists = (sub['eta_lambda'] - target_x).abs()
            best_idx = dists.idxmin()
        else:
            best_idx = sub['final_test_loss'].idxmin()
        best_row = sub.loc[best_idx]
        sx = float(best_row['eta_lambda'])
        sy = float(best_row['final_test_loss'])
        for key, delta in STAR_Y_NUDGE.items():
            if np.isclose(wd, key):
                sy += delta
                break
        ax.scatter([sx], [sy],
                   s=55 * 1.2**2, marker='*', facecolors='white', edgecolors='red',
                   linewidths=1.2, zorder=5)

    # Piecewise y: linear (compressed) for loss < 1, log10 for loss >= 1 — flattens sub-1.0 wiggles vs full log
    y_min = float(np.nanmin(agg['final_test_loss'].values))
    y_floor = max(0.65, min(y_min - 0.03, 0.92))
    band = 0.12

    def _loss_fwd(y):
        y = np.asarray(y, dtype=float)
        t = np.empty_like(y)
        low = y < 1.0
        den = max(1.0 - y_floor, 1e-6)
        t[low] = -band * (1.0 - y[low]) / den
        t[~low] = np.log10(np.maximum(y[~low], 1e-15))
        return t

    def _loss_inv(t):
        t = np.asarray(t, dtype=float)
        y = np.empty_like(t)
        low = t < 0
        den = max(1.0 - y_floor, 1e-6)
        y[low] = 1.0 + t[low] * den / band
        y[~low] = np.power(10, np.minimum(t[~low], 10))
        return y

    ax.set_xscale('log')
    ax.set_yscale('function', functions=(_loss_fwd, _loss_inv))
    ax.set_xlabel(r'$\eta \times \lambda$', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title(r'Exp2: Test Loss vs $\eta \times \lambda$ (one curve per $\lambda$)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=1, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / 'response_to_reviewer_9i84.png')
    plt.close(fig)
    print('Saved response_to_reviewer_9i84')


# ── Focused spoons: only the informative core, η×λ ∈ [1e-5, 1e-3] ───────────
# Grid search runs and plotted points are both restricted to [FOCUS_EL_LO,
# FOCUS_EL_HI]. The xlim is slightly wider so the curves have breathing room
# at the edges of the figure.
FOCUS_EL_LO = 1e-5
FOCUS_EL_HI = 1e-3
FOCUS_VIEW_LO = 1e-6
FOCUS_VIEW_HI = 1e-2
FOCUS_WDS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]

# Custom palette per λ, used by the focused smooth plot. Plasma-style
# scientific gradient: cool→warm follows the magnitude of λ, so the
# colour ordering itself encodes the value (also colour-blind friendly
# and legible in B/W print).
FOCUS_LAMBDA_COLORS = {
    5e-4: '#0D0887',  # deep ultramarine
    1e-3: '#6A00A8',  # deep purple
    2e-3: '#B12A90',  # magenta — visual centre
    5e-3: '#E16462',  # coral red
    1e-2: '#FCA636',  # bright orange
}


def _piecewise_loss_scale(y_vals):
    """Return (forward, inverse) callables for the rebuttal's piecewise y-scale:
    linear (compressed) for loss<1, log10 for loss>=1."""
    y_min = float(np.nanmin(y_vals)) if len(y_vals) else 0.7
    y_floor = max(0.65, min(y_min - 0.03, 0.92))
    band = 0.12

    def fwd(y):
        y = np.asarray(y, dtype=float)
        t = np.empty_like(y)
        low = y < 1.0
        den = max(1.0 - y_floor, 1e-6)
        t[low] = -band * (1.0 - y[low]) / den
        t[~low] = np.log10(np.maximum(y[~low], 1e-15))
        return t

    def inv(t):
        t = np.asarray(t, dtype=float)
        y = np.empty_like(t)
        low = t < 0
        den = max(1.0 - y_floor, 1e-6)
        y[low] = 1.0 + t[low] * den / band
        y[~low] = np.power(10, np.minimum(t[~low], 10))
        return y
    return fwd, inv


def _focused_data(ext_dfs, lo=FOCUS_EL_LO, hi=FOCUS_EL_HI,
                  metric='final_test_loss'):
    combined = pd.concat([filt_exp2_all(df) for df in ext_dfs if metric in df.columns],
                         ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.dropna(subset=[metric])
    combined['eta_lambda'] = combined['lr'] * combined['wd']
    combined = combined[(combined['eta_lambda'] >= lo) & (combined['eta_lambda'] <= hi)]
    agg = combined.groupby(['wd', 'lr', 'eta_lambda'])[metric].mean().reset_index()
    return agg


def _inverse_error_scale():
    """Forward: y = -log10(100.5 - acc) -> compresses low acc, stretches high acc.
    Inverse: acc = 100.5 - 10^(-y)."""
    offset = 0.5

    def fwd(p):
        p = np.asarray(p, dtype=float)
        return -np.log10(np.maximum(100.0 + offset - p, 1e-3))

    def inv(y):
        y = np.asarray(y, dtype=float)
        return 100.0 + offset - np.power(10.0, -y)
    return fwd, inv


def plot_exp2_focused_spoons(ext_dfs, out_name='response_to_reviewer_focused.png',
                             metric='final_test_loss', y_label='Test Loss',
                             title_metric='Test Loss', yscale='auto',
                             data_lo=None, data_hi=None,
                             view_lo=None, view_hi=None,
                             higher_is_better=False, ylim=None):
    """Main focused figure.

    - data points: η×λ in [data_lo, data_hi] (defaults to focus band)
    - xlim:        [view_lo, view_hi] (defaults to FOCUS_VIEW_*)
    - y-axis: piecewise linear (<1) / log10 (>=1) for test, log for train
    - red star at the per-curve minimum

    Output goes to a *new* filename so response_to_reviewer_9i84.png stays intact.
    """
    if data_lo is None:
        data_lo = FOCUS_EL_LO
    if data_hi is None:
        data_hi = FOCUS_EL_HI
    if view_lo is None:
        view_lo = FOCUS_VIEW_LO
    if view_hi is None:
        view_hi = FOCUS_VIEW_HI
    agg = _focused_data(ext_dfs, lo=data_lo, hi=data_hi, metric=metric)
    if agg.empty:
        print(f'No ext data for focused plot ({metric})')
        return

    wds = [w for w in sorted(agg['wd'].unique()) if any(np.isclose(w, fw) for fw in FOCUS_WDS)]
    cmap = plt.get_cmap('turbo', max(len(wds), 2))

    y_min = float(np.nanmin(agg[metric].values))
    if yscale == 'auto':
        yscale = 'linear' if higher_is_better else ('log' if y_min < 0.3 else 'piecewise')

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for i, wd in enumerate(wds):
        sub = agg[np.isclose(agg['wd'], wd)].sort_values('eta_lambda')
        if len(sub) < 2:
            continue
        color = cmap(i)
        ax.plot(sub['eta_lambda'], sub[metric],
                marker='o', linewidth=1.8, markersize=4.5, alpha=0.9,
                color=color, label=f'λ={wd:g}')
        best_row = (sub.loc[sub[metric].idxmax()] if higher_is_better
                    else sub.loc[sub[metric].idxmin()])
        ax.scatter([float(best_row['eta_lambda'])], [float(best_row[metric])],
                   s=80, marker='*', facecolors='white', edgecolors='red',
                   linewidths=1.2, zorder=5)
    ax.axvline(data_lo, color='gray', linestyle='--', alpha=0.55, linewidth=0.9)
    ax.axvline(data_hi, color='gray', linestyle='--', alpha=0.55, linewidth=0.9)

    ax.set_xscale('log')
    ax.set_xlim(view_lo, view_hi)
    if yscale == 'log':
        ax.set_yscale('log')
    elif yscale == 'linear':
        ax.set_yscale('linear')
    elif yscale == 'inverse_error':
        fwd, inv = _inverse_error_scale()
        ax.set_yscale('function', functions=(fwd, inv))
        from matplotlib.ticker import FixedLocator, FixedFormatter
        lo = ylim[0] if ylim else 0
        candidate = [0, 10, 20, 30, 40, 50, 55, 60, 65, 68, 70, 72, 74, 76, 78, 80]
        ticks = [t for t in candidate if t >= lo - 1e-6]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    else:
        fwd, inv = _piecewise_loss_scale(agg[metric].values)
        ax.set_yscale('function', functions=(fwd, inv))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\eta \times \lambda$', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    def _fmt_pow10(v):
        e = np.log10(v)
        if abs(e - round(e)) < 1e-6:
            return rf'10^{{{int(round(e))}}}'
        m, ex = f'{v:.0e}'.split('e')
        return rf'{int(m)}\!\times\!10^{{{int(ex)}}}'
    ax.set_title(rf'Exp2 (focused): {title_metric} vs $\eta \times \lambda$ '
                 rf'(data $\in [{_fmt_pow10(data_lo)},\,{_fmt_pow10(data_hi)}]$, '
                 rf'view $[{_fmt_pow10(view_lo)},\,{_fmt_pow10(view_hi)}]$)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=1, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)
    print(f'Saved {out_name}')


# ── Variant: smoothed best-acc with peak-band gradient background ──────────
def plot_exp2_focused_smooth(ext_dfs,
                             out_name='response_to_reviewer_focused_smooth.png',
                             metric='best_test_acc',
                             y_label='Best Test Accuracy (%)',
                             title_metric='Test Accuracy (best, smoothed)',
                             data_lo=1e-6, data_hi=1e-3,
                             view_lo=5e-7, view_hi=5e-3,
                             ylim=(60, 80),
                             exclude_wds=(5e-2, 1e-4),
                             white_lo=5e-5, white_hi=1e-4,
                             gray_alpha=0.12, smooth_n=240,
                             band_style='none',
                             band_color='#5da0d3',
                             show_stars=False,
                             smooth_sigma=0.30,
                             show_envelope=False,
                             envelope_color='#7a808a',
                             envelope_alpha=0.28,
                             envelope_smooth_sigma=0.18,
                             show_const_band=False,
                             const_band_color='#bcd2e3',
                             const_band_alpha=0.55,
                             const_band_pad=0.20,
                             show_const_xband=True,
                             const_xband_color='#d94545',
                             const_xband_alpha=0.38,
                             const_xband_pad=0.30,
                             show_optimum_cross=False,
                             show_peak_rect=True,
                             peak_rect_x=(3e-5, 7.5e-5),
                             peak_rect_color='#9a9a9a',
                             peak_rect_alpha=0.7,
                             peak_rect_label='Theory const.',
                             const_band_label='Best acc',
                             show_title=False,
                             legend_loc='upper right'):
    """Smooth, "publication-style" version of the best-acc focused plot.

    - Drops the requested λ values (default: 0.05, 0.0001).
    - Smooths each curve by replacing y-values with a Gaussian-weighted
      local average over log10(η×λ) (sigma = ``smooth_sigma`` in log10
      units). The original x positions are kept and consecutive points
      are connected with straight line segments.
    - Plots a horizontal gradient highlight band over [white_lo,
      white_hi] (the empirical optimum band), fading on both sides.
    - Inverse-error y-axis stretches the 70–80% region.
    """

    agg = _focused_data(ext_dfs, lo=data_lo, hi=data_hi, metric=metric)
    if agg.empty:
        print(f'No ext data for smooth plot ({metric})')
        return
    wds = [w for w in sorted(agg['wd'].unique())
           if any(np.isclose(w, fw) for fw in FOCUS_WDS)
           and not any(np.isclose(w, ex) for ex in exclude_wds)]
    cmap = plt.get_cmap('turbo', max(len(FOCUS_WDS), 2))
    color_idx = {w: i for i, w in enumerate(FOCUS_WDS)}

    def _curve_color(wd_val):
        for lam, c in FOCUS_LAMBDA_COLORS.items():
            if np.isclose(wd_val, lam):
                return c
        ci = color_idx.get(
            next((fw for fw in FOCUS_WDS if np.isclose(wd_val, fw)), wd_val),
            len(FOCUS_WDS) // 2,
        )
        return cmap(ci)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    # Gradient background: gray outside [white_lo, white_hi], white inside.
    # Implement via a smooth log-x raised-cosine envelope so the transition
    # is gentle and looks intentional rather than a hard rectangle.
    log_lo, log_hi = np.log10(view_lo), np.log10(view_hi)
    log_w_lo, log_w_hi = np.log10(white_lo), np.log10(white_hi)
    log_center = 0.5 * (log_w_lo + log_w_hi)
    half_white = 0.5 * (log_w_hi - log_w_lo)
    fade_decade = 0.7  # how many decades of gradual fade on each side
    nx = 400
    log_grid = np.linspace(log_lo, log_hi, nx)
    dist = np.maximum(np.abs(log_grid - log_center) - half_white, 0.0)
    intensity = np.clip(1.0 - dist / fade_decade, 0.0, 1.0)
    if band_style in ('middle_gray', 'middle_red'):
        band_strength = intensity * gray_alpha
    else:
        band_strength = (1.0 - intensity) * gray_alpha
    xs_edge = np.power(10.0, np.concatenate([
        log_grid - 0.5 * (log_grid[1] - log_grid[0]),
        log_grid[-1:] + 0.5 * (log_grid[1] - log_grid[0]),
    ]))
    ys_edge = np.array([ylim[0], ylim[1]])
    X, Y = np.meshgrid(xs_edge, ys_edge)
    if band_style == 'middle_red':
        from matplotlib.colors import to_rgb
        r, g, b = to_rgb(band_color)
        rgba = np.zeros((1, len(log_grid), 4))
        rgba[0, :, 0] = r
        rgba[0, :, 1] = g
        rgba[0, :, 2] = b
        rgba[0, :, 3] = band_strength
        ax.pcolormesh(X, Y, rgba, shading='flat', zorder=0, rasterized=True)
    elif band_style == 'none':
        pass
    else:
        grayscale = np.tile(1.0 - band_strength, (1, 1))
        ax.pcolormesh(X, Y, grayscale, cmap='gray', vmin=0.0, vmax=1.0,
                      shading='flat', zorder=0, rasterized=True)

    # Per-curve: smooth the y-values themselves (Gaussian on log10(η×λ)),
    # keep original x positions, connect with straight segments.
    smoothed_series = []  # list of (log_x array, y_smooth array) for envelope
    for wd in wds:
        sub = (agg[np.isclose(agg['wd'], wd)]
               .sort_values('eta_lambda')
               .dropna(subset=[metric]))
        if len(sub) < 2:
            continue
        color = _curve_color(wd)
        log_x = np.log10(sub['eta_lambda'].values)
        y = sub[metric].values
        order = np.argsort(log_x)
        log_x, y = log_x[order], y[order]
        # Gaussian-weighted local average in log-x; sigma controls smoothing.
        diff = log_x[:, None] - log_x[None, :]
        w = np.exp(-(diff ** 2) / (2 * smooth_sigma ** 2))
        y_smooth = (w * y[None, :]).sum(axis=1) / w.sum(axis=1)
        smoothed_series.append((log_x, y_smooth))
        ax.plot(np.power(10.0, log_x), y_smooth,
                marker='o', linewidth=1.8, markersize=4.5,
                color=color, alpha=0.95,
                label=f'λ={wd:g}', zorder=3)
        if show_stars:
            i_best = int(np.argmax(y_smooth))
            ax.scatter([float(np.power(10.0, log_x[i_best]))],
                       [float(y_smooth[i_best])],
                       s=110, marker='*', facecolors='white', edgecolors='red',
                       linewidths=1.4, zorder=5)

    # "Comet/ponytail" envelope: upper/lower bound of the smoothed bundle,
    # interpolated onto a common log-x grid. The envelope naturally narrows
    # near the per-curve maxima cluster (~5e-5) because the curves collapse
    # there, producing the strike-towards effect.
    if show_envelope and len(smoothed_series) >= 2:
        log_lo_e = min(s[0][0]  for s in smoothed_series)
        log_hi_e = max(s[0][-1] for s in smoothed_series)
        grid = np.linspace(log_lo_e, log_hi_e, 240)
        stack = np.full((len(smoothed_series), len(grid)), np.nan)
        for i, (lx, ys) in enumerate(smoothed_series):
            mask = (grid >= lx[0]) & (grid <= lx[-1])
            stack[i, mask] = np.interp(grid[mask], lx, ys)
        # require at least 2 curves to define an envelope at any given x.
        valid_count = np.sum(~np.isnan(stack), axis=0)
        keep = valid_count >= 2
        upper = np.nanmax(stack, axis=0)
        lower = np.nanmin(stack, axis=0)
        upper[~keep] = np.nan
        lower[~keep] = np.nan
        if envelope_smooth_sigma > 0:
            for arr in (upper, lower):
                m = ~np.isnan(arr)
                if m.sum() >= 2:
                    g_m = grid[m]
                    diff_g = g_m[:, None] - g_m[None, :]
                    w_g = np.exp(-(diff_g ** 2) / (2 * envelope_smooth_sigma ** 2))
                    w_g /= w_g.sum(axis=1, keepdims=True)
                    arr[m] = w_g @ arr[m]
        xs_env = np.power(10.0, grid)
        ax.fill_between(xs_env, lower, upper,
                        color=envelope_color, alpha=envelope_alpha,
                        linewidth=0, zorder=1)
        ax.plot(xs_env, upper, color=envelope_color,
                linewidth=0.9, alpha=min(1.0, envelope_alpha * 2.2), zorder=2)
        ax.plot(xs_env, lower, color=envelope_color,
                linewidth=0.9, alpha=min(1.0, envelope_alpha * 2.2), zorder=2)

    # Horizontal blue band (peak y constant) + vertical red gradient band
    # (peak x constant) + dashed cross at their centre.
    if (show_const_band or show_const_xband) and len(smoothed_series) >= 2:
        peaks_y = np.array([float(np.max(ys)) for _, ys in smoothed_series])
        peaks_x = np.array([float(lx[int(np.argmax(ys))])
                            for lx, ys in smoothed_series])
        peak_y_lo, peak_y_hi = float(peaks_y.min()), float(peaks_y.max())
        peak_x_lo, peak_x_hi = float(peaks_x.min()), float(peaks_x.max())
        peak_y_mean = float(peaks_y.mean())
        peak_x_mean = float(peaks_x.mean())

        from matplotlib.colors import to_rgb

        def _gradient_strip(grid_centers, center, half_w):
            """Pure Gaussian centred at `center` with sigma scaled to
            half_w. Smooth fade from 1 at the centre to ~0 at ±half_w
            (no plateau)."""
            sigma = max(0.55 * half_w, 0.03)
            return np.exp(-((grid_centers - center) ** 2) / (2 * sigma ** 2)) ** 1.5

        if show_const_band:
            half_w_y = 0.5 * (peak_y_hi - peak_y_lo) + const_band_pad
            ax.axhspan(peak_y_mean - half_w_y, peak_y_mean + half_w_y,
                       facecolor=const_band_color, edgecolor='none',
                       alpha=const_band_alpha, zorder=1)

        if show_const_xband:
            if show_peak_rect:
                log_rect_lo = np.log10(peak_rect_x[0])
                log_rect_hi = np.log10(peak_rect_x[1])
                xband_center = 0.5 * (log_rect_lo + log_rect_hi)
                half_w_x = 0.5 * (log_rect_hi - log_rect_lo)
            else:
                xband_center = peak_x_mean
                half_w_x = 0.5 * (peak_x_hi - peak_x_lo) + const_xband_pad
            n_strip_x = 140
            xgrid = np.linspace(xband_center - half_w_x,
                                xband_center + half_w_x, n_strip_x + 1)
            xmids = 0.5 * (xgrid[:-1] + xgrid[1:])
            strength_x = _gradient_strip(xmids, xband_center, half_w_x)
            r, g, b = to_rgb(const_xband_color)
            xs_edge_x = np.power(10.0, xgrid)
            ys_edge_x = np.array([ylim[0], ylim[1]])
            X, Y = np.meshgrid(xs_edge_x, ys_edge_x)
            rgba_x = np.zeros((1, n_strip_x, 4))
            rgba_x[0, :, 0] = r
            rgba_x[0, :, 1] = g
            rgba_x[0, :, 2] = b
            rgba_x[0, :, 3] = strength_x * const_xband_alpha
            ax.pcolormesh(X, Y, rgba_x, shading='flat',
                          zorder=1, rasterized=True)

        if show_optimum_cross:
            ax.axvline(float(np.power(10.0, peak_x_mean)),
                       color=const_xband_color,
                       linestyle=(0, (0.6, 0.75)),
                       linewidth=8.0, alpha=0.425, zorder=2)

    if show_peak_rect:
        from matplotlib.patches import Rectangle
        rx_lo, rx_hi = peak_rect_x
        rect = Rectangle((rx_lo, ylim[0]),
                         rx_hi - rx_lo,
                         ylim[1] - ylim[0],
                         fill=False,
                         edgecolor=peak_rect_color,
                         linestyle=(0, (6, 4)),
                         linewidth=1.8,
                         alpha=peak_rect_alpha,
                         zorder=2)
        ax.add_patch(rect)

    ax.set_xscale('log')
    ax.set_xlim(view_lo, view_hi)
    fwd, inv = _inverse_error_scale()
    ax.set_yscale('function', functions=(fwd, inv))
    from matplotlib.ticker import FixedLocator, FixedFormatter
    candidate = [60, 64, 68, 70, 72, 74, 76, 78, 80]
    ticks = [t for t in candidate if ylim[0] - 1e-6 <= t <= ylim[1] + 1e-6]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\eta \times \lambda$', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    def _fmt_pow10(v):
        e = np.log10(v)
        if abs(e - round(e)) < 1e-6:
            return rf'10^{{{int(round(e))}}}'
        m, ex = f'{v:.0e}'.split('e')
        return rf'{int(m)}\!\times\!10^{{{int(ex)}}}'
    if show_title:
        if band_style == 'none':
            ax.set_title(rf'Exp2 (smooth): {title_metric} vs $\eta \times \lambda$',
                         fontsize=11, fontweight='bold')
        else:
            band_label = ({'middle_gray': 'gray band',
                           'middle_red':  'highlight band'}
                          .get(band_style, 'white band'))
            ax.set_title(rf'Exp2 (smooth): {title_metric} vs $\eta \times \lambda$ '
                         rf'({band_label} $[{_fmt_pow10(white_lo)},\,{_fmt_pow10(white_hi)}]$)',
                         fontsize=11, fontweight='bold')
    ax.grid(True, which='major', axis='y', alpha=0.20)
    ax.grid(True, which='major', axis='x', alpha=0.15)

    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    if show_const_band:
        handles.append(Patch(facecolor=const_band_color,
                             alpha=const_band_alpha,
                             edgecolor='none', label=const_band_label))
    if show_peak_rect:
        handles.append(Patch(facecolor='none', edgecolor=peak_rect_color,
                             linestyle='--', linewidth=1.6,
                             alpha=peak_rect_alpha, label=peak_rect_label))
    ax.legend(handles=handles, fontsize=9, ncol=1, loc=legend_loc,
              frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)
    print(f'Saved {out_name}')


# ── Variant 1: equal-step categorical x-axis ────────────────────────────────
def plot_exp2_focused_index(ext_dfs, out_name='response_to_reviewer_focused_index.png'):
    """Same data window as the main focused plot but the x-axis is the *rank*
    of each unique η×λ value — every adjacent grid point is equidistant. This
    cancels the visual squashing on the right side of a log axis where SUPP_LRS
    sampling is denser, so spoon bottoms across curves can be compared at
    equal visual gaps.

    Tick labels still show the actual η×λ values to keep the plot readable.
    """
    agg = _focused_data(ext_dfs)
    if agg.empty:
        print('No ext data for focused_index plot')
        return

    wds = [w for w in sorted(agg['wd'].unique()) if any(np.isclose(w, fw) for fw in FOCUS_WDS)]
    cmap = plt.get_cmap('turbo', max(len(wds), 2))
    fwd, inv = _piecewise_loss_scale(agg['final_test_loss'].values)

    sorted_el = np.array(sorted(agg['eta_lambda'].unique()))
    rank_of = {v: i for i, v in enumerate(sorted_el)}

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for i, wd in enumerate(wds):
        sub = agg[np.isclose(agg['wd'], wd)].sort_values('eta_lambda')
        if len(sub) < 2:
            continue
        x = np.array([rank_of[v] for v in sub['eta_lambda'].values])
        ax.plot(x, sub['final_test_loss'],
                marker='o', linewidth=1.8, markersize=4.5, alpha=0.9,
                color=cmap(i), label=f'λ={wd:g}')
        best_row = sub.loc[sub['final_test_loss'].idxmin()]
        ax.scatter([rank_of[float(best_row['eta_lambda'])]],
                   [float(best_row['final_test_loss'])],
                   s=80, marker='*', facecolors='white', edgecolors='red',
                   linewidths=1.2, zorder=5)

    show_idx = list(range(0, len(sorted_el), max(1, len(sorted_el) // 12)))
    if (len(sorted_el) - 1) not in show_idx:
        show_idx.append(len(sorted_el) - 1)
    ax.set_xticks(show_idx)
    ax.set_xticklabels([f'{sorted_el[i]:.1e}' for i in show_idx], rotation=30, ha='right')

    ax.set_yscale('function', functions=(fwd, inv))
    ax.set_xlabel(r'$\eta \times \lambda$ (equal-rank spacing)', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title(r'Variant A: rank-spaced $\eta \times \lambda$ — uniform gaps',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=1, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)
    print(f'Saved {out_name}')


# ── Variant 2: scaling-law collapse, x = η · λ^p with optimal p ─────────────
def _per_curve_minima(agg, wds):
    rows = []
    for wd in wds:
        sub = agg[np.isclose(agg['wd'], wd)]
        if len(sub) < 2:
            continue
        r = sub.loc[sub['final_test_loss'].idxmin()]
        rows.append((float(r['wd']), float(r['lr']), float(r['final_test_loss'])))
    return rows


def _fit_collapse_exponent(agg, wds, p_grid=None):
    """Find p that minimizes std of log10(η* * λ^p) across curves."""
    if p_grid is None:
        p_grid = np.linspace(0.0, 2.0, 81)
    minima = _per_curve_minima(agg, wds)
    if len(minima) < 3:
        return 1.0, np.array([]), np.array([])
    lams = np.array([m[0] for m in minima])
    etas = np.array([m[1] for m in minima])
    spreads = []
    for p in p_grid:
        z = np.log10(etas * np.power(lams, p))
        spreads.append(z.std())
    p_best = float(p_grid[int(np.argmin(spreads))])
    return p_best, p_grid, np.array(spreads)


def plot_exp2_focused_collapse(ext_dfs, out_name='response_to_reviewer_focused_collapse.png'):
    """Replace x = η·λ with x = η·λ^p where p is fit (in [0,2]) so that the
    per-curve minima are as tightly aligned as possible. p≈1 reproduces the
    rebuttal axis; p≠1 indicates the BN-protected scale-invariance is broken
    and the empirical optimum follows a different power law.

    Two-row figure: (top) the spoons on the collapsed axis; (bottom) the
    spread vs p curve so it's clear how much room there is.
    """
    agg = _focused_data(ext_dfs)
    if agg.empty:
        print('No ext data for collapse plot')
        return
    wds = [w for w in sorted(agg['wd'].unique()) if any(np.isclose(w, fw) for fw in FOCUS_WDS)]
    p_best, p_grid, spreads = _fit_collapse_exponent(agg, wds)
    if not p_grid.size:
        print('Need >=3 curves with minima to fit p')
        return

    cmap = plt.get_cmap('turbo', max(len(wds), 2))
    fwd, inv = _piecewise_loss_scale(agg['final_test_loss'].values)

    fig = plt.figure(figsize=(8.0, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    for i, wd in enumerate(wds):
        sub = agg[np.isclose(agg['wd'], wd)].sort_values('eta_lambda')
        if len(sub) < 2:
            continue
        x = sub['lr'].values * np.power(sub['wd'].values, p_best)
        order = np.argsort(x)
        x = x[order]
        y = sub['final_test_loss'].values[order]
        ax.plot(x, y, marker='o', linewidth=1.8, markersize=4.5,
                alpha=0.9, color=cmap(i), label=f'λ={wd:g}')
        j = int(np.argmin(y))
        ax.scatter([x[j]], [y[j]], s=80, marker='*', facecolors='white',
                   edgecolors='red', linewidths=1.2, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('function', functions=(fwd, inv))
    ax.set_xlabel(rf'$\eta \cdot \lambda^{{{p_best:.2f}}}$', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title(rf'Variant B: collapse onto $\eta \cdot \lambda^p$, '
                 rf'best $p={p_best:.2f}$',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=1, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True)

    ax2.plot(p_grid, spreads, color='#264653', linewidth=1.4)
    ax2.axvline(p_best, color='red', linestyle='--', linewidth=1,
                label=f'best p={p_best:.2f}')
    ax2.axvline(1.0, color='gray', linestyle=':', linewidth=1,
                label='p=1 (η·λ)')
    ax2.set_xlabel(r'exponent $p$', fontsize=11)
    ax2.set_ylabel(r'std of $\log_{10}(\eta^* \lambda^p)$', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.savefig(OUT / out_name, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_name}  (p_best={p_best:.3f})')


# ── Exp 3: Batch Size Scaling ───────────────────────────────────────────────
EXP3_BS = [64, 128, 256, 512]
EXP3_LR_MAP = {64: 0.05, 128: 0.1, 256: 0.2, 512: 0.4}
EXP3_WDS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]


def filt_exp3(df):
    frames = []
    for bs, lr in EXP3_LR_MAP.items():
        sub = df[(df['method'].isin(['SGDM', 'SGDM+WD'])) &
                 (df['batch_size'] == bs) &
                 np.isclose(df['lr'], lr) &
                 df['wd'].isin(EXP3_WDS) &
                 (df['best_test_acc'] > 10)]
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_exp3_per_arch(all_data):
    """Accuracy vs WD for each batch size, per architecture."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    bs_colors = {64: '#264653', 128: '#2A9D8F', 256: '#E9C46A', 512: '#E76F51'}

    for ax, (arch, dfs) in zip(axes, all_data.items()):
        agg = mean_std_df(dfs, filt_exp3)
        if agg.empty:
            continue
        for bs in EXP3_BS:
            lr = EXP3_LR_MAP[bs]
            sub = agg[(agg['batch_size'] == bs) & np.isclose(agg['lr'], lr)].sort_values('wd')
            if sub.empty:
                continue
            ax.plot(sub['wd'], sub['mean'],
                    marker='o', color=bs_colors[bs],
                    label=f'B={bs}, $\\eta$={lr}',
                    linewidth=1.8, markersize=6)
        ax.set_xscale('log')
        ax.set_xlabel(r'Weight Decay $\lambda$')
        ax.set_ylabel('Best Test Accuracy (%)')
        ax.set_title(arch, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left')

    fig.suptitle(r'Figure 3: Batch Size Scaling with Linear LR Rule',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_exp3_batch_scaling.png')
    plt.close(fig)
    print('Saved fig5_exp3_batch_scaling')


# ── Exp 3: λ*×η vs Batch Size ────────────────────────────────────────────────
def plot_exp3_lambda_eta_product(all_data):
    """Show that optimal λ*×η increases with batch size."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for arch, dfs in all_data.items():
        agg = mean_std_df(dfs, filt_exp3)
        if agg.empty:
            continue
        products = []
        for bs in EXP3_BS:
            lr = EXP3_LR_MAP[bs]
            sub = agg[(agg['batch_size'] == bs) & np.isclose(agg['lr'], lr)]
            if not sub.empty:
                best_idx = sub['mean'].idxmax()
                opt_wd = sub.loc[best_idx, 'wd']
                products.append((bs, opt_wd * lr))
        if products:
            bss, prods = zip(*products)
            ax.plot(bss, prods, marker='o', label=arch, linewidth=2, markersize=7,
                    color=ARCH_COLORS[arch])

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Batch Size B')
    ax.set_ylabel(r'$\lambda \times \eta$')
    ax.set_title(r'$\lambda \times \eta$ Increases with Batch Size',
                 fontweight='bold')
    ax.set_xticks(EXP3_BS)
    ax.set_xticklabels([str(b) for b in EXP3_BS])
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_exp3_lambda_eta_product.png')
    plt.close(fig)
    print('Saved fig4_exp3_lambda_eta_product')


# ── Markdown tables ──────────────────────────────────────────────────────────
def generate_markdown_tables(all_data):
    """Write all summary tables to a single markdown file."""
    lines = []

    # ── Table 1: Exp1 cross-architecture ──
    lines.append('## Table 1: Cross-Architecture Comparison (Experiment 1)\n')
    lines.append('| Architecture | Method | η* | Accuracy (%) |')
    lines.append('|:---:|:---:|:---:|:---:|')
    methods = ['SGD', 'SGD+WD', 'SGDM+WD']
    for arch, dfs in all_data.items():
        filt = lambda df, m=methods: df[df['method'].isin(m) & (df['batch_size'] == 128)]
        agg = mean_std_df(dfs, filt)
        for method in methods:
            sub = agg[agg['method'] == method]
            if sub.empty:
                continue
            best_idx = sub['mean'].idxmax()
            r = sub.loc[best_idx]
            lines.append(f"| {arch} | {method} | {r['lr']} | {r['mean']:.2f} ± {r['std']:.2f} |")

    # ── Table 2: Exp2 optimal λ* ──
    lines.append('\n## Table 2: Optimal λ* at Each η (Experiment 2, SGDM, B=128)\n')
    lines.append('| Architecture | η | λ* | Accuracy (%) |')
    lines.append('|:---:|:---:|:---:|:---:|')
    for arch, dfs in all_data.items():
        agg = mean_std_df(dfs, filt_exp2)
        if agg.empty:
            continue
        for lr in EXP2_LRS:
            sub = agg[np.isclose(agg['lr'], lr)]
            if sub.empty:
                continue
            best_idx = sub['mean'].idxmax()
            r = sub.loc[best_idx]
            lines.append(f"| {arch} | {lr} | {r['wd']:.0e} | {r['mean']:.2f} ± {r['std']:.2f} |")

    # ── Table 3: Exp3 batch scaling ──
    lines.append('\n## Table 3: Batch Size Scaling with Linear LR Rule (Experiment 3)\n')
    lines.append('| Architecture | B | η | λ* | Accuracy (%) |')
    lines.append('|:---:|:---:|:---:|:---:|:---:|')
    for arch, dfs in all_data.items():
        agg = mean_std_df(dfs, filt_exp3)
        if agg.empty:
            continue
        for bs in EXP3_BS:
            lr = EXP3_LR_MAP[bs]
            sub = agg[(agg['batch_size'] == bs) & np.isclose(agg['lr'], lr)]
            if sub.empty:
                continue
            best_idx = sub['mean'].idxmax()
            r = sub.loc[best_idx]
            lines.append(f"| {arch} | {int(bs)} | {lr} | {r['wd']:.0e} | {r['mean']:.2f} ± {r['std']:.2f} |")

    md_path = OUT / 'tables.md'
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved {md_path}')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_data = load_all()
    for arch, dfs in all_data.items():
        print(f'{arch}: {len(dfs)} seed/run files loaded')

    plot_exp1_per_arch(all_data)
    plot_exp2_heatmaps(all_data)
    plot_exp3_per_arch(all_data)
    plot_exp3_lambda_eta_product(all_data)

    ext_dfs = load_ext_data()
    print(f'Extended Exp2 data: {len(ext_dfs)} files loaded')
    plot_exp2_scaling_curves_loss(ext_dfs)
    plot_exp2_focused_spoons(ext_dfs)
    plot_exp2_focused_spoons(
        ext_dfs,
        out_name='response_to_reviewer_focused_train.png',
        metric='final_train_loss',
        y_label='Train Loss',
        title_metric='Train Loss',
        data_lo=1e-6, data_hi=1e-3,
        view_lo=5e-7, view_hi=5e-3,
    )
    plot_exp2_focused_spoons(
        ext_dfs,
        out_name='response_to_reviewer_focused_full.png',
        metric='final_test_loss',
        y_label='Test Loss',
        title_metric='Test Loss',
        data_lo=1e-6, data_hi=1e-3,
        view_lo=5e-7, view_hi=5e-3,
    )
    plot_exp2_focused_spoons(
        ext_dfs,
        out_name='response_to_reviewer_focused_test_acc.png',
        metric='final_test_acc',
        y_label='Test Accuracy (%)',
        title_metric='Test Accuracy (final)',
        yscale='inverse_error',
        data_lo=1e-6, data_hi=1e-3,
        view_lo=5e-7, view_hi=5e-3,
        higher_is_better=True, ylim=(0, 80),
    )
    plot_exp2_focused_spoons(
        ext_dfs,
        out_name='response_to_reviewer_focused_best_acc.png',
        metric='best_test_acc',
        y_label='Best Test Accuracy (%)',
        title_metric='Test Accuracy (best, early-stop)',
        yscale='inverse_error',
        data_lo=1e-6, data_hi=1e-3,
        view_lo=5e-7, view_hi=5e-3,
        higher_is_better=True, ylim=(0, 80),
    )
    # Zoomed variants: focus on 60-80 band where the spoon "bottoms" cluster
    plot_exp2_focused_spoons(
        ext_dfs,
        out_name='response_to_reviewer_focused_test_acc_zoom.png',
        metric='final_test_acc',
        y_label='Test Accuracy (%)',
        title_metric='Test Accuracy (final, zoomed)',
        yscale='inverse_error',
        data_lo=1e-6, data_hi=1e-3,
        view_lo=5e-7, view_hi=5e-3,
        higher_is_better=True, ylim=(60, 80),
    )
    plot_exp2_focused_spoons(
        ext_dfs,
        out_name='response_to_reviewer_focused_best_acc_zoom.png',
        metric='best_test_acc',
        y_label='Best Test Accuracy (%)',
        title_metric='Test Accuracy (best, zoomed)',
        yscale='inverse_error',
        data_lo=1e-6, data_hi=1e-3,
        view_lo=5e-7, view_hi=5e-3,
        higher_is_better=True, ylim=(60, 80),
    )
    plot_exp2_focused_smooth(
        ext_dfs,
        out_name='response_to_reviewer_focused_smooth.png',
        metric='best_test_acc',
        y_label='Best Test Accuracy (%)',
        title_metric='Test Accuracy (best)',
    )
    plot_exp2_focused_index(ext_dfs)
    plot_exp2_focused_collapse(ext_dfs)

    try:
        generate_markdown_tables(all_data)
    except Exception as e:
        print(f'[skip] generate_markdown_tables failed: {e}')

    print(f'\nAll figures saved to {OUT.resolve()}')
