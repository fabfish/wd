"""
Regenerate the heatmap-style figures with the smooth-interpolated, colourful
look used by `wd/figures/WD and LR (1).jpg` (originally produced by
`analysis/plot_exp2_fixed.py`):

- exp2_heatmap_r18.png      single-panel ResNet-18 heatmap (main text Fig 2a)
- appendix_exp2_3arch.png   3 panels for ResNet-18 / VGG-16 / ResNet-50

Both use multi-seed mean over the available CSVs; no embedded titles
because the LaTeX caption already provides one.
"""
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

ROOT = Path('.').resolve()
OUT = ROOT / 'wd' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Smooth red→orange→yellow→green palette, ported from analysis/plot_exp2_fixed.py.
# Parameterised by `green_floor` (the accuracy at which yellow turns to green).
def make_smooth_cmap(vmin: float, vmax: float, green_floor: float):
    """Build a colormap whose yellow→green knee sits at `green_floor`.

    The shape (deep red → orange → yellow plateau → bright green) follows
    the original `analysis/plot_exp2_fixed.py` recipe; the only tunable is
    where the green knee falls inside [vmin, vmax].  A 0.6-unit knee
    matches the original smooth feel — narrower knees make the heatmap
    look angular under bicubic interpolation.
    """
    span = vmax - vmin
    knee_lo = max(0.0, min(1.0, (green_floor - 0.6 - vmin) / span))
    knee_hi = max(knee_lo + 0.005, min(1.0, (green_floor - vmin) / span))
    # Define the (red, green, blue, name) anchors and drop any whose
    # x-position would push beyond knee_lo, keeping the cdict strictly
    # monotonic for arbitrary `green_floor` (small values would otherwise
    # crowd out the late anchors).
    anchors = [
        (0.0,    0.55, 0.10, 0.10),
        (0.4375, 0.70, 0.35, 0.15),
        (0.60,   0.85, 0.55, 0.35),
        (0.85,   0.95, 0.65, 0.30),
    ]
    anchors = [a for a in anchors if a[0] < knee_lo - 1e-3]
    knee = [
        (knee_lo, 1.00, 0.90, 0.25),
        (knee_hi, 0.55, 0.80, 0.35),
        (1.0,     0.00, 0.55, 0.30),
    ]
    rows = anchors + knee
    cdict = {
        'red':   [(x, r, r) for (x, r, g, b) in rows],
        'green': [(x, g, g) for (x, r, g, b) in rows],
        'blue':  [(x, b, b) for (x, r, g, b) in rows],
    }
    return LinearSegmentedColormap('smooth_acc', cdict, N=256)


# Per-architecture green threshold (yellow→green knee).
# The yellow→green knee sits at this accuracy. For VGG-16, threshold at
# 71.9 so cells ≥ 71.9 render as green ("VGG-16 阈值至少调整到 71.9 及以上").
ARCH_GREEN_FLOOR = {'ResNet-18': 75.4, 'VGG-16': 71.9, 'ResNet-50': 76.7}


# ─── Data loading ───────────────────────────────────────────────────────
ARCH_FILES = {
    'ResNet-18': [
        ROOT / 'outputs/results/results.csv',
        ROOT / 'rebuttal/results/results_resnet18_seed42_run2.csv',
        ROOT / 'rebuttal/results/results_resnet18_seed123.csv',
        ROOT / 'rebuttal/results/results_resnet18_seed123_run2.csv',
    ],
    'VGG-16': [
        ROOT / 'rebuttal/results/results_vgg16_seed42.csv',
        ROOT / 'rebuttal/results/results_vgg16_seed123.csv',
    ],
    'ResNet-50': [
        ROOT / 'rebuttal/results/results_resnet50_seed42.csv',
        ROOT / 'rebuttal/results/results_resnet50_seed123.csv',
    ],
}

LRS = [0.01, 0.05, 0.1, 0.2, 0.3]
WDS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def arch_pivot(arch):
    rows = []
    for f in ARCH_FILES[arch]:
        if not f.exists():
            continue
        df = pd.read_csv(f)
        sub = df[(df['method'].isin(['SGDM', 'SGDM+WD']))
                 & (df['batch_size'] == 128)
                 & (np.isclose(df['momentum'].astype(float), 0.9))]
        sub = sub[sub['lr'].isin(LRS) & sub['wd'].isin(WDS)]
        rows.append(sub[['lr', 'wd', 'best_test_acc']])
    if not rows:
        return None
    cat = pd.concat(rows, ignore_index=True)
    pivot = (cat.groupby(['wd', 'lr'])['best_test_acc']
                .mean().reset_index()
                .pivot(index='wd', columns='lr', values='best_test_acc'))
    return pivot.sort_index(ascending=True)  # large wd at top w/ origin='lower'


# ─── Smooth-interpolated heatmap drawing routine ────────────────────────
def draw_smooth_heatmap(ax, pivot, vmin=0, vmax=80, zoom_factor=10,
                        green_floor=75.5):
    from scipy.ndimage import gaussian_filter
    data = pivot.values.astype(float)
    extent = [0, len(pivot.columns), 0, len(pivot.index)]
    # Two-stage zoom: nearest-neighbour blow-up keeps each cell's centre
    # at its true accuracy (no bicubic over/undershoot that would pull a
    # 72.5 cell down toward a 67-orange neighbour and stain it yellow),
    # then a Gaussian rounds the boundaries so the heatmap still reads
    # as the soft "topographic" style of the original.
    data_zoomed = zoom(data, zoom_factor, order=0)
    data_zoomed = gaussian_filter(data_zoomed, sigma=zoom_factor * 0.55)
    data_zoomed = np.clip(data_zoomed, vmin, vmax)

    cmap = make_smooth_cmap(vmin, vmax, green_floor)
    im = ax.imshow(
        data_zoomed,
        cmap=cmap,
        aspect='auto',
        vmin=vmin, vmax=vmax,
        extent=extent,
        origin='lower',
        interpolation='bilinear',
    )

    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([f'{x:g}' for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f'{y:g}' for y in pivot.index])

    for i, wd in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            val = pivot.loc[wd, lr]
            if not np.isnan(val):
                ax.text(j + 0.5, i + 0.5, f'{val:.1f}',
                        ha='center', va='center', fontsize=10,
                        color='black', fontweight='bold')

    ax.set_xlabel(r'Learning Rate ($\eta$)')
    ax.set_ylabel(r'Weight Decay ($\lambda$)')
    return im


# ─── Public APIs ────────────────────────────────────────────────────────
def heatmap_r18():
    pivot = arch_pivot('ResNet-18')
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    im = draw_smooth_heatmap(ax, pivot,
                             green_floor=ARCH_GREEN_FLOOR['ResNet-18'])
    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label('Test Accuracy (%)', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'exp2_heatmap_r18.png')
    plt.close(fig)
    print('Saved exp2_heatmap_r18.png')


def heatmap_3arch():
    """Three side-by-side heatmaps with independent colorbars and a wide
    inter-panel gap so that the y-axis label of the middle/right panel does
    not run into the previous panel."""
    archs = ['ResNet-18', 'VGG-16', 'ResNet-50']
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8),
                             gridspec_kw={'wspace': 0.55})
    for ax, arch in zip(axes, archs):
        pivot = arch_pivot(arch)
        if pivot is None:
            continue
        im = draw_smooth_heatmap(ax, pivot,
                                 green_floor=ARCH_GREEN_FLOOR[arch])
        ax.set_title(arch, fontweight='bold', fontsize=14)
        cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02, fraction=0.05)
        cbar.set_label('Test Accuracy (%)', fontsize=11)
    fig.savefig(OUT / 'appendix_exp2_3arch.png')
    plt.close(fig)
    print('Saved appendix_exp2_3arch.png')


# ─── Train-loss curves: aligned colours/fonts with the smooth panel ──────
def train_curve():
    """Train loss vs η×λ. Uses the same per-λ palette as the smooth panel."""
    EXT_FILES = [
        ROOT / 'rebuttal/results/results_resnet18_seed42_exp2_ext.csv',
        ROOT / 'rebuttal/results/results_resnet18_seed42_exp2_ext2.csv',
        ROOT / 'rebuttal/results/results_resnet18_seed123_exp2_ext.csv',
        ROOT / 'rebuttal/results/results_resnet18_exp2_supplement.csv',
        ROOT / 'rebuttal/results/results_resnet18_seed42_exp2_fill.csv',
    ]
    DATA_LO, DATA_HI = 1e-6, 1e-3
    VIEW_LO, VIEW_HI = 5e-7, 5e-3

    # Match `FOCUS_LAMBDA_COLORS` from rebuttal/generate_figures.py
    LAMBDA_COLORS = {
        5e-4: '#0D0887',
        1e-3: '#6A00A8',
        2e-3: '#B12A90',
        5e-3: '#E16462',
        1e-2: '#FCA636',
    }
    EXCLUDE = (1e-4, 5e-2)  # match the smooth panel's exclude_wds

    dfs = []
    for f in EXT_FILES:
        if f.exists():
            df = pd.read_csv(f)
            df['wd'] = df['wd'].astype(float)
            df['lr'] = df['lr'].astype(float)
            dfs.append(df)
    if not dfs:
        print('No ext data; skip train_curve regen')
        return
    cat = pd.concat(dfs, ignore_index=True)
    cat['eta_lambda'] = cat['lr'] * cat['wd']
    cat = cat[(cat['eta_lambda'] >= DATA_LO * 0.99) & (cat['eta_lambda'] <= DATA_HI * 1.01)]
    agg = (cat.groupby(['wd', 'lr', 'eta_lambda'])['final_train_loss']
              .mean().reset_index())

    wds_use = [w for w in sorted(agg['wd'].unique())
               if any(np.isclose(w, fw) for fw in LAMBDA_COLORS)
               and not any(np.isclose(w, ex) for ex in EXCLUDE)]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for wd in wds_use:
        color = next(c for lam, c in LAMBDA_COLORS.items() if np.isclose(wd, lam))
        sub = agg[np.isclose(agg['wd'], wd)].sort_values('eta_lambda')
        if len(sub) < 2:
            continue
        ax.plot(sub['eta_lambda'], sub['final_train_loss'],
                marker='o', linewidth=2.2, markersize=6.0, alpha=0.95,
                color=color, label=f'λ={wd:g}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(VIEW_LO, VIEW_HI)
    ax.set_xlabel(r'$\eta \times \lambda$', fontsize=14)
    ax.set_ylabel('Train Loss', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.10)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'exp2_train_curve.png')
    plt.close(fig)
    print('Saved exp2_train_curve.png')


# ─── Smooth panel: best-acc curves with theory band ─────────────────────
def smooth_panel():
    """Regenerate exp2_eta_lambda_smooth.png with the same line style /
    fonts as exp2_train_curve.png, and replace the boxed "Theory const."
    rectangle with two vertical dashed lines labelled simply "Theory".
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generate_figures as gf
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    # Load extended Exp-2 data as the rebuttal helper does.
    ext_dfs = gf.load_ext_data()

    DATA_LO, DATA_HI = 1e-6, 1e-3
    VIEW_LO, VIEW_HI = 5e-7, 5e-3
    YLIM = (60, 80)
    EXCLUDE = (5e-2, 1e-4)
    LAMBDA_COLORS = {
        5e-4: '#0D0887',
        1e-3: '#6A00A8',
        2e-3: '#B12A90',
        5e-3: '#E16462',
        1e-2: '#FCA636',
    }
    SMOOTH_SIGMA = 0.30

    agg = gf._focused_data(ext_dfs, lo=DATA_LO, hi=DATA_HI,
                           metric='best_test_acc')

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    # Theory band: two vertical dashed lines, no top/bottom edges.
    theory_lo, theory_hi = 3e-5, 7.5e-5
    theory_color = '#9a9a9a'
    for x in (theory_lo, theory_hi):
        ax.axvline(x, color=theory_color, linestyle=(0, (6, 4)),
                   linewidth=1.8, alpha=0.7, zorder=2)
    # Faint pink strip inside the band, fading at the edges.
    n_strip = 140
    log_lo, log_hi = np.log10(theory_lo), np.log10(theory_hi)
    log_centre = 0.5 * (log_lo + log_hi)
    log_half = 0.5 * (log_hi - log_lo)
    xgrid = np.linspace(log_lo, log_hi, n_strip + 1)
    xmids = 0.5 * (xgrid[:-1] + xgrid[1:])
    sigma = max(0.55 * log_half, 0.03)
    strength = np.exp(-((xmids - log_centre) ** 2) / (2 * sigma ** 2)) ** 1.5
    r, g, b = to_rgb('#d94545')
    rgba = np.zeros((1, n_strip, 4))
    rgba[0, :, 0] = r
    rgba[0, :, 1] = g
    rgba[0, :, 2] = b
    rgba[0, :, 3] = strength * 0.38
    xs_edge = np.power(10.0, xgrid)
    ys_edge = np.array([YLIM[0], YLIM[1]])
    Xg, Yg = np.meshgrid(xs_edge, ys_edge)
    ax.pcolormesh(Xg, Yg, rgba, shading='flat', zorder=1, rasterized=True)

    # Smoothed per-λ curves, line style aligned with train_curve.
    for wd, color in LAMBDA_COLORS.items():
        sub = (agg[np.isclose(agg['wd'], wd)]
               .sort_values('eta_lambda')
               .dropna(subset=['best_test_acc']))
        if len(sub) < 2:
            continue
        log_x = np.log10(sub['eta_lambda'].values)
        y = sub['best_test_acc'].values
        order = np.argsort(log_x)
        log_x, y = log_x[order], y[order]
        diff = log_x[:, None] - log_x[None, :]
        w = np.exp(-(diff ** 2) / (2 * SMOOTH_SIGMA ** 2))
        y_smooth = (w * y[None, :]).sum(axis=1) / w.sum(axis=1)
        ax.plot(np.power(10.0, log_x), y_smooth,
                marker='o', linewidth=2.2, markersize=6.0,
                color=color, alpha=0.95,
                label=f'λ={wd:g}', zorder=3)

    # Inverse-error y-scale (stretches the high-acc region).
    ax.set_xscale('log')
    fwd, inv = gf._inverse_error_scale()
    ax.set_yscale('function', functions=(fwd, inv))
    ticks = [60, 64, 68, 70, 72, 74, 76, 78, 80]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.set_ylim(*YLIM)
    ax.set_xlim(VIEW_LO, VIEW_HI)
    ax.set_xlabel(r'$\eta \times \lambda$', fontsize=14)
    ax.set_ylabel('Best Test Accuracy (%)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='major', axis='y', alpha=0.20)
    ax.grid(True, which='major', axis='x', alpha=0.15)

    # Legend: λ curves + a Theory entry (vertical dashed line).
    handles, labels = ax.get_legend_handles_labels()
    theory_handle = Line2D([], [], color=theory_color,
                           linestyle=(0, (6, 4)), linewidth=1.8,
                           alpha=0.85, label='Theory')
    handles.append(theory_handle)
    labels.append('Theory')
    ax.legend(handles, labels, loc='upper right',
              frameon=True, fontsize=11)

    # Black, fully-visible spines on all four sides — matches the
    # heatmap and batch-figure styling so the panel reads as part of
    # the same family.
    for side in ('left', 'bottom', 'right', 'top'):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_color('black')
        sp.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(OUT / 'exp2_eta_lambda_smooth.png')
    # Also overwrite the legacy lambda.png copy so the file the user
    # is viewing in the IDE picks up the same change.
    import shutil as _shutil
    _shutil.copy(OUT / 'exp2_eta_lambda_smooth.png', OUT / 'lambda.png')
    plt.close(fig)
    print('Saved exp2_eta_lambda_smooth.png and lambda.png')


if __name__ == '__main__':
    heatmap_r18()
    heatmap_3arch()
    train_curve()
    smooth_panel()
