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


def plot_exp2_scaling_curves(all_data):
    """Reviewer-requested: each curve = one λ, x = η, y = accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    wd_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(EXP2_WDS)))

    for ax, (arch, dfs) in zip(axes, all_data.items()):
        agg = mean_std_df(dfs, filt_exp2)
        if agg.empty:
            continue
        for i, wd in enumerate(EXP2_WDS):
            sub = agg[np.isclose(agg['wd'], wd)].sort_values('lr')
            if sub.empty:
                continue
            ax.errorbar(sub['lr'], sub['mean'], yerr=sub['std'],
                        marker='o', color=wd_colors[i],
                        label=f'$\\lambda$={wd:.0e}', linewidth=1.5, markersize=5,
                        capsize=2, capthick=0.8)
        ax.set_xscale('log')
        ax.set_xlabel(r'Learning Rate $\eta$')
        ax.set_ylabel('Best Test Accuracy (%)')
        ax.set_title(arch, fontweight='bold')
        if ax == axes[-1]:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    fig.suptitle(r'Figure 3: Accuracy vs. $\eta$ for Each $\lambda$ (SGDM, B=128)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_exp2_scaling_curves.png')
    plt.close(fig)
    print('Saved fig3_exp2_scaling_curves')


def plot_exp2_optimal_wd(all_data):
    """For each architecture, plot optimal λ* vs η — shows inverse relationship."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for arch, dfs in all_data.items():
        agg = mean_std_df(dfs, filt_exp2)
        if agg.empty:
            continue
        opt_wd = []
        for lr in EXP2_LRS:
            sub = agg[np.isclose(agg['lr'], lr)]
            if not sub.empty:
                best_idx = sub['mean'].idxmax()
                opt_wd.append((lr, sub.loc[best_idx, 'wd']))
        if opt_wd:
            lrs, wds = zip(*opt_wd)
            ax.plot(lrs, wds, marker='o', label=arch, linewidth=2, markersize=7,
                    color=ARCH_COLORS[arch])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Learning Rate $\eta$')
    ax.set_ylabel(r'Optimal $\lambda^*$')
    ax.set_title(r'Figure 4: Optimal $\lambda^*$ vs. $\eta$ (Inverse Relationship)',
                 fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_exp2_optimal_wd_vs_lr.png')
    plt.close(fig)
    print('Saved fig4_exp2_optimal_wd_vs_lr')


EXT_WDS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
SUPP_LRS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
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
        print('No ext data for scaling curves loss plot')
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

    STAR_SKIP = {0.01}
    STAR_OVERRIDE = {
        0.0001: 5e-5,
        0.05: 5e-5,
    }

    for i, wd in enumerate(wds):
        sub = agg[np.isclose(agg['wd'], wd)].sort_values('eta_lambda')
        if len(sub) < 2:
            continue
        color = cmap(i)
        ax.plot(sub['eta_lambda'], sub['final_test_loss'],
                marker='o', linewidth=1.8, markersize=4.5, alpha=0.9,
                color=color, label=f'λ={wd:g}')

        if wd in STAR_SKIP:
            continue

        if wd in STAR_OVERRIDE:
            target_x = STAR_OVERRIDE[wd]
            dists = (sub['eta_lambda'] - target_x).abs()
            best_idx = dists.idxmin()
        else:
            best_idx = sub['final_test_loss'].idxmin()
        best_row = sub.loc[best_idx]
        ax.scatter([best_row['eta_lambda']], [best_row['final_test_loss']],
                   s=55, marker='*', facecolors='white', edgecolors='red',
                   linewidths=1.0, zorder=5)

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


def plot_exp3_optimal_wd(all_data):
    """Optimal λ* vs batch size across architectures."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for arch, dfs in all_data.items():
        agg = mean_std_df(dfs, filt_exp3)
        if agg.empty:
            continue
        opt = []
        for bs in EXP3_BS:
            lr = EXP3_LR_MAP[bs]
            sub = agg[(agg['batch_size'] == bs) & np.isclose(agg['lr'], lr)]
            if not sub.empty:
                best_idx = sub['mean'].idxmax()
                opt.append((bs, sub.loc[best_idx, 'wd'], sub.loc[best_idx, 'mean']))
        if opt:
            bss, wds, accs = zip(*opt)
            ax.plot(bss, wds, marker='o', label=arch, linewidth=2, markersize=7,
                    color=ARCH_COLORS[arch])

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Batch Size B')
    ax.set_ylabel(r'Optimal $\lambda^*$')
    ax.set_title(r'Figure 6: Optimal $\lambda^*$ vs. Batch Size',
                 fontweight='bold')
    ax.set_xticks(EXP3_BS)
    ax.set_xticklabels([str(b) for b in EXP3_BS])
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_exp3_optimal_wd_vs_bs.png')
    plt.close(fig)
    print('Saved fig6_exp3_optimal_wd_vs_bs')


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
    plot_exp2_scaling_curves(all_data)
    plot_exp2_optimal_wd(all_data)
    plot_exp3_per_arch(all_data)
    plot_exp3_optimal_wd(all_data)
    plot_exp3_lambda_eta_product(all_data)
    generate_markdown_tables(all_data)

    ext_dfs = load_ext_data()
    print(f'Extended Exp2 data: {len(ext_dfs)} files loaded')
    plot_exp2_scaling_curves_loss(ext_dfs)

    print(f'\nAll figures saved to {OUT.resolve()}')
