"""
V3 补充实验可视化：3×2 布局

可视化设计：
- 列：左=BS 32，右=BS 128
- 第一排：固定λ，η变化的训练曲线 (Loss/Acc vs Epoch)
- 第二排：最优(η,λ)热力图 (Best Test Acc)
- 第三排：固定η，λ变化的训练曲线 (Loss/Acc vs Epoch)

Usage:
    python plot_v3_supplementary.py --history_dir outputs/history/v3_supplementary
    python plot_v3_supplementary.py --results outputs/results/v3_supplementary.csv
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150


def load_history_files(history_dir):
    """Load all history JSON files from directory."""
    history_dir = Path(history_dir)
    all_data = {}

    for json_file in history_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            config = data['config']
            key = (config['batch_size'], config['lr'], config['wd'])
            all_data[key] = {
                'config': config,
                'history': data['history']
            }

    return all_data


def load_results_csv(results_file):
    """Load results from CSV file."""
    return pd.read_csv(results_file)


def plot_training_curves(ax_loss, ax_acc, histories, title_prefix, legend_label_key='lr', legend_format=None):
    """Plot training curves (loss and accuracy) on given axes."""
    cmap = plt.cm.viridis
    n_curves = len(histories)
    colors = [cmap(i / max(n_curves - 1, 1)) for i in range(n_curves)]

    for i, (key, data) in enumerate(sorted(histories.items())):
        history = data['history']
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        test_acc = [h['test_acc'] for h in history]

        if legend_format:
            label = legend_format(data['config'][legend_label_key])
        else:
            label = f"{legend_label_key}={data['config'][legend_label_key]}"

        ax_loss.plot(epochs, train_loss, color=colors[i], label=label, linewidth=1.5)
        ax_acc.plot(epochs, test_acc, color=colors[i], label=label, linewidth=1.5)

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Train Loss')
    ax_loss.set_title(f'{title_prefix} - Train Loss')
    ax_loss.legend(fontsize=8, loc='upper right')

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Test Acc (%)')
    ax_acc.set_title(f'{title_prefix} - Test Accuracy')
    ax_acc.legend(fontsize=8, loc='lower right')


def plot_heatmap(ax, df, bs, title):
    """Plot heatmap of best test accuracy for given batch size."""
    bs_df = df[df['batch_size'] == bs].copy()

    if bs_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Create pivot table
    pivot = bs_df.pivot_table(values='best_test_acc', index='wd', columns='lr', aggfunc='max')

    # Sort index (WD) in descending order for display
    pivot = pivot.sort_index(ascending=False)

    # Plot heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=pivot.values.min() - 2, vmax=pivot.values.max())

    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.3g}' for x in pivot.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{y:.0e}' for y in pivot.index], fontsize=8)

    ax.set_xlabel('Learning Rate (η)')
    ax.set_ylabel('Weight Decay (λ)')
    ax.set_title(title)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < (pivot.values.min() + pivot.values.max()) / 2 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=7)

    # Mark maximum
    max_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
    ax.add_patch(plt.Rectangle((max_idx[1] - 0.5, max_idx[0] - 0.5), 1, 1,
                                 fill=False, edgecolor='blue', linewidth=2))

    return im


def create_3x2_visualization(all_data, results_df, output_path, fixed_wd=5e-4, fixed_lr=0.05):
    """Create the main 3×2 visualization."""
    fig = plt.figure(figsize=(14, 12))

    # Create grid for 3 rows × 2 columns
    # Each cell contains 2 subplots (loss + acc) for rows 1 and 3
    # Row 2 is a single heatmap per cell

    batch_sizes = [32, 128]
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    for col_idx, bs in enumerate(batch_sizes):
        # ========== Row 1: Fixed λ, varying η ==========
        # Filter data for this batch size and fixed WD
        row1_data = {k: v for k, v in all_data.items()
                     if k[0] == bs and abs(k[2] - fixed_wd) < 1e-10}

        if row1_data:
            ax_loss = fig.add_subplot(gs[0, col_idx * 2])
            ax_acc = fig.add_subplot(gs[0, col_idx * 2 + 1])
            plot_training_curves(ax_loss, ax_acc, row1_data,
                                 f'BS={bs}, λ={fixed_wd:.0e}',
                                 legend_label_key='lr',
                                 legend_format=lambda x: f'η={x}')
        else:
            ax = fig.add_subplot(gs[0, col_idx * 2: col_idx * 2 + 2])
            ax.text(0.5, 0.5, f'No data for BS={bs}, λ={fixed_wd:.0e}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'BS={bs} - Fixed λ')

        # ========== Row 2: Heatmap ==========
        ax_hm = fig.add_subplot(gs[1, col_idx * 2: col_idx * 2 + 2])
        im = plot_heatmap(ax_hm, results_df, bs, f'BS={bs} - Best Test Accuracy (%)')

        # ========== Row 3: Fixed η, varying λ ==========
        row3_data = {k: v for k, v in all_data.items()
                     if k[0] == bs and abs(k[1] - fixed_lr) < 1e-10}

        if row3_data:
            ax_loss = fig.add_subplot(gs[2, col_idx * 2])
            ax_acc = fig.add_subplot(gs[2, col_idx * 2 + 1])
            plot_training_curves(ax_loss, ax_acc, row3_data,
                                 f'BS={bs}, η={fixed_lr}',
                                 legend_label_key='wd',
                                 legend_format=lambda x: f'λ={x:.0e}')
        else:
            ax = fig.add_subplot(gs[2, col_idx * 2: col_idx * 2 + 2])
            ax.text(0.5, 0.5, f'No data for BS={bs}, η={fixed_lr}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'BS={bs} - Fixed η')

    # Add column titles
    fig.text(0.25, 0.98, 'Batch Size = 32', ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.98, 'Batch Size = 128', ha='center', va='top', fontsize=14, fontweight='bold')

    # Add row titles
    fig.text(0.02, 0.83, 'Row 1:\nFixed λ\nVarying η', ha='center', va='center', fontsize=10, rotation=0)
    fig.text(0.02, 0.50, 'Row 2:\nOptimal\n(η, λ)', ha='center', va='center', fontsize=10, rotation=0)
    fig.text(0.02, 0.17, 'Row 3:\nFixed η\nVarying λ', ha='center', va='center', fontsize=10, rotation=0)

    # Main title
    fig.suptitle('V3 Supplementary: Batch Size Scaling Analysis', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0.05, 0, 1, 0.98])

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    plt.show()


def find_optimal_fixed_params(results_df):
    """Find good fixed λ and η values based on results."""
    # For each batch size, find the optimal (lr, wd) pair
    optimal = {}
    for bs in [32, 128]:
        bs_df = results_df[results_df['batch_size'] == bs]
        if not bs_df.empty:
            best_row = bs_df.loc[bs_df['best_test_acc'].idxmax()]
            optimal[bs] = {'lr': best_row['lr'], 'wd': best_row['wd']}

    # Use median values as fixed params (or most common in experiment grid)
    all_wds = results_df['wd'].unique()
    all_lrs = results_df['lr'].unique()

    # Pick middle values
    fixed_wd = sorted(all_wds)[len(all_wds) // 2]
    fixed_lr = sorted(all_lrs)[len(all_lrs) // 2]

    return fixed_lr, fixed_wd, optimal


def main():
    parser = argparse.ArgumentParser(description='V3 补充实验可视化')
    parser.add_argument('--history_dir', type=str, default='outputs/history/v3_supplementary',
                        help='Directory containing history JSON files')
    parser.add_argument('--results', type=str, default='outputs/results/v3_supplementary.csv',
                        help='CSV file with aggregated results')
    parser.add_argument('--output', type=str, default='outputs/plots/v3_supplementary_analysis.png',
                        help='Output path for the figure')
    parser.add_argument('--fixed_wd', type=float, default=None,
                        help='Fixed WD for row 1 plots (auto-detect if not specified)')
    parser.add_argument('--fixed_lr', type=float, default=None,
                        help='Fixed LR for row 3 plots (auto-detect if not specified)')
    args = parser.parse_args()

    # Load data
    print(f"Loading history from: {args.history_dir}")
    all_data = load_history_files(args.history_dir)
    print(f"Loaded {len(all_data)} experiment histories")

    print(f"Loading results from: {args.results}")
    results_df = load_results_csv(args.results)
    print(f"Loaded {len(results_df)} result rows")

    # Find optimal fixed params
    fixed_lr, fixed_wd, optimal = find_optimal_fixed_params(results_df)
    if args.fixed_wd is not None:
        fixed_wd = args.fixed_wd
    if args.fixed_lr is not None:
        fixed_lr = args.fixed_lr

    print(f"\nUsing fixed params:")
    print(f"  Fixed λ (WD) for Row 1: {fixed_wd:.0e}")
    print(f"  Fixed η (LR) for Row 3: {fixed_lr}")
    print(f"\nOptimal configs per batch size:")
    for bs, opt in optimal.items():
        print(f"  BS={bs}: LR={opt['lr']}, WD={opt['wd']:.0e}")

    # Create visualization
    create_3x2_visualization(all_data, results_df, args.output,
                              fixed_wd=fixed_wd, fixed_lr=fixed_lr)


if __name__ == '__main__':
    main()
