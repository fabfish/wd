"""
V3 补充实验可视化：3×2 布局 (改进版)

可视化设计：
- 列：左=BS 32，右=BS 128
- 第一排：使用最优λ (1e-03)，η变化的训练曲线
- 第二排：最优(η,λ)热力图 (高质量渲染，带colorbar)
- 第三排：使用最优η (BS32: 0.02, BS128: 0.05)，λ变化的训练曲线

Usage:
    python plot_v3_supplementary_v2.py --history_dir outputs/history/v3_supplementary
    python plot_v3_supplementary_v2.py --results outputs/results/v3_supplementary.csv
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150


def create_smooth_colormap(vmin=60, vmax=80):
    """
    Create a smooth colormap for accuracy visualization.
    Maps from vmin to vmax:
    - vmin ~ 65%: red/orange
    - 70-75%: yellow
    - 75-78%: light green
    - 78+%: dark green
    """
    # Normalize key thresholds to [0, 1] range
    def norm(v):
        return (v - vmin) / (vmax - vmin)
    
    cdict = {
        'red': [
            (0.0, 0.65, 0.65),           # vmin: red
            (norm(70), 0.95, 0.95),      # 70%: orange
            (norm(75), 1.0, 1.0),        # 75%: yellow
            (norm(77), 0.55, 0.55),      # 77%: green
            (1.0, 0.0, 0.0),             # vmax: dark green
        ],
        'green': [
            (0.0, 0.15, 0.15),           # vmin: red
            (norm(70), 0.55, 0.55),      # 70%: orange
            (norm(75), 0.90, 0.90),      # 75%: yellow
            (norm(77), 0.80, 0.80),      # 77%: green
            (1.0, 0.55, 0.55),           # vmax: dark green
        ],
        'blue': [
            (0.0, 0.10, 0.10),           # vmin: red
            (norm(70), 0.25, 0.25),      # 70%: orange
            (norm(75), 0.25, 0.25),      # 75%: yellow
            (norm(77), 0.35, 0.35),      # 77%: green
            (1.0, 0.30, 0.30),           # vmax: dark green
        ]
    }
    return LinearSegmentedColormap('smooth_acc', cdict, N=256)


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


def plot_heatmap_improved(ax, df, bs, title, cax=None):
    """
    Plot improved heatmap of best test accuracy for given batch size.
    - Smooth interpolation
    - No grid lines
    - Colorbar
    - Blue box for best point
    """
    bs_df = df[df['batch_size'] == bs].copy()

    if bs_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return None

    # Create pivot table
    pivot = bs_df.pivot_table(values='best_test_acc', index='wd', columns='lr', aggfunc='max')

    # Sort index (WD) in ascending order for proper display with origin='lower'
    pivot = pivot.sort_index(ascending=True)

    # Get actual data range
    valid_data = pivot.values[~np.isnan(pivot.values)]
    data_min = max(valid_data.min() - 2, 60)  # At least 60
    data_max = min(valid_data.max() + 1, 80)  # At most 80
    
    # Create smooth colormap with actual data range
    cmap = create_smooth_colormap(vmin=data_min, vmax=data_max)

    # Prepare data for smooth interpolation
    data = pivot.values
    extent = [0, len(pivot.columns), 0, len(pivot.index)]

    # Upsample data for smoother interpolation
    zoom_factor = 10
    # Handle NaN values - use data_min for missing values
    data_filled = np.nan_to_num(data, nan=data_min)
    # Use order=1 (bilinear) to avoid overshoot artifacts from bicubic
    data_zoomed = zoom(data_filled, zoom_factor, order=1)
    # Clip to valid range to prevent color mapping issues
    data_zoomed = np.clip(data_zoomed, data_min, data_max)

    # Plot with smooth interpolation
    im = ax.imshow(
        data_zoomed,
        cmap=cmap,
        aspect='auto',
        vmin=data_min,
        vmax=data_max,
        extent=extent,
        origin='lower',
        interpolation='bilinear'
    )

    # Remove grid
    ax.grid(False)

    # Set tick positions and labels
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([f'{x:.3g}' for x in pivot.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f'{y:.0e}' for y in pivot.index], fontsize=8)

    ax.set_xlabel('Learning Rate (η)')
    ax.set_ylabel('Weight Decay (λ)')
    ax.set_title(title)

    # Add text annotations at cell centers
    for i, wd in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            val = pivot.loc[wd, lr]
            if not np.isnan(val):
                # Use black/white based on value
                color = 'white' if val < 60 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{val:.1f}',
                        ha='center', va='center', color=color, fontsize=7, fontweight='bold')

    # Mark maximum with blue rectangle
    max_val = pivot.values[~np.isnan(pivot.values)].max()
    max_idx = np.where(pivot.values == max_val)
    if len(max_idx[0]) > 0:
        row_idx, col_idx = max_idx[0][0], max_idx[1][0]
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1,
                                     fill=False, edgecolor='blue', linewidth=3))

    return im


def create_3x2_visualization(all_data, results_df, output_path):
    """
    Create the main 3×2 visualization with optimal configs.
    
    - Row 1: Fixed λ=1e-03 (optimal for both), varying η
    - Row 2: Heatmap with colorbar
    - Row 3: Fixed η (BS32: 0.02, BS128: 0.05), varying λ
    """
    fig = plt.figure(figsize=(15, 13))

    batch_sizes = [32, 128]
    optimal_wds = {32: 1e-03, 128: 1e-03}  # Optimal WD for both
    optimal_lrs = {32: 0.02, 128: 0.05}    # Optimal LR per batch size

    # Create grid: 3 rows × 4 columns + space for colorbar
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.35, width_ratios=[1, 1, 1, 1, 0.05])

    heatmap_ims = []

    for col_idx, bs in enumerate(batch_sizes):
        fixed_wd = optimal_wds[bs]
        fixed_lr = optimal_lrs[bs]

        # ========== Row 1: Fixed λ (optimal), varying η ==========
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
        im = plot_heatmap_improved(ax_hm, results_df, bs, f'BS={bs} - Best Test Accuracy (%)')
        if im is not None:
            heatmap_ims.append(im)

        # ========== Row 3: Fixed η (optimal for this BS), varying λ ==========
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

    # Add shared colorbar for heatmaps
    if heatmap_ims:
        cbar_ax = fig.add_subplot(gs[1, 4])
        cbar = fig.colorbar(heatmap_ims[0], cax=cbar_ax)
        cbar.set_label('Test Accuracy (%)', fontsize=11)

    # Add column titles (reduced whitespace)
    fig.text(0.27, 0.94, 'Batch Size = 32', ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.72, 0.94, 'Batch Size = 128', ha='center', va='top', fontsize=14, fontweight='bold')

    # Add row titles
    fig.text(0.02, 0.83, 'Row 1:\nFixed λ\n(optimal)\nVarying η', ha='center', va='center', fontsize=9, rotation=0)
    fig.text(0.02, 0.50, 'Row 2:\nOptimal\n(η, λ)', ha='center', va='center', fontsize=9, rotation=0)
    fig.text(0.02, 0.17, 'Row 3:\nFixed η\n(optimal)\nVarying λ', ha='center', va='center', fontsize=9, rotation=0)

    # Main title
    # fig.suptitle('V3 Supplementary: Batch Size Scaling Analysis\n(LR scaling: BS32→BS128 requires 2.5x higher LR)',
    #              fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0.05, 0, 0.98, 0.98])

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='V3 补充实验可视化 (改进版)')
    parser.add_argument('--history_dir', type=str, default='outputs/history/v3_supplementary',
                        help='Directory containing history JSON files')
    parser.add_argument('--results', type=str, default='outputs/results/v3_supplementary.csv',
                        help='CSV file with aggregated results')
    parser.add_argument('--output', type=str, default='outputs/plots/v3_supplementary_analysis_v2.png',
                        help='Output path for the figure')
    args = parser.parse_args()

    # Load data
    print(f"Loading history from: {args.history_dir}")
    all_data = load_history_files(args.history_dir)
    print(f"Loaded {len(all_data)} experiment histories")

    print(f"Loading results from: {args.results}")
    results_df = load_results_csv(args.results)
    print(f"Loaded {len(results_df)} result rows")

    # Print optimal configs
    print("\nOptimal configs used in visualization:")
    print("  Row 1 (fixed λ=1e-03 for both batch sizes)")
    print("  Row 3: BS=32 uses η=0.02, BS=128 uses η=0.05")
    print("         This shows the LR scaling: 0.02 → 0.05 (2.5x) when BS: 32 → 128 (4x)")

    # Create visualization
    create_3x2_visualization(all_data, results_df, args.output)


if __name__ == '__main__':
    main()
