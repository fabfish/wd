"""
Improved visualization script for experiments.
Generates charts to validate three core hypotheses.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_exp1_lr_ordering(df, output_dir):
    """
    Experiment 1: Learning Rate Ordering Validation
    Goal: Verify eta_SGD > eta_SGD+WD > eta_SGDM+WD (stability boundary)
    """
    # Filter experiment 1 data
    methods = ['SGD', 'SGD+WD', 'SGDM+WD']
    exp1_df = df[df['method'].isin(methods) & (df['batch_size'] == 128)]

    if exp1_df.empty:
        print("No data for Experiment 1")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'SGD': '#2196F3', 'SGD+WD': '#4CAF50', 'SGDM+WD': '#FF5722'}
    markers = {'SGD': 'o', 'SGD+WD': 's', 'SGDM+WD': '^'}

    # Plot 1: Test Accuracy vs LR
    ax1 = axes[0]
    for method in methods:
        subset = exp1_df[exp1_df['method'] == method].sort_values('lr')
        if not subset.empty:
            ax1.plot(subset['lr'], subset['best_test_acc'],
                     marker=markers[method], label=method, color=colors[method],
                     linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax1.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Test Accuracy vs Learning Rate', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Stability Analysis (Train Loss as indicator)
    ax2 = axes[1]
    for method in methods:
        subset = exp1_df[exp1_df['method'] == method].sort_values('lr')
        if not subset.empty:
            ax2.plot(subset['lr'], subset['final_train_loss'],
                     marker=markers[method], label=method, color=colors[method],
                     linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax2.set_ylabel('Final Train Loss (log scale)', fontsize=12)
    ax2.set_title('(b) Train Loss vs Learning Rate\n(Loss spikes when diverging)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    # Mark stability boundary
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Divergence threshold')

    # Plot 3: Optimal LR Comparison
    ax3 = axes[2]
    optimal_lrs = []
    optimal_accs = []
    stability_boundaries = []

    for method in methods:
        subset = exp1_df[exp1_df['method'] == method]
        if not subset.empty:
            # Optimal generalization LR
            best_row = subset.loc[subset['best_test_acc'].idxmax()]
            optimal_lrs.append(best_row['lr'])
            optimal_accs.append(best_row['best_test_acc'])

            # Stability boundary (max lr where train_loss < 1.0)
            stable = subset[subset['final_train_loss'] < 1.0]
            if not stable.empty:
                stability_boundaries.append(stable['lr'].max())
            else:
                stability_boundaries.append(0)

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(x - width/2, optimal_lrs, width, label='Optimal LR', color='steelblue')
    bars2 = ax3.bar(x + width/2, stability_boundaries, width, label='Stability Boundary', color='coral')

    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('(c) Optimal LR vs Stability Boundary', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.set_yscale('log')

    # Add value labels
    for bar, val in zip(bars1, optimal_lrs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, stability_boundaries):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = Path(output_dir) / 'exp1_lr_ordering_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print statistics
    print("\nExperiment 1 Results:")
    print("-" * 60)
    for i, method in enumerate(methods):
        print(f"{method:12s}: Optimal LR={optimal_lrs[i]:.3f}, Stability Boundary={stability_boundaries[i]:.1f}, Best Acc={optimal_accs[i]:.2f}%")

    # Verify hypothesis
    print("\nHypothesis Verification:")
    if stability_boundaries[0] >= stability_boundaries[1] >= stability_boundaries[2]:
        print("OK: Stability boundary ordering: SGD >= SGD+WD >= SGDM+WD (matches theory)")
    else:
        print("X: Stability boundary ordering does not match expectation")


def plot_exp2_eta_lambda_heatmap(df, output_dir):
    """
    Experiment 2: eta-lambda inverse relationship heatmap
    Theory: High accuracy region should follow eta * lambda = const
    """
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import zoom

    # Filter SGDM data with batch_size=128
    exp2_df = df[(df['method'] == 'SGDM') & (df['batch_size'] == 128)].copy()

    if exp2_df.empty:
        print("No data for Experiment 2")
        return

    # Create pivot table
    pivot = exp2_df.pivot_table(
        values='best_test_acc',
        index='wd',
        columns='lr',
        aggfunc='mean'
    )
    pivot = pivot.sort_index(ascending=True)

    if pivot.empty:
        print("No pivot data for Experiment 2")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Custom colormap for better distinction
    cdict = {
        'red': [
            (0.0, 0.55, 0.55),
            (0.4375, 0.70, 0.70),
            (0.60, 0.85, 0.85),
            (0.85, 0.95, 0.95),
            (0.94, 1.0, 1.0),
            (0.95, 0.55, 0.55),
            (1.0, 0.0, 0.0),
        ],
        'green': [
            (0.0, 0.10, 0.10),
            (0.4375, 0.35, 0.35),
            (0.60, 0.55, 0.55),
            (0.85, 0.65, 0.65),
            (0.94, 0.90, 0.90),
            (0.95, 0.80, 0.80),
            (1.0, 0.55, 0.55),
        ],
        'blue': [
            (0.0, 0.10, 0.10),
            (0.4375, 0.15, 0.15),
            (0.60, 0.35, 0.35),
            (0.85, 0.30, 0.30),
            (0.94, 0.25, 0.25),
            (0.95, 0.35, 0.35),
            (1.0, 0.30, 0.30),
        ]
    }
    cmap_smooth = LinearSegmentedColormap('smooth_acc', cdict, N=256)

    # Plot 1: Heatmap with smooth interpolation
    ax1 = axes[0]
    data = pivot.values
    extent = [0, len(pivot.columns), 0, len(pivot.index)]

    # Upsample for smoother visualization
    zoom_factor = 10
    data_zoomed = zoom(data, zoom_factor, order=3)

    im = ax1.imshow(
        data_zoomed,
        cmap=cmap_smooth,
        aspect='auto',
        vmin=0,
        vmax=80,
        extent=extent,
        origin='lower',
        interpolation='bilinear'
    )

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Test Accuracy (%)', fontsize=11)

    ax1.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax1.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
    ax1.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax1.set_yticklabels([f'{y:.4f}' for y in pivot.index])

    # Add value annotations
    for i, wd in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            val = pivot.loc[wd, lr]
            if not np.isnan(val):
                ax1.text(j + 0.5, i + 0.5, f'{val:.1f}',
                        ha='center', va='center', fontsize=9,
                        color='black', fontweight='bold')

    ax1.set_xlabel('Learning Rate (eta)', fontsize=12)
    ax1.set_ylabel('Weight Decay (lambda)', fontsize=12)
    ax1.set_title('Exp2: eta-lambda Interaction Heatmap', fontsize=14, fontweight='bold')

    # Plot 2: Optimal lambda vs eta
    ax2 = axes[1]
    lr_values = sorted(exp2_df['lr'].unique())
    optimal_wds = []

    for lr in lr_values:
        subset = exp2_df[exp2_df['lr'] == lr]
        if not subset.empty:
            best_row = subset.loc[subset['best_test_acc'].idxmax()]
            optimal_wds.append(best_row['wd'])
        else:
            optimal_wds.append(np.nan)

    ax2.plot(lr_values, optimal_wds, 'o-', markersize=10, linewidth=2,
             color='steelblue', label='Observed optimal lambda')

    # Fit inverse relationship
    valid_mask = ~np.isnan(optimal_wds)
    lr_fit = np.array(lr_values)[valid_mask]
    wd_fit = np.array(optimal_wds)[valid_mask]

    if len(lr_fit) >= 3:
        try:
            log_lr = np.log(lr_fit)
            log_wd = np.log(wd_fit)
            coeffs = np.polyfit(log_lr, log_wd, 1)
            b = -coeffs[0]
            a = np.exp(coeffs[1])

            lr_smooth = np.linspace(min(lr_values), max(lr_values), 100)
            wd_fitted = a / (lr_smooth ** b)
            ax2.plot(lr_smooth, wd_fitted, 'r--', linewidth=2,
                     label=f'Fit: lambda ~ eta^{-b:.2f}')

            wd_theory = optimal_wds[0] * lr_values[0] / np.array(lr_smooth)
            ax2.plot(lr_smooth, wd_theory, 'g:', linewidth=2, alpha=0.7,
                     label='Theory: lambda ~ 1/eta')
        except Exception as e:
            print(f"Fitting failed: {e}")

    ax2.set_xlabel('Learning Rate (eta)', fontsize=12)
    ax2.set_ylabel('Optimal Weight Decay (lambda)', fontsize=12)
    ax2.set_title('Optimal lambda vs eta\n(Inverse Relationship)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy vs eta*lambda
    ax3 = axes[2]
    exp2_df['eta_lambda'] = exp2_df['lr'] * exp2_df['wd']

    ax3.scatter(exp2_df['eta_lambda'], exp2_df['best_test_acc'],
                alpha=0.7, s=60, c='steelblue')

    ax3.set_xscale('log')
    ax3.set_xlabel('eta * lambda', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Accuracy vs eta*lambda\n(Verifying lambda~1/eta)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    best_idx = exp2_df['best_test_acc'].idxmax()
    best_row = exp2_df.loc[best_idx]
    ax3.scatter([best_row['eta_lambda']], [best_row['best_test_acc']],
                s=200, c='red', marker='*', zorder=5,
                label=f'Best: eta={best_row["lr"]}, lambda={best_row["wd"]}')
    ax3.legend(fontsize=10)

    plt.tight_layout()
    output_path = Path(output_dir) / 'exp2_eta_lambda_heatmap_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print statistics
    print("\nExperiment 2 Results:")
    print("-" * 60)
    print("Optimal Weight Decay for each Learning Rate:")
    for lr, wd in zip(lr_values, optimal_wds):
        if not np.isnan(wd):
            print(f"  eta={lr:.3f} -> optimal lambda={wd:.4f}")


def plot_exp3_batch_lambda(df, output_dir):
    """
    Experiment 3: B-lambda proportional relationship
    Theory: Optimal lambda is proportional to batch size B
    Uses linear LR scaling rule: lr = base_lr * (B / base_B)
    """
    # Define the expected lr for each batch size (linear scaling rule)
    # Base: B=128, lr=0.1
    bs_lr_map = {
        32: 0.025,
        64: 0.05,
        128: 0.1,
        256: 0.2,
        512: 0.4
    }

    # Filter exp3 data: SGDM with specific batch sizes and their corresponding lr
    exp3_data = []
    for bs, expected_lr in bs_lr_map.items():
        subset = df[(df['method'] == 'SGDM') &
                    (df['batch_size'] == bs) &
                    (np.isclose(df['lr'], expected_lr, rtol=0.01))]
        if not subset.empty:
            exp3_data.append(subset)

    if not exp3_data:
        print("No data for Experiment 3")
        return

    exp3_df = pd.concat(exp3_data, ignore_index=True)
    batch_sizes = sorted(exp3_df['batch_size'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Optimal wd for each batch size
    ax1 = axes[0]
    optimal_wds = []
    optimal_accs = []

    for bs in batch_sizes:
        subset = exp3_df[exp3_df['batch_size'] == bs]
        if not subset.empty:
            best_row = subset.loc[subset['best_test_acc'].idxmax()]
            optimal_wds.append(best_row['wd'])
            optimal_accs.append(best_row['best_test_acc'])
        else:
            optimal_wds.append(np.nan)
            optimal_accs.append(np.nan)

    ax1.plot(batch_sizes, optimal_wds, 'o-', markersize=10, linewidth=2, color='steelblue')
    ax1.set_xlabel('Batch Size (B)', fontsize=12)
    ax1.set_ylabel('Optimal Weight Decay (lambda)', fontsize=12)
    ax1.set_title('(a) Optimal lambda vs Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Add theoretical line (lambda proportional to B)
    if len(batch_sizes) > 1 and not np.isnan(optimal_wds[0]):
        base_bs = batch_sizes[0]
        base_wd = optimal_wds[0]
        theoretical_wds = [base_wd * (bs / base_bs) for bs in batch_sizes]
        ax1.plot(batch_sizes, theoretical_wds, '--', color='red',
                label='Theory (lambda prop. to B)', linewidth=2, alpha=0.7)
        ax1.legend()

    # Plot 2: Accuracy vs wd curves for each batch size
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

    for i, bs in enumerate(batch_sizes):
        subset = exp3_df[exp3_df['batch_size'] == bs].sort_values('wd')
        if not subset.empty:
            ax2.plot(subset['wd'], subset['best_test_acc'],
                    'o-', label=f'B={bs}', color=colors[i], markersize=6)

    ax2.set_xscale('log')
    ax2.set_xlabel('Weight Decay (lambda)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Test Acc vs lambda (Different Batch Sizes)', fontsize=14, fontweight='bold')
    ax2.legend(title='Batch Size')
    ax2.grid(True, alpha=0.3)

    # Plot 3: lambda/B normalized analysis
    ax3 = axes[2]
    exp3_df['lambda_over_B'] = exp3_df['wd'] / exp3_df['batch_size']

    for i, bs in enumerate(batch_sizes):
        subset = exp3_df[exp3_df['batch_size'] == bs].sort_values('lambda_over_B')
        if not subset.empty:
            ax3.plot(subset['lambda_over_B'], subset['best_test_acc'],
                    'o-', label=f'B={bs}', color=colors[i], markersize=6)

    ax3.set_xscale('log')
    ax3.set_xlabel('lambda/B (Normalized Weight Decay)', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('(c) Test Acc vs lambda/B\n(Curves should overlap if lambda prop. to B)', fontsize=14, fontweight='bold')
    ax3.legend(title='Batch Size')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'exp3_batch_lambda_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print statistics
    print("\nExperiment 3 Results:")
    print("-" * 60)
    print("Optimal Weight Decay for each Batch Size:")
    for i, bs in enumerate(batch_sizes):
        if not np.isnan(optimal_wds[i]):
            print(f"  B={bs:4d} -> optimal_lambda={optimal_wds[i]:.2e}, acc={optimal_accs[i]:.2f}%")

    print("\nHypothesis Verification:")
    # Check proportional relationship
    valid_data = [(bs, wd) for bs, wd in zip(batch_sizes, optimal_wds) if not np.isnan(wd)]
    if len(valid_data) >= 3:
        bs_arr = np.array([x[0] for x in valid_data])
        wd_arr = np.array([x[1] for x in valid_data])
        correlation = np.corrcoef(np.log(bs_arr), np.log(wd_arr))[0, 1]
        print(f"log(B) vs log(lambda_optimal) correlation: {correlation:.3f}")
        if correlation > 0.5:
            print("OK: B and lambda are positively correlated (supports proportional hypothesis)")
        else:
            print("X: No clear proportional relationship observed")

    # Compute ratio
    if len(valid_data) >= 2:
        ratios = [wd / bs for bs, wd in valid_data]
        print(f"lambda/B ratios: {[f'{r:.2e}' for r in ratios]}")
        print(f"lambda/B mean: {np.mean(ratios):.2e}, std: {np.std(ratios):.2e}")


def main():
    parser = argparse.ArgumentParser(description='Generate improved experiment visualizations')
    parser.add_argument('--input', type=str, default='outputs/results/results_v2.csv',
                        help='Input CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                        help='Output directory')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], default=None,
                        help='Only plot specified experiment')
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read data
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} results from {args.input}")
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        return

    # 生成图表
    if args.experiment is None or args.experiment == 1:
        plot_exp1_lr_ordering(df, output_dir)

    if args.experiment is None or args.experiment == 2:
        plot_exp2_eta_lambda_heatmap(df, output_dir)

    if args.experiment is None or args.experiment == 3:
        plot_exp3_batch_lambda(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
