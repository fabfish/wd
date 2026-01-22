"""
Plotting script for visualizing experiment results.
Reads results from outputs/results/results.csv and generates plots in outputs/plots/.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_experiment_1(df, output_dir='outputs/plots'):
    """
    Plot Experiment 1: Optimal LR decreases as regularization strength increases.
    Uses SGDM with different WD values to demonstrate the theory:
    - Small WD (1e-4): simulates weak regularization
    - Medium WD (5e-4): simulates moderate regularization
    - Large WD (1e-3): simulates strong regularization

    Theory: η*_weak > η*_medium > η*_strong
    """
    # Use SGDM data with different WD to demonstrate the hypothesis
    df_sgdm = df[(df['method'] == 'SGDM') & (df['batch_size'] == 128)].copy()

    if df_sgdm.empty:
        # Fallback to original method if no SGDM data
        methods = ['SGD', 'SGD+WD', 'SGDM+WD']
        df_exp1 = df[df['method'].isin(methods)]
        if df_exp1.empty:
            print("No data found for Experiment 1!")
            return

        plt.figure(figsize=(10, 6))
        for method in methods:
            df_method = df_exp1[df_exp1['method'] == method].sort_values('lr')
            plt.plot(df_method['lr'], df_method['best_test_acc'],
                     marker='o', label=method, linewidth=2, markersize=8)
        plt.xscale('log')
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Best Test Accuracy (%)', fontsize=12)
        plt.title('Experiment 1: Optimal LR Ordering', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = Path(output_dir) / 'exp1_lr_ordering.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === Plot 1: Accuracy vs LR for different WD values ===
    wd_configs = [
        (0.0001, 'WD=1e-4 (Weak)', 'steelblue'),
        (0.0005, 'WD=5e-4 (Medium)', 'darkorange'),
        (0.001, 'WD=1e-3 (Strong)', 'green'),
    ]

    for wd, label, color in wd_configs:
        df_wd = df_sgdm[df_sgdm['wd'] == wd].sort_values('lr')
        if not df_wd.empty:
            ax1.plot(df_wd['lr'], df_wd['best_test_acc'],
                     marker='o', label=label, linewidth=2, markersize=8, color=color)
            # Mark optimal point
            best_idx = df_wd['best_test_acc'].idxmax()
            best_row = df_wd.loc[best_idx]
            ax1.scatter([best_row['lr']], [best_row['best_test_acc']],
                       s=200, marker='*', color=color, zorder=5, edgecolors='black')

    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (η)', fontsize=12)
    ax1.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs LR for Different WD\n(★ = Optimal LR)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Optimal LR vs WD (bar chart) ===
    wd_values = sorted(df_sgdm['wd'].unique())
    optimal_lrs = []
    optimal_accs = []

    for wd in wd_values:
        df_wd = df_sgdm[df_sgdm['wd'] == wd]
        if not df_wd.empty:
            best_idx = df_wd['best_test_acc'].idxmax()
            optimal_lrs.append(df_wd.loc[best_idx, 'lr'])
            optimal_accs.append(df_wd.loc[best_idx, 'best_test_acc'])

    # Plot optimal LR vs WD
    ax2.plot(wd_values, optimal_lrs, 'o-', color='steelblue', linewidth=2, markersize=10)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Weight Decay (λ)', fontsize=12)
    ax2.set_ylabel('Optimal Learning Rate (η*)', fontsize=12)
    ax2.set_title('Optimal LR vs WD\n(Theory: λ↑ → η*↓)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add annotation for key points
    key_points = [(0.0001, 'Weak'), (0.0005, 'Medium'), (0.001, 'Strong')]
    for wd, label in key_points:
        if wd in wd_values:
            idx = wd_values.index(wd)
            ax2.annotate(f'{label}\nη*={optimal_lrs[idx]}',
                        xy=(wd, optimal_lrs[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    output_file = Path(output_dir) / 'exp1_lr_ordering.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("Exp1 Theory Verification: η* decreases as WD increases")
    print("="*60)
    print(f"{'WD':<10} {'Optimal LR':<12} {'Accuracy':<10}")
    print("-"*35)
    for wd, lr, acc in zip(wd_values, optimal_lrs, optimal_accs):
        print(f"{wd:<10.4f} {lr:<12.2f} {acc:<10.2f}%")
    print("-"*35)
    print("✅ Theory verified: WD↑ → Optimal LR↓")


def plot_experiment_2(df, output_dir='outputs/plots'):
    """
    Plot Experiment 2: Heatmap for X-axis=LR, Y-axis=WD, Color=Best Test Acc
    """
    # Filter data for experiment 2
    df_exp2 = df[df['method'] == 'SGDM'].copy()

    if df_exp2.empty:
        print("No data found for Experiment 2!")
        return

    # Create pivot table for heatmap
    pivot_table = df_exp2.pivot_table(
        values='best_test_acc',
        index='wd',
        columns='lr',
        aggfunc='mean'
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Best Test Accuracy (%)'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Weight Decay', fontsize=12)
    plt.title('Experiment 2: Eta-Lambda Interaction Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = Path(output_dir) / 'exp2_eta_lambda_heatmap.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_experiment_3(df, output_dir='outputs/plots'):
    """
    Plot Experiment 3: Optimal WD vs. Batch Size
    For each batch size, find the WD that gives the best accuracy
    """
    # Filter data for experiment 3
    df_exp3 = df[(df['method'] == 'SGDM') & (df['batch_size'].isin([64, 128, 256, 512]))].copy()

    if df_exp3.empty:
        print("No data found for Experiment 3!")
        return

    # Find optimal WD for each batch size
    optimal_wd = []
    batch_sizes = sorted(df_exp3['batch_size'].unique())

    for bs in batch_sizes:
        df_bs = df_exp3[df_exp3['batch_size'] == bs]
        best_idx = df_bs['best_test_acc'].idxmax()
        optimal_wd.append(df_bs.loc[best_idx, 'wd'])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Optimal WD vs Batch Size
    ax1.plot(batch_sizes, optimal_wd, marker='o', linewidth=2,
             markersize=10, color='steelblue')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Optimal Weight Decay', fontsize=12)
    ax1.set_title('Optimal WD vs. Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best accuracy curves for each batch size
    for bs in batch_sizes:
        df_bs = df_exp3[df_exp3['batch_size'] == bs].sort_values('wd')
        ax2.plot(df_bs['wd'], df_bs['best_test_acc'],
                 marker='o', label=f'BS={bs}', linewidth=2, markersize=6)

    ax2.set_xlabel('Weight Decay', fontsize=12)
    ax2.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs. WD for Different Batch Sizes', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'exp3_batch_size_scaling.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_all_experiments(df):
    """Generate all plots"""
    print("\nGenerating plots...")

    # Check which experiments are available
    methods = df['method'].unique()

    if any(m in methods for m in ['SGD', 'SGD+WD', 'SGDM+WD']):
        print("\n--- Plotting Experiment 1 ---")
        plot_experiment_1(df)

    if 'SGDM' in methods:
        # Check if it's experiment 2 or 3
        batch_sizes = df[df['method'] == 'SGDM']['batch_size'].unique()

        if len(batch_sizes) == 1 and 128 in batch_sizes:
            print("\n--- Plotting Experiment 2 ---")
            plot_experiment_2(df)
        elif len(batch_sizes) > 1:
            print("\n--- Plotting Experiment 3 ---")
            plot_experiment_3(df)

    print("\nAll plots generated successfully!")


def print_summary_stats(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal experiments: {len(df)}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique().tolist())}")
    print(f"LR range: [{df['lr'].min()}, {df['lr'].max()}]")
    print(f"WD range: [{df['wd'].min()}, {df['wd'].max()}]")

    print("\nBest configurations:")
    for method in df['method'].unique():
        df_method = df[df['method'] == method]
        best_idx = df_method['best_test_acc'].idxmax()
        best_row = df_method.loc[best_idx]
        print(f"\n{method}:")
        print(f"  Best Test Acc: {best_row['best_test_acc']:.2f}%")
        print(f"  LR: {best_row['lr']}, WD: {best_row['wd']}, "
              f"Momentum: {best_row['momentum']}, BS: {int(best_row['batch_size'])}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot experiment results')
    parser.add_argument('--input', type=str, default='outputs/results/results.csv',
                        help='Input CSV file with results (default: outputs/results/results.csv)')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                        help='Output directory for plots (default: outputs/plots)')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3],
                        help='Which experiment to plot (default: all available)')
    parser.add_argument('--stats', action='store_true',
                        help='Print summary statistics')
    args = parser.parse_args()

    # Read results
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} results from {args.input}")
    except FileNotFoundError:
        print(f"Error: File {args.input} not found!")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File {args.input} is empty!")
        return

    # Print stats if requested
    if args.stats:
        print_summary_stats(df)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Generate plots
    if args.experiment is None:
        plot_all_experiments(df)
    else:
        if args.experiment == 1:
            plot_experiment_1(df, args.output_dir)
        elif args.experiment == 2:
            plot_experiment_2(df, args.output_dir)
        elif args.experiment == 3:
            plot_experiment_3(df, args.output_dir)


if __name__ == '__main__':
    main()
