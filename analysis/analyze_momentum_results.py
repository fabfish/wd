"""
Analyze and visualize momentum search and SGD+WD supplement results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_data():
    """Load all experiment results."""
    results_dir = Path('outputs/results')
    
    # Load momentum search results
    momentum_df = pd.read_csv(results_dir / 'momentum_search.csv')
    print(f"Momentum search: {len(momentum_df)} experiments")
    
    # Load SGDM extended (momentum=0.9 baseline)
    sgdm_extended = pd.read_csv(results_dir / 'sgdm_extended.csv')
    # Filter only wd=0.002 for fair comparison
    sgdm_baseline = sgdm_extended[sgdm_extended['wd'] == 0.002].copy()
    sgdm_baseline['momentum'] = 0.9
    print(f"SGDM baseline (mom=0.9, wd=0.002): {len(sgdm_baseline)} experiments")
    
    # Combine momentum search with baseline
    all_momentum = pd.concat([momentum_df, sgdm_baseline], ignore_index=True)
    
    # Load SGD+WD supplement
    sgdwd_supp = pd.read_csv(results_dir / 'sgdwd_supplement.csv')
    print(f"SGD+WD supplement: {len(sgdwd_supp)} experiments")
    
    # Load existing SGD+WD results
    three_methods = pd.read_csv(results_dir / 'three_methods_comparison.csv')
    sgdwd_existing = three_methods[(three_methods['method'] == 'SGD+WD') & (three_methods['wd'] == 0.002)]
    print(f"SGD+WD existing (wd=0.002): {len(sgdwd_existing)} experiments")
    
    # Combine all SGD+WD
    all_sgdwd = pd.concat([sgdwd_existing, sgdwd_supp], ignore_index=True)
    all_sgdwd = all_sgdwd.drop_duplicates(subset=['lr'], keep='first')
    all_sgdwd = all_sgdwd.sort_values('lr')
    
    return all_momentum, all_sgdwd, three_methods


def analyze_momentum_results(df):
    """Analyze momentum search results."""
    print("\n" + "="*60)
    print("MOMENTUM SEARCH ANALYSIS (wd=0.002)")
    print("="*60)
    
    # Get unique momentums
    momentums = sorted(df['momentum'].unique())
    
    # Find best config for each momentum
    print("\nBest LR for each momentum:")
    print("-"*50)
    best_per_momentum = []
    for mom in momentums:
        mom_data = df[df['momentum'] == mom]
        best_idx = mom_data['best_test_acc'].idxmax()
        best_row = mom_data.loc[best_idx]
        best_per_momentum.append({
            'momentum': mom,
            'best_lr': best_row['lr'],
            'best_acc': best_row['best_test_acc']
        })
        print(f"  momentum={mom:.2f}: LR={best_row['lr']:.3f}, Acc={best_row['best_test_acc']:.2f}%")
    
    # Overall best
    overall_best_idx = df['best_test_acc'].idxmax()
    overall_best = df.loc[overall_best_idx]
    print("\n" + "-"*50)
    print(f"OVERALL BEST: momentum={overall_best['momentum']}, LR={overall_best['lr']}, Acc={overall_best['best_test_acc']:.2f}%")
    
    return pd.DataFrame(best_per_momentum)


def plot_momentum_heatmap(df, save_path):
    """Create heatmap of accuracy vs momentum and LR."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pivot for heatmap
    pivot = df.pivot_table(values='best_test_acc', index='momentum', columns='lr', aggfunc='first')
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=60, vmax=80)
    
    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Momentum')
    ax.set_title('SGDM+WD Test Accuracy (%) vs Momentum and LR (wd=0.002)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best Test Accuracy (%)')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < 65 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_momentum_curves(df, save_path):
    """Plot accuracy vs LR for each momentum."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    momentums = sorted(df['momentum'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(momentums)))
    
    for mom, color in zip(momentums, colors):
        mom_data = df[df['momentum'] == mom].sort_values('lr')
        ax.plot(mom_data['lr'], mom_data['best_test_acc'], 
                marker='o', linewidth=2, markersize=8, 
                label=f'mom={mom:.2f}', color=color)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('SGDM+WD: Effect of Momentum on Optimal Learning Rate (wd=0.002)')
    ax.legend(loc='lower right', title='Momentum')
    ax.set_ylim([20, 82])
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines at common optimal LRs
    ax.axhline(y=78, color='gray', linestyle='--', alpha=0.5, label='78% reference')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_best_momentum_summary(best_df, save_path):
    """Bar chart of best accuracy per momentum."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(best_df)), best_df['best_acc'], 
                  color=plt.cm.viridis(np.linspace(0, 1, len(best_df))))
    
    ax.set_xticks(range(len(best_df)))
    ax.set_xticklabels([f'{m:.2f}' for m in best_df['momentum']])
    ax.set_xlabel('Momentum')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('SGDM+WD: Best Accuracy Achieved at Each Momentum Value (wd=0.002)')
    ax.set_ylim([70, 80])
    
    # Add value labels on bars
    for bar, row in zip(bars, best_df.itertuples()):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%\n(LR={row.best_lr})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sgdwd_complete(df, save_path):
    """Plot complete SGD+WD curve with all LR values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_sorted = df.sort_values('lr')
    ax.plot(df_sorted['lr'], df_sorted['best_test_acc'], 
            marker='o', linewidth=2, markersize=10, color='blue')
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('SGD+WD (wd=0.002): Complete LR Curve')
    ax.grid(True, alpha=0.3)
    
    # Mark optimal point
    best_idx = df['best_test_acc'].idxmax()
    best = df.loc[best_idx]
    ax.scatter([best['lr']], [best['best_test_acc']], 
               s=200, c='red', marker='*', zorder=5, label=f"Best: LR={best['lr']}, Acc={best['best_test_acc']:.2f}%")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_comparison_plot(momentum_df, sgdwd_df, save_path):
    """Compare SGDM+WD (different momentums) with SGD+WD."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot SGD+WD
    sgdwd_sorted = sgdwd_df.sort_values('lr')
    ax.plot(sgdwd_sorted['lr'], sgdwd_sorted['best_test_acc'], 
            marker='s', linewidth=2.5, markersize=8, color='black', 
            label='SGD+WD (no momentum)')
    
    # Plot best momentums for SGDM+WD
    best_momentums = [0.8, 0.9, 0.7]  # Top performers
    colors = ['blue', 'green', 'orange']
    
    for mom, color in zip(best_momentums, colors):
        mom_data = momentum_df[momentum_df['momentum'] == mom].sort_values('lr')
        if len(mom_data) > 0:
            ax.plot(mom_data['lr'], mom_data['best_test_acc'], 
                    marker='o', linewidth=2, markersize=7, color=color, 
                    label=f'SGDM+WD (mom={mom})')
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Comparison: SGD+WD vs SGDM+WD at Different Momentums (wd=0.002)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([68, 80])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Loading experiment results...")
    all_momentum, all_sgdwd, three_methods = load_data()
    
    # Analyze
    best_df = analyze_momentum_results(all_momentum)
    
    # Print SGD+WD results
    print("\n" + "="*60)
    print("SGD+WD RESULTS (wd=0.002)")
    print("="*60)
    print(all_sgdwd[['lr', 'best_test_acc']].sort_values('lr').to_string(index=False))
    best_sgdwd = all_sgdwd.loc[all_sgdwd['best_test_acc'].idxmax()]
    print(f"\nBest SGD+WD: LR={best_sgdwd['lr']}, Acc={best_sgdwd['best_test_acc']:.2f}%")
    
    # Create visualizations
    plots_dir = Path('outputs/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    plot_momentum_heatmap(all_momentum, plots_dir / 'momentum_heatmap.png')
    plot_momentum_curves(all_momentum, plots_dir / 'momentum_curves.png')
    plot_best_momentum_summary(best_df, plots_dir / 'momentum_best_summary.png')
    plot_sgdwd_complete(all_sgdwd, plots_dir / 'sgdwd_complete_curve.png')
    create_comparison_plot(all_momentum, all_sgdwd, plots_dir / 'sgdm_vs_sgdwd_comparison.png')
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best SGDM+WD: momentum={best_df.loc[best_df['best_acc'].idxmax(), 'momentum']:.2f}, "
          f"LR={best_df.loc[best_df['best_acc'].idxmax(), 'best_lr']:.2f}, "
          f"Acc={best_df['best_acc'].max():.2f}%")
    print(f"Best SGD+WD: LR={best_sgdwd['lr']:.2f}, Acc={best_sgdwd['best_test_acc']:.2f}%")
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()
