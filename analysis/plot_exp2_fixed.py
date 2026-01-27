"""
Fixed Experiment 2 visualization script
- Y-axis inverted: large wd on top, small on bottom
- Better colormap for high accuracy distinction
- All English text
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit


def create_custom_colormap(vmin=0, vmax=80, high_acc_threshold=68):
    """
    Create a colormap with better distinction for high accuracy values.

    Strategy:
    - Below high_acc_threshold: uniform dark red (very compressed)
    - 68-80: rainbow-like gradient with maximum distinction
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Normalize threshold
    threshold_norm = (high_acc_threshold - vmin) / (vmax - vmin)

    # Build segmented colormap
    # Below 68%: single dark red color (almost no variation)
    # 68-80%: full color spectrum from orange to dark green

    cdict = {
        'red': [
            (0.0, 0.5, 0.5),           # 0%: dark red
            (threshold_norm - 0.01, 0.55, 0.55),  # Just below 68%: still dark red
            (threshold_norm, 0.95, 0.95),  # 68%: orange
            (threshold_norm + 0.25*(1-threshold_norm), 1.0, 1.0),   # 71%: yellow
            (threshold_norm + 0.5*(1-threshold_norm), 0.7, 0.7),    # 74%: yellow-green
            (threshold_norm + 0.75*(1-threshold_norm), 0.3, 0.3),   # 77%: green
            (1.0, 0.0, 0.0),           # 80%: dark green
        ],
        'green': [
            (0.0, 0.1, 0.1),           # 0%: dark red
            (threshold_norm - 0.01, 0.15, 0.15),  # Just below 68%: still dark red
            (threshold_norm, 0.5, 0.5),    # 68%: orange
            (threshold_norm + 0.25*(1-threshold_norm), 0.85, 0.85), # 71%: yellow
            (threshold_norm + 0.5*(1-threshold_norm), 0.9, 0.9),    # 74%: yellow-green
            (threshold_norm + 0.75*(1-threshold_norm), 0.8, 0.8),   # 77%: green
            (1.0, 0.55, 0.55),         # 80%: dark green
        ],
        'blue': [
            (0.0, 0.1, 0.1),           # 0%: dark red
            (threshold_norm - 0.01, 0.1, 0.1),   # Just below 68%: still dark red
            (threshold_norm, 0.2, 0.2),    # 68%: orange
            (threshold_norm + 0.25*(1-threshold_norm), 0.2, 0.2),   # 71%: yellow
            (threshold_norm + 0.5*(1-threshold_norm), 0.3, 0.3),    # 74%: yellow-green
            (threshold_norm + 0.75*(1-threshold_norm), 0.3, 0.3),   # 77%: green
            (1.0, 0.25, 0.25),         # 80%: dark green
        ]
    }

    return LinearSegmentedColormap('custom_acc', cdict, N=256)


def create_focused_colormap():
    """
    Create a colormap focused on 72-79 range with maximum distinction.
    Values below 68 are dark red, 68-72 orange-yellow, 72-79 yellow to dark green.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # We map to [0, 1] where:
    # 0-0.15: dark red (for values < 68)
    # 0.15-0.30: orange-red (68-72)
    # 0.30-1.0: yellow -> green (72-79)
    cdict = {
        'red': [
            (0.0, 0.45, 0.45),      # <68: dark red
            (0.15, 0.50, 0.50),     # ~68: dark red
            (0.20, 0.90, 0.90),     # 70: orange
            (0.30, 1.0, 1.0),       # 72: yellow
            (0.50, 0.85, 0.85),     # 74.5: yellow-green
            (0.70, 0.45, 0.45),     # 77: green
            (1.0, 0.0, 0.0),        # 79+: dark green
        ],
        'green': [
            (0.0, 0.08, 0.08),      # <68: dark red
            (0.15, 0.12, 0.12),     # ~68: dark red
            (0.20, 0.55, 0.55),     # 70: orange
            (0.30, 0.90, 0.90),     # 72: yellow
            (0.50, 0.92, 0.92),     # 74.5: yellow-green
            (0.70, 0.78, 0.78),     # 77: green
            (1.0, 0.50, 0.50),      # 79+: dark green
        ],
        'blue': [
            (0.0, 0.08, 0.08),      # <68: dark red
            (0.15, 0.10, 0.10),     # ~68: dark red
            (0.20, 0.20, 0.20),     # 70: orange
            (0.30, 0.25, 0.25),     # 72: yellow
            (0.50, 0.30, 0.30),     # 74.5: yellow-green
            (0.70, 0.28, 0.28),     # 77: green
            (1.0, 0.22, 0.22),      # 79+: dark green
        ]
    }
    return LinearSegmentedColormap('focused_acc', cdict, N=256)


def plot_exp2_heatmap_fixed(df, output_dir):
    """
    Experiment 2: eta-lambda inverse relationship heatmap
    Fixed version with:
    - Inverted Y-axis (large wd on top)
    - Better color distinction for high accuracy
    - All English labels
    """
    # Filter SGDM data with batch_size=128
    exp2_df_all = df[(df['method'] == 'SGDM') & (df['batch_size'] == 128)].copy()
    
    # For plots 1 and 2: use only original data (wd >= 0.0001, lr >= 0.01)
    # This excludes the new supplement data with smaller wd and lr values
    exp2_df = exp2_df_all[(exp2_df_all['wd'] >= 0.0001) & (exp2_df_all['lr'] >= 0.01)].copy()

    if exp2_df.empty:
        print("No data for Experiment 2")
        return

    # Create pivot table for heatmap (using original data only)
    pivot = exp2_df.pivot_table(
        values='best_test_acc',
        index='wd',
        columns='lr',
        aggfunc='mean'
    )

    # Sort index in ascending order - with origin='lower', this puts large wd at top
    pivot = pivot.sort_index(ascending=True)

    if pivot.empty:
        print("No pivot data for Experiment 2")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Use focused colormap with data-appropriate range
    # Data ranges roughly from 3-78.6, but we want to emphasize 72-79 range
    cmap = create_focused_colormap()

    # Plot 1: Heatmap with inverted Y-axis
    ax1 = axes[0]

    # Use continuous colormap with smooth interpolation
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    from scipy.ndimage import zoom

    # Create continuous colormap:
    # Below 35: deep red
    # 35-68: orange transition
    # 68-75.5: yellow zone
    # Above 75.5 (76+): green zone
    cdict = {
        'red': [
            (0.0, 0.55, 0.55),      # 0%: deep red
            (0.4375, 0.70, 0.70),   # 35%: red-orange
            (0.60, 0.85, 0.85),     # 48%: light orange
            (0.85, 0.95, 0.95),     # 68%: orange
            (0.94, 1.0, 1.0),       # 75.2%: yellow
            (0.95, 0.55, 0.55),     # 76%: green
            (1.0, 0.0, 0.0),        # 80%: dark green
        ],
        'green': [
            (0.0, 0.10, 0.10),      # 0%: deep red
            (0.4375, 0.35, 0.35),   # 35%: red-orange
            (0.60, 0.55, 0.55),     # 48%: light orange
            (0.85, 0.65, 0.65),     # 68%: orange
            (0.94, 0.90, 0.90),     # 75.2%: yellow
            (0.95, 0.80, 0.80),     # 76%: green
            (1.0, 0.55, 0.55),      # 80%: dark green
        ],
        'blue': [
            (0.0, 0.10, 0.10),      # 0%: deep red
            (0.4375, 0.15, 0.15),   # 35%: red-orange
            (0.60, 0.35, 0.35),     # 48%: light orange
            (0.85, 0.30, 0.30),     # 68%: orange
            (0.94, 0.25, 0.25),     # 75.2%: yellow
            (0.95, 0.35, 0.35),     # 76%: green
            (1.0, 0.30, 0.30),      # 80%: dark green
        ]
    }
    cmap_smooth = LinearSegmentedColormap('smooth_acc', cdict, N=256)

    # Use imshow with interpolation for smooth color transitions
    data = pivot.values
    extent = [0, len(pivot.columns), 0, len(pivot.index)]

    # Upsample data for smoother interpolation
    zoom_factor = 10
    data_zoomed = zoom(data, zoom_factor, order=3)  # Bicubic interpolation

    im = ax1.imshow(
        data_zoomed,
        cmap=cmap_smooth,
        aspect='auto',
        vmin=0,
        vmax=80,
        extent=extent,
        origin='lower',  # Row 0 at bottom
        interpolation='bilinear'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Test Accuracy (%)', fontsize=11)

    # Set tick positions and labels
    ax1.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax1.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
    ax1.set_yticks(np.arange(len(pivot.index)) + 0.5)
    # With origin='lower': row 0 at bottom, so labels go bottom to top
    ax1.set_yticklabels([f'{y:.4f}' for y in pivot.index])

    # Add value annotations at cell centers
    for i, wd in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            val = pivot.loc[wd, lr]
            if not np.isnan(val):
                # Use black font uniformly
                ax1.text(j + 0.5, i + 0.5, f'{val:.1f}',
                        ha='center', va='center', fontsize=9,
                        color='black', fontweight='bold')
    ax1.set_xlabel('Learning Rate (η)', fontsize=12)
    ax1.set_ylabel('Weight Decay (λ)', fontsize=12)
    ax1.set_title('Exp2: η-λ Interaction Heatmap', fontsize=14, fontweight='bold')

    # Plot 2: Optimal lambda vs eta (inverse relationship)
    ax2 = axes[1]

    # Find optimal wd for each lr
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
             color='steelblue', label='Observed optimal λ')

    # Fit inverse relationship: lambda = a / eta^b
    valid_mask = ~np.isnan(optimal_wds)
    lr_fit = np.array(lr_values)[valid_mask]
    wd_fit = np.array(optimal_wds)[valid_mask]

    if len(lr_fit) >= 3:
        try:
            # Fit power law: log(lambda) = log(a) - b * log(eta)
            log_lr = np.log(lr_fit)
            log_wd = np.log(wd_fit)
            coeffs = np.polyfit(log_lr, log_wd, 1)
            b = -coeffs[0]
            a = np.exp(coeffs[1])

            # Plot fitted curve
            lr_smooth = np.linspace(min(lr_values), max(lr_values), 100)
            wd_fitted = a / (lr_smooth ** b)
            ax2.plot(lr_smooth, wd_fitted, 'r--', linewidth=2,
                     label=f'Fit: λ ∝ η^{-b:.2f}')

            # Plot theoretical 1/eta curve for reference
            wd_theory = optimal_wds[0] * lr_values[0] / np.array(lr_smooth)
            ax2.plot(lr_smooth, wd_theory, 'g:', linewidth=2, alpha=0.7,
                     label='Theory: λ ∝ 1/η')
        except Exception as e:
            print(f"Fitting failed: {e}")

    ax2.set_xlabel('Learning Rate (η)', fontsize=12)
    ax2.set_ylabel('Optimal Weight Decay (λ)', fontsize=12)
    ax2.set_title('Optimal λ vs η\n(Inverse Relationship)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    # Plot 3: Accuracy vs eta*lambda (using ALL data including supplement)
    ax3 = axes[2]
    
    # Define transformation function for x-axis compression
    def transform_x(x_arr):
        """
        Compress range [10^-8, 10^-5] into visual width of 1 decade.
        Standard log scale for x >= 10^-5.
        Mapping:
          log(10^-5) = -5  -> -5
          log(10^-8) = -8  -> -6
        Formula for x < 10^-5:
          x_new = -5 + (log10(x) - (-5)) / 3
        """
        log_x = np.log10(np.array(x_arr))
        x_new = log_x.copy()
        
        # Mask for compressed region (x < 10^-5)
        mask = log_x < -5
        if np.any(mask):
            x_new[mask] = -5 + (log_x[mask] + 5) / 3
            
        return x_new

    # Calculate eta_lambda for all data
    exp2_df_all['eta_lambda'] = exp2_df_all['lr'] * exp2_df_all['wd']
    
    # Transform x coordinates
    x_transformed = transform_x(exp2_df_all['eta_lambda'])
    
    ax3.scatter(x_transformed, exp2_df_all['best_test_acc'],
                alpha=0.7, s=60, c='steelblue')

    # Custom x-axis ticks and labels
    # Major ticks: 10^-8, 10^-5, 10^-4, 10^-3, 10^-2
    # Transformed positions: -6, -5, -4, -3, -2
    major_ticks = [-6, -5, -4, -3, -2]
    major_labels = ['$10^{-8}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$']
    
    ax3.set_xticks(major_ticks)
    ax3.set_xticklabels(major_labels)
    
    # Add separating line at 10^-5 (visual indicator of scale change)
    ax3.axvline(x=-5, color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('η × λ', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Accuracy vs η×λ\n(Verifying λ∝1/η: constant η×λ)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Add light green background region for points with accuracy >= 75.5
    high_acc_threshold = 75.5
    high_acc_points = exp2_df_all[exp2_df_all['best_test_acc'] >= high_acc_threshold]
    if not high_acc_points.empty:
        # Use exact x range of high accuracy points (no margin)
        # Apply transformation to min/max
        raw_min = high_acc_points['eta_lambda'].min()
        raw_max = high_acc_points['eta_lambda'].max()
        
        box_x_min = transform_x([raw_min])[0]
        box_x_max = transform_x([raw_max])[0]

        from matplotlib.patches import Rectangle

        # Get current axis limits (after scatter and ylim are set)
        ylim = ax3.get_ylim()
        y_lower = ylim[0]
        y_upper = ylim[1]

        # Add very light green background below 75.5 threshold (lighter)
        rect_lower = Rectangle(
            (box_x_min, y_lower),
            box_x_max - box_x_min,
            high_acc_threshold - y_lower,
            facecolor='lightgreen',
            alpha=0.03,
            zorder=0
        )
        ax3.add_patch(rect_lower)

        # Add light green background above 75.5 threshold (darker)
        rect_upper = Rectangle(
            (box_x_min, high_acc_threshold),
            box_x_max - box_x_min,
            y_upper - high_acc_threshold,
            facecolor='lightgreen',
            alpha=0.35,
            zorder=0,
            label=f'Acc ≥ {high_acc_threshold}%'
        )
        ax3.add_patch(rect_upper)

        # Add horizontal green line at 75.5 (only between box_x_min and box_x_max)
        ax3.plot([box_x_min, box_x_max], [high_acc_threshold, high_acc_threshold],
                 color='green', linewidth=2, linestyle='-', zorder=2)

        # Add dark green vertical lines at start and end x positions
        ax3.axvline(x=box_x_min, color='darkgreen', linewidth=2.5, linestyle='-', zorder=2)
        ax3.axvline(x=box_x_max, color='darkgreen', linewidth=2.5, linestyle='-', zorder=2)

    # Mark optimal point (from all data)
    best_idx = exp2_df_all['best_test_acc'].idxmax()
    best_row = exp2_df_all.loc[best_idx]
    
    best_x = transform_x([best_row['eta_lambda']])[0]
    
    ax3.scatter([best_x], [best_row['best_test_acc']],
                s=200, c='red', marker='*', zorder=5,
                label=f'Best: η={best_row["lr"]}, λ={best_row["wd"]}')
    ax3.legend(fontsize=10)

    plt.tight_layout()
    output_path = Path(output_dir) / 'exp2_heatmap_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print analysis
    print("\nExperiment 2 Results:")
    print("-" * 60)
    print("Optimal λ for each learning rate:")
    for lr, wd in zip(lr_values, optimal_wds):
        if not np.isnan(wd):
            print(f"  η={lr:.3f} -> optimal λ={wd:.4f}")

    if len(lr_fit) >= 3:
        print(f"\nFitted relationship: λ ∝ η^{-b:.2f}")
        if abs(b - 1.0) < 0.3:
            print("✓ Close to inverse relationship (λ ∝ 1/η)")
        else:
            print(f"  Deviation from exact inverse: {abs(b-1.0):.2f}")


def main():
    input_file = 'outputs/results/results.csv'
    output_dir = 'outputs/plots'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} results from {input_file}")
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return

    plot_exp2_heatmap_fixed(df, output_dir)


if __name__ == '__main__':
    main()
