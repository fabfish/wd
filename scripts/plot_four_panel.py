#!/usr/bin/env python3
"""
Four-panel visualization for LR×WD vs Accuracy analysis.

Layout:
- Left column (narrow, gray): Overview plots from 0-shot (top) and 8-shot (bottom)
- Right column (wide, colored): Detailed analysis plots with stadium shapes

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as transforms
import os

# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load 0-shot and 8-shot CSV data."""
    df_0shot = pd.read_csv("outputs/results/results_0shot.csv")
    df_8shot = pd.read_csv("outputs/results/results_8shot.csv")
    
    for df in [df_0shot, df_8shot]:
        df['LR'] = df['LR'].astype(float)
        df['WD'] = df['WD'].astype(float)
        df['LR_x_WD'] = df['LR'] * df['WD']
    
    return df_0shot, df_8shot

# =============================================================================
# Stadium Shape Drawing
# =============================================================================

def draw_stadium_shape(ax, x1, y1, x2, y2, width, color_start, color_end, alpha=0.3, 
                       horizontal=True, zorder=0):
    """
    Draw a stadium shape (capsule/pill shape) between two points.
    
    Parameters:
    - ax: matplotlib axes
    - x1, y1: start point (in data coordinates)
    - x2, y2: end point (in data coordinates)
    - width: width of the stadium in data coordinates
    - color_start, color_end: colors for gradient
    - horizontal: if True, gradient goes from start to end
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    
    # Convert to display coordinates for proper calculation
    # For log scale x-axis, we work in log space
    log_x1, log_x2 = np.log10(x1), np.log10(x2)
    
    # Calculate angle and length
    dx = log_x2 - log_x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx)
    length = np.sqrt(dx**2 + dy**2)
    
    # Create stadium path in local coordinates
    # Stadium: rectangle with semicircles on both ends
    n_arc = 20
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, n_arc)
    theta_right = np.linspace(-np.pi/2, np.pi/2, n_arc)
    
    # Half width in appropriate units (we'll scale later)
    hw = width / 2
    
    # Left semicircle
    left_arc_x = hw * np.cos(theta_left)
    left_arc_y = hw * np.sin(theta_left)
    
    # Right semicircle  
    right_arc_x = length + hw * np.cos(theta_right)
    right_arc_y = hw * np.sin(theta_right)
    
    # Combine into full path
    path_x = np.concatenate([left_arc_x, right_arc_x])
    path_y = np.concatenate([left_arc_y, right_arc_y])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated_x = path_x * cos_a - path_y * sin_a + log_x1
    rotated_y = path_x * sin_a + path_y * cos_a + y1
    
    # Convert back from log space for x
    final_x = 10 ** rotated_x
    final_y = rotated_y
    
    # Create gradient fill using multiple polygons
    n_segments = 30
    cmap = LinearSegmentedColormap.from_list('custom', [color_start, color_end])
    
    for i in range(n_segments):
        t1 = i / n_segments
        t2 = (i + 1) / n_segments
        
        # Interpolate along the stadium
        seg_x1 = 10 ** (log_x1 + t1 * dx)
        seg_x2 = 10 ** (log_x1 + t2 * dx)
        seg_y1 = y1 + t1 * dy
        seg_y2 = y1 + t2 * dy
        
        # Create a thin slice of the stadium
        color = cmap(t1)
        
        # Perpendicular offset
        perp_dx = -sin_a * hw
        perp_dy = cos_a * hw
        
        slice_x = [10**(log_x1 + t1*dx - perp_dx), 10**(log_x1 + t1*dx + perp_dx),
                   10**(log_x1 + t2*dx + perp_dx), 10**(log_x1 + t2*dx - perp_dx)]
        slice_y = [y1 + t1*dy - perp_dy, y1 + t1*dy + perp_dy,
                   y1 + t2*dy + perp_dy, y1 + t2*dy - perp_dy]
        
        ax.fill(slice_x, slice_y, color=color, alpha=alpha, zorder=zorder, 
                edgecolor='none')
    
    # Draw outline
    ax.plot(final_x, final_y, color='gray', linewidth=0.8, alpha=0.5, zorder=zorder+1)
    # Close the path
    ax.plot([final_x[-1], final_x[0]], [final_y[-1], final_y[0]], 
            color='gray', linewidth=0.8, alpha=0.5, zorder=zorder+1)


def draw_stadium_simple(ax, points, padding_x=0.15, padding_y=0.02, 
                        color_left='#ffcccc', color_right='#cce5ff', alpha=0.4, zorder=0):
    """
    Draw a simple horizontal stadium shape around a group of points.
    Gradient from left (color_left) to right (color_right).
    
    points: list of (x, y) tuples in data coordinates
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    log_xs = [np.log10(x) for x in xs]
    
    min_log_x = min(log_xs) - padding_x
    max_log_x = max(log_xs) + padding_x
    min_y = min(ys) - padding_y
    max_y = max(ys) + padding_y
    
    center_y = (min_y + max_y) / 2
    height = max_y - min_y
    radius = height / 2
    
    # Semicircle radius in log space should match half the height visually
    # We need to scale it properly
    cap_radius_log = padding_x * 0.8  # radius of semicircle in log-x units
    
    # Create stadium path
    n_arc = 50
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, n_arc)
    theta_right = np.linspace(-np.pi/2, np.pi/2, n_arc)
    
    # Left semicircle center at (min_log_x, center_y)
    left_log_x = min_log_x + cap_radius_log * np.cos(theta_left)
    left_y = center_y + radius * np.sin(theta_left)
    
    # Right semicircle center at (max_log_x, center_y)  
    right_log_x = max_log_x + cap_radius_log * np.cos(theta_right)
    right_y = center_y + radius * np.sin(theta_right)
    
    # Gradient fill - include the caps properly
    n_segments = 60
    cmap = LinearSegmentedColormap.from_list('custom', [color_left, color_right])
    
    # Total span including caps
    total_span = (max_log_x + cap_radius_log) - (min_log_x - cap_radius_log)
    start_x = min_log_x - cap_radius_log
    
    for i in range(n_segments):
        t1 = i / n_segments
        t2 = (i + 1) / n_segments
        
        seg_log_x1 = start_x + t1 * total_span
        seg_log_x2 = start_x + t2 * total_span
        
        color = cmap((t1 + t2) / 2)
        
        # Calculate y bounds at this x position (considering stadium shape)
        def get_y_bounds(log_x):
            if log_x < min_log_x:
                # In left cap
                dx = log_x - min_log_x
                if abs(dx) <= cap_radius_log:
                    # Circle equation: (x-cx)^2/rx^2 + (y-cy)^2/ry^2 = 1
                    ratio = 1 - (dx / cap_radius_log) ** 2
                    if ratio > 0:
                        dy = radius * np.sqrt(ratio)
                        return center_y - dy, center_y + dy
                return center_y, center_y
            elif log_x > max_log_x:
                # In right cap
                dx = log_x - max_log_x
                if abs(dx) <= cap_radius_log:
                    ratio = 1 - (dx / cap_radius_log) ** 2
                    if ratio > 0:
                        dy = radius * np.sqrt(ratio)
                        return center_y - dy, center_y + dy
                return center_y, center_y
            else:
                # In rectangle part
                return min_y, max_y
        
        y1_low, y1_high = get_y_bounds(seg_log_x1)
        y2_low, y2_high = get_y_bounds(seg_log_x2)
        
        # Draw trapezoid slice
        ax.fill([10**seg_log_x1, 10**seg_log_x1, 10**seg_log_x2, 10**seg_log_x2],
                [y1_low, y1_high, y2_high, y2_low],
                color=color, alpha=alpha, zorder=zorder, edgecolor='none')
    
    # Clear outline
    outline_log_x = np.concatenate([left_log_x, right_log_x])
    outline_y = np.concatenate([left_y, right_y])
    ax.plot(10**outline_log_x, outline_y, color='#555555', linewidth=1.5, alpha=0.7, zorder=zorder+1)
    # Close the path
    ax.plot([10**outline_log_x[-1], 10**outline_log_x[0]], 
            [outline_y[-1], outline_y[0]], color='#555555', linewidth=1.5, alpha=0.7, zorder=zorder+1)


def draw_diagonal_ellipse(ax, points, pad_x=0.35, pad_y=0.04, color_start='#ffcccc', color_end='#cc0000', 
                          alpha=0.35, zorder=0):
    """
    Draw diagonal gradient using overlapping ellipses for a "painted/smudged" effect.
    
    points: list of (x, y) tuples in data coordinates
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # Sort by y (lower to upper)
    sort_idx = np.argsort(ys)
    xs_sorted = [xs[i] for i in sort_idx]
    ys_sorted = [ys[i] for i in sort_idx]
    
    # Get bounds
    log_xs = [np.log10(x) for x in xs_sorted]
    min_log_x, max_log_x = min(log_xs), max(log_xs)
    min_y, max_y = min(ys_sorted), max(ys_sorted)
    
    # Extend bounds with padding
    ext_min_log_x = min_log_x - pad_x
    ext_max_log_x = max_log_x + pad_x
    ext_min_y = min_y - pad_y
    ext_max_y = max_y + pad_y
    
    # Dimensions
    dx_log = ext_max_log_x - ext_min_log_x
    dy = ext_max_y - ext_min_y
    
    # Create gradient fill using multiple overlapping ellipses along the diagonal
    n_layers = 50
    cmap = LinearSegmentedColormap.from_list('custom', [color_start, color_end])
    
    # Each ellipse width/height
    ellipse_width_log = dx_log * 0.5  # width of each small ellipse
    ellipse_height = dy * 0.4  # height of each small ellipse
    
    for i in range(n_layers):
        t = i / (n_layers - 1) if n_layers > 1 else 0.5
        
        # Position along the diagonal (from lower-left to upper-right)
        layer_log_x = ext_min_log_x + t * dx_log
        layer_y = ext_min_y + t * dy
        
        color = cmap(t)
        
        # Create ellipse
        n_pts = 40
        theta = np.linspace(0, 2*np.pi, n_pts)
        
        ellipse_log_x = layer_log_x + (ellipse_width_log/2) * np.cos(theta)
        ellipse_y = layer_y + (ellipse_height/2) * np.sin(theta)
        
        # Convert x from log to linear
        ellipse_x = 10**ellipse_log_x
        
        ax.fill(ellipse_x, ellipse_y, color=color, alpha=alpha*0.7, 
                zorder=zorder, edgecolor='none')
    
    # Draw overall ellipse outline
    center_log_x = (ext_min_log_x + ext_max_log_x) / 2
    center_y = (ext_min_y + ext_max_y) / 2
    a_log_x = dx_log / 2
    b_y = dy / 2
    
    n_outline = 100
    theta = np.linspace(0, 2*np.pi, n_outline)
    outline_log_x = center_log_x + a_log_x * np.cos(theta)
    outline_y = center_y + b_y * np.sin(theta)
    outline_x = 10**outline_log_x
    
    ax.plot(outline_x, outline_y, color='#555555', linewidth=1.2, alpha=0.6, zorder=zorder+1)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_overview_gray(ax, df, title_suffix):
    """Plot gray overview scatter (left panels)."""
    
    # Markers: small=cross (hollow), large=circle
    small_df = df[df['BS_Type'] == 'small']
    large_df = df[df['BS_Type'] == 'large']
    
    # Plot small batch as hollow crosses
    ax.scatter(small_df['LR_x_WD'], small_df['Accuracy'], 
               marker='P', s=60, facecolors='none', edgecolors='#666666', 
               linewidths=1.5, zorder=5, label='small')
    
    # Plot large batch as circles
    ax.scatter(large_df['LR_x_WD'], large_df['Accuracy'], 
               marker='o', s=50, facecolors='#888888', edgecolors='#444444',
               linewidths=0.8, zorder=5, label='large')
    
    ax.set_xscale('log')
    
    # Minimal axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Simplified ticks
    ax.tick_params(axis='both', which='major', labelsize=8, length=3)
    ax.tick_params(axis='both', which='minor', length=2)
    
    # Set y limits to nice integers
    y_min = np.floor(df['Accuracy'].min() * 20) / 20
    y_max = np.ceil(df['Accuracy'].max() * 20) / 20
    ax.set_ylim(y_min, y_max)
    
    # Minimal labels
    ax.set_ylabel('Acc', fontsize=9)
    ax.set_xlabel('')
    
    # Remove grid
    ax.grid(False)


def plot_0shot_detail(ax, df, fig=None):
    """Plot 0-shot detailed analysis (right top panel)."""
    
    # Select the 6 specific points
    # Small batch: 2e-4×5e-3, 2e-4×1e-2, 5e-4×1e-2, 1e-3×1e-2
    # Large batch: 5e-3×1e-2, 1e-3×1e-2
    
    # Small batch points
    small_configs = [
        (2e-4, 5e-3),  # LR=2e-4, WD=5e-3
        (2e-4, 1e-2),  # LR=2e-4, WD=1e-2
        (5e-4, 1e-2),  # LR=5e-4, WD=1e-2
        (1e-3, 1e-2),  # LR=1e-3, WD=1e-2
    ]
    
    large_configs = [
        (5e-3, 1e-2),  # LR=5e-3, WD=1e-2
        (1e-3, 1e-2),  # LR=1e-3, WD=1e-2
    ]
    
    small_points = []
    large_points = []
    
    for lr, wd in small_configs:
        row = df[(df['BS_Type'] == 'small') & 
                 (np.isclose(df['LR'], lr, rtol=0.01)) & 
                 (np.isclose(df['WD'], wd, rtol=0.01))]
        if len(row) > 0:
            x = row['LR_x_WD'].values[0]
            y = row['Accuracy'].values[0]
            small_points.append((x, y, lr, wd))
    
    for lr, wd in large_configs:
        row = df[(df['BS_Type'] == 'large') & 
                 (np.isclose(df['LR'], lr, rtol=0.01)) & 
                 (np.isclose(df['WD'], wd, rtol=0.01))]
        if len(row) > 0:
            x = row['LR_x_WD'].values[0]
            y = row['Accuracy'].values[0]
            large_points.append((x, y, lr, wd))
    
    ax.set_xscale('log')
    
    # Group points by accuracy level
    # Low acc group (~0.31-0.33): small(2e-4×5e-3), small(2e-4×1e-2), large(1e-3×1e-2)
    # High acc group (~0.51-0.55): small(5e-4×1e-2), small(1e-3×1e-2), large(5e-3×1e-2)
    
    low_acc_small = [p for p in small_points if p[1] < 0.4]  # acc < 0.4
    high_acc_small = [p for p in small_points if p[1] >= 0.4]  # acc >= 0.4
    low_acc_large = [p for p in large_points if p[1] < 0.4]
    high_acc_large = [p for p in large_points if p[1] >= 0.4]
    
    # Draw stadium shapes (background first, lower zorder)
    # 1. Horizontal stadium for low acc group (red-white-blue)
    low_group_points = [(p[0], p[1]) for p in low_acc_small + low_acc_large]
    if len(low_group_points) >= 2:
        draw_stadium_simple(ax, low_group_points, padding_x=0.3, padding_y=0.025,
                           color_left='#ffdddd', color_right='#ddeeff', alpha=0.45, zorder=1)
    
    # 2. Horizontal stadium for high acc group (red-white-blue)
    high_group_points = [(p[0], p[1]) for p in high_acc_small + high_acc_large]
    if len(high_group_points) >= 2:
        draw_stadium_simple(ax, high_group_points, padding_x=0.3, padding_y=0.025,
                           color_left='#ffdddd', color_right='#ddeeff', alpha=0.45, zorder=1)
    
    # 3. Diagonal ellipse for small batch points (shallow red to deep red)
    all_small = [(p[0], p[1]) for p in small_points]
    if len(all_small) >= 2:
        draw_diagonal_ellipse(ax, all_small, pad_x=0.5, pad_y=0.06, 
                             color_start='#ffe8e8', color_end='#dd2222', alpha=0.4, zorder=2)
    
    # 4. Diagonal ellipse for large batch points (shallow blue to deep blue)
    all_large = [(p[0], p[1]) for p in large_points]
    if len(all_large) >= 2:
        draw_diagonal_ellipse(ax, all_large, pad_x=0.5, pad_y=0.06,
                             color_start='#e8f0ff', color_end='#2255dd', alpha=0.4, zorder=2)
    
    # Plot points with heatmap coloring
    # Lighter edge colors for better heatmap visibility
    
    # Small batch: red series (deeper = higher acc) - use cross marker
    small_accs = [p[1] for p in small_points]
    all_accs = small_accs + [p[1] for p in large_points]
    global_min_acc = min(all_accs) if all_accs else 0.3
    global_max_acc = max(all_accs) if all_accs else 0.55
    
    if small_accs:
        for x, y, lr, wd in small_points:
            # Normalize acc to 0-1 for color intensity (using global range)
            if global_max_acc > global_min_acc:
                intensity = (y - global_min_acc) / (global_max_acc - global_min_acc)
            else:
                intensity = 0.5
            # Red series: lighter to darker based on accuracy
            color = plt.cm.Reds(0.35 + 0.55 * intensity)
            ax.scatter(x, y, marker='P', s=220, facecolors=color, 
                      edgecolors='#cc6666', linewidths=1.2, zorder=10)
    
    # Large batch: blue series (deeper = higher acc) - use filled circle marker
    large_accs = [p[1] for p in large_points]
    if large_accs:
        for x, y, lr, wd in large_points:
            if global_max_acc > global_min_acc:
                intensity = (y - global_min_acc) / (global_max_acc - global_min_acc)
            else:
                intensity = 0.5
            color = plt.cm.Blues(0.35 + 0.55 * intensity)
            ax.scatter(x, y, marker='o', s=200, facecolors=color,
                      edgecolors='#6666cc', linewidths=1.2, zorder=10)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.set_xlabel('LR × WD', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('0-shot: Batch Size Effect on Optimal LR×WD', fontsize=12, fontweight='bold', pad=10)
    
    # Set axis limits for better view
    ax.set_xlim(3e-7, 2e-4)
    ax.set_ylim(0.27, 0.60)
    
    # Legend with heatmap indication
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='P', color='w', markerfacecolor='#dd6666', 
               markeredgecolor='#cc6666', markersize=11, markeredgewidth=1, label='Small batch'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6688dd',
               markeredgecolor='#6666cc', markersize=11, markeredgewidth=1, label='Large batch'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    # Add two colorbars for accuracy heatmap (red for small, blue for large)
    if fig is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Red colorbar for small batch (positioned at right side)
        cax_red = inset_axes(ax, width="2%", height="25%", loc='center right',
                             bbox_to_anchor=(0.12, 0, 1, 1), bbox_transform=ax.transAxes)
        sm_red = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                        norm=plt.Normalize(vmin=global_min_acc, vmax=global_max_acc))
        sm_red.set_array([])
        cbar_red = fig.colorbar(sm_red, cax=cax_red, orientation='vertical')
        cbar_red.ax.tick_params(labelsize=6)
        cbar_red.ax.set_title('Small', fontsize=7, pad=2)
        
        # Blue colorbar for large batch (next to red)
        cax_blue = inset_axes(ax, width="2%", height="25%", loc='center right',
                              bbox_to_anchor=(0.06, 0, 1, 1), bbox_transform=ax.transAxes)
        sm_blue = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                         norm=plt.Normalize(vmin=global_min_acc, vmax=global_max_acc))
        sm_blue.set_array([])
        cbar_blue = fig.colorbar(sm_blue, cax=cax_blue, orientation='vertical')
        cbar_blue.ax.tick_params(labelsize=6)
        cbar_blue.ax.set_title('Large', fontsize=7, pad=2)
    
    ax.grid(True, alpha=0.25, linestyle='--', color='#dddddd')


def plot_8shot_detail(ax, df):
    """Plot 8-shot detailed analysis (right bottom panel)."""
    
    # Select points where acc is between 0.10 and 0.14, LR×WD from 1e-6 to 1e-4
    mask = ((df['Accuracy'] >= 0.10) & (df['Accuracy'] <= 0.14) & 
            (df['LR_x_WD'] >= 1e-6) & (df['LR_x_WD'] <= 1e-4))
    
    selected_df = df[mask].copy()
    
    ax.set_xscale('log')
    
    small_df = selected_df[selected_df['BS_Type'] == 'small']
    large_df = selected_df[selected_df['BS_Type'] == 'large']
    
    # Collect all points for the big stadium
    all_points = [(row['LR_x_WD'], row['Accuracy']) 
                  for _, row in selected_df.iterrows()]
    
    # Draw one big stadium shape (red-white-blue gradient)
    # Very flat - minimal padding_y for compressed appearance
    if len(all_points) >= 2:
        draw_stadium_simple(ax, all_points, padding_x=0.5, padding_y=0.003,
                           color_left='#ffe0e0', color_right='#e0e8ff', alpha=0.5, zorder=1)
    
    # Plot points - small batch: red cross, large batch: blue circle
    ax.scatter(small_df['LR_x_WD'], small_df['Accuracy'],
               marker='P', s=180, facecolors='#e74c3c', edgecolors='#cc6666',
               linewidths=1.2, zorder=10, label='Small batch')
    
    ax.scatter(large_df['LR_x_WD'], large_df['Accuracy'],
               marker='o', s=160, facecolors='#3498db', edgecolors='#6699cc',
               linewidths=1.2, zorder=10, label='Large batch')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.set_xlabel('LR × WD', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('8-shot: Same Accuracy Requires Higher LR×WD for Large Batch', 
                 fontsize=12, fontweight='bold', pad=10)
    
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--', color='#dddddd')
    
    # Set axis limits - expand y-range so stadium doesn't fill everything
    if len(selected_df) > 0:
        x_min = selected_df['LR_x_WD'].min() / 3
        x_max = selected_df['LR_x_WD'].max() * 3
        # Expand y limits significantly beyond data range
        y_min = 0.08
        y_max = 0.16
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


# =============================================================================
# Main
# =============================================================================

def main():
    # Load data
    df_0shot, df_8shot = load_data()
    
    # Create figure with custom layout
    # Left column very narrow, right column wide
    # Left panels are flattened (short height)
    fig = plt.figure(figsize=(14, 7))
    
    # Grid with compressed left panels
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 1],
                          left=0.04, right=0.95, top=0.93, bottom=0.09,
                          wspace=0.12, hspace=0.30)
    
    # Create axes
    ax_left_top = fig.add_subplot(gs[0, 0])     # 0-shot overview (gray)
    ax_left_bottom = fig.add_subplot(gs[1, 0])  # 8-shot overview (gray)
    ax_right_top = fig.add_subplot(gs[0, 1])    # 0-shot detail
    ax_right_bottom = fig.add_subplot(gs[1, 1]) # 8-shot detail
    
    # Plot left panels (gray overviews) - compressed appearance
    plot_overview_gray(ax_left_top, df_0shot, '0-shot')
    plot_overview_gray(ax_left_bottom, df_8shot, '8-shot')
    
    # Heavily compress left panels' y-range for "flattened" look
    ax_left_top.set_ylim(0.15, 0.60)
    ax_left_bottom.set_ylim(0.02, 0.20)
    
    # Limit x-axis range
    ax_left_top.set_xlim(1e-7, 2e-4)
    ax_left_bottom.set_xlim(1e-7, 2e-4)
    
    # Add small labels
    ax_left_top.set_title('0-shot', fontsize=9, pad=3, color='#666666')
    ax_left_bottom.set_title('8-shot', fontsize=9, pad=3, color='#666666')
    ax_left_bottom.set_xlabel('LR×WD', fontsize=8, color='#777777')
    ax_left_top.set_xlabel('')
    
    # Make left panels look more muted
    for ax in [ax_left_top, ax_left_bottom]:
        ax.spines['left'].set_color('#bbbbbb')
        ax.spines['bottom'].set_color('#bbbbbb')
        ax.tick_params(colors='#888888', labelsize=7)
        ax.set_ylabel('Acc', fontsize=8, color='#777777')
        # Reduce number of y-ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    # Plot right panels (detailed analysis)
    plot_0shot_detail(ax_right_top, df_0shot, fig)
    plot_8shot_detail(ax_right_bottom, df_8shot)
    
    # Draw connecting arrows from left to right panels
    ax_left_top.annotate('', xy=(1.08, 0.5), xytext=(1.02, 0.5),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2))
    ax_left_bottom.annotate('', xy=(1.08, 0.5), xytext=(1.02, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2))
    
    # Save figure
    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "four_panel_analysis.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
