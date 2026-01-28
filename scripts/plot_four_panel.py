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
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator, NullFormatter
import os

# Typography constants
AXIS_LABEL_FS = 10
AXIS_TICK_FS = 9
TITLE_FS_RIGHT = 11


def _fmt_decimal_0x(y, _pos=None):
    """Format numbers in [0,1) as 0.x (no scientific notation)."""
    if y is None or not np.isfinite(y):
        return ""
    if y == 0:
        return "0"
    if 0 < y < 1:
        s = f"{y:.2f}"
        s = s.rstrip('0').rstrip('.')
        if s.startswith('.'):
            s = '0' + s
        return s
    # fallback
    return f"{y:.3g}"

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


def draw_stadium_simple(
    ax,
    points,
    padding_x=0.15,
    padding_y=0.02,
    color_left='#ffcccc',
    color_right='#cce5ff',
    alpha=0.4,
    zorder=0,
    *,
    y_log=False,
    x_min_override=None,
    x_max_override=None,
    cap_radius_log=None,
):
    """
    Draw a simple horizontal stadium shape around a group of points.
    Gradient from left (color_left) to right (color_right).
    
    points: list of (x, y) tuples in data coordinates
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    log_xs = [np.log10(x) for x in xs]

    if x_min_override is not None:
        min_log_x = np.log10(float(x_min_override))
    else:
        min_log_x = min(log_xs) - padding_x
    if x_max_override is not None:
        max_log_x = np.log10(float(x_max_override))
    else:
        max_log_x = max(log_xs) + padding_x

    # y-range padding is specified in *data* units (e.g. 0.003 accuracy)
    min_y_data = min(ys) - padding_y
    max_y_data = max(ys) + padding_y

    if y_log:
        # Build the stadium in log10(y) space then convert back.
        # This makes the shape visually stable under a log y-axis.
        min_y = np.log10(min_y_data)
        max_y = np.log10(max_y_data)
        center_y = (min_y + max_y) / 2
        height = max_y - min_y
        radius_y = height / 2
        radius_for_cap = radius_y
    else:
        min_y = min_y_data
        max_y = max_y_data
        center_y = (min_y + max_y) / 2
        height = max_y - min_y
        radius_y = height / 2
        radius_for_cap = radius_y

    # Semicircle radius in log-x space (bigger -> rounder ends)
    if cap_radius_log is None:
        cap_radius_log = max(0.12, padding_x * 0.35 + float(radius_for_cap) * 2.8)
    
    # Create stadium path
    n_arc = 50
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, n_arc)
    theta_right = np.linspace(-np.pi/2, np.pi/2, n_arc)
    
    # Left semicircle center at (min_log_x, center_y)
    left_log_x = min_log_x + cap_radius_log * np.cos(theta_left)
    left_y = center_y + radius_y * np.sin(theta_left)
    
    # Right semicircle center at (max_log_x, center_y)  
    right_log_x = max_log_x + cap_radius_log * np.cos(theta_right)
    right_y = center_y + radius_y * np.sin(theta_right)
    
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
                        dy = radius_y * np.sqrt(ratio)
                        return center_y - dy, center_y + dy
                return center_y, center_y
            elif log_x > max_log_x:
                # In right cap
                dx = log_x - max_log_x
                if abs(dx) <= cap_radius_log:
                    ratio = 1 - (dx / cap_radius_log) ** 2
                    if ratio > 0:
                        dy = radius_y * np.sqrt(ratio)
                        return center_y - dy, center_y + dy
                return center_y, center_y
            else:
                # In rectangle part
                return min_y, max_y
        
        y1_low, y1_high = get_y_bounds(seg_log_x1)
        y2_low, y2_high = get_y_bounds(seg_log_x2)
        
        # Draw trapezoid slice (convert y back if using log space)
        if y_log:
            y1_low_d, y1_high_d = 10**y1_low, 10**y1_high
            y2_low_d, y2_high_d = 10**y2_low, 10**y2_high
        else:
            y1_low_d, y1_high_d = y1_low, y1_high
            y2_low_d, y2_high_d = y2_low, y2_high

        ax.fill([10**seg_log_x1, 10**seg_log_x1, 10**seg_log_x2, 10**seg_log_x2],
                [y1_low_d, y1_high_d, y2_high_d, y2_low_d],
                color=color, alpha=alpha, zorder=zorder, edgecolor='none')
    
    # Clear outline
    outline_log_x = np.concatenate([left_log_x, right_log_x])
    outline_y = np.concatenate([left_y, right_y])
    if y_log:
        outline_y_d = 10**outline_y
    else:
        outline_y_d = outline_y
    ax.plot(10**outline_log_x, outline_y_d, color='#555555', linewidth=1.5, alpha=0.7, zorder=zorder+1)
    # Close the path
    ax.plot([10**outline_log_x[-1], 10**outline_log_x[0]], 
            [outline_y_d[-1], outline_y_d[0]], color='#555555', linewidth=1.5, alpha=0.7, zorder=zorder+1)


def draw_diagonal_stroke(ax, low_points, high_points, ellipse_width_log=0.25, ellipse_height_y=0.03,
                         color_start='#ffcccc', color_end='#cc0000', alpha=0.35, zorder=0):
    """
    Draw a diagonal stroke - an ellipse moving along a straight line from low to high points.
    Uniform ellipse size throughout (not varying).
    
    low_points: list of (x, y) tuples for the lower group
    high_points: list of (x, y) tuples for the upper group
    """
    # Calculate centers of low and high groups
    low_xs = [p[0] for p in low_points]
    low_ys = [p[1] for p in low_points]
    high_xs = [p[0] for p in high_points]
    high_ys = [p[1] for p in high_points]
    
    # Centers in log space
    low_center_log_x = np.mean([np.log10(x) for x in low_xs])
    low_center_y = np.mean(low_ys)
    high_center_log_x = np.mean([np.log10(x) for x in high_xs])
    high_center_y = np.mean(high_ys)
    
    # Start and end positions
    start_log_x, start_y = low_center_log_x, low_center_y
    end_log_x, end_y = high_center_log_x, high_center_y
    
    # Distance
    dx_log = end_log_x - start_log_x
    dy = end_y - start_y
    
    # Create gradient stroke using overlapping ellipses of UNIFORM size
    n_layers = 50
    cmap = LinearSegmentedColormap.from_list('custom', [color_start, color_end])
    
    for i in range(n_layers):
        t = i / (n_layers - 1) if n_layers > 1 else 0.5
        
        # Position along the stroke
        layer_log_x = start_log_x + t * dx_log
        layer_y = start_y + t * dy
        
        color = cmap(t)
        
        # Create ellipse - uniform size
        n_pts = 50
        theta = np.linspace(0, 2*np.pi, n_pts)
        
        ellipse_log_x = layer_log_x + (ellipse_width_log/2) * np.cos(theta)
        ellipse_y = layer_y + (ellipse_height_y/2) * np.sin(theta)
        
        # Convert x from log to linear
        ellipse_x = 10**ellipse_log_x
        
        ax.fill(ellipse_x, ellipse_y, color=color, alpha=alpha*0.6, 
                zorder=zorder, edgecolor='none')


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_overview_gray(ax, df, title_suffix, selection_box=None, selected_mask=None):
    """Plot overview scatter (left panels): mostly gray, with colored overlay for selected points."""

    if selected_mask is None:
        selected_mask = np.zeros(len(df), dtype=bool)

    # Base layer: unified gray points
    small_df = df[df['BS_Type'] == 'small']
    large_df = df[df['BS_Type'] == 'large']

    ax.scatter(small_df['LR_x_WD'], small_df['Accuracy'],
               marker='P', s=45, facecolors='#bbbbbb', edgecolors='#777777',
               linewidths=1.0, zorder=3)
    ax.scatter(large_df['LR_x_WD'], large_df['Accuracy'],
               marker='o', s=38, facecolors='#9a9a9a', edgecolors='#666666',
               linewidths=0.8, zorder=3)

    # Overlay: only selected points in red/blue (same markers)
    sel_df = df[selected_mask]
    sel_small = sel_df[sel_df['BS_Type'] == 'small']
    sel_large = sel_df[sel_df['BS_Type'] == 'large']

    if len(sel_small) > 0:
        ax.scatter(sel_small['LR_x_WD'], sel_small['Accuracy'],
                   marker='P', s=60, facecolors='#e74c3c', edgecolors='#cc6666',
                   linewidths=1.2, zorder=6)
    if len(sel_large) > 0:
        ax.scatter(sel_large['LR_x_WD'], sel_large['Accuracy'],
                   marker='o', s=55, facecolors='#3498db', edgecolors='#6699cc',
                   linewidths=1.2, zorder=6)
    
    ax.set_xscale('log')
    
    # Draw selection box if provided (at bottom layer)
    if selection_box is not None:
        x_min, x_max, y_min_box, y_max_box = selection_box
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_min, y_min_box), x_max - x_min, y_max_box - y_min_box,
                         fill=False, edgecolor='#777777', linewidth=2.5, 
                         linestyle='-', zorder=1)  # Bottom layer
        ax.add_patch(rect)
    
    # Minimal axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Simplified ticks
    ax.tick_params(axis='both', which='major', labelsize=7, length=3)
    ax.tick_params(axis='both', which='minor', length=2)
    
    # Minimal labels
    ax.set_ylabel('Acc', fontsize=8)
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
    # 1. Horizontal stadium for low acc group (red-white-blue) - more precise, tighter
    low_group_points = [(p[0], p[1]) for p in low_acc_small + low_acc_large]
    if len(low_group_points) >= 2:
        draw_stadium_simple(ax, low_group_points, padding_x=0.18, padding_y=0.018,
                           color_left='#ff8888', color_right='#88aaff', alpha=0.80, zorder=1)
    
    # 2. Horizontal stadium for high acc group (red-white-blue) - more precise, tighter
    high_group_points = [(p[0], p[1]) for p in high_acc_small + high_acc_large]
    if len(high_group_points) >= 2:
        draw_stadium_simple(ax, high_group_points, padding_x=0.18, padding_y=0.018,
                           color_left='#ff8888', color_right='#88aaff', alpha=0.80, zorder=1)
    
    # 3. Diagonal stroke for small batch points
    # Colors should match the heatmap - lighter for lower acc, darker for higher acc
    low_small = [(p[0], p[1]) for p in low_acc_small]
    high_small = [(p[0], p[1]) for p in high_acc_small]
    if len(low_small) >= 1 and len(high_small) >= 1:
        # Use same colormap as points, but start from very light (near white)
        low_color = plt.cm.Reds(0.15)  # Much lighter, closer to white
        high_color = plt.cm.Reds(0.35 + 0.55 * 1)  # high acc color
        draw_diagonal_stroke(ax, low_small, high_small,
                            ellipse_width_log=0.55, ellipse_height_y=0.055,
                            color_start=low_color, color_end=high_color, alpha=0.28, zorder=2)
    
    # 4. Diagonal stroke for large batch points
    low_large = [(p[0], p[1]) for p in low_acc_large]
    high_large = [(p[0], p[1]) for p in high_acc_large]
    if len(low_large) >= 1 and len(high_large) >= 1:
        low_color = plt.cm.Blues(0.15)  # Much lighter, closer to white
        high_color = plt.cm.Blues(0.35 + 0.55 * 1)
        draw_diagonal_stroke(ax, low_large, high_large,
                            ellipse_width_log=0.55, ellipse_height_y=0.055,
                            color_start=low_color, color_end=high_color, alpha=0.28, zorder=2)
    
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
            ax.scatter(x, y, marker='P', s=160, facecolors=color, 
                      edgecolors='#cc6666', linewidths=1.0, zorder=10)
            # Add label with lr×wd rotated (bottom-left to top-right direction)
            label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
            ax.text(x, y, label_text, fontsize=5.5, rotation=45, 
                   ha='left', va='bottom', color='#333333', zorder=11)
    
    # Large batch: blue series (deeper = higher acc) - use filled circle marker
    large_accs = [p[1] for p in large_points]
    if large_accs:
        for x, y, lr, wd in large_points:
            if global_max_acc > global_min_acc:
                intensity = (y - global_min_acc) / (global_max_acc - global_min_acc)
            else:
                intensity = 0.5
            color = plt.cm.Blues(0.35 + 0.55 * intensity)
            ax.scatter(x, y, marker='o', s=150, facecolors=color,
                      edgecolors='#6666cc', linewidths=1.0, zorder=10)
            # Add label with lr×wd rotated (bottom-left to top-right direction)
            label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
            ax.text(x, y, label_text, fontsize=5.5, rotation=45, 
                   ha='left', va='bottom', color='#333333', zorder=11)
    
    # Styling - full box on all four sides
    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FS)
    
    # Right column axes: remove text labels per request
    ax.set_xlabel('', fontsize=AXIS_LABEL_FS)
    ax.set_ylabel('', fontsize=AXIS_LABEL_FS, labelpad=0)
    
    # Set axis limits for better view
    ax.set_xlim(3e-7, 2e-4)
    ax.set_ylim(0.27, 0.60)
    
    # Add two colorbars for accuracy heatmap INSIDE the plot - moved to upper-left
    # Red on left, Blue on right, swapped with marker legend
    if fig is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Dual colorbars in upper-left, moved to the right
        cax_red = inset_axes(ax, width="2.3%", height="34%", loc='upper left',
                             bbox_to_anchor=(0.05, -0.05, 1, 1), bbox_transform=ax.transAxes,
                             borderpad=0)
        sm_red = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                        norm=plt.Normalize(vmin=global_min_acc, vmax=global_max_acc))
        sm_red.set_array([])
        cbar_red = fig.colorbar(sm_red, cax=cax_red, orientation='vertical')
        cbar_red.ax.tick_params(labelsize=6, length=2, labelleft=False, labelright=False)
        cbar_red.ax.set_yticklabels([])  # Remove all numeric labels
        # Remove offset text if present
        try:
            cbar_red.ax.yaxis.offsetText.set_visible(False)
        except:
            pass
        
        cax_blue = inset_axes(ax, width="2.3%", height="34%", loc='upper left',
                              bbox_to_anchor=(0.11, -0.05, 1, 1), bbox_transform=ax.transAxes,
                              borderpad=0)
        sm_blue = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                         norm=plt.Normalize(vmin=global_min_acc, vmax=global_max_acc))
        sm_blue.set_array([])
        cbar_blue = fig.colorbar(sm_blue, cax=cax_blue, orientation='vertical')
        cbar_blue.ax.tick_params(labelsize=6, length=2)

        # Centered label between the two bars, moved up
        ax.text(0.15, 0.6, 'Accuracy', transform=ax.transAxes,
                ha='center', va='top', fontsize=8, color='#333333')
    
    # Marker legend moved to right middle, smaller size
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='P', color='w', markerfacecolor='#dd6666', 
               markeredgecolor='#cc6666', markersize=6, markeredgewidth=0.6, label='Small batch'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6688dd',
               markeredgecolor='#6666cc', markersize=6, markeredgewidth=0.6, label='Large batch'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=6, framealpha=0.9)
    
    ax.grid(True, alpha=0.25, linestyle='--', color='#dddddd')


def plot_8shot_detail(ax, df):
    """Plot 8-shot detailed analysis (right bottom panel)."""
    
    # Select points where acc is between 0.10 and 0.14, LR×WD from 1e-6 to 1e-4
    mask = ((df['Accuracy'] >= 0.10) & (df['Accuracy'] <= 0.14) & 
            (df['LR_x_WD'] >= 1e-6) & (df['LR_x_WD'] <= 1e-4))
    
    selected_df = df[mask].copy()
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    small_df = selected_df[selected_df['BS_Type'] == 'small']
    large_df = selected_df[selected_df['BS_Type'] == 'large']
    
    # Collect all points for the big stadium
    all_points = [(row['LR_x_WD'], row['Accuracy']) 
                  for _, row in selected_df.iterrows()]
    
    # Draw one big stadium shape (red-white-blue gradient)
    # Reduced padding_y to make stadium visually flatter (half height)
    if len(all_points) >= 2:
        draw_stadium_simple(
            ax,
            all_points,
            padding_x=0.12,
            padding_y=0.0040,  # Increased for internal top/bottom whitespace
            color_left='#ff8888',
            color_right='#88aaff',
            alpha=0.80,
            zorder=1,
            y_log=True,
            x_min_override=1e-6,
            x_max_override=1e-4,
        )
    
    # Plot points - small batch: red cross, large batch: blue circle
    for idx, row in small_df.iterrows():
        x, y = row['LR_x_WD'], row['Accuracy']
        lr, wd = row['LR'], row['WD']
        ax.scatter(x, y, marker='P', s=130, facecolors='#e74c3c', edgecolors='#cc6666',
                  linewidths=1.0, zorder=10, label='Small batch' if idx == small_df.index[0] else '')
        # Add label with lr×wd rotated (bottom-left to top-right direction)
        label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
        ax.text(x, y, label_text, fontsize=5.5, rotation=45, 
               ha='left', va='bottom', color='#333333', zorder=11)
    
    for idx, row in large_df.iterrows():
        x, y = row['LR_x_WD'], row['Accuracy']
        lr, wd = row['LR'], row['WD']
        ax.scatter(x, y, marker='o', s=120, facecolors='#3498db', edgecolors='#6699cc',
                  linewidths=1.0, zorder=10, label='Large batch' if idx == large_df.index[0] else '')
        # Add label with lr×wd rotated (bottom-left to top-right direction)
        label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
        ax.text(x, y, label_text, fontsize=5.5, rotation=45, 
               ha='left', va='bottom', color='#333333', zorder=11)
    
    # Styling - full box on all four sides
    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FS)
    
    # Right column axes: remove text labels per request
    ax.set_xlabel('', fontsize=AXIS_LABEL_FS)
    ax.set_ylabel('', fontsize=AXIS_LABEL_FS, labelpad=0)
    
    ax.legend(loc='upper right', fontsize=6, framealpha=0.9, markerscale=0.6)
    ax.grid(True, alpha=0.25, linestyle='--', color='#dddddd')
    
    # Set axis limits - wider y-range (0.05 to 0.20) to center stadium and make it appear half height
    ax.set_xlim(5e-7, 2e-4)
    ax.set_ylim(0.08, 0.17)

    # Log y, but display tick labels as 0.x (no scientific notation)
    # Ticks adjusted for wider range (0.05 to 0.20)
    ax.yaxis.set_major_locator(FixedLocator([0.10, 0.15]))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_decimal_0x))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


# =============================================================================
# Main
# =============================================================================

def main():
    # Load data
    df_0shot, df_8shot = load_data()
    
    # Create figure with custom layout
    # Left column is a small "source" thumbnail column; right column holds main analysis
    fig = plt.figure(figsize=(9.2, 5.0))
    
    # Grid - right panels narrower than before, less vertical whitespace
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], height_ratios=[1, 1],
                          left=0.07, right=0.92, top=0.94, bottom=0.10,
                          wspace=0.35, hspace=0.00)
    
    # Create axes
    ax_left_top = fig.add_subplot(gs[0, 0])     # 0-shot overview (gray)
    ax_left_bottom = fig.add_subplot(gs[1, 0])  # 8-shot overview (gray)
    ax_right_top = fig.add_subplot(gs[0, 1])    # 0-shot detail
    ax_right_bottom = fig.add_subplot(gs[1, 1]) # 8-shot detail
    
    # Build masks for "selected" points that appear on the right panels
    # 0-shot selected: 6 configs (4 small + 2 large)
    sel0 = np.zeros(len(df_0shot), dtype=bool)
    for lr, wd in [(2e-4, 5e-3), (2e-4, 1e-2), (5e-4, 1e-2), (1e-3, 1e-2)]:
        sel0 |= ((df_0shot['BS_Type'] == 'small') &
                 np.isclose(df_0shot['LR'].astype(float), lr, rtol=0.01) &
                 np.isclose(df_0shot['WD'].astype(float), wd, rtol=0.01))
    for lr, wd in [(5e-3, 1e-2), (1e-3, 1e-2)]:
        sel0 |= ((df_0shot['BS_Type'] == 'large') &
                 np.isclose(df_0shot['LR'].astype(float), lr, rtol=0.01) &
                 np.isclose(df_0shot['WD'].astype(float), wd, rtol=0.01))

    # 8-shot selected: the same region used in the right-bottom panel
    sel8 = ((df_8shot['Accuracy'] >= 0.10) & (df_8shot['Accuracy'] <= 0.14) &
            (df_8shot['LR_x_WD'] >= 1e-6) & (df_8shot['LR_x_WD'] <= 1e-4)).values

    # Plot left panels with selection boxes showing the analysis region
    selection_0shot = (5e-7, 8e-5, 0.28, 0.58)
    plot_overview_gray(ax_left_top, df_0shot, '0-shot', selection_box=selection_0shot, selected_mask=sel0)
    
    # Selection box for 8-shot: acc 0.10-0.14, x: 1e-6 to 1e-4
    selection_8shot = (8e-7, 1.2e-4, 0.095, 0.145)
    plot_overview_gray(ax_left_bottom, df_8shot, '8-shot', selection_box=selection_8shot, selected_mask=sel8)
    
    # Left-top: log y-axis
    ax_left_top.set_yscale('log')
    ax_left_top.set_ylim(0.18, 0.60)
    ax_left_top.set_xlim(1e-7, 2e-4)
    ax_left_top.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.4, 0.6]))
    ax_left_top.yaxis.set_major_formatter(FuncFormatter(_fmt_decimal_0x))
    
    # Left-bottom: normal y-axis, compressed
    ax_left_bottom.set_ylim(0.04, 0.19)
    ax_left_bottom.set_xlim(1e-7, 2e-4)
    
    # Add small labels - titles moved up
    ax_left_top.set_title('0-shot', fontsize=9, pad=6, color='black', y=1.08)
    ax_left_bottom.set_title('8-shot', fontsize=9, pad=6, color='black', y=1.08)
    ax_left_top.set_xlabel('LR×WD', fontsize=8, color='#666666')
    ax_left_bottom.set_xlabel('LR×WD', fontsize=8, color='#666666')
    
    # Make left panels styling - unified font sizes
    for ax in [ax_left_top, ax_left_bottom]:
        # Full box: show all four spines
        for side in ['left', 'bottom', 'right', 'top']:
            ax.spines[side].set_visible(True)
        ax.spines['left'].set_color('#999999')
        ax.spines['bottom'].set_color('#999999')
        ax.spines['right'].set_color('#999999')
        ax.spines['top'].set_color('#999999')
        ax.tick_params(colors='#666666', labelsize=AXIS_TICK_FS)
        ax.set_ylabel('Accuracy', fontsize=AXIS_LABEL_FS, color='#666666')

    # Shrink panels; align to divider and center left thumbnails per-row
    def _shrink_axis(ax, width_scale, height_scale):
        bbox = ax.get_position()
        return bbox, bbox.width * width_scale, bbox.height * height_scale

    right_top_bbox, rt_w, rt_h = _shrink_axis(ax_right_top, 0.6, 0.75)
    right_bot_bbox, rb_w, rb_h = _shrink_axis(ax_right_bottom, 0.6, 0.75)

    # Divider is left edge of right axes
    divider_x = right_top_bbox.x0

    # Place right axes: left-aligned to divider, vertically centered within their grid cells
    rt_x0 = divider_x
    rt_y0 = right_top_bbox.y0 + (right_top_bbox.height - rt_h) / 2
    ax_right_top.set_position([rt_x0, rt_y0, rt_w, rt_h])

    rb_x0 = divider_x
    rb_y0 = right_bot_bbox.y0 + (right_bot_bbox.height - rb_h) / 2
    ax_right_bottom.set_position([rb_x0, rb_y0, rb_w, rb_h])

    # Left thumbnails: right-aligned to divider, vertically centered to corresponding right plot
    left_top_bbox, lt_w, lt_h = _shrink_axis(ax_left_top, 0.6, 0.5)
    left_bot_bbox, lb_w, lb_h = _shrink_axis(ax_left_bottom, 0.6, 0.5)

    gap = 0.06  # Further increased gap between left and right columns
    lt_x0 = divider_x - gap - lt_w
    lt_center = rt_y0 + rt_h / 2
    lt_y0 = lt_center - lt_h / 2
    ax_left_top.set_position([lt_x0, lt_y0, lt_w, lt_h])

    lb_x0 = divider_x - gap - lb_w
    lb_center = rb_y0 + rb_h / 2
    lb_y0 = lb_center - lb_h / 2
    ax_left_bottom.set_position([lb_x0, lb_y0, lb_w, lb_h])
    
    # Plot right panels (detailed analysis)
    plot_0shot_detail(ax_right_top, df_0shot, fig)
    plot_8shot_detail(ax_right_bottom, df_8shot)
    
    # Draw connecting lines forming full rectangular frames between left and right panels
    # Connect all four corners of each row's left and right subplots
    fig.canvas.draw()  # Ensure positions are updated
    lt_pos = ax_left_top.get_position()
    rt_pos = ax_right_top.get_position()
    lb_pos = ax_left_bottom.get_position()
    rb_pos = ax_right_bottom.get_position()
    
    # Draw full rectangular frames connecting left and right panels
    from matplotlib.lines import Line2D
    # Top row: connect all four sides
    line_top_bottom = Line2D([lt_pos.x1, rt_pos.x0], [lt_pos.y0, rt_pos.y0],
                             transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    line_top_top = Line2D([lt_pos.x1, rt_pos.x0], [lt_pos.y1, rt_pos.y1],
                          transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    line_top_left = Line2D([lt_pos.x1, lt_pos.x1], [lt_pos.y0, lt_pos.y1],
                           transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    line_top_right = Line2D([rt_pos.x0, rt_pos.x0], [rt_pos.y0, rt_pos.y1],
                            transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    
    # Bottom row: connect all four sides
    line_bot_bottom = Line2D([lb_pos.x1, rb_pos.x0], [lb_pos.y0, rb_pos.y0],
                             transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    line_bot_top = Line2D([lb_pos.x1, rb_pos.x0], [lb_pos.y1, rb_pos.y1],
                          transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    line_bot_left = Line2D([lb_pos.x1, lb_pos.x1], [lb_pos.y0, lb_pos.y1],
                           transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    line_bot_right = Line2D([rb_pos.x0, rb_pos.x0], [rb_pos.y0, rb_pos.y1],
                             transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0)
    
    # Add all lines
    for line in [line_top_bottom, line_top_top, line_top_left, line_top_right,
                 line_bot_bottom, line_bot_top, line_bot_left, line_bot_right]:
        fig.add_artist(line)
    
    # Save figure
    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "four_panel_analysis.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
