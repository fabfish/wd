#!/usr/bin/env python3
"""
Modified four-panel visualization: full 4-panel layout with Chinese labels.

Based on four_panel_analysis.png:
- ACC -> 测试精度
- LR×WD -> 学习率与权重衰减的联合取值
- Chinese font support
- Output: four_panel_analysis_modified.png / .pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Droid Sans Fallback', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

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

# Chinese labels (unified font size and darker color)
LABEL_ACC = "测试精度"
LABEL_LR_WD = "学习率与权重衰减的联合取值"
CHINESE_FONTSIZE = 9
CHINESE_COLOR = "#1a1a1a"


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
    return f"{y:.3g}"


def _fmt_log_x(x, _pos=None):
    """Format log-scale x values as plain text (e.g. 1e-6) to avoid mathtext ^ display issues."""
    if x is None or x <= 0 or not np.isfinite(x):
        return ""
    s = f"{x:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def load_data():
    """Load 0-shot and 8-shot CSV data."""
    df_0shot = pd.read_csv("outputs/results/results_0shot.csv")
    df_8shot = pd.read_csv("outputs/results/results_8shot.csv")
    for df in [df_0shot, df_8shot]:
        df['LR'] = df['LR'].astype(float)
        df['WD'] = df['WD'].astype(float)
        df['LR_x_WD'] = df['LR'] * df['WD']
    return df_0shot, df_8shot


def draw_stadium_simple(ax, points, padding_x=0.15, padding_y=0.02,
                        color_left='#ffcccc', color_right='#cce5ff', alpha=0.4, zorder=0,
                        *, y_log=False, x_min_override=None, x_max_override=None, cap_radius_log=None):
    """Draw a simple horizontal stadium shape around a group of points."""
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
    min_y_data = min(ys) - padding_y
    max_y_data = max(ys) + padding_y
    if y_log:
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
    if cap_radius_log is None:
        cap_radius_log = max(0.12, padding_x * 0.35 + float(radius_for_cap) * 2.8)
    n_arc = 50
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, n_arc)
    theta_right = np.linspace(-np.pi/2, np.pi/2, n_arc)
    left_log_x = min_log_x + cap_radius_log * np.cos(theta_left)
    left_y = center_y + radius_y * np.sin(theta_left)
    right_log_x = max_log_x + cap_radius_log * np.cos(theta_right)
    right_y = center_y + radius_y * np.sin(theta_right)
    n_segments = 60
    cmap = LinearSegmentedColormap.from_list('custom', [color_left, color_right])
    total_span = (max_log_x + cap_radius_log) - (min_log_x - cap_radius_log)
    start_x = min_log_x - cap_radius_log

    def get_y_bounds(log_x):
        if log_x < min_log_x:
            dx = log_x - min_log_x
            if abs(dx) <= cap_radius_log:
                ratio = 1 - (dx / cap_radius_log) ** 2
                if ratio > 0:
                    dy = radius_y * np.sqrt(ratio)
                    return center_y - dy, center_y + dy
            return center_y, center_y
        elif log_x > max_log_x:
            dx = log_x - max_log_x
            if abs(dx) <= cap_radius_log:
                ratio = 1 - (dx / cap_radius_log) ** 2
                if ratio > 0:
                    dy = radius_y * np.sqrt(ratio)
                    return center_y - dy, center_y + dy
            return center_y, center_y
        else:
            return min_y, max_y

    for i in range(n_segments):
        t1 = i / n_segments
        t2 = (i + 1) / n_segments
        seg_log_x1 = start_x + t1 * total_span
        seg_log_x2 = start_x + t2 * total_span
        color = cmap((t1 + t2) / 2)
        y1_low, y1_high = get_y_bounds(seg_log_x1)
        y2_low, y2_high = get_y_bounds(seg_log_x2)
        if y_log:
            y1_low_d, y1_high_d = 10**y1_low, 10**y1_high
            y2_low_d, y2_high_d = 10**y2_low, 10**y2_high
        else:
            y1_low_d, y1_high_d = y1_low, y1_high
            y2_low_d, y2_high_d = y2_low, y2_high
        ax.fill([10**seg_log_x1, 10**seg_log_x1, 10**seg_log_x2, 10**seg_log_x2],
                [y1_low_d, y1_high_d, y2_high_d, y2_low_d],
                color=color, alpha=alpha, zorder=zorder, edgecolor='none')

    outline_log_x = np.concatenate([left_log_x, right_log_x])
    outline_y = np.concatenate([left_y, right_y])
    outline_y_d = 10**outline_y if y_log else outline_y
    ax.plot(10**outline_log_x, outline_y_d, color='#555555', linewidth=1.5, alpha=0.7, zorder=zorder+1)
    ax.plot([10**outline_log_x[-1], 10**outline_log_x[0]], [outline_y_d[-1], outline_y_d[0]],
            color='#555555', linewidth=1.5, alpha=0.7, zorder=zorder+1)


def draw_diagonal_stroke(ax, low_points, high_points, ellipse_width_log=0.25, ellipse_height_y=0.03,
                         color_start='#ffcccc', color_end='#cc0000', alpha=0.35, zorder=0):
    """Draw a diagonal stroke - ellipse moving along line from low to high points."""
    low_xs = [p[0] for p in low_points]
    low_ys = [p[1] for p in low_points]
    high_xs = [p[0] for p in high_points]
    high_ys = [p[1] for p in high_points]
    low_center_log_x = np.mean([np.log10(x) for x in low_xs])
    low_center_y = np.mean(low_ys)
    high_center_log_x = np.mean([np.log10(x) for x in high_xs])
    high_center_y = np.mean(high_ys)
    start_log_x, start_y = low_center_log_x, low_center_y
    end_log_x, end_y = high_center_log_x, high_center_y
    dx_log = end_log_x - start_log_x
    dy = end_y - start_y
    n_layers = 50
    cmap = LinearSegmentedColormap.from_list('custom', [color_start, color_end])
    for i in range(n_layers):
        t = i / (n_layers - 1) if n_layers > 1 else 0.5
        layer_log_x = start_log_x + t * dx_log
        layer_y = start_y + t * dy
        color = cmap(t)
        n_pts = 50
        theta = np.linspace(0, 2*np.pi, n_pts)
        ellipse_log_x = layer_log_x + (ellipse_width_log/2) * np.cos(theta)
        ellipse_y = layer_y + (ellipse_height_y/2) * np.sin(theta)
        ax.fill(10**ellipse_log_x, ellipse_y, color=color, alpha=alpha*0.6, zorder=zorder, edgecolor='none')


def plot_overview_gray(ax, df, title_suffix, selection_box=None, selected_mask=None, exclude_mask=None):
    """Plot overview scatter (left panels): mostly gray, with colored overlay for selected points."""
    if selected_mask is None:
        selected_mask = np.zeros(len(df), dtype=bool)
    if exclude_mask is None:
        exclude_mask = np.zeros(len(df), dtype=bool)
    small_df = df[(df['BS_Type'] == 'small') & ~exclude_mask]
    large_df = df[(df['BS_Type'] == 'large') & ~exclude_mask]
    ax.scatter(small_df['LR_x_WD'], small_df['Accuracy'],
               marker='P', s=45, facecolors='#bbbbbb', edgecolors='#777777', linewidths=1.0, zorder=3)
    ax.scatter(large_df['LR_x_WD'], large_df['Accuracy'],
               marker='o', s=38, facecolors='#9a9a9a', edgecolors='#666666', linewidths=0.8, zorder=3)
    sel_df = df[selected_mask & ~exclude_mask]
    sel_small = sel_df[sel_df['BS_Type'] == 'small']
    sel_large = sel_df[sel_df['BS_Type'] == 'large']
    if len(sel_small) > 0:
        ax.scatter(sel_small['LR_x_WD'], sel_small['Accuracy'],
                   marker='P', s=60, facecolors='#e74c3c', edgecolors='#cc6666', linewidths=1.2, zorder=6)
    if len(sel_large) > 0:
        ax.scatter(sel_large['LR_x_WD'], sel_large['Accuracy'],
                   marker='o', s=55, facecolors='#3498db', edgecolors='#6699cc', linewidths=1.2, zorder=6)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_log_x))
    if selection_box is not None:
        x_min, x_max, y_min_box, y_max_box = selection_box
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_min, y_min_box), x_max - x_min, y_max_box - y_min_box,
                         fill=False, edgecolor='#777777', linewidth=2.5, linestyle='-', zorder=1)
        ax.add_patch(rect)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=7, length=3)
    ax.tick_params(axis='both', which='minor', length=2)
    ax.set_ylabel(LABEL_ACC, fontsize=CHINESE_FONTSIZE, color=CHINESE_COLOR)
    ax.set_xlabel('')
    ax.grid(False)


def plot_0shot_detail(ax, df, fig=None):
    """Plot 0-shot detailed analysis (right top panel)."""
    small_configs = [(2e-4, 5e-3), (2e-4, 1e-2), (5e-4, 1e-2), (1e-3, 1e-2)]
    large_configs = [(5e-3, 1e-2), (1e-3, 1e-2)]
    small_points = []
    large_points = []
    for lr, wd in small_configs:
        row = df[(df['BS_Type'] == 'small') & np.isclose(df['LR'], lr, rtol=0.01) & np.isclose(df['WD'], wd, rtol=0.01)]
        if len(row) > 0:
            small_points.append((row['LR_x_WD'].values[0], row['Accuracy'].values[0], lr, wd))
    for lr, wd in large_configs:
        row = df[(df['BS_Type'] == 'large') & np.isclose(df['LR'], lr, rtol=0.01) & np.isclose(df['WD'], wd, rtol=0.01)]
        if len(row) > 0:
            large_points.append((row['LR_x_WD'].values[0], row['Accuracy'].values[0], lr, wd))
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_log_x))
    low_acc_small = [p for p in small_points if p[1] < 0.4]
    high_acc_small = [p for p in small_points if p[1] >= 0.4]
    low_acc_large = [p for p in large_points if p[1] < 0.4]
    high_acc_large = [p for p in large_points if p[1] >= 0.4]
    low_group_points = [(p[0], p[1]) for p in low_acc_small + low_acc_large]
    if len(low_group_points) >= 2:
        draw_stadium_simple(ax, low_group_points, padding_x=0.18, padding_y=0.018,
                           color_left='#ff8888', color_right='#88aaff', alpha=0.80, zorder=1)
    high_group_points = [(p[0], p[1]) for p in high_acc_small + high_acc_large]
    if len(high_group_points) >= 2:
        draw_stadium_simple(ax, high_group_points, padding_x=0.18, padding_y=0.018,
                           color_left='#ff8888', color_right='#88aaff', alpha=0.80, zorder=1)
    low_small = [(p[0], p[1]) for p in low_acc_small]
    high_small = [(p[0], p[1]) for p in high_acc_small]
    if len(low_small) >= 1 and len(high_small) >= 1:
        draw_diagonal_stroke(ax, low_small, high_small, ellipse_width_log=0.55, ellipse_height_y=0.055,
                            color_start=plt.cm.Reds(0.15), color_end=plt.cm.Reds(0.9), alpha=0.28, zorder=2)
    low_large = [(p[0], p[1]) for p in low_acc_large]
    high_large = [(p[0], p[1]) for p in high_acc_large]
    if len(low_large) >= 1 and len(high_large) >= 1:
        draw_diagonal_stroke(ax, low_large, high_large, ellipse_width_log=0.55, ellipse_height_y=0.055,
                            color_start=plt.cm.Blues(0.15), color_end=plt.cm.Blues(0.9), alpha=0.28, zorder=2)
    all_accs = [p[1] for p in small_points] + [p[1] for p in large_points]
    global_min_acc = min(all_accs) if all_accs else 0.3
    global_max_acc = max(all_accs) if all_accs else 0.55
    for x, y, lr, wd in small_points:
        intensity = (y - global_min_acc) / (global_max_acc - global_min_acc) if global_max_acc > global_min_acc else 0.5
        color = plt.cm.Reds(0.35 + 0.55 * intensity)
        ax.scatter(x, y, marker='P', s=110, facecolors=color, edgecolors='#cc6666', linewidths=0.8, zorder=10)
        label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
        ax.text(x, y, label_text, fontsize=6.5, rotation=45, ha='left', va='bottom', color='#333333', zorder=11)
    for x, y, lr, wd in large_points:
        intensity = (y - global_min_acc) / (global_max_acc - global_min_acc) if global_max_acc > global_min_acc else 0.5
        color = plt.cm.Blues(0.35 + 0.55 * intensity)
        ax.scatter(x, y, marker='o', s=100, facecolors=color, edgecolors='#6666cc', linewidths=0.8, zorder=10)
        label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
        ax.text(x, y, label_text, fontsize=6.5, rotation=45, ha='left', va='bottom', color='#333333', zorder=11)
    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FS)
    ax.set_xlabel('', fontsize=AXIS_LABEL_FS)
    ax.set_ylabel('', fontsize=AXIS_LABEL_FS, labelpad=0)
    ax.set_xlim(3e-7, 2e-4)
    ax.set_ylim(0.27, 0.60)
    if fig is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax_red = inset_axes(ax, width="2.3%", height="34%", loc='upper left',
                             bbox_to_anchor=(0.05, -0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        sm_red = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=global_min_acc, vmax=global_max_acc))
        sm_red.set_array([])
        cbar_red = fig.colorbar(sm_red, cax=cax_red, orientation='vertical')
        cbar_red.ax.tick_params(labelsize=6, length=2, labelleft=False, labelright=False)
        cbar_red.ax.set_yticklabels([])
        try:
            cbar_red.ax.yaxis.offsetText.set_visible(False)
        except Exception:
            pass
        cax_blue = inset_axes(ax, width="2.3%", height="34%", loc='upper left',
                              bbox_to_anchor=(0.11, -0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        sm_blue = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=global_min_acc, vmax=global_max_acc))
        sm_blue.set_array([])
        cbar_blue = fig.colorbar(sm_blue, cax=cax_blue, orientation='vertical')
        cbar_blue.ax.tick_params(labelsize=6, length=2)
        ax.text(0.13, 0.55, LABEL_ACC, transform=ax.transAxes, ha='center', va='top', fontsize=CHINESE_FONTSIZE, color=CHINESE_COLOR)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='P', color='w', markerfacecolor='#dd6666', markeredgecolor='#cc6666',
               markersize=6, markeredgewidth=0.6, label='Small batch'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6688dd', markeredgecolor='#6666cc',
               markersize=6, markeredgewidth=0.6, label='Large batch'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=6, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--', color='#dddddd')


def plot_8shot_detail(ax, df):
    """Plot 8-shot detailed analysis (right bottom panel)."""
    mask = ((df['Accuracy'] >= 0.10) & (df['Accuracy'] <= 0.14) &
            (df['LR_x_WD'] >= 1e-6) & (df['LR_x_WD'] <= 1e-4))
    selected_df = df[mask].copy()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_log_x))
    small_df = selected_df[selected_df['BS_Type'] == 'small']
    large_df = selected_df[selected_df['BS_Type'] == 'large']
    all_points = [(row['LR_x_WD'], row['Accuracy']) for _, row in selected_df.iterrows()]
    if len(all_points) >= 2:
        draw_stadium_simple(ax, all_points, padding_x=0.12, padding_y=0.0040,
                           color_left='#ff8888', color_right='#88aaff', alpha=0.80, zorder=1,
                           y_log=True, x_min_override=1e-6, x_max_override=1e-4)
    for idx, row in small_df.iterrows():
        x, y = row['LR_x_WD'], row['Accuracy']
        lr, wd = row['LR'], row['WD']
        ax.scatter(x, y, marker='P', s=90, facecolors='#e74c3c', edgecolors='#cc6666',
                  linewidths=0.8, zorder=10, label='Small batch' if idx == small_df.index[0] else '')
        label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
        ax.text(x, y, label_text, fontsize=6.5, rotation=45, ha='left', va='bottom', color='#333333', zorder=11)
    for idx, row in large_df.iterrows():
        x, y = row['LR_x_WD'], row['Accuracy']
        lr, wd = row['LR'], row['WD']
        ax.scatter(x, y, marker='o', s=85, facecolors='#3498db', edgecolors='#6699cc',
                  linewidths=0.8, zorder=10, label='Large batch' if idx == large_df.index[0] else '')
        label_text = f'{lr:.0e}×{wd:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
        ax.text(x, y, label_text, fontsize=6.5, rotation=45, ha='left', va='bottom', color='#333333', zorder=11)
    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FS)
    ax.set_xlabel('', fontsize=AXIS_LABEL_FS)
    ax.set_ylabel('', fontsize=AXIS_LABEL_FS, labelpad=0)
    ax.legend(loc='upper right', fontsize=6, framealpha=0.9, markerscale=0.6)
    ax.grid(True, alpha=0.25, linestyle='--', color='#dddddd')
    ax.set_xlim(5e-7, 2e-4)
    ax.set_ylim(0.08, 0.17)
    ax.yaxis.set_major_locator(FixedLocator([0.10, 0.15]))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_decimal_0x))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    df_0shot, df_8shot = load_data()
    fig = plt.figure(figsize=(9.2, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], height_ratios=[1, 1],
                          left=0.07, right=0.92, top=0.94, bottom=0.10, wspace=0.35, hspace=0.00)
    ax_left_top = fig.add_subplot(gs[0, 0])
    ax_left_bottom = fig.add_subplot(gs[1, 0])
    ax_right_top = fig.add_subplot(gs[0, 1])
    ax_right_bottom = fig.add_subplot(gs[1, 1])

    sel0 = np.zeros(len(df_0shot), dtype=bool)
    for lr, wd in [(2e-4, 5e-3), (2e-4, 1e-2), (5e-4, 1e-2), (1e-3, 1e-2)]:
        sel0 |= ((df_0shot['BS_Type'] == 'small') & np.isclose(df_0shot['LR'].astype(float), lr, rtol=0.01) &
                 np.isclose(df_0shot['WD'].astype(float), wd, rtol=0.01))
    for lr, wd in [(5e-3, 1e-2), (1e-3, 1e-2)]:
        sel0 |= ((df_0shot['BS_Type'] == 'large') & np.isclose(df_0shot['LR'].astype(float), lr, rtol=0.01) &
                 np.isclose(df_0shot['WD'].astype(float), wd, rtol=0.01))
    sel8 = ((df_8shot['Accuracy'] >= 0.10) & (df_8shot['Accuracy'] <= 0.14) &
            (df_8shot['LR_x_WD'] >= 1e-6) & (df_8shot['LR_x_WD'] <= 1e-4)).values

    exclude_0shot = np.zeros(len(df_0shot), dtype=bool)
    exclude_0shot |= ((df_0shot['BS_Type'] == 'large') & (df_0shot['LR_x_WD'] < 1e-5) & (df_0shot['LR'].astype(float) < 0.25))
    exclude_0shot |= ((df_0shot['BS_Type'] == 'small') & (df_0shot['LR_x_WD'] >= 1e-5) & (df_0shot['Accuracy'] < 0.3))
    exclude_0shot |= ((df_0shot['BS_Type'] == 'small') & (df_0shot['LR_x_WD'] > 2e-5))
    exclude_8shot = np.zeros(len(df_8shot), dtype=bool)
    exclude_8shot |= (df_8shot['Accuracy'] > 0.15)
    exclude_8shot |= ((df_8shot['BS_Type'] == 'small') & (df_8shot['Accuracy'] < 0.10))
    exclude_8shot |= ((df_8shot['BS_Type'] == 'large') & (df_8shot['LR_x_WD'] < 1e-5) & (df_8shot['LR_x_WD'] > 8e-7) &
                      ~(np.isclose(df_8shot['LR'].astype(float), 5e-4, rtol=0.01) &
                        np.isclose(df_8shot['WD'].astype(float), 1e-2, rtol=0.01)))

    selection_0shot = (5e-7, 8e-5, 0.28, 0.58)
    plot_overview_gray(ax_left_top, df_0shot, '0-shot', selection_box=selection_0shot,
                       selected_mask=sel0, exclude_mask=exclude_0shot)
    selection_8shot = (8e-7, 1.2e-4, 0.095, 0.145)
    plot_overview_gray(ax_left_bottom, df_8shot, '8-shot', selection_box=selection_8shot,
                       selected_mask=sel8, exclude_mask=exclude_8shot)

    ax_left_top.set_yscale('log')
    ax_left_top.set_ylim(0.18, 0.60)
    ax_left_top.set_xlim(1e-7, 2e-4)
    ax_left_top.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.4, 0.6]))
    ax_left_top.yaxis.set_major_formatter(FuncFormatter(_fmt_decimal_0x))
    ax_left_bottom.set_ylim(0.04, 0.19)
    ax_left_bottom.set_xlim(1e-7, 2e-4)
    ax_left_top.set_title('', fontsize=9, pad=6, color='black', y=1.08)
    ax_left_bottom.set_title('8-shot', fontsize=9, pad=6, color='black', y=1.08)
    ax_left_top.set_xlabel(LABEL_LR_WD, fontsize=CHINESE_FONTSIZE, color=CHINESE_COLOR)
    ax_left_bottom.set_xlabel(LABEL_LR_WD, fontsize=CHINESE_FONTSIZE, color=CHINESE_COLOR)

    for ax in [ax_left_top, ax_left_bottom]:
        for side in ['left', 'bottom', 'right', 'top']:
            ax.spines[side].set_visible(True)
        ax.spines['left'].set_color('#999999')
        ax.spines['bottom'].set_color('#999999')
        ax.spines['right'].set_color('#999999')
        ax.spines['top'].set_color('#999999')
        ax.tick_params(colors='#666666', labelsize=AXIS_TICK_FS)
        ax.set_ylabel(LABEL_ACC, fontsize=CHINESE_FONTSIZE, color=CHINESE_COLOR)

    def _shrink_axis(ax, width_scale, height_scale):
        bbox = ax.get_position()
        return bbox, bbox.width * width_scale, bbox.height * height_scale
    right_top_bbox, rt_w, rt_h = _shrink_axis(ax_right_top, 0.6, 0.75)
    right_bot_bbox, rb_w, rb_h = _shrink_axis(ax_right_bottom, 0.6, 0.75)
    divider_x = right_top_bbox.x0
    rt_x0, rt_y0 = divider_x, right_top_bbox.y0 + (right_top_bbox.height - rt_h) / 2
    ax_right_top.set_position([rt_x0, rt_y0, rt_w, rt_h])
    rb_x0, rb_y0 = divider_x, right_bot_bbox.y0 + (right_bot_bbox.height - rb_h) / 2
    ax_right_bottom.set_position([rb_x0, rb_y0, rb_w, rb_h])
    left_top_bbox, lt_w, lt_h = _shrink_axis(ax_left_top, 0.6, 0.5)
    left_bot_bbox, lb_w, lb_h = _shrink_axis(ax_left_bottom, 0.6, 0.5)
    gap = 0.06
    lt_x0 = divider_x - gap - lt_w
    lt_y0 = (rt_y0 + rt_h / 2) - lt_h / 2
    ax_left_top.set_position([lt_x0, lt_y0, lt_w, lt_h])
    lb_x0 = divider_x - gap - lb_w
    lb_y0 = (rb_y0 + rb_h / 2) - lb_h / 2
    ax_left_bottom.set_position([lb_x0, lb_y0, lb_w, lb_h])

    plot_0shot_detail(ax_right_top, df_0shot, fig)
    plot_8shot_detail(ax_right_bottom, df_8shot)

    fig.canvas.draw()
    lt_pos = ax_left_top.get_position()
    rt_pos = ax_right_top.get_position()
    lb_pos = ax_left_bottom.get_position()
    rb_pos = ax_right_bottom.get_position()
    from matplotlib.lines import Line2D
    for xdata, ydata in [
        ([lt_pos.x1, rt_pos.x0], [lt_pos.y0, rt_pos.y0]), ([lt_pos.x1, rt_pos.x0], [lt_pos.y1, rt_pos.y1]),
        ([lt_pos.x1, lt_pos.x1], [lt_pos.y0, lt_pos.y1]), ([rt_pos.x0, rt_pos.x0], [rt_pos.y0, rt_pos.y1]),
        ([lb_pos.x1, rb_pos.x0], [lb_pos.y0, rb_pos.y0]), ([lb_pos.x1, rb_pos.x0], [lb_pos.y1, rb_pos.y1]),
        ([lb_pos.x1, lb_pos.x1], [lb_pos.y0, lb_pos.y1]), ([rb_pos.x0, rb_pos.x0], [rb_pos.y0, rb_pos.y1]),
    ]:
        fig.add_artist(Line2D(xdata, ydata, transform=fig.transFigure, color='#aaaaaa', linewidth=1.2, zorder=0))

    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "four_panel_analysis_modified.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")
    pdf_path = os.path.join(output_dir, "four_panel_analysis_modified.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    plt.close()


if __name__ == "__main__":
    main()
