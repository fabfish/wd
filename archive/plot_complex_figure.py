
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from matplotlib.transforms import Bbox

def create_gradient_patch(ax, bbox, color_start, color_end, color_mid=None, vertical=True, alpha=0.3):
    """
    Simulates a gradient patch by creating an image within a bounding box.
    Uses clip_path to restrict it to a fancy box shape.
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap

    # Create the fancy box path
    x, y = bbox.x0, bbox.y0
    w, h = bbox.width, bbox.height
    
    # Box style: 'round,pad=0.1' or similar
    # We want "elliptical rectangle" -> basically a rounded rectangle with high rounding
    # But user said "rectangular ellipse". FancyBboxPatch with boxstyle="round" is closest.
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                         ec="none", fc="none", transform=ax.transData, zorder=0)
    ax.add_patch(box)

    # Create gradient image
    if color_mid:
        colors = [color_start, color_mid, color_end]
    else:
        colors = [color_start, color_end]
    
    cmap = LinearSegmentedColormap.from_list("custom_grad", colors, N=100)
    
    # We need to render the gradient image over the bbox area
    # Create a meshgrid for the image
    # Note: imshow needs data coordinates? No, it works in axes coords or data.
    # We'll use extent to map to data coordinates.
    
    # Add some padding for the extent
    pad = 0.0 # already handled by box pad?
    extent = (x, x+w, y, y+h)
    
    if vertical:
        grad = np.linspace(0, 1, 100).reshape(-1, 1) # Vertical gradient
    else:
        grad = np.linspace(0, 1, 100).reshape(1, -1) # Horizontal gradient
        
    img = ax.imshow(grad, extent=extent, cmap=cmap, aspect='auto', alpha=alpha, zorder=0)
    img.set_clip_path(box)
    
    return box

def plot_complex_figure():
    file_0shot = "outputs/results/results_0shot.csv"
    file_8shot = "outputs/results/results_8shot.csv"
    
    if not os.path.exists(file_0shot) or not os.path.exists(file_8shot):
        print("Error: Input files not found.")
        return

    df0 = pd.read_csv(file_0shot)
    df8 = pd.read_csv(file_8shot)
    
    for df in [df0, df8]:
        df['LR'] = df['LR'].astype(float)
        df['WD'] = df['WD'].astype(float)
        df['LR_x_WD'] = df['LR'] * df['WD']

    # Setup Figure and GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, width_ratios=[1, 2.5], height_ratios=[1, 1], wspace=0.15, hspace=0.2)

    # --- Left Column: Context Plots ---
    
    # Helper for context plots
    def plot_context(ax, df, title):
        ax.scatter(df['LR_x_WD'], df['Accuracy'], color='silver', s=50, alpha=0.6, edgecolors='grey')
        ax.set_xscale('log')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("LR $\\times$ WD (Log Scale)")
        ax.grid(True, linestyle='--', alpha=0.5)

    ax_left_top = fig.add_subplot(gs[0, 0])
    plot_context(ax_left_top, df0, "0-shot Overview (All Points)")
    
    ax_left_bottom = fig.add_subplot(gs[1, 0])
    plot_context(ax_left_bottom, df8, "8-shot Overview (All Points)")


    # --- Top Right: 0-shot Focused (The 6 points) ---
    ax_main_top = fig.add_subplot(gs[0, 1])
    
    # Filter the 6 specific points for 0-shot
    # Small batch: 2e-4*5e-3, 2e-4*1e-2, 5e-4*1e-2, 1e-3*1e-2
    # Large batch: 5e-3*1e-2, 1e-3*1e-2
    
    # We need to match these in the dataframe closely (floating point tolerance)
    tol = 1e-9
    
    # Define targets
    targets_small = [
        (2e-4, 5e-3), (2e-4, 1e-2), (5e-4, 1e-2), (1e-3, 1e-2)
    ]
    targets_large = [
        (5e-3, 1e-2), (1e-3, 1e-2) 
        # Wait, user said "large batch: 5e03*1e02, 1e03*1e02" -> 5e-3, 1e-3
    ]
    
    # Filter function
    def is_target(row, targets):
        for t_lr, t_wd in targets:
            if abs(row['LR'] - t_lr) < tol and abs(row['WD'] - t_wd) < tol:
                return True
        return False
        
    df0_small_target = df0[ (df0['BS_Type'] == 'small') & df0.apply(lambda x: is_target(x, targets_small), axis=1) ].copy()
    df0_large_target = df0[ (df0['BS_Type'] == 'large') & df0.apply(lambda x: is_target(x, targets_large), axis=1) ].copy()
    
    # Combine for plotting logic
    # We want Heatmap style coloring: Large->Blue, Small->Red. Darker -> Higher Acc.
    
    # Custom cmap for points
    cmap_red = sns.light_palette("#e74c3c", as_cmap=True, n_colors=10) # Red
    cmap_blue = sns.light_palette("#3498db", as_cmap=True, n_colors=10) # Blue
    
    # We will scatter manually to control colors perfectly
    # Normalize Acc for color mapping relative to their own range? Or global?
    # User says: "Darker represents higher acc"
    
    # Plot Small Points
    sc_small = ax_main_top.scatter(
        df0_small_target['LR_x_WD'], df0_small_target['Accuracy'],
        c=df0_small_target['Accuracy'], cmap='Reds', vmin=0.2, vmax=0.6, # heuristic range
        s=300, marker='o', edgecolors='k', zorder=10, label='Small BS'
    )
    
    # Plot Large Points
    sc_large = ax_main_top.scatter(
        df0_large_target['LR_x_WD'], df0_large_target['Accuracy'],
        c=df0_large_target['Accuracy'], cmap='Blues', vmin=0.2, vmax=0.6,
        s=400, marker='^', edgecolors='k', zorder=10, label='Large BS'
    )

    ax_main_top.set_xscale('log')
    ax_main_top.set_title("0-shot Analysis: Batch Size vs LR*WD Trade-off", fontsize=14)
    ax_main_top.set_ylabel("Accuracy", fontsize=12)
    ax_main_top.grid(True, linestyle=':', alpha=0.4)
    
    # Annotations (Gradient Rectangles)
    # Ref: "将上方的和下方的代表不同acc水平的各两组每组2小1大各三个点用这种中间是矩形的椭圆框住，背景采用淡淡的 红色-白色-蓝色 转换"
    # "以下方2小的中心，和上方两小的中心，你要画一个椭圆矩形...浅红到深红...左下到右上"
    
    # Let's identify the groups coordinates
    # Group Lower Acc: Small(2e-4*5e-3 ~ 1e-6), Small(2e-4*1e-2 ~ 2e-6), Large(1e-3*1e-2 ~ 1e-5)?
    # Wait, check Accuracy values.
    # Small 2e-4*5e-3 Acc=0.22/0.33?, Small 2e-4*1e-2 Acc=0.3167
    # Large 1e-3*1e-2 Acc=0.31
    # Large 5e-3*1e-2 Acc=0.5533
    # Small 5e-4*1e-2 Acc=0.5333
    # Small 1e-3*1e-2 Acc=0.5167
    
    # High Acc Group: Small(5e-4*1e-2), Small(1e-3*1e-2), Large(5e-3*1e-2). (Acc ~0.53-0.55)
    # Low Acc Group: Small(2e-4*5e-3), Small(2e-4*1e-2), Large(1e-3*1e-2) (Wait, 1e-3*1e-2 is 0.31, others are ~0.33, 0.31)
    # Yes, looks like two data levels: ~0.3 and ~0.55.
    
    # Bbox for High Group
    high_group = pd.concat([
        df0_small_target[df0_small_target['Accuracy'] > 0.5],
        df0_large_target[df0_large_target['Accuracy'] > 0.5]
    ])
    
    if not high_group.empty:
        xmin, xmax = high_group['LR_x_WD'].min(), high_group['LR_x_WD'].max()
        ymin, ymax = high_group['Accuracy'].min(), high_group['Accuracy'].max()
        # Add padding
        w, h = xmax - xmin, ymax - ymin
        # Log scale width adjustment is tricky for bbox...
        # We define bbox in data coords, plotting logic handles visual.
        # But rect is linear in data space. Log scale makes it warped?
        # FancyBboxPatch works in data space but shape might be distorted on log axis?
        # Actually, patches on log axes are tricky.
        # Better to just use spans or matplotlib's transform to display coords if we want geometric shapes to look right.
        # But for "Rectangular Ellipse", we can just define a box that covers them.
        
        # NOTE: Visual rectangle on log plot = log-width box.
        
        # High Group Bbox
        # Use log10 for padding calculation logic if needed, but for patch we pass actual data coords
        # Padding factor
        x_pad_factor = 2 # multiplier
        y_pad = 0.05
        
        bbox_high = Bbox.from_extents(xmin / x_pad_factor, xmax * x_pad_factor, ymin - y_pad, ymax + y_pad)
        create_gradient_patch(ax_main_top, bbox_high, '#ffebee', '#e3f2fd', color_mid='#ffffff', vertical=False, alpha=0.3) # Red-White-Blue
        
        ax_main_top.text(np.sqrt(xmin*xmax), ymax + y_pad, "High Accuracy Group", ha='center', va='bottom', fontweight='bold')

    # Low Group Bbox
    low_group = pd.concat([
        df0_small_target[df0_small_target['Accuracy'] < 0.4],
        df0_large_target[df0_large_target['Accuracy'] < 0.4]
    ])
    
    if not low_group.empty:
        xmin, xmax = low_group['LR_x_WD'].min(), low_group['LR_x_WD'].max()
        ymin, ymax = low_group['Accuracy'].min(), low_group['Accuracy'].max()
        x_pad_factor = 2
        y_pad = 0.05
        
        bbox_low = Bbox.from_extents(xmin / x_pad_factor, xmax * x_pad_factor, ymin - y_pad, ymax + y_pad)
        create_gradient_patch(ax_main_top, bbox_low, '#ffebee', '#e3f2fd', color_mid='#ffffff', vertical=False, alpha=0.3)
        ax_main_top.text(np.sqrt(xmin*xmax), ymin - y_pad, "Baseline Accuracy Group", ha='center', va='top', fontweight='bold')

    # Diagonal Groups (Small & Large Trends)
    # The user asks for "Elliptical Rectangle" connecting centers of smalls and centers of larges.
    # This implies showing the "Improvement Path".
    # Small path: Low Small to High Small.
    # Large path: Low Large (which is the middle one? 1e-3*1e-2) to High Large.
    
    # Since these are diagonal on the plot, a simple axis-aligned rect box isn't perfect, but "rectangular ellipse" usually implies axis aligned.
    # If standard axis aligned box covers them, let's use that.
    
    # Small Trend Box
    s_xmin, s_xmax = df0_small_target['LR_x_WD'].min(), df0_small_target['LR_x_WD'].max()
    s_ymin, s_ymax = df0_small_target['Accuracy'].min(), df0_small_target['Accuracy'].max()
    bbox_small = Bbox.from_extents(s_xmin/1.5, s_xmax*1.5, s_ymin-0.03, s_ymax+0.03)
    # Shallow Red to Deep Red
    # To avoid overlap clutter, maybe make this more subtle or outline?
    # User asked for "background color from light red to deep red".
    # Overlapping background patches can get messy. Let's try to put these "trend" boxes behind the "group" boxes if possible, or vice versa.
    # Let's put trend boxes at zorder=-1, group boxes at zorder=-2?
    
    create_gradient_patch(ax_main_top, bbox_small, '#ffcdd2', '#b71c1c', vertical=False, alpha=0.2) 
    
    # Large Trend Box
    l_xmin, l_xmax = df0_large_target['LR_x_WD'].min(), df0_large_target['LR_x_WD'].max()
    l_ymin, l_ymax = df0_large_target['Accuracy'].min(), df0_large_target['Accuracy'].max()
    bbox_large = Bbox.from_extents(l_xmin/1.5, l_xmax*1.5, l_ymin-0.03, l_ymax+0.03)
    # Light Blue to Deep Blue
    create_gradient_patch(ax_main_top, bbox_large, '#bbdefb', '#0d47a1', vertical=False, alpha=0.2)


    # --- Bottom Right: 8-shot Focused ---
    ax_main_bot = fig.add_subplot(gs[1, 1])
    
    # Filter 8-shot points
    # Acc in 0.10 * 0.14
    # LR*WD in 1e-6 to 1e-4
    
    mask_8 = (df8['Accuracy'] >= 0.10) & (df8['Accuracy'] <= 0.14) & \
             (df8['LR_x_WD'] >= 1e-6) & (df8['LR_x_WD'] <= 1e-4) # Wait, 10e-4 is 1e-3. User: "10e-6 to 10e-4" -> 1e-5 to 1e-3? Or 10*10^-6? Usually 10e-6 means 1e-5. 
             # Let's assume user meant 1e-5 to 1e-3 area based on typical log ranges. But csv has points around 1e-6 too.
             # Let's query the specific points to be sure.
             # From CSV:
             # bs_large..wd_5e-1 -> 2e-4*5e-1 = 1e-4
             # ..wd_1e-1 -> 2e-4*1e-1 = 2e-5
             # ..wd_1e-2 -> 2e-4*1e-2 = 2e-6
             # Acc for these in 8shot is around 0.10, 0.11
             # So range 1e-6 to 1e-3 seems correct.
    
    df8_target = df8[mask_8].copy()
    
    # Separate Small/Large
    df8_target_small = df8_target[df8_target['BS_Type']=='small']
    df8_target_large = df8_target[df8_target['BS_Type']=='large']
    
    # Plotting
    ax_main_bot.scatter(df8_target_small['LR_x_WD'], df8_target_small['Accuracy'], c='#e74c3c', s=200, label='Small BS', edgecolors='k')
    ax_main_bot.scatter(df8_target_large['LR_x_WD'], df8_target_large['Accuracy'], c='#3498db', s=200, marker='^', label='Large BS', edgecolors='k')
    
    ax_main_bot.set_xscale('log')
    ax_main_bot.set_title("8-shot Analysis: Sensitivity Region", fontsize=14)
    ax_main_bot.set_ylabel("Accuracy", fontsize=12)
    ax_main_bot.set_xlabel("LR $\\times$ WD (Log Scale)", fontsize=12)
    ax_main_bot.grid(True, linestyle=':', alpha=0.4)
    
    # Annotation: One large Ellipse enclosing all, Red-White-Blue gradient
    if not df8_target.empty:
        xmin, xmax = df8_target['LR_x_WD'].min(), df8_target['LR_x_WD'].max()
        ymin, ymax = df8_target['Accuracy'].min(), df8_target['Accuracy'].max()
        
        bbox_8 = Bbox.from_extents(xmin/2, xmax*2, ymin-0.01, ymax+0.01)
        create_gradient_patch(ax_main_bot, bbox_8, '#ffebee', '#e3f2fd', color_mid='#ffffff', vertical=False, alpha=0.3)
        ax_main_bot.text(np.sqrt(xmin*xmax), ymax+0.015, "Parameter Sensitivity Zone", ha='center', va='bottom', style='italic')

    # Legends
    ax_main_top.legend(loc='lower right')
    ax_main_bot.legend(loc='lower right')
    
    # Layout adjustments
    plt.tight_layout()
    
    outfile = "outputs/plots/complex_analysis.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved complex figure to {outfile}")
    plt.close()

if __name__ == "__main__":
    plot_complex_figure()
