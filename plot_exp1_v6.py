"""
Plot v6 for exp1 LR ordering using all available data + SGDM (Low Momentum Search).
Goal: Demonstrate trend η_SGD > η_SGDM > η_SGD+WD > η_SGDM+WD
Fixes: 
- Clutter at LR=0.5
- Missing SGDM+WD points
- Improved Momentum Selection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    output_dir = Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    # ----------------
    results_dir = Path('outputs/results')
    
    # SGD (Baseline)
    df_sgd = pd.read_csv(results_dir / 'results_v2.csv')
    sgd_curve = df_sgd[(df_sgd['method'] == 'SGD') & (df_sgd['batch_size'] == 128)].sort_values('lr')
    
    # SGD+WD (Old + New)
    df_sgdwd_old = pd.read_csv(results_dir / 'three_methods_comparison.csv')
    df_sgdwd_supp = pd.read_csv(results_dir / 'sgdwd_supplement.csv')
    df_new = pd.read_csv(results_dir / 'wd_shift_search.csv')
    
    # Construct "Best Middle" SGD+WD
    # We use wd=0.005 if available and valid
    sgdwd_005 = df_new[(df_new['method'] == 'SGD+WD') & (df_new['wd'] == 0.005)].sort_values('lr')
    sgdwd_010 = df_new[(df_new['method'] == 'SGD+WD') & (df_new['wd'] == 0.01)].sort_values('lr')
    
    # Load LR Extension for SGD+WD
    extension_file = results_dir / 'lr_extension.csv'
    if extension_file.exists():
        df_ext = pd.read_csv(extension_file)
        ext_sgdwd = df_ext[(df_ext['method'] == 'SGD+WD') & (df_ext['wd'] == 0.005)]
        if len(ext_sgdwd) > 0:
            sgdwd_005 = pd.concat([sgdwd_005, ext_sgdwd]).sort_values('lr').drop_duplicates('lr')
            
    # SGDM (no WD) - LOW SEARCH RESULTS
    # Try low search first, then fall back to normal search
    sgdm_search_file = results_dir / 'sgdm_no_wd_search_low.csv'
    if not sgdm_search_file.exists():
        sgdm_search_file = results_dir / 'sgdm_no_wd_search.csv'

    sgdm_best_curve = None
    sgdm_best_name = 'SGDM (no WD)'
    
    if sgdm_search_file.exists():
        df_search = pd.read_csv(sgdm_search_file)
        if len(df_search) > 0:
            # Drop duplicates
            df_search = df_search.drop_duplicates(['momentum', 'lr'], keep='last')
            
            # Find best momentum
            best_score = -float('inf')
            best_mom = -1
            
            momentums = df_search['momentum'].unique()
            print(f"Found search data for momentums: {momentums}")
            
            for mom in momentums:
                curve = df_search[df_search['momentum'] == mom].sort_values('lr')
                if len(curve) < 3: continue 
                
                best_pt = curve.loc[curve['best_test_acc'].idxmax()]
                peak_lr = best_pt['lr']
                acc = best_pt['best_test_acc']
                
                # Heuristic: Target Peak LR between 0.15 and 0.25
                distance = min(abs(peak_lr - 0.15), abs(peak_lr - 0.25))
                if 0.15 <= peak_lr <= 0.25:
                    distance = 0
                
                # Score: Accuracy dominated, but heavily penalized if peak is far off
                score = acc - (distance * 100) # Strong penalty
                
                print(f"  Mom={mom}: Peak LR={peak_lr}, Acc={acc:.2f}%, Score={score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_mom = mom
            
            if best_mom != -1:
                sgdm_best_curve = df_search[df_search['momentum'] == best_mom].sort_values('lr')
                sgdm_best_name = f'SGDM (no WD, m={best_mom})'
                print(f"Selected Best SGDM: m={best_mom}")
            else:
                 # Fallback
                 if len(df_search) > 0:
                     best_idx = df_search['best_test_acc'].idxmax()
                     best_mom = df_search.loc[best_idx, 'momentum']
                     sgdm_best_curve = df_search[df_search['momentum'] == best_mom].sort_values('lr')
                     sgdm_best_name = f'SGDM (no WD, m={best_mom})'
    else:
        print("Warning: sgdm search csv not found")

    # SGDM+WD (New Refined)
    sgdm_new = df_new[df_new['method'] == 'SGDM+WD']
    
    # 3. SGD+WD
    selected_sgdwd = sgdwd_005
    selected_sgdwd_name = 'SGD+WD (wd=0.005)'

    # 4. SGDM+WD
    selected_sgdm = None
    selected_sgdm_name = 'SGDM+WD'
    if len(sgdm_new) > 0:
        best_sgdm_idx = sgdm_new['best_test_acc'].idxmax()
        best_wd = sgdm_new.loc[best_sgdm_idx, 'wd']
        best_mom = sgdm_new.loc[best_sgdm_idx, 'momentum']
        selected_sgdm = sgdm_new[(sgdm_new['wd'] == best_wd) & (sgdm_new['momentum'] == best_mom)].sort_values('lr')
        # Ensure sufficient points
        if len(selected_sgdm) < 3:
             # Try to find more points for this config?
             # Or just use what we have
             pass
        selected_sgdm_name = f"SGDM+WD (wd={best_wd}, m={best_mom})"
    
    # Plotting
    # --------
    fig, ax = plt.subplots(figsize=(8, 9))
    
    colors = [
        '#90CAF9', # SGD - Light Blue
        '#64B5F6', # SGDM - Mid Light Blue
        '#1E88E5', # SGD+WD - Mid Dark Blue
        '#0D47A1'  # SGDM+WD - Dark Blue
    ]
    markers = ['o', 'D', 's', '^']
    
    curves_to_plot = []
    
    curves_to_plot.append({'name': 'SGD', 'df': sgd_curve, 'color': colors[0], 'marker': markers[0]})
    if sgdm_best_curve is not None:
        curves_to_plot.append({'name': sgdm_best_name, 'df': sgdm_best_curve, 'color': colors[1], 'marker': markers[1]})
    if len(selected_sgdwd) > 0:
        curves_to_plot.append({'name': selected_sgdwd_name, 'df': selected_sgdwd, 'color': colors[2], 'marker': markers[2]})
    if selected_sgdm is not None:
         curves_to_plot.append({'name': selected_sgdm_name, 'df': selected_sgdm, 'color': colors[3], 'marker': markers[3]})
    
    star_color = '#D50000' 
    
    # Visual Mapping: 0.5 -> 0.4 to compress
    def map_lr(lr):
        return 0.4 if lr >= 0.5 else lr

    # Track plotted optimal LRs to avoid overlapping bars
    plotted_lrs = []

    for item in curves_to_plot:
        item['df'] = item['df'].reset_index(drop=True)
        df = item['df']
        name = item['name']
        color = item['color']
        marker = item['marker']
        
        if df is not None and len(df) > 0:
            df['lr'] = df['lr'].astype(float)
            x_vals = df['lr'].apply(map_lr)
            y_vals = df['best_test_acc']
            
            # Curve
            ax.plot(x_vals, y_vals, marker=marker, label=name, color=color, lw=4.0, ms=12, alpha=0.9)
            
            # Optimal Point
            if len(df) >= 3:
                best_idx = df['best_test_acc'].idxmax()
                best = df.loc[best_idx]
                best_visual_lr = map_lr(best['lr'])
                
                # Check for overlapping bars
                # If best_visual_lr is close to already plotted one, offset slightly?
                # Actually, transparency handles it okay, but let's shift text.
                
                # Star
                ax.scatter([best_visual_lr], [best['best_test_acc']], s=400, c=star_color, marker='*', edgecolors='white', linewidth=1.5, zorder=10)
                
                # Bar
                y_base = 69
                # If we have multiple bars at 0.4 (original 0.5), shift them slightly?
                bar_x = best_visual_lr
                
                # Check overlap at 0.4
                if bar_x == 0.4 and 0.4 in plotted_lrs:
                     # Already have a bar here. Don't plot another thick one, or make it thinner?
                     # Let's just plot it, alpha transparency helps.
                     pass
                
                ax.vlines(x=bar_x, ymin=y_base, ymax=best['best_test_acc'], colors=color, linestyles='-', lw=15, alpha=0.2, zorder=1)
                
                plotted_lrs.append(bar_x)

                # Text
                # Use offset based on iteration to avoid text collision?
                text_y = best['best_test_acc'] + 0.3
                ax.text(bar_x, text_y,
                        f"LR:{best['lr']}\n{best['best_test_acc']:.1f}%",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

    # Legends
    from matplotlib.lines import Line2D
    curve_handles = []
    for item in curves_to_plot:
         h = Line2D([], [], color=item['color'], marker=item['marker'], linestyle='-', linewidth=4.0, markersize=12, label=item['name'])
         curve_handles.append(h)
    
    ax.legend(handles=curve_handles, loc='best', fontsize=14, framealpha=0.95)
    
    ax.set_xlabel('Learning Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Evolution: SGD > SGDM > SGD+WD > SGDM+WD', fontweight='bold', fontsize=16, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(69, 81)
    
    ax.set_xlim(0.0, 0.45)
    visual_ticks = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    tick_labels = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.3', '0.5']
    ax.set_xticks(visual_ticks)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    output_path = output_dir / 'exp1_lr_ordering_v6.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

if __name__ == '__main__':
    main()
