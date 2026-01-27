"""
Plot v7 for exp1 LR ordering.
Based on plot_exp1_v3.py.
Changes:
- Include SGDM (no WD) with m=0.1 (Peaking at 0.25).
- Restore dense SGDM+WD curve (fix missing points).
- Remove LR=0.5 points (clutter).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

def main():
    output_dir = Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    # ----------------
    results_dir = Path('outputs/results')
    
    # A. SGD (Baseline)
    df_sgd = pd.read_csv(results_dir / 'results_v2.csv')
    sgd_curve = df_sgd[(df_sgd['method'] == 'SGD') & (df_sgd['batch_size'] == 128)].sort_values('lr')
    
    # B. SGDM (no WD)
    # User wants result at 0.25 -> m=0.1 from low search
    sgdm_files = [
        results_dir / 'sgdm_no_wd_search_low.csv',
        results_dir / 'sgdm_no_wd_search.csv'
    ]
    
    sgdm_nowd_curve = None
    sgdm_nowd_name = 'SGDM (no WD)'
    
    for f in sgdm_files:
        if f.exists():
            df = pd.read_csv(f)
            # Filter for m=0.1
            target_mom = 0.1
            curve = df[df['momentum'] == target_mom].sort_values('lr')
            if len(curve) > 0:
                sgdm_nowd_curve = curve
                sgdm_nowd_name = f'SGDM (no WD, m={target_mom})'
                print(f"Loaded SGDM (no WD) m={target_mom} from {f.name}")
                break
    
    # C. SGD+WD
    # We want a curve peaking < 0.3. 
    # v3 logic: wd=0.005 (new) or 0.002 (old)
    df_new = pd.read_csv(results_dir / 'wd_shift_search.csv')
    
    # Try finding dense wd=0.005 first
    sgdwd_005 = df_new[(df_new['method'] == 'SGD+WD') & (df_new['wd'] == 0.005)].sort_values('lr')
    
    # Extension?
    extension_file = results_dir / 'lr_extension.csv'
    if extension_file.exists():
         df_ext = pd.read_csv(extension_file)
         ext_sgdwd = df_ext[(df_ext['method'] == 'SGD+WD') & (df_ext['wd'] == 0.005)]
         if len(ext_sgdwd) > 0:
             sgdwd_005 = pd.concat([sgdwd_005, ext_sgdwd]).sort_values('lr').drop_duplicates('lr')
             
    selected_sgdwd = sgdwd_005
    selected_sgdwd_name = 'SGD+WD (wd=0.005)'
    
    if len(selected_sgdwd) < 4:
         # Fallback to Old 0.002 if 0.005 is sparse
         print("SGD+WD (0.005) is sparse, falling back to 0.002")
         df_sgdwd_old = pd.read_csv(results_dir / 'three_methods_comparison.csv')
         df_sgdwd_supp = pd.read_csv(results_dir / 'sgdwd_supplement.csv')
         
         base = df_sgdwd_old[(df_sgdwd_old['method'] == 'SGD+WD') & (df_sgdwd_old['wd'] == 0.002)]
         supp = df_sgdwd_supp[(df_sgdwd_supp['method'] == 'SGD+WD') & (df_sgdwd_supp['wd'] == 0.002)]
         selected_sgdwd = pd.concat([base, supp]).sort_values('lr').drop_duplicates('lr')
         selected_sgdwd_name = 'SGD+WD (wd=0.002)'

    # D. SGDM+WD
    # User complained about missing points.
    # Check wd_shift_search (df_new)
    sgdm_wd_new = df_new[df_new['method'] == 'SGDM+WD']
    
    selected_sgdm_wd = None
    selected_sgdm_wd_name = 'SGDM+WD'
    
    # Try to find a dense curve in new Data
    # We need something that looks like the "Lowest" curve (peak ~0.05-0.1)
    
    # Check momentum_search.csv (Old robust data)
    # v3 fallback: wd=0.002, m=0.7
    mom_search_file = results_dir / 'momentum_search.csv'
    if mom_search_file.exists():
        df_mom = pd.read_csv(mom_search_file)
        # Check m=0.7, wd=0.002
        fallback_curve = df_mom[(df_mom['momentum'] == 0.7) & (df_mom['wd'] == 0.002)].sort_values('lr')
        
        # Also check comparison file (wd=5e-4, m=0.9)
        # This was the "Low" curve in v2
        df_comp = pd.read_csv(results_dir / 'three_methods_comparison.csv')
        comp_curve = df_comp[df_comp['method'] == 'SGDM+WD'].sort_values('lr')
        
        # Pick the one with more points or better fit?
        # User wants "COMPLETE" curve.
        if len(fallback_curve) > len(comp_curve):
             selected_sgdm_wd = fallback_curve
             selected_sgdm_wd_name = 'SGDM+WD (wd=0.002, m=0.7)'
             print("Selected SGDM+WD from momentum_search (dense)")
        else:
             selected_sgdm_wd = comp_curve
             selected_sgdm_wd_name = 'SGDM+WD (wd=5e-4, m=0.9)'
             print("Selected SGDM+WD from three_methods (dense)")
             
    # 2. Filtering
    # ------------
    # Remove LRs >= 0.5 as requested ("don't need results at 0.5")
    
    def filter_lr(df):
        if df is None: return None
        return df[df['lr'] < 0.5].reset_index(drop=True)
        
    sgd_curve = filter_lr(sgd_curve)
    sgdm_nowd_curve = filter_lr(sgdm_nowd_curve)
    selected_sgdwd = filter_lr(selected_sgdwd)
    selected_sgdm_wd = filter_lr(selected_sgdm_wd)

    # 3. Plotting
    # -----------
    fig, ax = plt.subplots(figsize=(7, 8))
    
    # Colors: Light -> Dark
    colors = [
        '#90CAF9', # SGD
        '#64B5F6', # SGDM (no WD)
        '#1E88E5', # SGD+WD
        '#0D47A1'  # SGDM+WD
    ]
    markers = ['o', 'D', 's', '^']
    
    curves = [
        ('SGD', sgd_curve, colors[0], markers[0]),
        (sgdm_nowd_name, sgdm_nowd_curve, colors[1], markers[1]),
        (selected_sgdwd_name, selected_sgdwd, colors[2], markers[2]),
        (selected_sgdm_wd_name, selected_sgdm_wd, colors[3], markers[3])
    ]
    
    star_color = '#D50000'
    
    for name, df, color, marker in curves:
        if df is not None and len(df) > 0:
            df['lr'] = df['lr'].astype(float)
            
            # Line
            ax.plot(df['lr'], df['best_test_acc'], marker=marker, label=name, color=color, lw=4.5, ms=12, alpha=0.9)
            
            # Optimal
            if len(df) >= 3:
                best = df.loc[df['best_test_acc'].idxmax()]
                
                # Star
                ax.scatter([best['lr']], [best['best_test_acc']], s=400, c=star_color, marker='*', edgecolors='white', linewidth=1.5, zorder=10)
                
                # Bar
                y_base = 69
                ax.vlines(x=best['lr'], ymin=y_base, ymax=best['best_test_acc'], colors=color, linestyles='-', lw=15, alpha=0.2, zorder=1)
                
                # Text
                ax.text(best['lr'], best['best_test_acc'] + 0.3,
                        f"LR:{best['lr']}\n{best['best_test_acc']:.1f}%",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

    # Legend
    curve_handles = []
    for name, _, color, marker in curves:
         h = Line2D([], [], color=color, marker=marker, linestyle='-', linewidth=4.5, markersize=12, label=name)
         curve_handles.append(h)
         
    ax.legend(handles=curve_handles, loc='upper right', fontsize=14, framealpha=0.95)
    
    ax.set_xlabel('Learning Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Evolution: SGD > SGDM > SGD+WD > SGDM+WD', fontweight='bold', fontsize=18, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(69, 81)
    ax.set_xlim(0.0, 0.45)
    
    # Ticks
    # v3 had custom ticks logic, but here we just use linear space since we cutoff 0.5
    visual_ticks = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    ax.set_xticks(visual_ticks)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    output_path = output_dir / 'exp1_lr_ordering_v7.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

if __name__ == '__main__':
    main()
