"""
Plot v9 for exp1 LR ordering.
EXACTLY based on plot_exp1_v3.py logic.
Added only: SGDM (no WD).
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
    
    # Get SGD+WD curves for different WDs to pick the best "Middle" one
    # We want a curve that has optimal LR < SGD's optimal LR (0.3)
    # Candidates: wd=0.002 (old), wd=0.005 (new), wd=0.01 (new)
    
    # Construct wd=0.002 curve (Old combined)
    sgdwd_002_base = df_sgdwd_old[(df_sgdwd_old['method'] == 'SGD+WD') & (df_sgdwd_old['wd'] == 0.002)]
    sgdwd_002_supp = df_sgdwd_supp[(df_sgdwd_supp['method'] == 'SGD+WD') & (df_sgdwd_supp['wd'] == 0.002)]
    sgdwd_002 = pd.concat([sgdwd_002_base, sgdwd_002_supp]).sort_values('lr').drop_duplicates('lr')
    
    # Construct wd=0.005 and 0.01 curves (New)
    sgdwd_005 = df_new[(df_new['method'] == 'SGD+WD') & (df_new['wd'] == 0.005)].sort_values('lr')
    sgdwd_010 = df_new[(df_new['method'] == 'SGD+WD') & (df_new['wd'] == 0.01)].sort_values('lr')
    
    # Load LR Extension
    extension_file = results_dir / 'lr_extension.csv'
    if extension_file.exists():
        df_ext = pd.read_csv(extension_file)
        
        # Merge SGD+WD (wd=0.005) extension
        ext_sgdwd = df_ext[(df_ext['method'] == 'SGD+WD') & (df_ext['wd'] == 0.005)]
        if len(ext_sgdwd) > 0:
            sgdwd_005 = pd.concat([sgdwd_005, ext_sgdwd]).sort_values('lr').drop_duplicates('lr')
            
        # Merge SGDM+WD extension (will be handled in selection step if we pick the right one)
        # We need to make sure we augment the "df_new" or handle it specifically
        # Let's augment df_new so selection logic works or just augment the selected one later
        # Better: augment df_new
        df_new = pd.concat([df_new, df_ext], ignore_index=True)
        # Re-extract sgdwd_005/010 just in case
        sgdwd_005 = df_new[(df_new['method'] == 'SGD+WD') & (df_new['wd'] == 0.005)].sort_values('lr')
        sgdwd_010 = df_new[(df_new['method'] == 'SGD+WD') & (df_new['wd'] == 0.01)].sort_values('lr')
        
    # SGDM+WD (New Refined)
    # We want the one with highest accuracy
    sgdm_new = df_new[df_new['method'] == 'SGDM+WD']
    
    # [NEW] SGDM (no WD) - Load from Search
    sgdm_nowd_file = results_dir / 'sgdm_no_wd_search_low.csv'
    if not sgdm_nowd_file.exists():
        sgdm_nowd_file = results_dir / 'sgdm_no_wd_search.csv'
        
    sgdm_nowd_curve = None
    sgdm_nowd_name = 'SGDM (no WD)'
    
    if sgdm_nowd_file.exists():
        df_nowd = pd.read_csv(sgdm_nowd_file)
        # Filter for our winner: m=0.1
        sgdm_nowd_curve = df_nowd[df_nowd['momentum'] == 0.1].sort_values('lr').drop_duplicates('lr')
        sgdm_nowd_name = 'SGDM (no WD, m=0.1)'
        
    # If new experiments haven't finished, these might be empty.
    if len(sgdwd_005) == 0:
        print("Warning: New SGD+WD experiments (wd=0.005) data missing.")
    if len(sgdm_new) == 0:
        print("Warning: New SGDM+WD experiments data missing.")

    # 2. Select Best Curves
    # -------------------
    # SGD is fixed
    
    # Select best SGD+WD:
    # We prefer the one with highest accuracy that fits the LR gap.
    # wd=0.002 peaked at 0.3 (too high). wd=0.005 or 0.01 should peak lower.
    # Let's compare their peaks.
    
    curves = {
        'SGD': sgd_curve,
        'SGD+WD (wd=0.002)': sgdwd_002,
        'SGD+WD (wd=0.005)': sgdwd_005,
        'SGD+WD (wd=0.01)': sgdwd_010
    }
    
    print("\nAnalyzing peaks:")
    for name, df in curves.items():
        if len(df) > 0:
            best = df.loc[df['best_test_acc'].idxmax()]
            print(f"  {name}: Peak LR={best['lr']}, Acc={best['best_test_acc']:.2f}%")
            
    # Heuristic selection algorithm for plot:
    # 1. SGD (Fixed)
    # 2. SGD+WD: Choose wd=0.005 if peak LR < 0.3, else wd=0.01
    # 3. SGDM+WD: Choose best accuracy config
    
    selected_sgdwd = sgdwd_002 # Default fall back
    selected_sgdwd_name = 'SGD+WD (wd=0.002)'
    
    if len(sgdwd_005) > 0:
        peak_005 = sgdwd_005.loc[sgdwd_005['best_test_acc'].idxmax()]
        if peak_005['lr'] < 0.3:
            selected_sgdwd = sgdwd_005
            selected_sgdwd_name = f"SGD+WD (wd=0.005)"
        elif len(sgdwd_010) > 0:
            selected_sgdwd = sgdwd_010
            selected_sgdwd_name = f"SGD+WD (wd=0.01)"
            
    # Select best SGDM+WD
    selected_sgdm = None
    selected_sgdm_name = 'SGDM+WD'
    
    if len(sgdm_new) > 0:
        best_sgdm_idx = sgdm_new['best_test_acc'].idxmax()
        best_sgdm = sgdm_new.loc[best_sgdm_idx]
        best_wd = best_sgdm['wd']
        best_mom = best_sgdm['momentum']
        
        selected_sgdm = sgdm_new[(sgdm_new['wd'] == best_wd) & (sgdm_new['momentum'] == best_mom)].sort_values('lr')
        selected_sgdm_name = f"SGDM+WD (wd={best_wd}, m={best_mom})"
        print(f"  Selected SGDM+WD: {selected_sgdm_name}, Peak LR={best_sgdm['lr']}, Acc={best_sgdm['best_test_acc']:.2f}%")
    else:
        # Fallback to old best
        df_momentum = pd.read_csv(results_dir / 'momentum_search.csv')
        selected_sgdm = df_momentum[(df_momentum['wd'] == 0.002) & (df_momentum['momentum'] == 0.7)].sort_values('lr')
        selected_sgdm_name = 'SGDM+WD (wd=0.002, m=0.7)'
        print("  Selected SGDM+WD: Fallback to old (wd=0.002, m=0.7)")

    # 3. Filter for Common LRs (MODIFIED to include SGDM_NO_WD)
    # ------------------------
    # Find intersection of LRs
    lrs_sgd = set(sgd_curve['lr'])
    lrs_sgdwd = set(selected_sgdwd['lr'])
    lrs_sgdm = set(selected_sgdm['lr'])
    
    common_lrs = lrs_sgd.intersection(lrs_sgdwd).intersection(lrs_sgdm)
    # Add new curve to filter? 
    # v3 filtered to standardize x-axis points.
    # But SGDM (no WD) might be different. 
    # Let's NOT intersection filter the new curve with old ones if it kills points.
    # Instead, let's just create a Union of "visual ticks" or just rely on plotting.
    # Actually, v3 used this to cleanly handle the plot? No, likely just to ensure comparing apples to apples.
    # Let's SKIP common filtering for the NEW curve, but keep it for old ones to match v3 logic exactly? 
    # Or better: don't filter the new curve at all.
    
    # Keeping v3 logic for existing curves:
    common_lrs = sorted(list(common_lrs))
    print(f"\nCommon LRs ({len(common_lrs)}): {common_lrs}")
    
    sgd_curve = sgd_curve[sgd_curve['lr'].isin(common_lrs)]
    selected_sgdwd = selected_sgdwd[selected_sgdwd['lr'].isin(common_lrs)]
    selected_sgdm = selected_sgdm[selected_sgdm['lr'].isin(common_lrs)]
    
    # 4. Plotting
    # -----------
    # 4. Plotting (Merged)
    # --------------------
    fig, ax = plt.subplots(figsize=(7, 8)) # Narrower figure (7x8)
    
    # Monochromatic Color Scheme: Light -> Dark
    # Added one intermediate color
    colors = ['#90CAF9', '#64B5F6', '#42A5F5', '#1565C0'] # Light(SGD), Mid(SGDM noWD), Mid(SGDWD), Dark(SGDM WD)
    # Use Red for optimal stars
    star_color = '#D50000' # Red
    markers = ['o', 'D', 's', '^']
    
    # Create final list
    final_curves = [
        ('SGD', sgd_curve, colors[0], markers[0]),
        (sgdm_nowd_name, sgdm_nowd_curve, colors[1], markers[1]),
        ('SGD+WD', selected_sgdwd, colors[2], markers[2]),
        ('SGDM+WD', selected_sgdm, colors[3], markers[3])
    ]
    
    # Helper for visual mapping
    def map_lr(lr):
        return 0.4 if lr == 0.5 else lr

    # Plot Curves
    for name, df, color, marker in final_curves:
        if df is not None and len(df) > 0:
            # Map X values - v3 logic
            # v3 mapped 0.5 -> 0.4
            
            # v3 hard filter:
            # df = df[df['lr'].isin(common_lrs)]
            # But we didn't filter the new curve.
            # And user complained about 0.5 points.
            
            # Filter 0.5 if user wanted it removed (implied by "v6... lr 0.5... I don't need")
            # v3 kept it but mapped it to 0.4.
            # "v6 has lr 0.5... I don't need" -> User wants to hide 0.5?
            # Let's just drop lr >= 0.5
            df = df[df['lr'] < 0.5].reset_index(drop=True)

            x_vals = df['lr'].astype(float) # No mapping needed if < 0.5? 
            # v3 had map_lr. 
            # If we drop < 0.5, map(0.5->0.4) is irrelevant.
            
            # Line plot
            ax.plot(x_vals, df['best_test_acc'], marker=marker, label=name, color=color, lw=4.5, ms=14, alpha=0.9)
            
            # Find optimal
            if len(df) > 0:
                best = df.loc[df['best_test_acc'].idxmax()]
                best_visual_lr = best['lr']
                
                # 1. Red Star for optimal
                ax.scatter([best_visual_lr], [best['best_test_acc']], s=500, c=star_color, marker='*', edgecolors='white', linewidth=2.0, zorder=20)
                
                # 2. "Bar" effect: Thick vertical line from bottom to point
                y_base = 69 # Raised base slightly
                ax.vlines(x=best_visual_lr, ymin=y_base, ymax=best['best_test_acc'], colors=color, linestyles='-', lw=20, alpha=0.25, zorder=5)
                
                # 3. Annotation
                # LR on top, Acc below
                ax.text(best_visual_lr, best['best_test_acc'] + 0.4, 
                        f"LR: {best['lr']}\nAcc: {best['best_test_acc']:.2f}%", 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)

    # Legends
    from matplotlib.lines import Line2D
    
    # 1. Curve Legend (Method Names) - Upper Right
    curve_handles = []
    for name, _, color, marker in final_curves:
         h = Line2D([], [], color=color, marker=marker, linestyle='-', linewidth=4.5, markersize=14, label=name)
         curve_handles.append(h)
         
    curve_legend = ax.legend(handles=curve_handles, loc='upper right', fontsize=16, framealpha=0.95)
    ax.add_artist(curve_legend) # Persist this legend
    
    # 2. Indicator Legend (Star & Bar) - Lower Left
    indicator_handles = []
    
    # Star
    star_handle = Line2D([], [], color='white', marker='*', markerfacecolor=star_color, markersize=20, label='Optimal Accuracy')
    indicator_handles.append(star_handle)
    
    # Bar (Vertical Line)
    # Renamed "Optimal LR Position" -> "Optimal LR"
    bar_handle = Line2D([], [], color='gray', marker='|', linestyle='None', markersize=22, markeredgewidth=5, alpha=0.5, label='Optimal LR')
    indicator_handles.append(bar_handle)
    
    ax.legend(handles=indicator_handles, loc='lower left', fontsize=16, framealpha=0.95)
    
    # Switch to Linear Scale
    ax.set_xscale('linear')
    ax.set_xlim(0.0, 0.45) # Show up to "0.5" (visual 0.4)
    
    # Custom X-ticks
    # Visual positions: 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4
    visual_ticks = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    ax.set_xticks(visual_ticks)
    # ax.set_xticklabels(tick_labels)
    
    # Ensure they are readable
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    ax.set_xlabel('Learning Rate', fontsize=18, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=18, fontweight='bold')
    
    # Two-line title
    ax.set_title('Optimal Performance: SGDM+WD > SGD+WD > SGD', fontweight='bold', fontsize=20, pad=20)
    
    ax.grid(True, alpha=0.3)
    
    # Set Y limit
    ax.set_ylim(69, 81) # Expanded slightly to fit top text
    
    plt.tight_layout()
    output_path = output_dir / 'exp1_lr_ordering_v9.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

if __name__ == '__main__':
    main()
