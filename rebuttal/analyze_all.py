"""Compute mean ± half-range for all 4 experiment groups (2 models × 2 seeds)."""
import pandas as pd
import numpy as np

# Load all data
r18_s42 = pd.read_csv('outputs/results/results.csv')
r18_s123 = pd.read_csv('rebuttal/results/results_resnet18_seed123.csv')
v16_s42 = pd.read_csv('rebuttal/results/results_vgg16_seed42.csv')
v16_s123 = pd.read_csv('rebuttal/results/results_vgg16_seed123.csv')

def fmt(a, b):
    m = (a + b) / 2
    h = abs(a - b) / 2
    return f"{m:.2f} ± {h:.2f}"

def get_exp1(df, method):
    if method == 'SGD':
        return df[(df['method']=='SGD') & (df['wd']==0) & (df['momentum']==0) & (df['batch_size']==128)]
    elif method == 'SGD+WD':
        return df[(df['method']=='SGD+WD') & (df['batch_size']==128)]
    else:
        return df[(df['method']=='SGDM+WD') & (df['batch_size']==128)]

lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
lr_vals2 = [0.01, 0.05, 0.1, 0.2, 0.3]
wd_vals2 = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
bss = [64, 128, 256, 512]
wd_vals3 = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

SEP = "=" * 80

# =================== VGG-16 REPORT ===================
print(SEP)
print("VGG-16 RESULTS: seed=42 vs seed=123")
print(SEP)

# --- EXP1 ---
print(f"\n--- Exp1: Stability Boundary Ordering ---")
methods = ['SGD', 'SGD+WD', 'SGDM+WD']
for mname in methods:
    s42 = get_exp1(v16_s42, mname)
    s123 = get_exp1(v16_s123, mname)
    print(f"\n  {mname}:")
    for lr in lrs:
        r42 = s42[s42['lr']==lr]
        r123 = s123[s123['lr']==lr]
        if len(r42) > 0 and len(r123) > 0:
            print(f"    LR={lr}: {fmt(r42['best_test_acc'].values[0], r123['best_test_acc'].values[0])}")
    b42 = s42.loc[s42['best_test_acc'].idxmax()]
    b123 = s123.loc[s123['best_test_acc'].idxmax()]
    print(f"    Peak: {fmt(b42['best_test_acc'], b123['best_test_acc'])} (η*: s42={b42['lr']}, s123={b123['lr']})")

# --- EXP2 ---
print(f"\n--- Exp2: η-λ Heatmap ---")
v_sgdm42 = v16_s42[(v16_s42['momentum']==0.9) & (v16_s42['batch_size']==128)]
v_sgdm123 = v16_s123[(v16_s123['momentum']==0.9) & (v16_s123['batch_size']==128)]

for lr in lr_vals2:
    row = f"  η={lr}:"
    for wd in wd_vals2:
        m42 = v_sgdm42[(abs(v_sgdm42['lr']-lr)<1e-6) & (abs(v_sgdm42['wd']-wd)<1e-6)]
        m123 = v_sgdm123[(abs(v_sgdm123['lr']-lr)<1e-6) & (abs(v_sgdm123['wd']-wd)<1e-6)]
        if len(m42)>0 and len(m123)>0:
            row += f"  {fmt(m42['best_test_acc'].values[0], m123['best_test_acc'].values[0])}"
        else:
            row += "  ---"
    print(row)

print(f"\n  Optimal λ* per η:")
for lr in lr_vals2:
    bl = v_sgdm42[(abs(v_sgdm42['lr']-lr)<1e-6) & (v_sgdm42['wd'].isin(wd_vals2))]
    nl = v_sgdm123[(abs(v_sgdm123['lr']-lr)<1e-6) & (v_sgdm123['wd'].isin(wd_vals2))]
    if len(bl)>0 and len(nl)>0:
        bo = bl.loc[bl['best_test_acc'].idxmax()]
        no = nl.loc[nl['best_test_acc'].idxmax()]
        print(f"    η={lr}: λ*=({bo['wd']:.0e}, {no['wd']:.0e}), peak={fmt(bo['best_test_acc'], no['best_test_acc'])}")

# --- EXP3 ---
print(f"\n--- Exp3: Batch Size Scaling ---")
for bs in bss:
    lr = 0.1 * (bs / 128)
    print(f"  BS={bs}, η={lr:.3f}:")
    for wd in wd_vals3:
        m42 = v16_s42[(v16_s42['batch_size']==bs) & (abs(v16_s42['lr']-lr)<1e-6) & (abs(v16_s42['wd']-wd)<1e-6) & (v16_s42['momentum']==0.9)]
        m123 = v16_s123[(v16_s123['batch_size']==bs) & (abs(v16_s123['lr']-lr)<1e-6) & (abs(v16_s123['wd']-wd)<1e-6) & (v16_s123['momentum']==0.9)]
        if len(m42)>0 and len(m123)>0:
            print(f"    λ={wd:g}: {fmt(m42['best_test_acc'].values[0], m123['best_test_acc'].values[0])}")

print(f"\n  Optimal λ* per BS:")
for bs in bss:
    lr = 0.1*(bs/128)
    bl = v16_s42[(v16_s42['batch_size']==bs) & (abs(v16_s42['lr']-lr)<1e-6) & (v16_s42['momentum']==0.9) & (v16_s42['wd'].isin(wd_vals3))]
    nl = v16_s123[(v16_s123['batch_size']==bs) & (abs(v16_s123['lr']-lr)<1e-6) & (v16_s123['momentum']==0.9) & (v16_s123['wd'].isin(wd_vals3))]
    if len(bl)>0 and len(nl)>0:
        bo = bl.loc[bl['best_test_acc'].idxmax()]
        no = nl.loc[nl['best_test_acc'].idxmax()]
        print(f"    BS={bs}: λ*=({bo['wd']:.0e},{no['wd']:.0e}), peak={fmt(bo['best_test_acc'], no['best_test_acc'])}")

# =================== CROSS-MODEL: 4-run average ===================
print(f"\n{SEP}")
print("CROSS-MODEL SUMMARY: ResNet-18 vs VGG-16 (each with 2-seed mean)")
print(SEP)

def quad_fmt(a, b, c, d):
    """4-value: mean ± std"""
    vals = [a, b, c, d]
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

def dual_fmt(a, b):
    return f"{(a+b)/2:.2f} ± {abs(a-b)/2:.2f}"

# --- EXP1 Cross-model ---
print(f"\n--- Exp1: Peak Accuracy by Optimizer ---")
print(f"  {'Method':<10s} | {'ResNet-18 (2-seed)':>20s} | {'VGG-16 (2-seed)':>20s} | {'All 4 runs':>20s}")
print(f"  {'-'*10}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")
for mname in methods:
    r18a = get_exp1(r18_s42, mname)['best_test_acc'].max()
    r18b = get_exp1(r18_s123, mname)['best_test_acc'].max()
    v16a = get_exp1(v16_s42, mname)['best_test_acc'].max()
    v16b = get_exp1(v16_s123, mname)['best_test_acc'].max()
    print(f"  {mname:<10s} | {dual_fmt(r18a, r18b):>20s} | {dual_fmt(v16a, v16b):>20s} | {quad_fmt(r18a, r18b, v16a, v16b):>20s}")

# --- EXP2 Cross-model heatmap ---
print(f"\n--- Exp2: Cross-model Heatmap (4-run mean ± std) ---")
r_sgdm42 = r18_s42[(r18_s42['momentum']==0.9) & (r18_s42['batch_size']==128)]
r_sgdm123 = r18_s123[(r18_s123['momentum']==0.9) & (r18_s123['batch_size']==128)]

for lr in lr_vals2:
    row = f"  η={lr}:"
    for wd in wd_vals2:
        vals = []
        for df in [r_sgdm42, r_sgdm123, v_sgdm42, v_sgdm123]:
            m = df[(abs(df['lr']-lr)<1e-6) & (abs(df['wd']-wd)<1e-6)]
            if len(m)>0:
                vals.append(m['best_test_acc'].values[0])
        if len(vals)==4:
            row += f"  {np.mean(vals):.2f}±{np.std(vals):.2f}"
        else:
            row += "  ---"
    print(row)

# --- EXP2 optimal λ* cross-model ---
print(f"\n  Optimal λ* (4-run consensus):")
for lr in lr_vals2:
    opts = []
    for df_sgdm in [r_sgdm42, r_sgdm123, v_sgdm42, v_sgdm123]:
        sl = df_sgdm[(abs(df_sgdm['lr']-lr)<1e-6) & (df_sgdm['wd'].isin(wd_vals2))]
        if len(sl)>0:
            opts.append(sl.loc[sl['best_test_acc'].idxmax(), 'wd'])
    if opts:
        print(f"    η={lr}: λ* = {opts} → range [{min(opts):.0e}, {max(opts):.0e}]")

# --- EXP3 Cross-model ---
print(f"\n--- Exp3: Cross-model Batch Size Scaling ---")
print(f"  {'BS':>4s} {'η':>5s} | {'ResNet-18 (2-seed)':>20s} | {'VGG-16 (2-seed)':>20s} | {'All 4 runs':>20s}")
print(f"  {'-'*4} {'-'*5}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

for bs in bss:
    lr = 0.1*(bs/128)
    # find peak for each
    peaks = []
    for df in [r18_s42, r18_s123, v16_s42, v16_s123]:
        sub = df[(df['batch_size']==bs) & (abs(df['lr']-lr)<1e-6) & (df['momentum']==0.9) & (df['wd'].isin(wd_vals3))]
        if len(sub)>0:
            peaks.append(sub['best_test_acc'].max())
    if len(peaks)==4:
        print(f"  {bs:>4d} {lr:>5.3f} | {dual_fmt(peaks[0], peaks[1]):>20s} | {dual_fmt(peaks[2], peaks[3]):>20s} | {quad_fmt(*peaks):>20s}")

# --- EXP3 optimal λ* cross-model ---
print(f"\n  Optimal λ* (4-run):")
for bs in bss:
    lr = 0.1*(bs/128)
    opts = []
    for df in [r18_s42, r18_s123, v16_s42, v16_s123]:
        sub = df[(df['batch_size']==bs) & (abs(df['lr']-lr)<1e-6) & (df['momentum']==0.9) & (df['wd'].isin(wd_vals3))]
        if len(sub)>0:
            opts.append(sub.loc[sub['best_test_acc'].idxmax(), 'wd'])
    if opts:
        print(f"    BS={bs}: λ* = {opts} → range [{min(opts):.0e}, {max(opts):.0e}]")
