"""
SGDM+WD 扩展实验可视化
分析 Weight Decay 与 Learning Rate 的最优配置关系
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

# 设置绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 读取数据
data_path = Path('outputs/results/sgdm_extended.csv')
df = pd.read_csv(data_path)

# 创建输出目录
output_dir = Path('outputs/plots')
output_dir.mkdir(parents=True, exist_ok=True)

# 定义颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
wd_values = sorted(df['wd'].unique())
wd_labels = ['1e-4', '5e-4', '1e-3', '2e-3', '5e-3']

# ============================================================
# 图1: 不同 WD 下 LR vs Accuracy 曲线
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

for i, wd in enumerate(wd_values):
    subset = df[df['wd'] == wd].sort_values('lr')
    ax1.plot(subset['lr'], subset['best_test_acc'], 
             marker='o', linewidth=2, markersize=8,
             color=colors[i], label=f'WD={wd_labels[i]}')
    
    # 标记最优点
    best_idx = subset['best_test_acc'].idxmax()
    best_row = subset.loc[best_idx]
    ax1.scatter(best_row['lr'], best_row['best_test_acc'], 
                s=150, color=colors[i], edgecolors='black', 
                linewidth=2, zorder=5)

ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Best Test Accuracy (%)')
ax1.set_title('SGDM+WD: Impact of Weight Decay on Optimal Learning Rate')
ax1.legend(title='Weight Decay', loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_xlim(0.008, 0.12)
ax1.set_ylim(66, 80)

plt.tight_layout()
fig1.savefig(output_dir / 'sgdm_wd_lr_curves.png')
print(f"保存: {output_dir / 'sgdm_wd_lr_curves.png'}")

# ============================================================
# 图2: 热力图 - LR vs WD 的 Accuracy
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# 创建透视表
pivot = df.pivot_table(values='best_test_acc', 
                       index='wd', columns='lr', aggfunc='mean')

# 绘制热力图
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
            linewidths=0.5, ax=ax2, cbar_kws={'label': 'Test Accuracy (%)'})

# 设置标签
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Weight Decay')
ax2.set_title('SGDM+WD: Accuracy Heatmap (LR × WD)')

# 使用科学计数法显示 WD
ax2.set_yticklabels([f'{wd:.0e}' for wd in sorted(df['wd'].unique())])

plt.tight_layout()
fig2.savefig(output_dir / 'sgdm_wd_lr_heatmap.png')
print(f"保存: {output_dir / 'sgdm_wd_lr_heatmap.png'}")

# ============================================================
# 图3: 最优 LR 随 WD 变化的趋势
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

# 找出每个 WD 的最优配置
optimal_configs = []
for wd in wd_values:
    subset = df[df['wd'] == wd]
    best_idx = subset['best_test_acc'].idxmax()
    best_row = subset.loc[best_idx]
    optimal_configs.append({
        'wd': wd,
        'best_lr': best_row['lr'],
        'best_acc': best_row['best_test_acc']
    })

opt_df = pd.DataFrame(optimal_configs)

# 左图: 最优 LR 随 WD 变化
ax3a.plot(opt_df['wd'], opt_df['best_lr'], 'bo-', linewidth=2, markersize=10)
ax3a.set_xscale('log')
ax3a.set_xlabel('Weight Decay')
ax3a.set_ylabel('Optimal Learning Rate')
ax3a.set_title('Optimal LR vs Weight Decay')
ax3a.grid(True, alpha=0.3)

# 添加数值标注
for _, row in opt_df.iterrows():
    ax3a.annotate(f'{row["best_lr"]:.2f}', 
                  (row['wd'], row['best_lr']),
                  textcoords="offset points", xytext=(0, 10),
                  ha='center', fontsize=10, fontweight='bold')

# 右图: 最佳精度随 WD 变化
ax3b.bar(range(len(opt_df)), opt_df['best_acc'], color=colors)
ax3b.set_xticks(range(len(opt_df)))
ax3b.set_xticklabels(wd_labels)
ax3b.set_xlabel('Weight Decay')
ax3b.set_ylabel('Best Test Accuracy (%)')
ax3b.set_title('Best Achievable Accuracy per WD')
ax3b.set_ylim(72, 80)

# 标注最佳值
best_wd_idx = opt_df['best_acc'].idxmax()
bars = ax3b.patches
bars[best_wd_idx].set_edgecolor('red')
bars[best_wd_idx].set_linewidth(3)

for i, (bar, acc) in enumerate(zip(bars, opt_df['best_acc'])):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig3.savefig(output_dir / 'sgdm_optimal_config_trend.png')
print(f"保存: {output_dir / 'sgdm_optimal_config_trend.png'}")

# ============================================================
# 图4: 3D 表面图
# ============================================================
fig4 = plt.figure(figsize=(12, 8))
ax4 = fig4.add_subplot(111, projection='3d')

# 准备网格数据
lr_unique = sorted(df['lr'].unique())
wd_unique = sorted(df['wd'].unique())
LR, WD = np.meshgrid(lr_unique, wd_unique)

# 创建 Accuracy 矩阵
Z = np.zeros_like(LR)
for i, wd in enumerate(wd_unique):
    for j, lr in enumerate(lr_unique):
        row = df[(df['wd'] == wd) & (df['lr'] == lr)]
        if len(row) > 0:
            Z[i, j] = row['best_test_acc'].values[0]

# 绘制 3D 表面
surf = ax4.plot_surface(np.log10(LR), np.log10(WD), Z, 
                        cmap='viridis', alpha=0.8,
                        linewidth=0, antialiased=True)

ax4.set_xlabel('log₁₀(LR)')
ax4.set_ylabel('log₁₀(WD)')
ax4.set_zlabel('Test Accuracy (%)')
ax4.set_title('SGDM+WD: Accuracy Landscape')

# 添加颜色条
fig4.colorbar(surf, shrink=0.5, aspect=10, label='Accuracy (%)')

plt.tight_layout()
fig4.savefig(output_dir / 'sgdm_3d_surface.png')
print(f"保存: {output_dir / 'sgdm_3d_surface.png'}")

# ============================================================
# 输出汇总信息
# ============================================================
print("\n" + "="*60)
print("实验结果汇总")
print("="*60)
print("\n各 WD 值的最优配置:")
for _, row in opt_df.iterrows():
    wd_str = f"{row['wd']:.0e}"
    print(f"  WD={wd_str}: 最优LR={row['best_lr']:.3f}, Acc={row['best_acc']:.2f}%")

print(f"\n全局最优配置:")
global_best = opt_df.loc[opt_df['best_acc'].idxmax()]
print(f"  WD={global_best['wd']:.0e}, LR={global_best['best_lr']:.3f}, Acc={global_best['best_acc']:.2f}%")

print(f"\n关键发现:")
print(f"  1. WD 增大时，最优 LR 先增后减")
print(f"  2. 小 WD (1e-4) 需要较高 LR (0.05) 获得最佳效果")
print(f"  3. 大 WD (5e-3) 需要较低 LR (0.02) 才能收敛")
print(f"  4. WD=2e-3 达到全局最佳精度 {global_best['best_acc']:.2f}%")

plt.show()
