"""
改进实验的可视化脚本
针对三个核心假设生成验证图表
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_exp1_lr_ordering(df, output_dir):
    """
    实验一：学习率排序验证图
    目标：验证 η_SGD > η_SGD+WD > η_SGDM+WD (稳定性边界)
    """
    # 筛选实验一的数据
    methods = ['SGD', 'SGD+WD', 'SGDM+WD']
    exp1_df = df[df['method'].isin(methods) & (df['batch_size'] == 128)]

    if exp1_df.empty:
        print("No data for Experiment 1")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'SGD': '#2196F3', 'SGD+WD': '#4CAF50', 'SGDM+WD': '#FF5722'}
    markers = {'SGD': 'o', 'SGD+WD': 's', 'SGDM+WD': '^'}

    # 图1: Test Accuracy vs LR
    ax1 = axes[0]
    for method in methods:
        subset = exp1_df[exp1_df['method'] == method].sort_values('lr')
        if not subset.empty:
            ax1.plot(subset['lr'], subset['best_test_acc'],
                     marker=markers[method], label=method, color=colors[method],
                     linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax1.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Test Accuracy vs Learning Rate', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 图2: 稳定性分析 (Train Loss 作为指标)
    ax2 = axes[1]
    for method in methods:
        subset = exp1_df[exp1_df['method'] == method].sort_values('lr')
        if not subset.empty:
            ax2.plot(subset['lr'], subset['final_train_loss'],
                     marker=markers[method], label=method, color=colors[method],
                     linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax2.set_ylabel('Final Train Loss (log scale)', fontsize=12)
    ax2.set_title('(b) Train Loss vs Learning Rate\n(发散时loss急剧上升)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    # 标记稳定性边界
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Divergence threshold')

    # 图3: 最优学习率对比
    ax3 = axes[2]
    optimal_lrs = []
    optimal_accs = []
    stability_boundaries = []

    for method in methods:
        subset = exp1_df[exp1_df['method'] == method]
        if not subset.empty:
            # 最优泛化学习率
            best_row = subset.loc[subset['best_test_acc'].idxmax()]
            optimal_lrs.append(best_row['lr'])
            optimal_accs.append(best_row['best_test_acc'])

            # 稳定性边界 (train_loss < 1.0 的最大lr)
            stable = subset[subset['final_train_loss'] < 1.0]
            if not stable.empty:
                stability_boundaries.append(stable['lr'].max())
            else:
                stability_boundaries.append(0)

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(x - width/2, optimal_lrs, width, label='最优泛化LR', color='steelblue')
    bars2 = ax3.bar(x + width/2, stability_boundaries, width, label='稳定性边界', color='coral')

    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('(c) 最优LR vs 稳定性边界对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.set_yscale('log')

    # 添加数值标签
    for bar, val in zip(bars1, optimal_lrs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, stability_boundaries):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = Path(output_dir) / 'exp1_lr_ordering_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # 打印统计信息
    print("\n实验一结果统计:")
    print("-" * 60)
    for i, method in enumerate(methods):
        print(f"{method:12s}: 最优LR={optimal_lrs[i]:.3f}, 稳定性边界≈{stability_boundaries[i]:.1f}, Best Acc={optimal_accs[i]:.2f}%")

    # 验证假设
    print("\n假设验证:")
    if stability_boundaries[0] >= stability_boundaries[1] >= stability_boundaries[2]:
        print("✓ 稳定性边界排序: SGD ≥ SGD+WD ≥ SGDM+WD (符合理论预期)")
    else:
        print("✗ 稳定性边界排序不符合预期，需要进一步分析")


def plot_exp2_eta_lambda_heatmap(df, output_dir):
    """
    实验二：η-λ 反比例关系热力图
    理论预期：高准确率区域应呈现 η × λ ≈ const 的反比例曲线
    """
    exp2_df = df[df['method'] == 'SGDM'].copy()

    if exp2_df.empty:
        print("No data for Experiment 2")
        return

    # 创建透视表
    pivot = exp2_df.pivot_table(
        values='best_test_acc',
        index='wd',
        columns='lr',
        aggfunc='mean'
    )

    if pivot.empty:
        print("No pivot data for Experiment 2")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 热力图
    ax1 = axes[0]
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1,
                cbar_kws={'label': 'Test Accuracy (%)'})
    ax1.set_xlabel('Learning Rate (η)', fontsize=12)
    ax1.set_ylabel('Weight Decay (λ)', fontsize=12)
    ax1.set_title('(a) η-λ 交互热力图\n高准确率区域应呈反比例曲线', fontsize=14, fontweight='bold')

    # 图2: η×λ 分析
    ax2 = axes[1]
    exp2_df['eta_lambda'] = exp2_df['lr'] * exp2_df['wd']

    # 按 η×λ 分组取平均
    grouped = exp2_df.groupby('eta_lambda').agg({
        'best_test_acc': ['mean', 'std', 'count']
    }).reset_index()
    grouped.columns = ['eta_lambda', 'acc_mean', 'acc_std', 'count']

    ax2.scatter(grouped['eta_lambda'], grouped['acc_mean'],
                s=grouped['count']*20, alpha=0.7, c='steelblue')
    ax2.set_xscale('log')
    ax2.set_xlabel('η × λ (有效正则化强度)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('(b) 准确率 vs η×λ\n应存在最优的η×λ值', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 标记最优点
    best_idx = grouped['acc_mean'].idxmax()
    best_eta_lambda = grouped.loc[best_idx, 'eta_lambda']
    best_acc = grouped.loc[best_idx, 'acc_mean']
    ax2.scatter([best_eta_lambda], [best_acc], s=200, c='red', marker='*',
                label=f'最优: η×λ={best_eta_lambda:.2e}')
    ax2.legend()

    plt.tight_layout()
    output_path = Path(output_dir) / 'exp2_eta_lambda_heatmap_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # 找出每个lr对应的最优wd
    print("\n实验二结果统计:")
    print("-" * 60)
    print("每个学习率对应的最优Weight Decay:")
    for lr in sorted(exp2_df['lr'].unique()):
        subset = exp2_df[exp2_df['lr'] == lr]
        best_row = subset.loc[subset['best_test_acc'].idxmax()]
        print(f"  lr={lr:.3f} -> optimal_wd={best_row['wd']:.2e}, acc={best_row['best_test_acc']:.2f}%")

    print("\n假设验证:")
    # 检查是否满足反比例关系
    lr_optimal_wd = []
    for lr in sorted(exp2_df['lr'].unique()):
        subset = exp2_df[exp2_df['lr'] == lr]
        best_row = subset.loc[subset['best_test_acc'].idxmax()]
        lr_optimal_wd.append((lr, best_row['wd']))

    # 检查趋势：lr增大时，optimal_wd是否减小
    increasing_lr = [x[0] for x in lr_optimal_wd]
    optimal_wds = [x[1] for x in lr_optimal_wd]

    # 计算相关性
    if len(increasing_lr) > 2:
        correlation = np.corrcoef(np.log(increasing_lr), np.log(optimal_wds))[0, 1]
        print(f"log(η) vs log(λ_optimal) 相关系数: {correlation:.3f}")
        if correlation < -0.3:
            print("✓ η 与 λ 呈负相关 (符合反比例假设)")
        else:
            print("✗ 未观察到明显的反比例关系")


def plot_exp3_batch_lambda(df, output_dir):
    """
    实验三：B-λ 正比例关系验证
    理论预期：最优 λ 与 batch size B 成正比
    """
    exp3_df = df[df['method'] == 'SGDM'].copy()

    if exp3_df.empty:
        print("No data for Experiment 3")
        return

    batch_sizes = sorted(exp3_df['batch_size'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 图1: 每个batch size的最优wd
    ax1 = axes[0]
    optimal_wds = []
    optimal_accs = []

    for bs in batch_sizes:
        subset = exp3_df[exp3_df['batch_size'] == bs]
        if not subset.empty:
            best_row = subset.loc[subset['best_test_acc'].idxmax()]
            optimal_wds.append(best_row['wd'])
            optimal_accs.append(best_row['best_test_acc'])
        else:
            optimal_wds.append(np.nan)
            optimal_accs.append(np.nan)

    ax1.plot(batch_sizes, optimal_wds, 'o-', markersize=10, linewidth=2, color='steelblue')
    ax1.set_xlabel('Batch Size (B)', fontsize=12)
    ax1.set_ylabel('Optimal Weight Decay (λ)', fontsize=12)
    ax1.set_title('(a) Optimal λ vs Batch Size\n理论预期: 正比例关系', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 添加理论预期线 (λ ∝ B)
    if len(batch_sizes) > 1 and not np.isnan(optimal_wds[0]):
        base_bs = batch_sizes[0]
        base_wd = optimal_wds[0]
        theoretical_wds = [base_wd * (bs / base_bs) for bs in batch_sizes]
        ax1.plot(batch_sizes, theoretical_wds, '--', color='red',
                label='理论预期 (λ ∝ B)', linewidth=2, alpha=0.7)
        ax1.legend()

    # 图2: 每个batch size的accuracy vs wd曲线
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

    for i, bs in enumerate(batch_sizes):
        subset = exp3_df[exp3_df['batch_size'] == bs].sort_values('wd')
        if not subset.empty:
            ax2.plot(subset['wd'], subset['best_test_acc'],
                    'o-', label=f'B={bs}', color=colors[i], markersize=6)

    ax2.set_xscale('log')
    ax2.set_xlabel('Weight Decay (λ)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Test Acc vs λ (不同Batch Size)', fontsize=14, fontweight='bold')
    ax2.legend(title='Batch Size')
    ax2.grid(True, alpha=0.3)

    # 图3: λ/B 归一化分析
    ax3 = axes[2]
    exp3_df['lambda_over_B'] = exp3_df['wd'] / exp3_df['batch_size']

    for i, bs in enumerate(batch_sizes):
        subset = exp3_df[exp3_df['batch_size'] == bs].sort_values('lambda_over_B')
        if not subset.empty:
            ax3.plot(subset['lambda_over_B'], subset['best_test_acc'],
                    'o-', label=f'B={bs}', color=colors[i], markersize=6)

    ax3.set_xscale('log')
    ax3.set_xlabel('λ/B (归一化Weight Decay)', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('(c) Test Acc vs λ/B\n如果 λ∝B, 曲线应重合', fontsize=14, fontweight='bold')
    ax3.legend(title='Batch Size')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'exp3_batch_lambda_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # 打印统计信息
    print("\n实验三结果统计:")
    print("-" * 60)
    print("每个Batch Size对应的最优Weight Decay:")
    for i, bs in enumerate(batch_sizes):
        if not np.isnan(optimal_wds[i]):
            print(f"  B={bs:4d} -> optimal_λ={optimal_wds[i]:.2e}, acc={optimal_accs[i]:.2f}%")

    print("\n假设验证:")
    # 检查正比例关系
    valid_data = [(bs, wd) for bs, wd in zip(batch_sizes, optimal_wds) if not np.isnan(wd)]
    if len(valid_data) >= 3:
        bs_arr = np.array([x[0] for x in valid_data])
        wd_arr = np.array([x[1] for x in valid_data])
        correlation = np.corrcoef(np.log(bs_arr), np.log(wd_arr))[0, 1]
        print(f"log(B) vs log(λ_optimal) 相关系数: {correlation:.3f}")
        if correlation > 0.5:
            print("✓ B 与 λ 呈正相关 (符合正比例假设)")
        else:
            print("✗ 未观察到明显的正比例关系")

    # 计算比例系数
    if len(valid_data) >= 2:
        ratios = [wd / bs for bs, wd in valid_data]
        print(f"λ/B 比值: {[f'{r:.2e}' for r in ratios]}")
        print(f"λ/B 平均值: {np.mean(ratios):.2e}, 标准差: {np.std(ratios):.2e}")


def main():
    parser = argparse.ArgumentParser(description='生成改进实验的可视化图表')
    parser.add_argument('--input', type=str, default='outputs/results/results_v2.csv',
                        help='输入CSV文件')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                        help='输出目录')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], default=None,
                        help='只绘制指定实验的图表')
    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} results from {args.input}")
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        return

    # 生成图表
    if args.experiment is None or args.experiment == 1:
        plot_exp1_lr_ordering(df, output_dir)

    if args.experiment is None or args.experiment == 2:
        plot_exp2_eta_lambda_heatmap(df, output_dir)

    if args.experiment is None or args.experiment == 3:
        plot_exp3_batch_lambda(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
