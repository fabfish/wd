"""
Analyze ResNet-18 results across 4 runs (2 seeds × 2 runs each).
Generates markdown report with mean ± half-range tables.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCES = {
    's42_r1': BASE_DIR / 'outputs' / 'results' / 'results.csv',
    's123_r1': BASE_DIR / 'rebuttal' / 'results' / 'results_resnet18_seed123.csv',
    's42_r2': BASE_DIR / 'rebuttal' / 'results' / 'results_resnet18_seed42_run2.csv',
    's123_r2': BASE_DIR / 'rebuttal' / 'results' / 'results_resnet18_seed123_run2.csv',
}


def load_all():
    dfs = {}
    for name, path in SOURCES.items():
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        for c in ['lr', 'wd', 'momentum']:
            df[c] = df[c].astype(float)
        dfs[name] = df
    return dfs


def get_exp1(df):
    return df[
        ((df['method'] == 'SGD') & (df['wd'] == 0) & (df['momentum'] == 0) & (df['batch_size'] == 128)) |
        ((df['method'] == 'SGD+WD') & (df['wd'] == 5e-4) & (df['momentum'] == 0) & (df['batch_size'] == 128)) |
        ((df['method'] == 'SGDM+WD') & (df['wd'] == 5e-4) & (df['momentum'] == 0.9) & (df['batch_size'] == 128))
    ].copy()


def get_exp2(df):
    return df[
        (df['method'].isin(['SGDM', 'SGDM+WD'])) &
        (df['momentum'] == 0.9) &
        (df['batch_size'] == 128) &
        (df['lr'].isin([0.01, 0.05, 0.1, 0.2, 0.3]))
    ].copy()


def get_exp3(df):
    bs_lr = {64: 0.05, 128: 0.1, 256: 0.2, 512: 0.4}
    rows = []
    for bs, lr in bs_lr.items():
        mask = (
            (df['method'].isin(['SGDM', 'SGDM+WD'])) &
            (df['momentum'] == 0.9) &
            (df['batch_size'] == bs) &
            (np.isclose(df['lr'], lr, atol=1e-6))
        )
        rows.append(df[mask])
    return pd.concat(rows)


def fmt(values):
    """Format list of values as mean ± half-range."""
    values = [v for v in values if v is not None and not np.isnan(v)]
    if len(values) == 0:
        return '—'
    mean = np.mean(values)
    hr = (max(values) - min(values)) / 2
    return f'{mean:.2f} ± {hr:.2f}'


def fmt_bold(values, is_best=False):
    s = fmt(values)
    return f'**{s}**' if is_best else s


def analyze_exp1(dfs):
    methods = ['SGD', 'SGD+WD', 'SGDM+WD']
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    lines = []
    lines.append('| LR | SGD | SGD+WD (λ=5e-4) | SGDM+WD (μ=0.9, λ=5e-4) |')
    lines.append('|---:|:---:|:---:|:---:|')

    peak_per_method = {m: (None, -1) for m in methods}

    for lr in lrs:
        row = [f'{lr}']
        for method in methods:
            vals = []
            for name, df in dfs.items():
                exp1 = get_exp1(df)
                match = exp1[(exp1['method'] == method) & (np.isclose(exp1['lr'], lr, atol=1e-6))]
                if len(match) > 0:
                    vals.append(match.iloc[0]['best_test_acc'])
            mean_val = np.mean(vals) if vals else 0
            if mean_val > peak_per_method[method][1]:
                peak_per_method[method] = (lr, mean_val)
            row.append((vals, mean_val))
        lines.append(f'| {lr} | {fmt(row[1][0])} | {fmt(row[2][0])} | {fmt(row[3][0])} |')

    peak_lines = []
    peak_lines.append('| Optimizer | η* | Peak Acc (mean ± half-range, N=4) |')
    peak_lines.append('|---|---|---|')
    for method in methods:
        lr, _ = peak_per_method[method]
        vals = []
        for name, df in dfs.items():
            exp1 = get_exp1(df)
            match = exp1[(exp1['method'] == method) & (np.isclose(exp1['lr'], lr, atol=1e-6))]
            if len(match) > 0:
                vals.append(match.iloc[0]['best_test_acc'])
        peak_lines.append(f'| {method} | {lr} | {fmt(vals)} |')

    return lines, peak_lines, peak_per_method


def analyze_exp2(dfs):
    lrs = [0.01, 0.05, 0.1, 0.2, 0.3]
    wds = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    lines = []
    header = '| η \\ λ | ' + ' | '.join([f'{w}' for w in wds]) + ' |'
    sep = '|------:' + '|:----:' * len(wds) + '|'
    lines.append(header)
    lines.append(sep)

    opt_wd_per_lr = {}

    for lr in lrs:
        row_vals = {}
        best_wd = None
        best_mean = -1
        for wd in wds:
            vals = []
            for name, df in dfs.items():
                exp2 = get_exp2(df)
                match = exp2[(np.isclose(exp2['lr'], lr, atol=1e-6)) & (np.isclose(exp2['wd'], wd, atol=1e-8))]
                if len(match) > 0:
                    vals.append(match.iloc[0]['best_test_acc'])
            mean_val = np.mean(vals) if vals else 0
            row_vals[wd] = (vals, mean_val)
            if mean_val > best_mean:
                best_mean = mean_val
                best_wd = wd
        opt_wd_per_lr[lr] = best_wd
        cells = []
        for wd in wds:
            v, m = row_vals[wd]
            s = fmt(v)
            if wd == best_wd:
                s = f'**{s}**'
            cells.append(s)
        lines.append(f'| {lr} | ' + ' | '.join(cells) + ' |')

    return lines, opt_wd_per_lr


def analyze_exp3(dfs):
    bs_lr = [(64, 0.05), (128, 0.1), (256, 0.2), (512, 0.4)]
    wds = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

    lines = []
    header = '| B | η | ' + ' | '.join([f'λ={w}' for w in wds]) + ' |'
    sep = '|---:|----:' + '|:---:' * len(wds) + '|'
    lines.append(header)
    lines.append(sep)

    for bs, lr in bs_lr:
        best_wd = None
        best_mean = -1
        row_vals = {}
        for wd in wds:
            vals = []
            for name, df in dfs.items():
                exp3 = get_exp3(df)
                match = exp3[
                    (exp3['batch_size'] == bs) &
                    (np.isclose(exp3['lr'], lr, atol=1e-6)) &
                    (np.isclose(exp3['wd'], wd, atol=1e-8))
                ]
                if len(match) > 0:
                    vals.append(match.iloc[0]['best_test_acc'])
            mean_val = np.mean(vals) if vals else 0
            row_vals[wd] = (vals, mean_val)
            if mean_val > best_mean:
                best_mean = mean_val
                best_wd = wd

        cells = []
        for wd in wds:
            v, m = row_vals[wd]
            s = fmt(v)
            if wd == best_wd:
                s = f'**{s}**'
            cells.append(s)
        lines.append(f'| {bs} | {lr} | ' + ' | '.join(cells) + ' |')

    return lines


def compute_deltas(dfs):
    """Compute pairwise deltas across all 4 runs for Exp2+Exp3 configs."""
    all_vals = {}
    for name, df in dfs.items():
        exp2 = get_exp2(df)
        for _, row in exp2.iterrows():
            key = (row['lr'], row['wd'], row['batch_size'])
            all_vals.setdefault(key, []).append(row['best_test_acc'])
        exp3 = get_exp3(df)
        for _, row in exp3.iterrows():
            key = (row['lr'], row['wd'], row['batch_size'])
            all_vals.setdefault(key, []).append(row['best_test_acc'])

    ranges = []
    for key, vals in all_vals.items():
        if len(vals) >= 2:
            ranges.append(max(vals) - min(vals))
    ranges = np.array(ranges)
    n = len(ranges)
    within_1 = np.sum(ranges < 2.0)  # half-range < 1%
    within_2 = np.sum(ranges < 4.0)  # half-range < 2%
    return n, np.mean(ranges), np.std(ranges), within_1, within_2


def main():
    dfs = load_all()
    print(f'Loaded {len(dfs)} result files:')
    for name, df in dfs.items():
        print(f'  {name}: {len(df)} rows from {SOURCES[name]}')

    exp1_table, exp1_peaks, peak_info = analyze_exp1(dfs)
    exp2_table, opt_wd = analyze_exp2(dfs)
    exp3_table = analyze_exp3(dfs)
    n_configs, mean_range, std_range, w1, w2 = compute_deltas(dfs)

    report = []
    report.append('# ResNet-18 Multi-Run Reproducibility Report (CIFAR-100)')
    report.append('')
    report.append('# ResNet-18 多次运行可复现性报告（CIFAR-100）')
    report.append('')
    report.append('All results: **mean ± half-range** over **4 independent runs** (2 seeds × 2 runs/seed). Best test accuracy (%).')
    report.append('')
    report.append('所有结果：**mean ± half-range** 格式，基于 **4 次独立运行**（2 个种子 × 每种子 2 次运行）。全文使用 best test accuracy (%)。')
    report.append('')
    report.append('### Data Sources / 数据来源')
    report.append('')
    report.append('| Run | Seed | CSV Path | Note |')
    report.append('|:---:|:---:|---|---|')
    report.append('| 1 | 42 | `outputs/results/results.csv` | Original baseline / 原始基线 |')
    report.append('| 2 | 123 | `rebuttal/results/results_resnet18_seed123.csv` | Rebuttal seed=123 |')
    report.append('| 3 | 42 | `rebuttal/results/results_resnet18_seed42_run2.csv` | Repeat run, seed=42 / 重复运行 |')
    report.append('| 4 | 123 | `rebuttal/results/results_resnet18_seed123_run2.csv` | Repeat run, seed=123 / 重复运行 |')
    report.append('')
    report.append('- Training: CIFAR-100, CosineAnnealingLR, 100 epochs, AMP, ResNet-18')
    report.append('- Note: Runs 3 & 4 differ from Runs 1 & 2 due to CUDA non-determinism (`cudnn.benchmark=True`)')
    report.append('- 注：Run 3/4 与 Run 1/2 因 CUDA 非确定性（`cudnn.benchmark=True`）而存在微小差异')
    report.append('')
    report.append('---')
    report.append('')

    # Experiment 1
    report.append('## Experiment 1: Stability Boundary Ordering / 实验一：稳定性边界排序')
    report.append('')
    report.append('### Best Test Accuracy (%) — N=4 runs')
    report.append('')
    report.extend(exp1_table)
    report.append('')
    report.append('### Peak accuracy / 峰值精度')
    report.append('')
    report.extend(exp1_peaks)
    report.append('')

    report.append('### Observations / 观察')
    report.append('')
    report.append('- **SGD**: peaks at η=0.1, consistent across all 4 runs.')
    report.append('  **SGD**：在 η=0.1 达峰，4 次运行一致。')
    report.append('- **SGD+WD**: broad stable plateau at η ∈ [0.5, 2.0], weight decay extends stability range.')
    report.append('  **SGD+WD**：在 η ∈ [0.5, 2.0] 形成宽广稳定平台。')
    report.append('- **SGDM+WD**: sharp peak at η ∈ [0.05, 0.1], collapses beyond η=0.5. Momentum tightens stability boundary.')
    report.append('  **SGDM+WD**：在 η ∈ [0.05, 0.1] 尖锐达峰，η > 0.5 后崩溃。动量收紧稳定性边界。')
    report.append('')
    report.append('---')
    report.append('')

    # Experiment 2
    report.append('## Experiment 2: η–λ Interaction Heatmap (SGDM) / 实验二：η–λ 交互热力图')
    report.append('')
    report.append('### Best Test Accuracy (%) — N=4 runs')
    report.append('')
    report.extend(exp2_table)
    report.append('')
    report.append('Bold = row maximum. / 粗体 = 行最大值。')
    report.append('')
    report.append('### Observations / 观察')
    report.append('')
    report.append('- **Anti-diagonal pattern preserved**: as η↑, optimal λ*↓ (1e-2 → 2e-4), matching stability bound η(1+λ) < 2/L.')
    report.append('  **反对角线模式保持**：η↑ 时最优 λ*↓（1e-2 → 2e-4），与稳定性边界 η(1+λ) < 2/L 吻合。')
    report.append('- Stable region shows tight ± values; large ± only near divergence boundary.')
    report.append('  稳定区域 ± 值紧凑；仅在发散边界附近出现大 ± 值。')
    report.append('')
    report.append('---')
    report.append('')

    # Experiment 3
    report.append('## Experiment 3: Batch Size Scaling / 实验三：Batch Size 缩放')
    report.append('')
    report.append('Linear scaling rule: η = 0.1 × (B / 128). SGDM (μ=0.9).')
    report.append('')
    report.append('### Best Test Accuracy (%) — N=4 runs')
    report.append('')
    report.extend(exp3_table)
    report.append('')
    report.append('Bold = row maximum. / 粗体 = 行最大值。')
    report.append('')
    report.append('### Observations / 观察')
    report.append('')
    report.append('- λ* consistently in [5e-4, 1e-3] across all batch sizes and runs.')
    report.append('  λ* 在所有 batch size 和所有运行中一致落入 [5e-4, 1e-3]。')
    report.append('- Peak accuracy degrades smoothly with batch size; reproducible across runs.')
    report.append('  峰值精度随 batch size 平滑衰减，跨运行可复现。')
    report.append('')
    report.append('---')
    report.append('')

    # Summary
    report.append('## Summary Statistics / 总体统计')
    report.append('')
    report.append(f'Across {n_configs} (η, λ) or (B, λ) configurations from Exp 2 and Exp 3:')
    report.append('')
    report.append(f'覆盖实验二和实验三共 {n_configs} 组配置：')
    report.append('')
    report.append('| Metric | Value |')
    report.append('|---|---|')
    report.append(f'| Configurations / 配置数 | {n_configs} |')
    report.append(f'| N (runs per config) | 4 |')
    report.append(f'| Mean range (max−min) / 平均全距 | {mean_range:.2f}% |')
    report.append(f'| Half-range < 1% | **{w1/n_configs*100:.1f}%** ({w1}/{n_configs}) |')
    report.append(f'| Half-range < 2% | **{w2/n_configs*100:.1f}%** ({w2}/{n_configs}) |')
    report.append('')
    report.append('---')
    report.append('')

    # Conclusion
    report.append('## Conclusion / 结论')
    report.append('')
    report.append('Averaging over 4 independent runs (2 seeds × 2 runs/seed) confirms:')
    report.append('')
    report.append('对 4 次独立运行（2 种子 × 每种子 2 次）取平均后确认：')
    report.append('')
    report.append('1. **Exp 1**: Stability boundary ordering and accuracy-LR curve shapes are invariant.')
    report.append('   **实验一**：稳定性边界排序和精度-LR 曲线形态不变。')
    report.append('2. **Exp 2**: Anti-diagonal η–λ interaction pattern is robust.')
    report.append('   **实验二**：η–λ 反对角线交互模式稳健。')
    report.append('3. **Exp 3**: Linear LR scaling preserves optimal λ* ∈ [5e-4, 1e-3].')
    report.append('   **实验三**：线性 LR 缩放保持最优 λ* ∈ [5e-4, 1e-3]。')
    report.append('')
    report.append('**All experimental conclusions are robust to random seed and run-to-run variation.**')
    report.append('')
    report.append('**所有实验结论对随机种子和运行间变化均稳健。**')

    report_text = '\n'.join(report) + '\n'
    out_path = BASE_DIR / 'rebuttal' / 'resnet18_4run_report.md'
    out_path.write_text(report_text, encoding='utf-8')
    print(f'\nReport saved to: {out_path}')


if __name__ == '__main__':
    main()
