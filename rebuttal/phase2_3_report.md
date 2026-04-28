# Cross-Architecture Generalization — VGG-16 on CIFAR-100

# 跨架构泛化验证 — VGG-16 on CIFAR-100

All results reported as **mean ± half-range** over 2 seeds (seed=42, seed=123). Best test accuracy (%).

所有结果以 **mean ± half-range** 格式报告（seed=42, seed=123 双种子），best test accuracy (%)。

### Data Sources / 数据来源

| Seed | Model | CSV Path | Unique Configs |
|---:|---|---|---:|
| 42 | ResNet-18 | [`outputs/results/results.csv`](../outputs/results/results.csv) | 77 |
| 123 | ResNet-18 | [`rebuttal/results/results_resnet18_seed123.csv`](results/results_resnet18_seed123.csv) | 77 |
| 42 | VGG-16 | [`rebuttal/results/results_vgg16_seed42.csv`](results/results_vgg16_seed42.csv) | 77 |
| 123 | VGG-16 | [`rebuttal/results/results_vgg16_seed123.csv`](results/results_vgg16_seed123.csv) | 77 |

- Model definitions / 模型定义: [`wd_core/models.py`](../wd_core/models.py) — `resnet18()` (11.2M params), `vgg16()` (14.8M params)
- Experiment runner / 实验运行器: [`rebuttal/run_rebuttal.py`](run_rebuttal.py)
- All experiments: CIFAR-100, CosineAnnealingLR, 100 epochs, AMP enabled.
- Total runs / 总运行数: **308** (2 models × 2 seeds × 77 unique configurations)

### Reliability Check / 可靠性验证

| Check | Result |
|---|---|
| VGG-16 ≠ ResNet-18 (same seed=42) | **100%** configs differ (mean \|Δ\| = 5.96%) |
| VGG-16 seed consistency | 90% configs \|Δ\| < 1%, 96% < 2% |
| Architecture fingerprint | VGG crashes at SGDM+WD η=1.0 (3.04%), R18 survives (54.84%) |

---

## Experiment 1: Stability Boundary Ordering / 实验一：稳定性边界排序

### VGG-16 (2-seed mean ± half-range)

| LR | SGD | SGD+WD (λ=5e-4) | SGDM+WD (μ=0.9, λ=5e-4) |
|---:|:---:|:---:|:---:|
| 0.001 | 55.37 ± 0.02 | 55.20 ± 0.12 | 65.37 ± 0.01 |
| 0.005 | 62.20 ± 0.47 | 62.73 ± 0.05 | 70.75 ± 0.25 |
| 0.01 | 65.09 ± 0.27 | 65.68 ± 0.29 | 71.27 ± 0.22 |
| 0.05 | 68.63 ± 0.25 | 70.36 ± 0.18 | 72.99 ± 0.21 |
| 0.1 | **69.12 ± 0.45** | 71.20 ± 0.37 | **73.00 ± 0.09** |
| 0.5 | 66.85 ± 0.75 | 72.60 ± 0.30 | 66.12 ± 0.05 |
| 1.0 | 65.65 ± 0.37 | **72.92 ± 0.29** | 2.02 ± 1.02 |
| 2.0 | 63.43 ± 0.20 | 72.66 ± 0.27 | 1.14 ± 0.14 |

### Cross-Architecture Peak Comparison / 跨架构峰值对比

| Optimizer | ResNet-18 η\* | ResNet-18 Peak | VGG-16 η\* | VGG-16 Peak | Δ |
|---|:---:|:---:|:---:|:---:|:---:|
| SGD | 0.1 | 73.23 ± 0.14 | 0.1 | 69.12 ± 0.45 | −4.11 |
| SGD+WD | 0.5 (plateau) | 76.70 ± 0.42 | 0.5–1.0 (plateau) | 72.92 ± 0.29 | −3.79 |
| SGDM+WD | 0.05–0.1 | 77.29 ± 0.16 | 0.05–0.1 | 73.00 ± 0.09 | −4.29 |

### Key Observations / 核心观察

- **Stability boundary ordering preserved**: On VGG-16, SGDM+WD collapses at η=0.5 (66.12%) and fully diverges at η≥1.0 (2–3%), while SGD+WD maintains a broad plateau up to η=2.0 (72.66%). SGD remains stable but with lower accuracy.

  **稳定性边界排序保持不变**：VGG-16 上，SGDM+WD 在 η=0.5 处大幅衰减 (66.12%)，η≥1.0 完全发散 (2–3%)；SGD+WD 保持宽平台至 η=2.0 (72.66%)。

- **VGG-16 has a tighter stability boundary**: SGDM+WD crashes at η=1.0 (3.04%) on VGG vs η=2.0 (4.07%) on ResNet-18. This is consistent with VGG's larger Lipschitz constant L (no residual skip connections).

  **VGG-16 稳定性边界更窄**：SGDM+WD 在 VGG 上 η=1.0 即崩溃 (3.04%)，ResNet-18 在 η=2.0 才崩溃 (4.07%)。与 VGG 更大的 Lipschitz 常数 L（无残差跳连）一致。

- **Absolute accuracy is ~4% lower** across all optimizers, but the **relative ordering and curve shapes are identical**.

  **绝对精度低约 4%**，但**相对排序和曲线形态完全一致**。

---

## Experiment 2: η–λ Interaction Heatmap / 实验二：η–λ 交互热力图

### VGG-16 (2-seed mean ± half-range)

| η \ λ | 1e-4 | 2e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 | 1e-2 |
|------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 0.01 | 70.2 ± 0.1 | 70.5 ± 0.1 | 71.3 ± 0.2 | 72.5 ± 0.4 | 73.6 ± 0.2 | 74.8 ± 0.1 | **74.9 ± 0.1** |
| 0.05 | 68.9 ± 0.1 | 70.8 ± 0.6 | 73.0 ± 0.2 | **73.7 ± 0.0** | 73.4 ± 0.2 | 72.3 ± 0.4 | 67.7 ± 0.1 |
| 0.10 | 69.5 ± 0.4 | 71.4 ± 0.5 | **73.0 ± 0.1** | 73.0 ± 0.0 | 72.4 ± 0.1 | 66.9 ± 0.2 | 3.7 ± 0.3 |
| 0.20 | 70.8 ± 0.4 | 72.2 ± 0.0 | **73.0 ± 0.2** | 71.9 ± 0.1 | 68.3 ± 0.1 | 4.1 ± 0.0 | 1.1 ± 0.1 |
| 0.30 | 71.4 ± 0.1 | 72.0 ± 0.3 | **72.3 ± 0.0** | 69.7 ± 0.6 | 58.8 ± 3.7 | 2.3 ± 0.0 | 1.1 ± 0.0 |

Bold = row maximum. / 粗体 = 行最大值。

### Optimal λ\* per η (4 runs: R18×2, VGG×2) / 各 η 最优 λ\*

| η | R18 s42 | R18 s123 | VGG s42 | VGG s123 | Consensus Range |
|----:|:---:|:---:|:---:|:---:|:---:|
| 0.01 | 1e-2 | 1e-2 | 5e-3 | 1e-2 | **[5e-3, 1e-2]** |
| 0.05 | 1e-3 | 2e-3 | 1e-3 | 1e-3 | **[1e-3, 2e-3]** |
| 0.10 | 5e-4 | 1e-3 | 1e-3 | 5e-4 | **[5e-4, 1e-3]** |
| 0.20 | 5e-4 | 5e-4 | 5e-4 | 5e-4 | **5e-4** (unanimous) |
| 0.30 | 2e-4 | 5e-4 | 5e-4 | 2e-4 | **[2e-4, 5e-4]** |

### Key Observations / 核心观察

- **Anti-diagonal pattern confirmed on VGG-16**: λ\* decreases from [5e-3, 1e-2] at η=0.01 to [2e-4, 5e-4] at η=0.3 — a clear monotonic decrease across both architectures.

  **反对角线模式在 VGG-16 上得到确认**：λ\* 从 η=0.01 时的 [5e-3, 1e-2] 递减至 η=0.3 时的 [2e-4, 5e-4] — 两种架构上均呈清晰单调递减。

- At η=0.2, **all 4 runs unanimously select λ\*=5e-4** — the strongest consensus point.

  η=0.2 时，**全部 4 组实验一致选择 λ\*=5e-4** — 最强共识点。

- **VGG-16's unstable region is larger**: η=0.2/λ=5e-3 gives 4.1% on VGG vs ~39% on R18; η=0.3/λ=2e-3 gives 58.8% on VGG vs ~66% on R18.

  **VGG-16 的不稳定区域更大**：η=0.2/λ=5e-3 时 VGG 仅 4.1%（R18 ~39%）。

---

## Experiment 3: Batch Size Scaling / 实验三：Batch Size 缩放

Linear scaling rule: η = 0.1 × (B / 128). SGDM with μ=0.9.

### VGG-16 (2-seed mean ± half-range)

| B | η | λ=1e-4 | λ=2e-4 | λ=5e-4 | λ=1e-3 | λ=2e-3 | λ=5e-3 |
|---:|----:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64 | 0.05 | 70.9 ± 0.3 | 72.3 ± 0.1 | 73.4 ± 0.0 | **73.6 ± 0.1** | 73.1 ± 0.3 | 68.8 ± 0.2 |
| 128 | 0.10 | 69.5 ± 0.4 | 71.4 ± 0.5 | **73.0 ± 0.1** | 73.0 ± 0.0 | 72.4 ± 0.1 | 66.9 ± 0.2 |
| 256 | 0.20 | 68.9 ± 0.3 | 71.4 ± 0.2 | 72.2 ± 0.1 | **73.0 ± 0.0** | 71.8 ± 0.0 | 62.8 ± 0.0 |
| 512 | 0.40 | 67.6 ± 0.6 | 69.4 ± 0.2 | 71.3 ± 0.6 | **71.9 ± 0.1** | 70.8 ± 0.0 | 9.6 ± 6.3 |

### Optimal λ\* per B (4 runs) / 各 B 最优 λ\*

| B | η | R18 s42 | R18 s123 | VGG s42 | VGG s123 | Consensus |
|---:|----:|:---:|:---:|:---:|:---:|:---:|
| 64 | 0.05 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | **1e-3** (unanimous) |
| 128 | 0.10 | 5e-4 | 1e-3 | 1e-3 | 5e-4 | **[5e-4, 1e-3]** |
| 256 | 0.20 | 5e-4 | 1e-3 | 1e-3 | 1e-3 | **[5e-4, 1e-3]** |
| 512 | 0.40 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | **1e-3** (unanimous) |

### Key Observations / 核心观察

- λ\* consistently stays in **[5e-4, 1e-3]** across all batch sizes and both architectures. At B=64 and B=512, all 4 runs unanimously select λ\*=1e-3.

  λ\* 在所有 batch size 和两种架构下一致落入 **[5e-4, 1e-3]**。B=64 和 B=512 时全部 4 组一致选择 λ\*=1e-3。

- Peak accuracy degrades gracefully with batch size on VGG-16: 73.6 → 73.0 → 73.0 → 71.9 (±0.0–0.6).

  VGG-16 峰值精度随 batch size 平滑衰减：73.6 → 73.0 → 73.0 → 71.9 (±0.0–0.6)。

- B=512/λ=5e-3 crashes on VGG-16 (9.6 ± 6.3%) but stays at 53.01% on R18, again reflecting VGG-16's tighter stability boundary.

  B=512/λ=5e-3 时 VGG-16 崩溃 (9.6±6.3%)，R18 仍有 53.01%，再次体现 VGG-16 更窄的稳定性边界。

---

## Overall Conclusion / 总体结论

With **308 total runs** (2 architectures × 2 seeds × 77 configurations), we demonstrate:

通过 **308 次实验**（2 种架构 × 2 个种子 × 77 种配置），我们证明：

1. **Architecture generalization / 架构泛化**: All three theoretical predictions hold on VGG-16 (a fundamentally different architecture with no residual connections), with the same qualitative patterns despite ~4% lower absolute accuracy.

   全部三个理论预测在 VGG-16（无残差连接的根本不同架构）上成立，定性模式一致，尽管绝对精度低 ~4%。

2. **Tighter stability boundary on VGG-16 / VGG-16 稳定性边界更窄**: VGG-16's SGDM+WD diverges at η=1.0 (vs η=2.0 on ResNet-18), and its unstable region in the η-λ heatmap is larger. This is consistent with a larger Lipschitz constant L for networks without skip connections.

   VGG-16 的 SGDM+WD 在 η=1.0 发散（R18 在 η=2.0），η-λ 热力图的不稳定区域更大。与无跳连网络更大的 Lipschitz 常数 L 一致。

3. **Optimal hyperparameter ranges are shared / 最优超参数范围一致**: Despite architectural differences, the optimal λ\* falls in the same [5e-4, 1e-3] range for both models, and the η-λ inverse relationship is quantitatively similar.

   尽管架构不同，两种模型的最优 λ\* 落在相同的 [5e-4, 1e-3] 范围，η-λ 反比关系定量相似。

**The theoretical predictions generalize across architecturally diverse models and multiple random seeds.**

**理论预测在架构多样的模型和多个随机种子上均得到泛化验证。**
