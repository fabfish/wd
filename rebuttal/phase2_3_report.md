# Phase 2–3: Cross-Architecture Generalization — VGG-16 on CIFAR-100

# Phase 2–3：跨架构泛化验证 — VGG-16 on CIFAR-100

All results reported as **mean ± half-range** over 2 seeds (seed=42, seed=123). Best test accuracy (%).

所有结果以 **mean ± half-range** 格式报告（seed=42, seed=123 双种子），best test accuracy (%)。

### Data Sources / 数据来源

| Seed | Model | CSV Path | Rows |
|---:|---|---|---:|
| 42 | ResNet-18 | [`outputs/results/results.csv`](outputs/results/results.csv) | 87 |
| 123 | ResNet-18 | [`rebuttal/results/results_resnet18_seed123.csv`](rebuttal/results/results_resnet18_seed123.csv) | 84 |
| 42 | VGG-16 | [`rebuttal/results/results_vgg16_seed42.csv`](rebuttal/results/results_vgg16_seed42.csv) | 84 |
| 123 | VGG-16 | [`rebuttal/results/results_vgg16_seed123.csv`](rebuttal/results/results_vgg16_seed123.csv) | 84 |

- Analysis script / 分析脚本: [`rebuttal/analyze_all.py`](rebuttal/analyze_all.py)
- Model definitions / 模型定义: [`wd_core/models.py`](wd_core/models.py) — `resnet18()`, `vgg16()`
- Total runs / 总运行数: **332** (2 models × 2 seeds × 83 configurations)

---

## Part A: VGG-16 Results (2-Seed) / VGG-16 结果（双种子）

### Experiment 1: Stability Boundary Ordering / 实验一：稳定性边界排序

| LR | SGD | SGD+WD (λ=5e-4) | SGDM+WD (μ=0.9, λ=5e-4) |
|---:|:---:|:---:|:---:|
| 0.001 | 43.56 ± 0.86 | 43.64 ± 0.86 | 68.45 ± 0.00 |
| 0.005 | 64.44 ± 0.13 | 64.47 ± 0.16 | 73.77 ± 0.16 |
| 0.01 | 68.28 ± 0.17 | 68.58 ± 0.07 | 75.06 ± 0.05 |
| 0.05 | 72.44 ± 0.04 | 74.09 ± 0.14 | 77.01 ± 0.18 |
| 0.1 | **73.21 ± 0.12** | 75.22 ± 0.09 | **77.31 ± 0.18** |
| 0.5 | 71.25 ± 0.22 | **76.49 ± 0.21** | 73.88 ± 1.01 |
| 1.0 | 68.37 ± 2.10 | 75.61 ± 0.05 | 56.98 ± 2.16 |
| 2.0 | 67.52 ± 1.14 | 76.62 ± 0.29 | 4.29 ± 0.04 |

#### Peak accuracy / 峰值精度

| Optimizer | η\* | Peak Acc |
|---|---|---|
| SGD | 0.1 | 73.21 ± 0.12 |
| SGD+WD | 0.5–2.0 (plateau) | 76.80 ± 0.10 |
| SGDM+WD | 0.05–0.1 | 77.34 ± 0.15 |

The same pattern observed in ResNet-18 is reproduced: SGD+WD maintains a broad high-LR plateau, while SGDM+WD collapses sharply beyond η=0.1.

与 ResNet-18 相同的模式被复现：SGD+WD 在高 LR 区间保持宽广平台，SGDM+WD 在 η>0.1 后急剧崩溃。

### Experiment 2: η–λ Interaction Heatmap / 实验二：η–λ 交互热力图

| η \ λ | 1e-4 | 2e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 | 1e-2 |
|------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 0.01 | 73.66 ± 0.05 | 74.10 ± 0.20 | 75.06 ± 0.05 | 75.89 ± 0.07 | 77.02 ± 0.27 | **77.62 ± 0.06** | 77.53 ± 0.06 |
| 0.05 | 73.91 ± 0.04 | 75.19 ± 0.30 | 77.01 ± 0.18 | 78.16 ± 0.20 | **78.38 ± 0.18** | 75.51 ± 0.09 | 68.95 ± 0.06 |
| 0.10 | 73.82 ± 0.60 | 75.10 ± 0.10 | 77.31 ± 0.18 | **77.45 ± 0.11** | 76.22 ± 0.18 | 67.41 ± 1.11 | 41.30 ± 0.02 |
| 0.20 | 73.57 ± 0.41 | 75.33 ± 0.06 | **76.36 ± 0.09** | 75.53 ± 0.05 | 72.41 ± 0.44 | 38.88 ± 0.59 | 5.42 ± 0.25 |
| 0.30 | 74.16 ± 0.19 | **75.34 ± 0.46** | 75.59 ± 0.18 | 73.12 ± 0.32 | 59.69 ± 6.35 | 15.12 ± 6.20 | 2.37 ± 0.77 |

#### Optimal λ\* per η

| η | λ\* (seed=42) | λ\* (seed=123) | Peak Acc | Agreement |
|----:|:---:|:---:|:---:|:---:|
| 0.01 | 5e-3 | 1e-2 | 77.64 ± 0.04 | ≈ adjacent |
| 0.05 | 2e-3 | 2e-3 | 78.38 ± 0.18 | ✓ exact |
| 0.10 | 1e-3 | 1e-3 | 77.45 ± 0.11 | ✓ exact |
| 0.20 | 5e-4 | 5e-4 | 76.36 ± 0.09 | ✓ exact |
| 0.30 | 2e-4 | 5e-4 | 75.60 ± 0.19 | ≈ adjacent |

The anti-diagonal pattern (higher η → lower optimal λ\*) is clearly preserved on VGG-16: λ\* decreases from 5e-3 → 2e-4 as η increases from 0.01 → 0.3. Notably, **3/5 learning rates show exact λ\* match** between seeds — stronger consistency than ResNet-18 (2/5 exact).

反对角线模式在 VGG-16 上清晰保持：λ\* 从 5e-3 → 2e-4 随 η 从 0.01 → 0.3 递减。值得注意的是，**3/5 的学习率在两个种子间 λ\* 完全一致** — 比 ResNet-18 (2/5) 更强的一致性。

### Experiment 3: Batch Size Scaling / 实验三：Batch Size 缩放

| B | η | λ=1e-4 | λ=2e-4 | λ=5e-4 | λ=1e-3 | λ=2e-3 | λ=5e-3 |
|---:|----:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64 | 0.05 | 74.97 ± 0.09 | 76.17 ± 0.02 | 77.72 ± 0.26 | **78.31 ± 0.38** | 77.28 ± 0.03 | 71.72 ± 0.20 |
| 128 | 0.10 | 73.82 ± 0.60 | 75.10 ± 0.10 | 77.31 ± 0.18 | **77.45 ± 0.11** | 76.22 ± 0.18 | 67.41 ± 1.11 |
| 256 | 0.20 | 71.92 ± 0.46 | 74.48 ± 0.45 | **76.56 ± 0.35** | 76.31 ± 0.08 | 74.69 ± 0.17 | 63.86 ± 0.11 |
| 512 | 0.40 | 68.78 ± 1.84 | 71.53 ± 0.38 | 75.38 ± 0.48 | **75.48 ± 0.46** | 73.52 ± 0.14 | 51.64 ± 1.29 |

#### Optimal λ\* per batch size

| B | η | λ\* (seed=42) | λ\* (seed=123) | Peak Acc | Agreement |
|---:|----:|:---:|:---:|:---:|:---:|
| 64 | 0.05 | 1e-3 | 1e-3 | 78.31 ± 0.38 | ✓ exact |
| 128 | 0.10 | 1e-3 | 1e-3 | 77.45 ± 0.11 | ✓ exact |
| 256 | 0.20 | 5e-4 | 1e-3 | 76.57 ± 0.34 | ≈ adjacent |
| 512 | 0.40 | 1e-3 | 1e-3 | 75.48 ± 0.46 | ✓ exact |

λ\* stays in **[5e-4, 1e-3]** across all batch sizes — identical to ResNet-18 — confirming the linear scaling rule on a different architecture.

λ\* 在所有 batch size 下保持在 **[5e-4, 1e-3]** — 与 ResNet-18 完全一致 — 确认线性缩放规则在不同架构上成立。

---

## Part B: Cross-Architecture Comparison / 跨架构对比

### Experiment 1: Peak Accuracy by Optimizer / 各优化器峰值精度

| Optimizer | ResNet-18 (2-seed) | VGG-16 (2-seed) | All 4 runs (mean ± std) |
|---|:---:|:---:|:---:|
| SGD | 73.23 ± 0.14 | 73.21 ± 0.12 | **73.22 ± 0.13** |
| SGD+WD | 77.02 ± 0.11 | 76.80 ± 0.10 | **76.91 ± 0.15** |
| SGDM+WD | 77.32 ± 0.13 | 77.34 ± 0.15 | **77.33 ± 0.14** |

Both models yield nearly identical peak accuracies (within 0.2%), confirming the conclusions do not depend on the specific architecture.

两种模型的峰值精度几乎完全一致（差异 <0.2%），确认结论不依赖于特定架构。

### Experiment 2: η–λ Heatmap (4-Run Mean ± Std) / η–λ 热力图（4 组平均）

| η \ λ | 1e-4 | 2e-4 | 5e-4 | 1e-3 | 2e-3 | 5e-3 | 1e-2 |
|------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 0.01 | 73.66 ± 0.05 | 74.14 ± 0.17 | 75.10 ± 0.11 | 76.02 ± 0.28 | 76.97 ± 0.23 | **77.58 ± 0.06** | 77.56 ± 0.05 |
| 0.05 | 74.06 ± 0.28 | 75.08 ± 0.24 | 76.95 ± 0.25 | 78.21 ± 0.26 | **78.25 ± 0.20** | 75.53 ± 0.12 | 68.94 ± 0.08 |
| 0.10 | 73.41 ± 0.61 | 75.19 ± 0.22 | **77.30 ± 0.17** | 77.30 ± 0.21 | 76.20 ± 0.16 | 67.28 ± 0.99 | 42.75 ± 2.52 |
| 0.20 | 73.53 ± 0.38 | 75.37 ± 0.12 | **76.34 ± 0.08** | 75.45 ± 0.13 | 72.21 ± 0.69 | 37.85 ± 1.51 | 5.19 ± 0.57 |
| 0.30 | 74.52 ± 0.55 | **75.41 ± 0.54** | 75.48 ± 0.16 | 73.06 ± 0.28 | 59.01 ± 5.75 | 17.87 ± 5.19 | 2.34 ± 0.74 |

#### Optimal λ\* consensus (4 runs: R18-s42, R18-s123, V16-s42, V16-s123) / 最优 λ\* 共识

| η | Run 1 (R18-s42) | Run 2 (R18-s123) | Run 3 (V16-s42) | Run 4 (V16-s123) | Consensus Range |
|----:|:---:|:---:|:---:|:---:|:---:|
| 0.01 | 1e-2 | 1e-2 | 5e-3 | 1e-2 | **[5e-3, 1e-2]** |
| 0.05 | 1e-3 | 2e-3 | 2e-3 | 2e-3 | **[1e-3, 2e-3]** |
| 0.10 | 5e-4 | 1e-3 | 1e-3 | 1e-3 | **[5e-4, 1e-3]** |
| 0.20 | 5e-4 | 5e-4 | 5e-4 | 5e-4 | **5e-4** (unanimous) |
| 0.30 | 2e-4 | 5e-4 | 2e-4 | 5e-4 | **[2e-4, 5e-4]** |

The inverse η–λ relationship is robust across all 4 runs. The consensus range narrows from [5e-3, 1e-2] at η=0.01 to [2e-4, 5e-4] at η=0.3 — a clear monotonic decrease.

η–λ 反比关系在全部 4 组实验中稳健成立。共识范围从 η=0.01 时的 [5e-3, 1e-2] 收窄到 η=0.3 时的 [2e-4, 5e-4] — 清晰的单调递减。

### Experiment 3: Batch Size Scaling (Peak Accuracy) / Batch Size 缩放（峰值精度）

| B | η | ResNet-18 (2-seed) | VGG-16 (2-seed) | All 4 runs (mean ± std) |
|---:|----:|:---:|:---:|:---:|
| 64 | 0.05 | 78.22 ± 0.29 | 78.31 ± 0.38 | **78.27 ± 0.34** |
| 128 | 0.10 | 77.40 ± 0.05 | 77.45 ± 0.11 | **77.42 ± 0.09** |
| 256 | 0.20 | 76.78 ± 0.55 | 76.57 ± 0.34 | **76.68 ± 0.46** |
| 512 | 0.40 | 75.22 ± 0.20 | 75.48 ± 0.46 | **75.35 ± 0.38** |

#### Optimal λ\* consensus (4 runs)

| B | Run 1 (R18-s42) | Run 2 (R18-s123) | Run 3 (V16-s42) | Run 4 (V16-s123) | Consensus |
|---:|:---:|:---:|:---:|:---:|:---:|
| 64 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | **1e-3** (unanimous) |
| 128 | 5e-4 | 1e-3 | 1e-3 | 1e-3 | **[5e-4, 1e-3]** |
| 256 | 5e-4 | 1e-3 | 5e-4 | 1e-3 | **[5e-4, 1e-3]** |
| 512 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | **1e-3** (unanimous) |

For B=64 and B=512, all 4 runs agree on exactly λ\*=1e-3. For B=128 and B=256, the range is [5e-4, 1e-3]. The linear scaling rule holds across both architectures.

B=64 和 B=512 时，全部 4 组实验一致选择 λ\*=1e-3。B=128 和 B=256 时，范围为 [5e-4, 1e-3]。线性缩放规则在两种架构上均成立。

---

## Overall Conclusion / 总体结论

With **332 total runs** (2 architectures × 2 seeds × 83 configurations), we demonstrate:

通过 **332 次实验**（2 种架构 × 2 个种子 × 83 种配置），我们证明：

1. **Seed robustness / 种子稳健性**: Within each architecture, the ± half-range is consistently small (0.02–0.5% in stable regions). Large variance occurs only in unstable regimes near the divergence boundary.

   在每种架构内部，± half-range 始终很小（稳定区域 0.02–0.5%）。大方差仅出现在接近发散边界的不稳定区域。

2. **Architecture generalization / 架构泛化**: ResNet-18 and VGG-16 yield nearly identical results (peak accuracy within 0.2%, same optimal hyperparameter ranges), despite fundamentally different architectures (residual connections vs. plain deep convolutions).

   ResNet-18 和 VGG-16 产生几乎一致的结果（峰值精度差异 <0.2%，最优超参数范围相同），尽管它们的架构根本不同（残差连接 vs. 纯深度卷积）。

3. **Theoretical predictions confirmed across all conditions / 理论预测在所有条件下得到确认**:
   - Stability boundary ordering: SGD+WD widest, SGDM+WD tightest — **4/4 runs**
   - η–λ inverse relationship: monotonic decrease of λ\* with η — **4/4 runs**
   - Batch size scaling: λ\* ∈ [5e-4, 1e-3] across all B — **4/4 runs**

   - 稳定性边界排序：SGD+WD 最宽、SGDM+WD 最紧 — **4/4 组**
   - η–λ 反比关系：λ\* 随 η 单调递减 — **4/4 组**
   - Batch size 缩放：所有 B 下 λ\* ∈ [5e-4, 1e-3] — **4/4 组**

**The theoretical predictions are validated across multiple random seeds and architecturally diverse models.**

**理论预测在多个随机种子和架构多样的模型上均得到验证。**
