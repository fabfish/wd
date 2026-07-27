# Rebuttal → `wd/` 论文补充对照表

本文件落盘 rebuttal 中相对于原投稿（`Understanding_the_Interplay_of__Weight_Decay_and_Hyperparameters__A_Stability_Perspective/`）新增的全部内容，以及它们在当前 NeurIPS 2026 版本（`wd/`）中的处理状态，便于增量推进。

- **Scope = current**：本轮已纳入实施；
- **Scope = deferred**：留待后续，但本表保留以防遗漏。

## 实验数据 / 多种子 / 跨架构

| # | 类别 | Rebuttal 新增内容 | 来源 | `wd/` 当前状态 | Scope | 状态 |
|---|---|---|---|---|---|---|
| 1 | 实验-多种子 | 全部实验 seed=42 / seed=123 双种子，mean ± half-range 报告 | `phase1_report.md` | 实验章节仍报单种子最佳值 | current | done（附录扩写） |
| 2 | 实验-跨架构 | VGG-16 (14.8M) + ResNet-50 (23.5M) 跨架构验证（≈640 runs） | `phase2_3_report.md`, `resnet50_report.md` | 仅 ResNet-18 + Qwen3 | current | done（附录新节） |
| 3 | 实验-多次重跑 | ResNet-18 共 4 次独立运行（2 seed × 2 run） | `resnet18_4run_report.md` | 缺失 | current | done（正文/附录均提及） |
| 4 | 表格 | Table 1：3 架构 Exp 1 峰值精度对照 | `figures/tables.md` | 仅 R18 单种子的 `tab:exp1` | current | done（Appendix Table A） |
| 5 | 表格 | Table 2：3 架构 Exp 2 各 η 最优 λ\* | `figures/tables.md` | 缺失 | current | done（Appendix Table B） |
| 6 | 表格 | Table 3：3 架构 Exp 3 batch-size scaling | `figures/tables.md` | 缺失 | current | done（Appendix Table C） |
| 7 | 图 | `fig1_exp1_stability_boundary.png`（3 架构 LR 曲线） | `rebuttal/figures/` | 用 `performance.jpg`（仅 R18） | current | done（附录图） |
| 8 | 图 | `fig2_exp2_heatmap.png`（3 架构 η-λ 热图） | `rebuttal/figures/` | 用 `WD and LR (1).jpg`（仅 R18） | current | done（附录图） |
| 9 | 图 | `fig3_exp3_batch_scaling.png`（3 架构 batch scaling） | `rebuttal/figures/` | 用 `batch.png`（仅 R18） | current | done（附录图） |
| 10 | 图 | `fig4_exp3_lambda_eta_product.png`（λ×η ↑ with B） | `rebuttal/figures/` | 缺失 | current | done（附录图） |
| 11 | 图 | `response_to_reviewer_focused_smooth.png`（每条曲线一个 λ，X=η×λ，所有曲线在同一 η×λ 处达峰） | `rebuttal/figures/` | 缺失，且为审稿人 9i84 明确要求 | current | done（正文 Fig 2 Right） |
| 12 | 图 | `response_to_reviewer_focused_train.png`（train curve） | `rebuttal/figures/` | 缺失 | current | done（正文 Fig 2 Middle） |
| 13 | 实验扩展 | Exp2 扩展到 7 个 λ × η∈[0.0002, 5.0]，验证 η×λ 集中在 ≈10⁻⁴ | `response_to_qZ4a_experiments.md` | 缺失 | current | done（与 #11 配合） |
| 14 | 正文图1 排版 | NeurIPS 单栏排版 | — | 需要切 R18 单子图 | current | done（替换 `performance.jpg`） |
| 15 | 正文图3 | `fig:batch` 改造 | — | 暂不动 | deferred | deferred |

## 章节内容（Limitations / REH 直觉 / Muon）

| # | 类别 | Rebuttal 新增内容 | 来源 | `wd/` 当前状态 | Scope | 状态 |
|---|---|---|---|---|---|---|
| 16 | Limitations 三条扩展 | 凸假设；上界非紧→近似关系；适用范围如 dropout 可，early stopping 不可 | `response.md`（多回复合并） | `08_Conclusion.tex` 中只有一条短 Limitation | deferred | deferred |
| 17 | REH 直觉补段 | LR vs WD 等价性 + 导弹/泰勒-小波类比 + 为何匹配 SWA 与 SGD-WD | `response.md`（核心回复） | Intro 中只有"Case 1/Case 2"两个简短例子 | deferred | deferred |
| 18 | Muon 推测段 | 在凸/有界梯度假设下，Muon 中 η 与 λ 的相互作用大概率仍呈类似缩放 | `response.md`（KQ2） | 缺失 | deferred | deferred |

## Related Work 补引

| # | 类别 | Rebuttal 新增内容 | 来源 | `wd/` 当前状态 | Scope | 状态 |
|---|---|---|---|---|---|---|
| 19 | Related Work | Jia et al. 2022 — Weight decay with tailored Adam | `response.md`（W1） | ref.bib 不含 | deferred | deferred |
| 20 | Related Work | Chen et al. 2024 — WD induces low-rank bias | `response.md`（W1） | ref.bib 不含 | deferred | deferred |
| 21 | Related Work | Zhang et al. 2018 — Three mechanisms of WD regularization | `response.md`（W1） | ref.bib 不含 | deferred | deferred |
| 22 | Related Work | Xie et al. 2020 — Understanding and scheduling WD（非凸，η ∝ 1/λ） | `response.md`（KQ1） | ref.bib 不含 | deferred | deferred |

## 措辞 / Typo / 数学表述

| # | 类别 | Rebuttal 新增内容 | 来源 | `wd/` 当前状态 | Scope | 状态 |
|---|---|---|---|---|---|---|
| 23 | 措辞 | Remark 5.2 中 "SWA stability is half of SGD" → "SWA improves the generalization performance over SGD" | `response.md`（W2） | 当前 `06_Parameter.tex` L28-30 仍是旧表述 | deferred | deferred |
| 24 | Typo 全替 | "SGD-DW" → "SGD-WD" 全部替换 | `response.md`（minor） | `05_Generalization.tex` L75 + `09_appendix.tex` L28/85/180 仍 4 处 | deferred | deferred |
| 25 | SWA 定义 typo | Eq.(8) 用 `1/T·Σθ_i` 而非递归形式 | `response.md`（5.3） | `04_Preliminary.tex` L70-72 仍是递归 | deferred | deferred |
| 26 | τ_β 说明 | τ_β 实际依赖 t，文本应注明这点 | `response.md`（W5.2） | 已有但可加注一句 | deferred | deferred |
| 27 | 证明完整性 (31)/(40) | 分开 bound `‖∇F(θ)‖` 与 `‖∇F(θ')‖` | `response.md`（W5.6） | `09_appendix.tex` 仅一条 bound | deferred | deferred |
| 28 | 证明完整性 L675/731 | 学习率约束的非扩张推导细节标注出处 | `response.md`（W5.7） | 证明已有但可加注 | deferred | deferred |

## Abstract / Setup 配套

| # | 类别 | Rebuttal 新增内容 | 来源 | `wd/` 当前状态 | Scope | 状态 |
|---|---|---|---|---|---|---|
| 29 | Abstract 架构列表 | 把架构列表扩为 R18 + VGG + R50 + Qwen3 | 派生 | 仅 R18 + Qwen3 | current | done |
| 30 | Appendix Setup | "ResNet-18" → "ResNet-18 / VGG-16 / ResNet-50" + seeds & runs 说明 | 派生 | `09_appendix.tex` 仅 R18 | current | done |

---

## 推进备忘

- 本轮（current）已落实 #1–14, #29, #30。
- 后续（deferred）建议优先级：#16 Limitations 扩展 → #23 Remark 5.2 措辞 → #24 SGD-DW → SGD-WD typo 全替 → #25 SWA 定义 → #19–22 Related Work → #17 REH 直觉 → #18 Muon → #27/#28 证明补丁。
- 任何新增改动请同步更新本表的「状态」列。
