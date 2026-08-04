# Placeholder registry

Every `[[TOKEN]]` in the reviewer responses is a number that is not measured
yet. Fill them from the experiment listed here, then delete the row.

Run `python -m analysis.nips26_report` after each wave: it prints the current
value of every token that can already be resolved.

| Token | Filled by | Meaning | Status |
|---|---|---|---|
| `[[E1-T-SLOPE]]` | E1-fine | slope of `log lambda*` vs `log T` at eta=0.1 | resolved (−0.226) |
| `[[E1-T-CI]]` | E1-fine | 95% bootstrap interval for the above | resolved ([−0.28, −0.17]) |
| `[[E1-T-LAMBDA-25]]` | E1-fine | lambda* at T=25 | resolved (0.0012) |
| `[[E1-T-LAMBDA-100]]` | E1-fine | lambda* at T=100 | resolved (0.000877) |
| `[[E1-T-LAMBDA-200]]` | E1-fine | lambda* at T=200 | resolved (0.000737) |
| `[[E1-PRODUCT-DRIFT]]` | E1-fine | factor by which the optimal `eta*lambda` moves from T=25 to T=200 | resolved (1.62×) |
| `[[E1-LOWLR-SLOPE]]` | E1-full / rescue | same slope measured at eta=0.02 | resolved (−0.52) |
| `[[E1-ETAT-COLLAPSE]]` | E1-full / rescue | residual spread of C after collapsing both eta arms onto `sum_lr` | resolved |
| `[[E1-SCHED-RATIO]]` | E1-rescue | ratio of lambda* between constant-LR and cosine at matched eta, T | pending (ladder edge) |
| `[[E2B-ISO-DROP]]` | E2b | accuracy lost at the ends of the iso-product line relative to its peak | resolved (10.3) |
| `[[E2B-ISO-RANGE]]` | E2b | eta range over which the iso-product line stays within 1 point of its peak | resolved (factor of 5) |
| `[[E3-SLOPE]]` | E3 | fitted slope of `1/eta_max` against lambda (theory: 1) | resolved (0.67) |
| `[[E3-INTERCEPT]]` | E3 | intercept of that fit, i.e. the implied `L/2` | resolved (0.08) |
| `[[E3-LMAX]]` | E3 Hessian | top Hessian eigenvalue measured by power iteration | resolved (417.4 at epoch 15, λ=0) |
| `[[E3-MOM-RATIO]]` | E3 | ratio of eta_max between beta=0 and beta=0.9 (theory: 1-beta = 0.1) | resolved (0.23) |
| `[[E4-TABLE]]` | E4 | full accuracy-gap-to-oracle table over the five strategies | resolved (`_data/e4_transfer_table.md`) |
| `[[E4-OURS-MEAN]]` | E4 | mean gap to oracle, our rule | resolved (0.87) |
| `[[E4-OURS-WORST]]` | E4 | worst gap to oracle, our rule | resolved (1.58) |
| `[[E4-DEFAULT-MEAN]]` | E4 | mean gap to oracle, fixed lambda=5e-4 | resolved (0.70) |
| `[[E4-KOSSON-MEAN]]` | E4 | mean gap to oracle, constant `eta*lambda` | resolved (1.63) |
| `[[E4-WANG-MEAN]]` | E4 | mean gap to oracle, `1/(eta*T)` | resolved (1.60) |
| `[[E5B-3X]]` | E5b (CIFAR, exact `sum_lr`) | accuracy lost when C is wrong by a factor of 3 | resolved (15.11) |
| `[[E5B-10X]]` | E5b (CIFAR, exact `sum_lr`) | accuracy lost when C is wrong by a factor of 10 | resolved (69.74) |
| `[[E5C-C-SGD]]` | **E5c** MNIST-MLP | fitted C under SGD (mom=0) | resolved (0.44) |
| `[[E5C-C-SGDM]]` | **E5c** MNIST-MLP | fitted C under SGDM (mom=0.9) | resolved (0.32) |
| `[[E5C-C-RATIO]]` | E5c | max(C)/min(C) across SGD/SGDM (vs E5a range) | resolved (1.38×) |
| `[[E5C-3X]]` | E5c | cost of wrong C by ×3 (worse of under/over) | resolved (0.018 test-loss) |
| `[[E5C-10X]]` | E5c | cost of wrong C by ×10 (worse of under/over) | resolved (0.083 test-loss) |
| `[[E5C-FIG]]` | E5c | three-panel figure path | resolved (`outputs/plots/nips26/e5c_mnist_mlp_C.png`) |
| `[[E6B-LAMBDA-SLOPE]]` | E6b | slope of `log lambda*` against `log(1-beta)` (prediction: 1) | resolved (0.42) |
| `[[E6B-GAP-SGD]]` | E6b | train-test accuracy gap, SGD+WD at its optimum | resolved (22.6) |
| `[[E6B-GAP-SGDM]]` | E6b | train-test accuracy gap, SGDM+WD at its optimum | resolved (23.7) |
| `[[E7-DIVERGENCE-RATIO]]` | E7a | ratio of `\|\|theta_T - theta'_T\|\|` between lambda=0 and lambda>0 | resolved (1.10) |
| `[[E7-PLATEAU]]` | E7a | whether the weight-decay run plateaus, and at what step | resolved (yes under λ=1e-3) |
| `[[E7-BN]]` | E7c | whether the coupling survives without normalization layers | resolved (yes) |

E8 scheduled-WD 对比见 [`common/e8_wd_schedule.md`](common/e8_wd_schedule.md)（无 `[[TOKEN]]`，数字已写死）。

## Follow-up 2026-08-03（reviewer xkCF：E9 / E10 两点）

E9/E10 的数字**全部由脚本产出**，已直接写进 `xkCF/response.md` 的
"Follow-up (2026-08-03)" 一节（无未解析占位符）。来源与复现口径：

| 组 | 生成脚本 | 产物 | 关键数字 |
|---|---|---|---|
| **E9** iso-product / matched-contraction | `rebuttal/run_nips26_wd_sched.py --sweep iso\|matched` → `analysis/nips26_e9_iso_sched.py` | `_data/e9_iso_matched.csv`、`_data/e9_table.md`、`_data/e9_tokens.md`、`outputs/plots/nips26/e9_iso_matched.png` | `C = 1.181`；matched 最佳 **78.22**（iso @ `C`）；同预算跨形状 spread **1.70 / 2.72 / 3.53 pp**；iso 梯峰值 **77.98** |
| **E10** held-out `C`（宽度） | `mlp_wd/scripts/run_e10_c_width.py` → `mlp_wd/analysis/report_e10_c_width.py` | `_data/e10_predictions_{mnist,cifar10}.json`、`_data/e10_heldout_table_*.md`、`_data/e10_heldout_*.csv`、`_data/e10_C_by_width_*.csv`、`_data/e10_tokens_*.md`、`outputs/plots/nips26/e10_c_width_*.png` | C-vs-width slope **−0.00 [−0.24,+0.24]**(SGD) / **−0.27 [−0.40,−0.14]**(SGDM)；C 外推误差 **1.03–1.28×**；gap(ours) MNIST **0.08/0.19 pp**、CIFAR-10 **2.01/3.79 pp** |

实验笔记：[`common/e9_iso_matched.md`](common/e9_iso_matched.md)、
[`common/e10_c_width.md`](common/e10_c_width.md)。

**盲预测证据**：E10 的 `predict` 阶段在任何 held-out 训练开始前把`λ_pred`
连时间戳写入 `_data/e10_predictions_<ds>.json`，两份文件均为 `blind: true`；
`predict` 默认拒绝覆盖已有预测文件。

**需要改正文（paper edit，自行修正，非审稿意见）**：删掉 Exp. 3 中
「optimal `λ` grows with batch size」这句，改为「在线性 LR scaling 下 `ηλ*`
随 B 增长而 `λ*` 基本不动，因为 `Σ_tη_t` 被固定住了；`λ*∝B` 是**固定 η** 下的
读法」。C 在 B 上的残余漂移 1.38/1.89/1.40/1.60/1.71 与已发布数字逐字一致，
不存在 Eq.(17) 与 Table 4 的矛盾，故 reviewer 的 F1 不采纳。

## Numbers that are already measured

These are in the drafts as literal values, from Wave 0 (no new training):

- envelope varies by 2.8 points over two decades of learning rate
- fixed `lambda = 5e-4` gives up 1.44 points on average, 3.77 at worst
- no fixed `lambda` does better than 3.77 points worst-case
- `log lambda*` vs `log eta` slope: -0.87, 95% interval [-0.94, -0.75]
- `C = lambda* * sum_lr`: geometric mean 1.48, spread x/1.70, 65 settings
- C by architecture: ResNet-18 1.42, ResNet-50 1.42, VGG-16 1.72
- momentum at its own optimal eta changes accuracy by 0.25 points
  (`lambda=0`, beta in [0, 0.8]) and 1.38 points (`lambda=2e-3`, beta in [0, 0.95])
- `log eta*` vs `log(1-beta)` slope: 1.24 [0.80, 2.32] without weight decay,
  0.72 [0.58, 0.93] with it

## E5b note (CIFAR)

Re-ran with exact `sum_lr`. Tokens filled: `[[E5B-3X]]=15.11`,
`[[E5B-10X]]=69.74` (worst accuracy drop across the two `(η,T)` settings).
Asymmetry: undershooting `C` by 3–10× costs ≤~5 pp; overshooting dominates.