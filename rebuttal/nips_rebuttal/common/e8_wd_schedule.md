# E8: Fixed Learning Rate vs Scheduled Weight Decay

xkCF Q3 要求对比至少一种 existing scaling rule **或** scheduled weight-decay baseline。
E4 已经覆盖「跨 run 选常数 λ」的 scaling-rule 对比；本实验 **E8** 单独测
**run 内对 λ 做 schedule**（学习率固定），形状取自 Loshchilov & Hutter
(AdamW / SGDW) 的 schedule multiplier，而不是 Xie et al.。

相关代码与产物：

| 项 | 路径 |
|---|---|
| Runner | [`rebuttal/run_nips26_wd_sched.py`](../../rebuttal/run_nips26_wd_sched.py) |
| 分析 | [`analysis/nips26_e8_wd_sched.py`](../../analysis/nips26_e8_wd_sched.py) |
| 结果表 | [`_data/e8_wd_sched_peaks.csv`](../_data/e8_wd_sched_peaks.csv) |
| 峰表 markdown | [`_data/e8_wd_sched_table.md`](../_data/e8_wd_sched_table.md) |
| 图 SGD | `outputs/plots/nips26/e8_wd_sched_sgd.png` |
| 图 SGDM | `outputs/plots/nips26/e8_wd_sched_sgdm.png` |
| 训练 CSV | `rebuttal/results/nips26_runs.csv`（`exp=e8_wd_sched`） |

---

## 1. 设定

| 项 | 值 |
|---|---|
| 模型 / 数据 | ResNet-18 / CIFAR-100 |
| B / T / seed | 128 / 100 / 42 |
| 学习率 | **全程固定** `η = 0.1`（无 CosineAnnealingLR） |
| Phase A | SGD，`momentum = 0` |
| Phase B | SGDM，`momentum = 0.9` |
| λ₀ 网格 | `{1e-4, 5e-4, 1e-3, 2e-3, 5e-3}` |

每个 epoch 开始时写入 `optimizer.param_groups[*]['weight_decay']`：

| `wd_sched` | 公式（`t` = 0-based epoch，`T = 100`） |
|---|---|
| `fixed` | `λ_t = λ₀` |
| `cosine` | `λ_t = λ₀ · ½(1 + cos(π t / T))` → 0 |
| `linear` | `λ_t = λ₀ · (1 − t / T)` |
| `step` | `t/T < 0.5 → λ₀`；`< 0.75 → 0.1 λ₀`；否则 `0.01 λ₀` |
| `cosine_restarts` | SGDR，`Te = 50`，`Tmult = 2`（t=50 重启一次） |

公平比法：同一 λ₀ 画曲线；主数字取每个 `(optimizer, schedule)` 在网格上的
**best test accuracy**，以及相对 `fixed` 的增益。

SGDM 的 `fixed` 臂尽量复用已有 `scheduler=const` 格子（缺 `λ₀=2e-3` 时补跑）。

---

## 2. 结果（peak over λ₀）

| optimizer | fixed | cosine | linear | step | cosine_restarts |
|---|---:|---:|---:|---:|---:|
| **SGD** (mom=0) | 73.20 | 73.50 (+0.30) | 73.22 (+0.02) | **74.24 (+1.04)** | 73.43 (+0.23) |
| **SGDM** (mom=0.9) | 66.67 | 71.34 (+4.67) | 70.32 (+3.65) | **73.10 (+6.43)** | 70.13 (+3.46) |

Peak λ₀：

| optimizer | fixed | cosine | linear | step | cosine_restarts |
|---|---:|---:|---:|---:|---:|
| SGD | 1e-4 | 5e-4 | 5e-4 | **1e-3** | 1e-4 |
| SGDM | 1e-4 | 5e-4 | 1e-4 | **5e-4** | 1e-4 |

曲线见上表链接的两张图。要点：

1. **固定 LR 下，常数 λ 很脆**：λ₀ 稍大准确率掉得很快（SGD fixed 在 5e-3 到
   ~57%；SGDM fixed 到 ~21%）。
2. **对 λ 做衰减能托住高 λ₀**：cosine / linear / step / restarts 在大 λ₀ 处
   都明显高于 fixed。
3. **drop-step 峰值最高**（SGD +1.0pp，SGDM +6.4pp）。SGDM 增益更大，因为
   mom=0.9 + 固定 η=0.1 时，常数 λ 与「缺 LR 退火」特别不匹配。
4. **cosine-with-restarts 并未更好**：在本预算（T=100、一次 mid-run restart）
   下不如 plain step / cosine。

---

## 3. 和论文主张怎么对齐

- **E8 测的是**：给定固定 η，run 内 `λ(t)` 是否比常数 λ 更好。
- **本文主张测的是**：跨设置选一个**常数** `λ* ≈ C / Σ_t η_t`（E1/E4）。
- 二者是不同旋钮：schedule 重分配正则化的时间分布；我们的规则预测该用多大的
  常数。可以组合（cosine LR + 常数 λ，或固定 LR + scheduled λ），**不互相替代**。
- 主实验默认本来就是 **cosine LR + 常数 λ**；E8 故意把 LR 钉死，才能把
  「只 schedule λ」的效果从 LR 退火里拆出来。

审稿回复里（`xkCF/response.md` Q3、`AC_vXFZ/response.md` §3）已用上表数字。

---

## 4. 成本与复现

- 格子：SGD 25 + SGDM 21 新跑（4 个 fixed 复用）≈ **46** 单位。
- 复现：

```bash
cd /home/yzy/GitHub/wd
PY=/home/yzy/.conda/envs/trace/bin/python
$PY rebuttal/run_nips26_wd_sched.py --phase all --gpus 0,2,3 --workers_per_gpu 2
$PY -m analysis.nips26_e8_wd_sched
```

中断后续跑会按 `(model,B,lr,wd,mom,epochs,scheduler,seed,wd_sched)` 去重跳过。

---

## 5. 未做（有充裕再加）

- 同样 schedule，但学习率也用 cosine（更接近 AdamW 原文「同一 multiplier 乘
  η 与 λ」）。
- 更长 T / 多次 restart（SGDWR 完整设定）。
- 与 E4 的「选常数 λ」在同一张表里直接比最终准确率（需统一 LR schedule）。
