# E8: Fixed Learning Rate vs Scheduled Weight Decay

xkCF Q3 要求对比至少一种 existing scaling rule **或** scheduled weight-decay baseline。
E4 已经覆盖「跨 run 选常数 λ」的 scaling-rule 对比；本实验 **E8** 单独测
**run 内对 λ 做 schedule**（学习率固定），形状取自 Loshchilov & Hutter
(AdamW / SGDW) 的 schedule multiplier，而不是 Xie et al.。

相关代码与产物：

| 项 | 路径 |
|---|---|
| Runner | [`rebuttal/run_nips26_wd_sched.py`](../../rebuttal/run_nips26_wd_sched.py) |
| Follow-up 队列 | [`rebuttal/run_nips26_e8_followup.sh`](../../rebuttal/run_nips26_e8_followup.sh) |
| 分析 | [`analysis/nips26_e8_wd_sched.py`](../../analysis/nips26_e8_wd_sched.py) |
| 结果表 | [`_data/e8_wd_sched_peaks.csv`](../_data/e8_wd_sched_peaks.csv) |
| 峰表 markdown | [`_data/e8_wd_sched_table.md`](../_data/e8_wd_sched_table.md) |
| Follow-up 表 | [`_data/e8_followup_table.md`](../_data/e8_followup_table.md) |
| 图 SGD / SGDM | `outputs/plots/nips26/e8_wd_sched_{sgd,sgdm}.png` |
| 图 joint / long | `outputs/plots/nips26/e8_{joint,long}_sgdm.png` |
| 训练 CSV | `rebuttal/results/nips26_runs.csv`（`exp∈{e8_wd_sched,e8_joint,e8_long,e8_e4_baseline}`） |

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

- E8 主实验：SGD 25 + SGDM 21 新跑（4 个 fixed 复用）≈ **46** 单位。
- Follow-up（§5）：joint T=100 ≈ 25 + long T=200 ≈ 9 + E4 baseline 复用 ≈ **~34** 单位。
- 复现：

```bash
cd /home/yzy/GitHub/wd
PY=/home/yzy/.conda/envs/trace/bin/python
$PY rebuttal/run_nips26_wd_sched.py --phase all --gpus 0,2,3 --workers_per_gpu 2
bash rebuttal/run_nips26_e8_followup.sh   # joint / long / e4 baseline
$PY -m analysis.nips26_e8_wd_sched
```

中断后续跑会按 `(model,B,lr,wd,mom,epochs,scheduler,seed,wd_sched)` 去重跳过。

---

## 5. Follow-up（已完成）

统一设定：SGDM，`η₀ = 0.1`，`B = 128`，R18/CIFAR-100。完整表见
[`_data/e8_followup_table.md`](../_data/e8_followup_table.md)。

### 5.1 Joint multiplier（同一 `m(t)` 乘 η 与 λ；T=100）

更接近 AdamW/SGDW 原文：每个 epoch `η_t = η₀·m(t)`，`λ_t = λ₀·m(t)`（不用
CosineAnnealingLR）。`fixed` ≡ 常数 LR + 常数 λ（与 §2 SGDM fixed 同点）。

| wd_sched | peak_acc | peak λ₀ | Δ vs fixed |
|---|---:|---:|---:|
| fixed | 66.67 | 1e-4 | — |
| cosine | **76.42** | 1e-3 | **+9.75** |
| linear | 76.17 | 1e-3 | +9.50 |
| step | 75.36 | 5e-4 | +8.69 |
| cosine_restarts | 75.11 | 1e-3 | +8.44 |

要点：joint 退火把 SGDM 从 ~67% 拉到 ~76%；**大部分增益来自 η 退火**，不是
单独 schedule λ（对照 §2：固定 η 时 step 只到 73.10）。

### 5.2 更长 T / restarts（T=200，joint；`Te=50`, `Tmult=2`）

| wd_sched | peak_acc | peak λ₀ | Δ vs fixed |
|---|---:|---:|---:|
| fixed | 62.39 | 5e-4 | — |
| step | 76.31 | 5e-4 | +13.92 |
| cosine_restarts | **76.88** | 1e-3 | **+14.49** |

T=200 下 cosine-with-restarts 略优于 step；相对 fixed 的增益更大（固定 η 更久
更脆）。峰值仍在 ~77%，与 T=100 cosine LR + 常数 λ 同量级。

### 5.3 vs E4：选常数 λ（cosine LR） vs joint schedule

同一 `η₀=0.1`、T=100、SGDM：

| method | best_acc | note |
|---|---:|---|
| cosine LR + fixed λ（default 5e-4） | 76.73 | 常用默认 |
| cosine LR + fixed λ（E4-ours `C/∑η`） | 76.72 | `λ=5.982e-4` |
| cosine LR + fixed λ（oracle over λ₀） | **77.28** | `λ₀=1e-3` |
| joint `m(t)=cosine`（peak over λ₀） | 76.42 | AdamW-style |
| joint `m(t)=linear` | 76.17 | |
| joint `m(t)=step` | 75.36 | |
| joint `m(t)=cosine_restarts` | 75.11 | |
| joint / const LR `fixed` | 66.67 | 无退火 |

**结论（对 xkCF Q3）**：

1. Run 内 schedule λ（或 η+λ 同乘）在**缺 LR 退火**时帮助很大。
2. 一旦采用主实验设定（**cosine LR + 常数 λ**），E4/默认常数已达
   **76.7–77.3%**，与 joint cosine **76.42%** 持平或更好——within-run schedule
   **不替代**跨设置选常数 `λ* ≈ C/∑η`。
3. E4 测的是后者；E8 测的是前者；二者可组合，叙事上已对齐。
