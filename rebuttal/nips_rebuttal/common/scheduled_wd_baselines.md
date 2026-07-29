# Scheduled Weight Decay Baselines

审稿人要求对比至少一种 **scheduled weight-decay** baseline。
本页汇总固定学习率下的 λ-schedule、AdamW 式 joint multiplier、以及与
**cosine LR + 常数 λ**（主实验设定）的对照。Schedule 形状取自
Loshchilov & Hutter（AdamW / SGDW），不是 Xie et al.。

| 项 | 路径 |
|---|---|
| Runner | `rebuttal/run_nips26_wd_sched.py` |
| 队列脚本 | `rebuttal/run_nips26_e8_followup.sh` |
| 分析 | `analysis/nips26_e8_wd_sched.py` |
| 训练结果 | `rebuttal/results/nips26_runs.csv` |
| 峰表 | `_data/e8_wd_sched_table.md`、`_data/e8_followup_table.md` |
| 图 | `outputs/plots/nips26/e8_wd_sched_{sgd,sgdm}.png`、`e8_{joint,long}_sgdm.png` |

统一设定（除非另注）：ResNet-18 / CIFAR-100，`B = 128`，`η₀ = 0.1`，seed 42。
λ₀ 网格：`{1e-4, 5e-4, 1e-3, 2e-3, 5e-3}`（T=200 子网格略少）。
主数字均为每个 schedule 在网格上的 **best test accuracy**。

---

## 1. 固定 LR，只 schedule λ（T=100）

学习率全程钉死 `η = 0.1`（无 CosineAnnealingLR）。每个 epoch 写
`weight_decay = λ₀ · m(t)`：

| schedule | `m(t)` |
|---|---|
| fixed | 1 |
| cosine | `½(1 + cos(π t / T))` → 0 |
| linear | `1 − t/T` |
| step | `t/T < 0.5 → 1`；`< 0.75 → 0.1`；否则 `0.01` |
| cosine_restarts | SGDR，`Te = 50`，`Tmult = 2` |

| optimizer | fixed | cosine | linear | step | cosine_restarts |
|---|---:|---:|---:|---:|---:|
| SGD (mom=0) | 73.20 | 73.50 (+0.30) | 73.22 (+0.02) | **74.24 (+1.04)** | 73.43 (+0.23) |
| SGDM (mom=0.9) | 66.67 | 71.34 (+4.67) | 70.32 (+3.65) | **73.10 (+6.43)** | 70.13 (+3.46) |

要点：

- 固定 LR 下常数 λ 很脆；对 λ 做衰减能托住更大的 λ₀。
- **drop-step 峰值最高**（SGD +1.0pp，SGDM +6.4pp）。SGDM 增益更大，因为
  momentum + 固定 η 与常数正则特别不匹配。
- cosine-with-restarts 在 T=100、一次 mid-run restart 下并不更好。

---

## 2. Joint multiplier：同一 `m(t)` 乘 η 与 λ（SGDM，T=100）

更接近 AdamW/SGDW 原文：`η_t = η₀·m(t)`，`λ_t = λ₀·m(t)`。
`fixed` ≡ 常数 LR + 常数 λ（与上表 SGDM fixed 同点）。

| schedule | peak_acc | peak λ₀ | Δ vs fixed |
|---|---:|---:|---:|
| fixed | 66.67 | 1e-4 | — |
| cosine | **76.42** | 1e-3 | **+9.75** |
| linear | 76.17 | 1e-3 | +9.50 |
| step | 75.36 | 5e-4 | +8.69 |
| cosine_restarts | 75.11 | 1e-3 | +8.44 |

对照 §1：固定 η 时最好的 λ-schedule 只到 **73.10**；joint 退火到 **~76%**。
大部分增益来自 **η 退火**，不是单独 schedule λ。

---

## 3. 更长训练 + restarts（SGDM，T=200，joint）

`Te = 50`，`Tmult = 2`（两次 restart 周期）。

| schedule | peak_acc | peak λ₀ | Δ vs fixed |
|---|---:|---:|---:|
| fixed | 62.39 | 5e-4 | — |
| step | 76.31 | 5e-4 | +13.92 |
| cosine_restarts | **76.88** | 1e-3 | **+14.49** |

T=200 下 cosine-with-restarts 略优于 step；相对无退火 baseline 增益更大。
峰值仍在 ~77%，与下面 cosine LR + 常数 λ 同量级。

---

## 4. 对照：cosine LR + 常数 λ（主实验设定）

同一 `η₀ = 0.1`、T=100、SGDM。左侧是论文默认训练方式（LR 用 cosine，
λ 全程常数）；右侧是 §2 的 joint schedule。

| method | best_acc | note |
|---|---:|---|
| cosine LR + fixed λ（default `5e-4`） | 76.73 | 常用默认 |
| cosine LR + fixed λ（`λ = C / ∑η`） | 76.72 | `λ ≈ 5.982×10⁻⁴` |
| cosine LR + fixed λ（oracle over λ₀） | **77.28** | `λ₀ = 1e-3` |
| joint cosine（peak over λ₀） | 76.42 | AdamW-style |
| joint linear | 76.17 | |
| joint step | 75.36 | |
| joint cosine_restarts | 75.11 | |
| const LR + fixed λ | 66.67 | 无任何退火 |

---

## 5. 结论（给审稿回复用）

1. **缺 LR 退火时**，run 内衰减 λ（或 η+λ 同乘）帮助很大——SGDM 固定 LR
   下 drop-step **+6.4pp**；joint cosine **+9.8pp**。
2. **一旦采用主设定（cosine LR + 常数 λ）**，默认 / `C/∑η` 已达
   **76.7%**，oracle **77.3%**，与 joint cosine **76.4%** 持平或更好。
3. Scheduled WD 是时间上重分配正则；我们的规则是跨设置选一个常数
   `λ* ≈ C / ∑_t η_t`。二者可组合，**不互相替代**。

复现：

```bash
PY=/home/yzy/.conda/envs/trace/bin/python
$PY rebuttal/run_nips26_wd_sched.py --phase all --gpus 0,1,2,3 --workers_per_gpu 2
bash rebuttal/run_nips26_e8_followup.sh
$PY -m analysis.nips26_e8_wd_sched
```



**Within-run scheduled weight decay.** Separately from across-run scaling rules,
we hold the learning rate fixed at `η = 0.1` (ResNet-18 / CIFAR-100, `B = 128`,
`T = 100`) and compare a constant `λ` against the AdamW/SGDW schedule shapes of
Loshchilov & Hutter applied *only* to weight decay: cosine, linear, drop-step,
and cosine-with-restarts (`Te = 50`, `Tmult = 2`). Each schedule is swept over
`λ₀ ∈ {1e-4, 5e-4, 1e-3, 2e-3, 5e-3}`; we report the best accuracy in that grid.

| optimizer | fixed | cosine | linear | step | cosine_restarts |
|---|---:|---:|---:|---:|---:|
| SGD (mom=0) | 73.20 | 73.50 (+0.30) | 73.22 (+0.02) | **74.24 (+1.04)** | 73.43 (+0.23) |
| SGDM (mom=0.9) | 66.67 | 71.34 (+4.67) | 70.32 (+3.65) | **73.10 (+6.43)** | 70.13 (+3.46) |

Under a fixed learning rate, decaying `λ` helps — especially for SGDM, where a
constant `λ` is badly mismatched to the lack of LR annealing, and drop-step
recovers **+6.4** points.

We also apply the same AdamW-style multiplier *jointly* to `η` and `λ`
(SGDM, `T = 100`): joint cosine peaks at **76.42**, versus **76.72–76.73** for
cosine LR with a fixed `λ` (our `C/∑η` rule / the common default `5e-4`) and
**77.28** for an oracle over the same `λ₀` grid. Extending to `T = 200` with
restarts, joint cosine-with-restarts reaches **76.88**, still in the same range.
So within-run scheduling is useful when the learning rate is held fixed, but
under our default cosine-LR setup a constant `λ` already matches or beats it.

That is a different knob from our claim: we select one constant `λ` as a
function of `(η, T, B)`, whereas these schedules redistribute regularization
over time. Both can be used together; they do not substitute for each other.
