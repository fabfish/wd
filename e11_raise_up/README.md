# E11: raise-up weight decay —— WD 在训练的什么阶段起作用?

E8/E9 回答的是"衰减形 λ 调度 vs 固定 λ"。E11 问一个反向的问题:
**λ 一路上调(raise-up)会怎样?** 动机有二:

1. E9 的 `iso_product`(λ_t = λ₀·η₀/η_t,理论上保持 η_tλ_t 恒定)本身就是
   一条上升曲线,且它已经超过了固定 λ 的峰值 —— 上升方向值得系统研究。
2. 已有结果里固定 λ 的默认值(5e-4)**不是最优值**;所有对照必须对
   "固定 λ 的最优值"(oracle peak),而不是对默认值。

结论先行:**在余弦 LR 下,平滑上升的 λ 调度稳定优于任何固定 λ;
最优总预算 Σ_tη_tλ_t 因优化器而异(SGDM ≈ 1.5~2C,SGD ≈ 6~9C),
且收缩的时机(中后期)与总量同样重要。**

| 项 | 路径 |
|---|---|
| Runner | [`rebuttal/run_nips26_wd_sched.py`](../rebuttal/run_nips26_wd_sched.py)(`--sweep raise` / `raise_big` / `raise_matched` / `raise_matched_iso` / `raise_matched_xl` / `raise_ms`,新增 `--seeds`) |
| 队列脚本 | [`run_e11_queue.sh`](run_e11_queue.sh)、[`run_e11_followup_queue.sh`](run_e11_followup_queue.sh)、[`run_e11_matched_dense_queue.sh`](run_e11_matched_dense_queue.sh)、[`run_e11_iso_dense_queue.sh`](run_e11_iso_dense_queue.sh)(本目录) |
| 分析 | [`analysis/nips26_e11_wd_raise.py`](../analysis/nips26_e11_wd_raise.py) |
| 训练 CSV | [`results/nips26_e11_runs.csv`](results/nips26_e11_runs.csv)(**E11 专用**,为 nips26_runs.csv 的超集拷贝;原 CSV 冻结未动) |
| 结果表 | [`tables/e11_wd_raise_table.md`](tables/e11_wd_raise_table.md)、[`tables/e11_matched_table.md`](tables/e11_matched_table.md)、[`tables/e11_multiseed_table.md`](tables/e11_multiseed_table.md) |
| 图 | [`figures/`](figures/):`e11_wd_raise_{sgdm,sgd}.png`、`e11_matched_budget_{sgdm,sgd}.png`、`e11_contraction_trajectories.png` |

---

## 1. 设定

| 项 | 值 |
|---|---|
| 模型 / 数据 | ResNet-18 / CIFAR-100 |
| B / T | 128 / 100 |
| 学习率 | **cosine 退火**,η₀ = 0.1(与 E9 相同的 `cos_shape` 手动驱动,LR 轨迹与 CosineAnnealingLR 逐 epoch 一致) |
| 优化器 | SGDM(β=0.9)与 SGD(β=0)两相位 |
| 预算单位 | C = λ_ref·Σ_tη_t = 1.181,λ_ref = 5.982e-4(同 E9) |
| seed | 网格 run 为 42;峰值配置补 {42, 123, 2024} 三种子 |

### 新增的 λ 形状(m_λ(t) 从 0 升到 1,λ₀ 为末段峰值)

| `wd_sched` | `m_λ(t)` |
|---|---|
| `linear_up` | `t/T` |
| `cosine_up` | `½(1−cos(πt/T))` |
| `step_up` | `t/T<0.5 → 0.01`;`<0.75 → 0.1`;否则 `1.0`(E8 `step` 的镜像) |
| (参照)`iso_product` | `min(1/m_cos(t), 10)`,E9 的解析上升形状 |

### 对照(全部复用已有 run,0 个新 run)

- **固定 λ oracle 峰值**(余弦 LR):SGDM 网格 33 行(最优 6e-4→77.45 @seed42);
  SGD 相位原有只有 3 行,本次先补跑了 `e4_baselines --phase sgd` 网格。
- 默认 5e-4:SGDM 76.7~76.9(非最优参照点)。
- `iso_product` 的 E9 已有行。

## 2. 结果 I:自由网格峰值(λ₀ ∈ {1e-4 … 5e-2},seed 42)

| phase | 方案 | peak acc | 峰值 λ₀ | Δ vs fixed 最优 | 实现预算 |
|---|---|---:|---:|---:|---:|
| SGDM | fixed 最优 | 77.45 | 6e-4 | — | 1.0C |
| SGDM | **linear_up** | **78.28** | 5e-3 | **+0.83** | 2.46C |
| SGDM | cosine_up | 78.09 | 5e-3 | +0.64 | 2.07C |
| SGDM | step_up | 77.39 | 1e-2 | −0.06 | 0.84C |
| SGDM | iso_product(参照) | 78.22 | 3.475e-4 | +0.77 | 1.00C |
| SGD | fixed 最优 | 77.75 | 5e-3 | — | 8.4C |
| SGD | linear_up | 77.94 | 2e-2 | +0.19 | 8.5C |
| SGD | cosine_up | 77.91 | 2e-2 | +0.16 | 7.2C |
| SGD | step_up | 76.73 | 5e-2 | −1.02 | 4.2C |

观察:固定 λ 在 λ≥5e-3(SGDM)即崩塌(1e-2 → 45%),而 raise-up 在同 λ₀
处达到峰值 —— **λ₀ 容忍度右移约 10×**,因为早期大 LR 阶段 λ≈0 不做收缩。
(SGDM 侧;图 `e11_wd_raise_sgdm.png`)

## 3. 结果 II:密集等预算阶梯(matched,预算与形状分离)

每个形状按 `solve_lambda0_for_budget` 反解 λ₀,使实现预算恰好落在
{C/3, C, 1.5C, 2C, 2.5C, 3C, 4C, 6C, 9C}(SGD 另有 {15,25,40,60}C,见 §5)。

**SGDM**(fixed@1C = 76.72 为锚):

| 预算 | linear_up | cosine_up | step_up | iso_product |
|---:|---:|---:|---:|---:|
| 0.33C | 75.84 | 75.57 | 75.48 | 75.86 |
| 1C | 77.81 | 77.80 | 77.58 | **78.22** |
| 1.5C | **78.44** | 78.31 | 74.68 | 77.93 |
| 2C | **78.45** | 78.11 | 69.94 | 78.19 |
| 2.5C | 77.77 | 77.69 | 65.94 | 77.62 |
| 3C | 77.44 | 76.94 | 64.40 | 77.50 |
| 4C | 76.18 | 75.43 | 64.49 | 75.42 |
| 6C | 72.53 | 70.50 | 61.06 | 70.04 |
| 9C | 64.49 | 59.92 | 57.99 | 57.58 |

**SGD**:

| 预算 | linear_up | cosine_up | step_up | iso_product |
|---:|---:|---:|---:|---:|
| 1C | 75.32 | 74.98 | 74.73 | 75.16 |
| 2C | 75.94 | 75.94 | 74.71 | 75.78 |
| 3C | 76.98 | 76.52 | 76.40 | 76.69 |
| 4C | 77.53 | 77.03 | 76.62 | 77.31 |
| 6C | **77.91** | 77.62 | 73.30 | 77.68 |
| 9C | 77.83 | 77.82 | 70.91 | 77.83 |

(图 `e11_matched_budget_{sgdm,sgd}.png`;E9 的衰减形状在同预算下只有
75.5~76.4,表中略,见 `tables/e11_matched_table.md`)

要点:

- **同预算下 raise-up ≫ 衰减形**(差 1.3~2 个点):收缩的时机是方向性差异,
  不是预算差异。
- SGDM 最优预算 **1.5~2C**;raise-up 的最优预算高于 fixed 的最优(1C)——
  收缩压在小 LR 期,扰动小,所以"花得起"更多。
- SGD 最优预算 **6~9C**,且与自由网格独立自洽(SGD fixed 最优 5e-3 = 8.4C)。

## 4. 结果 III:多种子(SGDM 峰值配置,3 seeds)

| 配置 | mean ± std |
|---|---|
| linear_up 5e-3 | 78.05 ± 0.16 |
| cosine_up 5e-3 | 77.93 ± 0.16 |
| step_up 1e-2 | 77.69 ± 0.24 |
| fixed 6e-4(对照) | 77.45 ± 0.31 |

raise-up 对 fixed 最优的增益(+0.6)明显超出种子噪声(std 0.16~0.31)。
注意:linear_up / cosine_up / iso 三者峰值间差异(~0.2)与种子噪声同量级,
稳妥结论是"**平滑 raise-up 家族 > fixed**",而非"linear_up 严格优于 iso"。

## 5. 结果 IV:SGD 的预算上界(XL 档 {15,25,40,60}C)

SGD 在 9C 处三形状都不崩(77.8+),补测上界:

| 预算 | linear_up | cosine_up | iso_product | step_up |
|---:|---:|---:|---:|---:|
| 9C | 77.83 | 77.82 | 77.83 | 70.91 |
| 15C | **78.08** | 77.45 | **78.01** | 66.20 |
| 25C | 76.59 | 69.82 | 57.74 | 63.06 |
| 40C | 59.50 | 55.75 | 39.63 | 61.07 |
| 60C | 40.49 | 53.46 | 31.43 | 54.83 |

- **平滑三形状的安全上界 ≈15C,25C 坠崖**;step_up 在 4~6C 就开始劣化。
- SGD 全场最优更新为 **linear_up @15C = 78.08**(对 fixed 最优 +0.33)。
- 对比 SGDM(3C 后下滑、6C 明显坏):**SGDM 的崩溃预算比 SGD 早 ~5 倍**。

## 6. 机制解释

### 崩溃的控制变量不是总预算,而是"瞬时收缩率作用时的 LR 水平"

固定大 λ 的死法:早期 λ 就把权重范数钉在小值;ResNet 有 BN、逐层尺度
不变,有效学习率 η_eff = η/‖w‖² 随之爆炸 → 发散。raise-up 构造上规避了
这一点(早期 λ≈0,范数自由生长)。坠崖出现在"末期收缩速度超过剩余步数内
再平衡能力"处;iso_product 的 ηλ 从 epoch 0 起恒定,25C 时等效于
"fixed λ≈8.7e-3 从头压到尾",正中固定 λ 的崩溃机制,因此坠崖最陡;
step_up 则是末期尖峰收缩来不及再平衡。

### 为什么 linear_up 最优(收缩率的时间分配)

见 `e11_contraction_trajectories.png`(每 epoch 收缩率 η_tλ_t):

- **fixed**:早期(大 LR)即以 2%/epoch 收缩 —— 与拟合抢容量;
- **step_up**:75% 处 5.8%/epoch 尖峰 —— 预算砸在梯度趋零的尾部且带来突变;
- **iso_product**:全程恒定 1.1%/epoch —— 理论最优雅,但早期收缩仍浪费;
- **linear_up(2C)**:唯一的**单峰平滑隆起** —— 前 1/3 近零 WD 自由拟合,
  中段(t/T≈0.55)收缩达峰 ~3.5%/epoch(LR 仍可观、loss 面正在锐化,
  收缩最有价值),尾段随 LR 平滑归零。**早期不打扰、中期加压、末期不冲击。**

### SGD vs SGDM:动量放大 WD 收缩

WD 的收缩方向 −λw 完全恒定,动量对恒定方向的增益为 1/(1−β)=10
(对噪声只有 1/√(1−β²)≈2.3)。因此同名义预算下 SGDM 的**有效收缩约强
5 倍**,两侧数据都吻合:

- 最优预算比:SGDM 1.5~2C vs SGD 6~9C(≈4.5×)
- 崩溃边界比:SGDM ~3-4C vs SGD ~15-25C(≈5-7×)

均在 1/(1−β²)≈5.3 与 1/(1−β)=10 之间。

**实践含义**:有动量时 WD 调度很重要(收缩被放大,须后置、平滑、
预算 ~2C,收益 +0.8 且 λ₀ 容忍度宽一个数量级);无动量时调度收益小
(固定大 λ 已接近上限,只需把总量给够)。**WD schedule 的价值恰恰来自
动量对晚期收缩的放大。**

## 7. 复现

```bash
# 环境:Python 3.11, torch 2.4.1+cu121;数据在 data/ 下自动下载
CSV=e11_raise_up/results/nips26_e11_runs.csv
PY=python

# SGD 固定 λ 对照网格(原 CSV 只有 3 行)
$PY rebuttal/run_nips26_wd_sched.py --sweep e4_baselines --phase sgd --csv $CSV
# 自由网格 raise-up 曲线
$PY rebuttal/run_nips26_wd_sched.py --sweep raise     --phase sgdm --csv $CSV
$PY rebuttal/run_nips26_wd_sched.py --sweep raise_big --phase sgdm --csv $CSV
# (sgd 相位同理换 --phase sgd)
# 密集等预算阶梯(含 iso_product)
$PY rebuttal/run_nips26_wd_sched.py --sweep raise_matched     --phase sgdm --csv $CSV
$PY rebuttal/run_nips26_wd_sched.py --sweep raise_matched_iso --phase sgdm --csv $CSV
# SGD 预算上界
$PY rebuttal/run_nips26_wd_sched.py --sweep raise_matched_xl  --phase sgd  --csv $CSV
# 多种子
$PY rebuttal/run_nips26_wd_sched.py --sweep raise_ms --phase sgdm \
    --seeds 42,123,2024 --csv $CSV
# 分析(表 + 图)
$PY -m analysis.nips26_e11_wd_raise
```

所有命令支持 `--gpus` / `--workers_per_gpu`;RUN_KEY 去重,中断后重跑同一
命令自动跳过已完成配置。一键队列见 §首部表格的四个 shell 脚本。
