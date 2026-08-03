# E9: schedules that preserve the coupling, and a matched-contraction comparison

Reviewer **xkCF**（2026-08-03 follow-up 第 2 点）指出：E8 的 joint multiplier
把 `η_t = η₀m(t)`、`λ_t = λ₀m(t)` 同乘，于是
`η_tλ_t = η₀λ₀m(t)²` —— **恰好破坏了本文主张的耦合**。他给了两个替代方案：

> (a) compare against a schedule with approximately constant `η_tλ_t`, such as
> `λ_t = λ₀·η₀/η_t`; or (b) match methods by the same cumulative contraction
> `Σ_t η_tλ_t`.

E9 就是这两条，都做了。

| 项 | 路径 |
|---|---|
| Runner | [`rebuttal/run_nips26_wd_sched.py`](../../run_nips26_wd_sched.py)（`--sweep iso` / `--sweep matched`） |
| 队列脚本 | [`rebuttal/run_nips26_e9_queue.sh`](../../run_nips26_e9_queue.sh) |
| 分析 | [`analysis/nips26_e9_iso_sched.py`](../../../analysis/nips26_e9_iso_sched.py) |
| 逐 run 明细 | [`_data/e9_iso_matched.csv`](../_data/e9_iso_matched.csv) |
| 结果表 | [`_data/e9_table.md`](../_data/e9_table.md) |
| Token | [`_data/e9_tokens.md`](../_data/e9_tokens.md) |
| 图 | `outputs/plots/nips26/e9_iso_matched.png` |
| 训练 CSV | `rebuttal/results/nips26_runs.csv`（`exp∈ {e9_iso, e9_matched}`） |

---

## 1. 设定

| 项 | 值 |
|---|---|
| 模型 / 数据 | ResNet-18 / CIFAR-100 |
| B / T / seed | 128 / 100 / 42 |
| 优化器 | SGDM，`β = 0.9` |
| 学习率 | **cosine 退火**，`η₀ = 0.1` |
| 收缩预算单位 | `C = λ_ref · Σ_tη_t = 1.181`，其中 `λ_ref = 5.982e-4` |

关键实现选择：cosine LR 由 `lr_schedule_fn` **手动**驱动（`scheduler=None`），
而不是 `CosineAnnealingLR`。两者逐 epoch 完全等价（都是每 epoch step 一次，
乘子 `½(1+cos(πt/T))`），但手动驱动使脚本在开跑前就能**解析地**算出
`Σ_tη_tλ_t`，这是预算反解的前提。CSV 里 `scheduler` 仍写 `cosine`。

### λ 形状

| `wd_sched` | `m_λ(t)` |
|---|---|
| `fixed` | 1 |
| `cosine` | `½(1+cos(πt/T))` |
| `linear` | `1 − t/T` |
| `step` | `t/T<0.5 → 1`；`<0.75 → 0.1`；否则 `0.01` |
| `iso_product` | `min(1/m_cos(t), 10)` ← **审稿人的(a)** |

`iso_product` 即 `λ_t = λ₀·η₀/η_t`。cosine 尾部 `m_cos → 0` 会让 λ 放大约
4000 倍，因此乘子上限取 **10×**（等价于在 λ 公式里把 η 下限钉在 `η₀/10`）。
这个截断是**协议的一部分**，在回复里明说，不是偷偷修正。截断后
`η_tλ_t` 在 `m_cos ≥ 0.1` 期间严格恒定，之后随 η 一起衰减。

### 预算反解（审稿人的 (b)）

`Σ_tη_tλ_t` 对 λ₀ 是线性的，所以

```
λ₀ = budget / (steps_per_epoch · Σ_e η₀ m_cos(e) m_λ(e))
```

三档预算 `{C/3, C, 3C}`。选这三档的原因：与 E5b/E5c 的 3×/10× 敏感度实验同一
倍数刻度，且能直接回答「**是预算决定成败，还是形状决定成败**」。

**一致性红利**：`fixed` 形状下 `Σ_tη_tλ = λ₀·Σ_tη_t`，所以预算 `{C/3, C, 3C}`
反解出的 λ₀ 恰好是 `{1.994e-4, 5.982e-4, 1.795e-3}` —— 正是 **E5b 的
wrong-C 扫描点**和 **E4-ours 基线**。这三格直接命中已有 run被复用，既省卡
又保证与已发布数字逐字一致。

---

## 2. 结果

数字全部来自 [`_data/e9_table.md`](../_data/e9_table.md)，此处只摘要点。

### (a) 匹配累积收缩后（同一 `Σ_tη_tλ_t`）

| budget | fixed | cosine | linear | step | **iso_product** |
|---|---:|---:|---:|---:|---:|
| `C/3` | 74.62 | 74.34 | 74.43 | 74.16 | **75.86** |
| `C` | 76.72 | 75.50 | 76.44 | 75.85 | **78.22** |
| `3C` | 76.19 | 75.03 | 75.53 | 73.97 | **77.50** |

- **同预算下形状仍有差别**：spread 1.70 / 2.72 / 3.53 pp（预算越大差别越大）。
  所以「只要收缩预算相同就等价」是不成立的 —— 收缩**怎么在时间上分布**也重要。
- **保持耦合的形状在每一档预算上都最好**。`iso_product` 相对同预算的常数λ
  分别 +1.24 / +1.50 / +1.31 pp。
- 固定形状、改变预算的 spread 是 1.16–2.36 pp，与同预算改形状的 spread 同量级：
  两个因素都不可忽略。

### (b) `iso_product` 在标准 λ₀ 梯上

| λ₀ | 实测 `Σηλ` | best acc | train acc |
|---:|---:|---:|---:|
| 1e-4 | 0.29 C | 75.56 | 100.0 |
| **5e-4** | 1.44 C | **77.98** | 99.5 |
| 1e-3 | 2.88 C | 77.98 | 96.1 |
| 2e-3 | 5.76 C | 70.36 | 79.5 |
| 5e-3 | 14.39 C | 22.13 | 21.1 |

### (c) 与同 `η₀`、T、优化器的所有参照点比较

| method | best acc |
|---|---:|
| **E9 iso-product（matched at `C`）** | **78.22** |
| E9 iso-product（λ₀ 梯峰值） | 77.98 |
| cosine LR + 常数 λ（oracle over λ₀） | 77.28 |
| cosine LR + 常数 λ（default 5e-4） | 76.73 |
| cosine LR + 常数 λ（ours `C/Σηₜ`） | 76.72 |
| joint `m(t)` on η and λ（最好形状 cosine） | 76.42 |
| 固定 LR + scheduled λ（最好形状 step） | 73.10 |
| 固定 LR + 常数 λ | 66.67 |

---

## 3. 结论（给审稿回复用）

1. **审稿人是对的**：joint multiplier 让 `η_tλ_t ∝ m(t)²`，不保持耦合，因此
   E8 §2 那组对比不能用来评价本文的规则。
2. 换成他建议的 `λ_t = λ₀η₀/η_t` 后，结果**更支持**本文的观点而非更弱：
   保持 `η_tλ_t` 恒定的调度是所有被测调度中最好的，**78.22**，比常数 λ 的
   per-cell oracle（77.28）高 0.94 pp，比 joint cosine（76.42）高 1.80 pp。
3. 匹配累积收缩后形状仍有 1.7–3.5 pp 的差别，说明 `Σ_tη_tλ_t` 是**必要但不充分**
   的描述量；本文规则给出的是该预算该取多大（`≈ C`），而调度形状决定这份预算
   怎么花。两者互补。
4. 诚实边界：`iso_product` 的 10× 截断是必要的数值处理；`Σηλ` 在 `m_cos < 0.1`
   的尾部不再恒定（实测值已在表中列出）。单seed。

---

## 4. 成本与复现

新跑 **17** 个100-epoch run（iso 5 + matched 15 − 3 复用），4×A100 中只用
GPU 1,2,3、`workers_per_gpu=2`，约 1.5 小时。

```bash
cd /home/yzy/GitHub/wd
GPUS=1,2,3 WPG=2 bash rebuttal/run_nips26_e9_queue.sh
# 或分步
PY=/home/yzy/.conda/envs/trace/bin/python
$PY rebuttal/run_nips26_wd_sched.py --sweep iso     --phase sgdm --gpus 1,2,3
$PY rebuttal/run_nips26_wd_sched.py --sweep matched --phase sgdm --gpus 1,2,3
$PY -m analysis.nips26_e9_iso_sched
```

中断后续跑按 `(model,B,lr,wd,mom,epochs,scheduler,seed,wd_sched)` 去重跳过。
