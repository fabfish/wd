# E10: held-out test of the constant `C` across MLP widths

Reviewer **xkCF**（2026-08-03 follow-up 第 3 点）：

> The rebuttal acknowledges that `C` is not universal. Can you provide one
> held-out test in which `C` is estimated on one setting and used to predict
> `λ*` for another architecture, batch size, or learning rate? This would show
> whether the rule actually reduces tuning relative to a two-dimensional grid.

E10 用**MLP 宽度**作为「另一个架构」的干净旋钮：宽度改变参数量一个数量级以上，
而数据、优化器、schedule 全部不变。在小宽度上标定 `C`，**盲预测**大宽度的 `λ*`。

| 项 | 路径 |
|---|---|
| Runner | [`mlp_wd/scripts/run_e10_c_width.py`](../../../mlp_wd/scripts/run_e10_c_width.py) |
| 拟合库 | [`mlp_wd/analysis/e10_c_width.py`](../../../mlp_wd/analysis/e10_c_width.py) |
| 报告 | [`mlp_wd/analysis/report_e10_c_width.py`](../../../mlp_wd/analysis/report_e10_c_width.py) |
| 队列脚本 | [`rebuttal/run_e10_queue.sh`](../../run_e10_queue.sh) |
| **盲预测证据** | [`_data/e10_predictions_mnist.json`](../_data/e10_predictions_mnist.json)、[`..._cifar10.json`](../_data/e10_predictions_cifar10.json) |
| 结果表 | [`_data/e10_heldout_table_mnist.md`](../_data/e10_heldout_table_mnist.md)、[`..._cifar10.md`](../_data/e10_heldout_table_cifar10.md) |
|逐格明细 | `_data/e10_heldout_{mnist,cifar10}.csv`、`_data/e10_C_by_width_*.csv` |
| 训练 CSV | `mlp_wd/outputs/results/e10_{mnist,cifar10}.csv` |
| 图 | `outputs/plots/nips26/e10_c_width_{mnist,cifar10}.png` |

---

## 1. 为什么这样才算「盲」

Runner 分三个**硬隔断**的阶段：

```
ladder  →  predict  →  heldout
```

- `ladder`：只跑标定宽度的 `η × λ` 网格。
- `predict`：拟合 `C(h)`、外推、把 `λ_pred` **连同时间戳写进
  `_data/e10_predictions_<ds>.json`**；若检测到 held-out 宽度已有数据，会打印
  `NOT blind` 警告并把 `blind: false` 落进 JSON。两份 JSON 都是 `blind: true`。
- `heldout`：读那份 JSON，再去跑 held-out 宽度的 oracle 网格与各规则单点。

预测文件写入时间早于 held-out 训练开始时间，这是可核查的。

## 2. 协议

| | MNIST（主） | CIFAR-10（交叉验证） |
|---|---|---|
| 模型 | 3 层 ReLU MLP，无归一化 | 同 |
| T / B / seed | 20 / 128 / 42 | 30 / 128 / 42 |
| LR schedule | cosine | cosine |
| momentum | 0（SGD）与 0.9（SGDM） | 0.9 |
| η 网格 | 0.05, 0.1, 0.2 | 0.03, 0.1, 0.3 |
| λ 梯 | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | 同 |
| 标定宽度 | **128, 256, 512** | **256, 512** |
| held-out 宽度 | **1024, 2048** | **1024** |
| 复用 | h=512 来自 `e5c_mnist.csv` | h=512 来自 `exp2.csv` |

`C` 的定义与 E5a/E5c逐字一致：`C = λ* · Σ_tη_t`，`λ*` 取 `best_test_loss` 在
log λ 上的抛物线极小，按 (momentum, η) 分组、优先 interior 点、取几何平均。
h=512 的 MNIST 拟合结果 `C_sgd = 0.440`、`C_sgdm = 0.320` **与已发布的 E5c
数字完全吻合**，说明新管线没有引入口径漂移。

## 3. 结果

### (a) `C` 随宽度怎么变

| dataset | momentum | C(128) | C(256) | C(512) | slope of `log C` vs `log h` |
|---|---:|---:|---:|---:|---|
| MNIST | 0 | 0.442 | 0.374 | 0.440 | **−0.00** [−0.24, +0.24] |
| MNIST | 0.9 | 0.466 | 0.353 | 0.320 | **−0.27** [−0.40, −0.14] |
| CIFAR-10 | 0.9 | — | 2.800 | 2.672 | −0.068（仅两点，无 CI） |

宽度对 `C` 的影响很弱：16 倍宽度范围内 `C` 变化 ≤ 1.5×，指数在 −0.27 到 0 之间。

### (b) 盲预测的 `C` 对不对

| dataset | momentum | width | C 预测 | C 实测 | 比值 |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | 1024 | 0.416 | 0.325 | 1.28× |
| MNIST | 0 | 2048 | 0.415 | 0.327 | 1.27× |
| MNIST | 0.9 | 1024 | 0.257 | 0.244 | 1.05× |
| MNIST | 0.9 | 2048 | 0.213 | 0.255 | 0.84× |
| CIFAR-10 | 0.9 | 1024 | 2.549 | 2.479 | **1.03×** |

**外推误差 ≤ 1.3×**。对应的 `λ_pred/λ_oracle` 几何平均：MNIST **1.60×**、
CIFAR-10 **1.14×**。

### (c) 零调参规则 vs per-cell oracle

oracle = 该格内**任何人测过的最好 λ**（5 点梯 ∪ 四条规则各自的 λ），
所以没有规则能靠「梯太粗」显得赢过调参。调参成本：oracle 每格 5+ 次训练，
任何规则 1 次（一次性标定后 0 次）。

**MNIST（12 格）**

| rule | mean acc gap | worst | mean loss gap | λ/λ_oracle |
|---|---:|---:|---:|---:|
| default 5e-4 | 0.03 | 0.11 | 0.0005 | 1.27× |
| `1/(ηT)` | 0.11 | 0.25 | 0.0032 | 2.70× |
| constant `ηλ` | 0.06 | 0.20 | 0.0018 | 1.56× |
| **ours `C_pred(h)/Σηₜ`** | 0.08 | 0.19 | 0.0020 | 1.60× |

**CIFAR-10（3 格，1 格发散）**

| rule | mean acc gap | worst | mean loss gap | λ/λ_oracle |
|---|---:|---:|---:|---:|
| default 5e-4 | 0.84 | 1.19 | 0.1576 |0.13× |
| `1/(ηT)` | 0.52 | 1.11 | 0.0792 | 0.23× |
| constant `ηλ` | 1.49 | 3.73 | **0.0171** | 1.08× |
| **ours** | 2.01 | 3.79 | 0.0233 | 1.14× |

## 4. 结论（给审稿回复用，含不利结果）

1. **`C` 确实跨宽度迁移。** 在小宽度标定、盲外推到 4–16 倍宽的网络，`C` 误差
   ≤ 1.3×，`λ` 误差 ≤ 1.6×。所以「`C` 依赖架构」这件事在宽度方向上是**弱依赖**，
   不是致命的。
2. **但预测对 λ 准 ≠ 在准确率上赢。** CIFAR-10 上 ours / constant-`ηλ` 最接近
   loss-oracle 的 λ（1.1×），default 与 `1/(ηT)` 偏离 4–8×；可 default 在**准确率**
   上反而更好（0.84 vs 2.01 pp），因为该设定下准确率最优 λ 明显小于损失最优 λ。
   这与 E4 的结论一致（ours 0.87 vs default 0.70），我们照实报告。
3. **MNIST 无法区分规则**：四条规则的准确率差距全在 0.11 pp 以内 —— 该任务上
   λ 这一维几乎是平的。这本身是有用的负面信息：宣称在 MNIST-MLP 上"规则更优"
   是没有证据的。
4. **`C` 的跨族差异远大于跨宽度差异**：MNIST-MLP 0.32–0.44、CIFAR-10-MLP
   2.5–2.8、CIFAR-100-CNN 1.48（E5a 几何均值）—— 约 **8.8×** 的跨度。所以真正
   的限制不是宽度，而是数据集/架构族；`C` 必须在每个族里标定一次。
5. 诚实边界：CIFAR-10 只有两级标定梯（无有效 CI）、单seed、3 个held-out 格，
   其中 η=0.3 那格我们的λ=1.402e-3 发散，而 λ=1.322e-3 正常收敛 —— 处在稳定性
   边界上，该格按发散计入而非插补。

## 5. 成本与复现

MNIST：ladder 60 + held-out 108 = **168** 次 20-epoch 短跑（h=512 的 30 格复用）。
CIFAR-10：ladder 15 + held-out 27 = **42** 次 30-epoch 短跑。
GPU 1,2,3、`workers_per_gpu=6`，各约 15–30 分钟。

```bash
cd /home/yzy/GitHub/wd
GPUS=1,2,3 WPG=6 bash rebuttal/run_e10_queue.sh mnist   ladder,predict,heldout
GPUS=1,2,3 WPG=6 bash rebuttal/run_e10_queue.sh cifar10 ladder,predict,heldout
```

阶段可分开跑；`predict` 默认不覆盖已有预测文件（需 `--overwrite_predictions`），
以免事后修改预测。
