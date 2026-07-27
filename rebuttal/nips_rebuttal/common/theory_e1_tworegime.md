# E1 理论救援：双机制（时间尺度 × 平衡态地板）

## 1. 投稿理论到底预测了什么

SGD-WD 的一致稳定性界（Thm.）是

\[
\varepsilon_{\mathrm{WD}} \le \frac{2G^2}{\lambda n},
\]

**本身不含 \(T\)**。\(λ ∝ 1/(ηT)\) 并不是这条界的直接推论，而是
**Regularization Equivalence**：把上述界与 SWA 的界
\(\varepsilon_{\mathrm{SWA}} \le η G^2 T / n\) 对齐，得到

\[
λ_{\mathrm{REH}} \;\approx\; \frac{C}{S}, \qquad S=\sum_{t<T}η_t.
\]

与此同时，Kosson 等的旋转平衡态给出另一条约束：对近似尺度不变的参数，

\[
η\,λ \;\approx\; P_\star
\quad\Rightarrow\quad
λ_{\mathrm{eq}} \;\approx\; \frac{P_\star}{η},
\]

与训练长度无关。

## 2. 观测到的“假阴性”如何产生

在 \(η=0.1\) 密阶梯上，网格 argmax 钉在 \(λ=10^{-3}\)（四个 \(T\) 相同），
插值斜率只有 \(-0.23\)。这被读成对 \(1/T\) 的否定——但同一批数据里还有：

| 证据 | 结果 |
|---|---|
| 软峰值（峰值 1pp 内 acc 加权） | \(1.30\!\to\!0.99\!\to\!0.81\!\to\!0.73\times10^{-3}\)，斜率 \(\approx-0.28\) |
| \(η=0.02\) 臂 | \(λ^\star\approx 5.1\!\to\!3.3\!\to\!1.3\times10^{-3}\)，斜率 \(\approx\mathbf{-0.61}\) CI \([-1.34,-0.32]\) |
| 跨 \(η\) 的 \(λ\) 比（\(T=25\)） | \(λ(0.02)/λ(0.1)\approx 4.3\)（预测 5） |

解释：**两个约束同时存在**。实用最优点近似满足

\[
λ^\star \;\approx\; \max\!\big(λ_{\mathrm{REH}}(η,T),\; λ_{\mathrm{eq}}(η)\big)
\;=\; \max\!\Big(\frac{C}{S},\;\frac{P_\star}{η}\Big).
\]

- 当 \(η\) 大、\(T\) 长时，\(C/S\) 落到 \(P_\star/η\) **之下**，最优点被平衡态地板钉住
  → 网格上看起来 \(λ^\star\) 不随 \(T\) 动（我们在 \(η=0.1\) 上看到的）。
- 当 \(η\) 小（或 \(T\) 短）时，\(C/S\) 高于地板 → **时间尺度重新可见**
  （\(η=0.02\) 臂斜率 \(-0.61\)）。

这不是否定 REH，而是：**REH 给出随 \(T\) 下降的软约束，平衡态给出不随 \(T\)
下降的硬地板；先前只在地板上方采样，所以只看见地板。**

## 3. 与 Kosson 的正确对位

Kosson 的 \(ηλ=\mathrm{const}\) 描述的是**稳态**。他们自己也指出乘积相同的
\((η,λ)\) 在瞬态不同。我们的 \(1/T\) 来自整段轨迹的稳定性预算。
二者在 \(λ^\star=\max(C/S,\,P_\star/η)\) 下兼容：

- 长训 + 常规 \(η\)：地板主导 → 看起来像 Kosson；
- 短训 / 小 \(η\) / 无归一化：时间尺度主导 → 看起来像 REH。

E2b（行为在等积线上不平坦）仍然必要：地板只钉住乘积量级，并不钉住
\((η,λ)\) 在等积线上的位置。

## 4. 救援实验（`e1_rescue`）要验证的预言

1. **加密峰值**（\(η=0.1$，$λ\in[4\mathrm{e}{-4},2.5\mathrm{e}{-3}]\)）：软 \(λ^\star\) 随 \(T\) 单调降。
2. **短 \(T\in\{5,10,15\}\)**：\(λ^\star\) 明显高于 \(10^{-3}\)，斜率更接近 \(-1\)。
3. **加密 \(η=0.02\)**：斜率稳定在 \([-1,-0.5]\)，且 \(C=λ^\star S\) 比 \(η=0.1\) 臂更稳。
4. **常数 LR + 更低 \(λ\) 阶梯**：去掉 cosine 退火后，\(1/T\) 应更干净；旧 const 臂峰在左边界是阶梯不够低。

## 5. 写进 rebuttal 的一句话

> The headline \(η=0.1\) grid is dominated by an equilibrium floor on \(ηλ\); once
> we move below that floor (smaller \(η\), shorter \(T\), soft-peak statistics),
> the \(λ^\star\!\propto\!1/T\) drift predicted by matching stability budgets
> reappears. The two accounts are not mutually exclusive: \(λ^\star\approx\max(C/S,\,P_\star/η)\).
