# E12 下一步计划（next steps）

基于 E12 完整结果的空白分析与后续实验设计，按优先级排序。

## 1. 多种子补齐（最高优先，成本低）

e12_ms 已覆盖 R18/SGD、VGG、MLP 的 peak configs，但缺：
- **R50/C100 双相位**：fixed oracle（SGD 5e-3 / SGDM 9.62e-4）与
  linear_up peak（SGD 1.698e-2 / SGDM 3.057e-3）× seeds {42,123,2024}。
- **R18/SGD linear_down 新峰值 78.04@1.063e-2**（fill 新发现，只 1 种子）
  需要多种子确认它真的与 linear_up/iso 同处第一梯队。
- MLP/MNIST 跳过（分辨率 <0.5%）。
预计 ~16 runs，数小时即可完成。若多种子确认 linear_down@1.5C 与
linear_up@1.8C 并列，R18/SGD 的"最优 WD 形状"就不是单峰而是
1.5–1.8C 的平台，需要重写对应结论的措辞。

## 2. SGD 大预算上界与崩塌边界

- R18/SGD：1.8C 之后（2.9C/4.8C/7.2C）已有崩塌数据 ✓ 无需补。
- R50/SGD：阶梯只到 15C（E4 单位对应 1.8C 本地），fill 未测 2C 以上——
  若想报告"R50/SGD 的崩塌边界"，需补 3C/6C 两三个点。
- VGG/SGD：1.2C/1.5C 已补，崩塌定位在 ~1.5–2C ✓。

## 3. fixed 曲线加密（E4 锚的普适性）

R50/VGG 的 SGDM fixed 峰在 9.62e-4（E4 理论 λ），且 6e-4→1e-3 之间极尖。
值得在 **MLP/C10 的 SGDM**（当前 oracle 1e-3，网格 {6e-4,1e-3} 之间无点）
补 8e-4/9.62e-4 两点，检验"9.62e-4 是跨架构最优 fixed λ"的猜想。
另可补 R18 的 9.62e-4（E11 已有 e4 行：R18@9.62e-4 未在 B128/lr0.1/T100
协议下测过，可 1 个 run 补齐对照）。

## 4. 与论文/rebuttal 衔接

- **把 E12 写进 rebuttal**：核心卖点 = "raise-up 优于 fixed 在 4 个架构×2
  优化器上复现；iso 的最优预算始终压在理论点 1C；跨架构最优预算集中在
  ~1–2C（setting-local C 单位）"。同时诚实报告反例：MLP/C10 SGD 上
  fixed 最优、MNIST 无分辨率。
- 机制解释待强化：VGG 崩塌边界更早（WD 容差窗口窄）与
  "BN+scale-invariance" 的关系；R50 SGD > SGDM 与 R18 不同的原因
  （深度 vs 宽度）。这些可结合 stability 探针（run_nips26_stability.py）
  或 hessian 分析（analysis/hessian_top_eig）补实验。

## 5. 其他 setting 扩展（可选）

- 数据维度目前只有 CIFAR-100/CIFAR-10/MNIST；可加 CIFAR-100 上
  VGG-13/ResNet-34 等中间规模，或 mlp_wd 的 BN 版 MLP（use_bn=1）以检验
  "无 BN → SGD 相位 raise-up 失效"的猜想（E12 MLP 无 BN）。
- wd_core 的 loader 已支持多数据集，新增 setting 只需排产。

## 执行约定

- 所有新 run 继续走 e12_multi_setting/ 目录与 e12_runs.csv（RUN_KEY 去重）。
- 多种子用 run_e12_ms.py + ms_configs.json 追加。
- 新网格用 make_fill_configs.py 生成 fill_configs.json 追加（注意 anchors
  更新后预算反解会自动跟随）。
- 完成后同样 git commit+push（cron 已配好收尾流程）。
