# SGD Weight Decay 实验框架

基于 CIFAR-100 + ResNet-18 的 SGD 权重衰减与泛化能力理论验证实验框架。

## 实验目标

回答核心问题：**"How do the weight decay coefficient, learning rate scheduler, and batch size coexist when setting parameters for maximum generalization ability?"**

验证基于 SGD/SGDM 的三个核心理论推论：

1. **稳定性边界与学习率排序**：验证最优学习率遵循 `η_SGD > η_SGD+WD > η_SGDM+WD`
2. **η 与 λ 的反比关系**：验证 `λ_opt ∝ 1/η`，即在热力图中呈现反比例的高准确率区域
3. **Batch Size 与 λ 的缩放律**：验证 `λ_opt ∝ B`（或相关线性缩放关系）

## 实验环境

- **硬件**：NVIDIA A100 (80GB/40GB) × N
- **数据集**：CIFAR-100
- **模型**：ResNet-18 (从头训练，不使用预训练权重)
- **优化器**：PyTorch 原生 SGD + CosineAnnealingLR

## Quick Start

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

```bash
# 单个实验
python main.py --batch_size 128 --lr 0.1 --wd 5e-4 --momentum 0.9 --epochs 100

# 实验集（多GPU并行）
python run_experiments_v2.py --experiment 1 --gpus all --epochs 100
python run_experiments_v2.py --experiment 2 --gpus all --epochs 100
python run_experiments_v2.py --experiment 3 --gpus all --epochs 100

# 一键运行所有实验
./run_all_experiments.sh
./run_all_experiments.sh --parallel  # 并行模式
```

### 生成可视化

```bash
python plot_results_v2.py
python plot_results_v2.py --experiment 1 --stats
```

## 文件结构

```
.
├── main.py                    # 单实验入口
├── models.py                  # ResNet-18 模型定义
├── utils.py                   # 训练/评估循环、检查点管理
├── run_experiments.py         # 原版实验运行器
├── run_experiments_v2.py      # 改进版实验运行器（推荐）
├── gpu_scheduler.py           # 多GPU任务调度器
├── logger.py                  # 日志工具
├── plot_results.py            # 原版可视化
├── plot_results_v2.py         # 改进版可视化
├── requirements.txt           # Python 依赖
└── outputs/                   # 输出目录
    ├── logs/                  # 实验日志
    ├── results/               # CSV 结果
    ├── checkpoints/           # 模型检查点
    └── plots/                 # 生成的图表
```

## 三组实验设计

### 实验一：最优学习率排序 (Stability Bound Verification)

**目的**：验证 `η_SGD > η_SGD+WD > η_SGDM+WD`

| Method | Momentum | Weight Decay |
|--------|----------|--------------|
| SGD | 0 | 0 |
| SGD+WD | 0 | 5e-4 |
| SGDM+WD | 0.9 | 5e-4 |

- **搜索空间**：LR ∈ [0.01, 3.0]，15个点
- **预期结果**：Test Accuracy vs. LR 曲线中，SGD 峰值最靠右，SGDM+WD 峰值最靠左

### 实验二：η-λ 协同作用 (Heatmap Analysis)

**目的**：验证最优的 η 和 λ 满足反比例关系

- **配置**：SGDM (momentum=0.9), Batch Size = 128
- **搜索空间**：
  - LR: [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
  - WD: [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
- **预期结果**：2D 热力图中高准确率区域呈"对角线"趋势

### 实验三：Batch Size 与 Weight Decay 的缩放律

**目的**：验证 `λ_opt ∝ B` 的正比例关系

- **配置**：SGDM (momentum=0.9), Linear LR Scaling: `η = 0.1 × (B / 128)`
- **搜索空间**：
  - Batch Size: [32, 64, 128, 256, 512]
  - WD: [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
- **预期结果**：Optimal λ vs. Batch Size 曲线呈线性正相关

## 输出说明

**结果 CSV** (`outputs/results/results_v2.csv`)：
- 列：`method, batch_size, lr, wd, momentum, final_test_acc, final_train_loss, best_test_acc`
- 支持断点续跑（追加模式）

**日志** (`outputs/logs/`)：
- 时间戳命名：`exp1_v2_20260114_194919.log`

**可视化** (`outputs/plots/`)：
- `exp1_lr_ordering_v2.png`: LR vs. Test Accuracy 曲线
- `exp2_heatmap_analysis.png`: η-λ 热力图
- `exp3_batch_size_scaling.png`: B-λ 关系图

## 多GPU使用

```bash
# 使用所有可用GPU
python run_experiments_v2.py --experiment 1 --gpus all

# 指定GPU
python run_experiments_v2.py --experiment 2 --gpus 0,1,2,3

# GPU范围
python run_experiments_v2.py --experiment 3 --gpus 0-7

# 监控GPU使用
watch -n 1 nvidia-smi
```

## Troubleshooting

**Out of memory**: 减小 batch size 或使用更少的 num_workers

**CUDA not available**: 检查 `python -c "import torch; print(torch.cuda.is_available())"`

**多GPU进程卡住**: `pkill -f run_experiments` 然后减少 num_workers
