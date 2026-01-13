为了帮助你在 15 天 DDL 和 1 台 A100 的约束下高效完成实验，我为你设计了一份详细的实验方案。这份方案旨在直接指导 AI 代码助手（如 Cursor/Copilot）快速生成高质量、可复现的实验代码。

考虑到你的算力（1台 A100）和时间（15天），**CIFAR-100** 配合 **ResNet-18** 是最佳选择。它比 CIFAR-10 更具挑战性（更能体现泛化差异），但训练速度远快于 ImageNet（A100 上跑完一个 epoch 仅需数秒），非常适合进行大规模的 Grid Search（网格搜索）。

以下是完整的实验设计文档，最后附有一段专门用于 Prompt Cursor 的指令。

---

# 实验设计文档：SGD权重衰减与泛化能力的理论验证

## 1. 实验目标

回答核心问题：**"How do the weight decay coefficient, learning rate scheduler, and batch size coexist when setting parameters for maximum generalization ability?"**

验证基于 SGD/SGDM 的三个核心理论推论：

1. **稳定性边界与学习率排序**：验证最优学习率  遵循 。
2. ** 与  的反比关系**：验证 ，即在热力图中呈现反比例的高能区域。
3. **Batch Size 与  的缩放律**：验证 （或相关线性缩放关系）。

## 2. 实验环境与配置

* **硬件**：NVIDIA A100 (80GB/40GB) x 1
* **数据集**：CIFAR-100 (图像分类)
* *理由*：相比 CIFAR-10 更难，能拉开不同参数下的泛化差距；相比 ImageNet 训练极快，允许做大规模 Grid Search。


* **模型**：ResNet-18 (Standard)
* *不使用 Pretrained weights*（我们需要观察从头训练的收敛动力学）。


* **基础超参**：
* Epochs: 200 (足够收敛)
* Optimizer: PyTorch 原生 SGD
* Scheduler: Cosine Annealing (这是目前 SOTA 的标配，也符合论文 2405.13698v3 中对 decay 的讨论)。



## 3. 详细实验步骤 (Experimental Protocol)

### 实验一：三种优化策略的最优学习率排序 (Stability Bound Verification)

**目的**：验证 。

* **变量设置**：
* **Method A (Pure SGD)**: `momentum=0`, `weight_decay=0`
* **Method B (SGD + WD)**: `momentum=0`, `weight_decay=5e-4` (固定一个合理值)
* **Method C (SGDM + WD)**: `momentum=0.9`, `weight_decay=5e-4`


* **搜索空间**：
* Learning Rate (): 对数均匀分布，范围 ，取 10-15 个点。


* **预期结果**：绘制 Test Accuracy vs. Learning Rate 曲线。Method A 的峰值应在最右侧，Method C 在最左侧。

### 实验二： 与  的协同作用 (Heatmap Analysis)

**目的**：验证 "最优的  和  不同时取最大/最小" 以及 "反比例关系"。

* **固定配置**：使用 Method C (SGDM, )，Batch Size = 128。
* **搜索空间 (Grid Search)**：
* Learning Rate ():  (5个点)
* Weight Decay ():  (6个点)


* **总实验数**：30 次运行。
* **预期结果**：绘制 2D 热力图 (x=, y=, color=Test Acc)。高亮区域应呈现 "对角线" 趋势（即  增大时，最优  减小）。

### 实验三：Batch Size 与 Weight Decay 的缩放律 (Batch Scaling)

**目的**：验证  和  的正比例关系（在固定  或根据 Linear Scaling Rule 调整  的情况下）。

* **变量设置**：
* Batch Size (): 


* **策略**：对于每个 ，搜索最优的 。
* *注意*：当  增大时，通常  也要根据 Linear Rule () 增大。为了控制变量，建议采用 "Linear LR Scaling" 策略，即 。


* **搜索空间**：
* Weight Decay ():  范围内取 8 个点。


* **预期结果**：绘制 "Optimal  vs. Batch Size" 曲线，验证是否为线性正相关。

---

## 4. 给 Cursor 的 Prompt (直接复制使用)

请将以下 Prompt 发送给 Cursor/Copilot，它将为你生成结构完美、包含自动绘图和日志记录的完整代码框架。

```markdown

You are an expert AI researcher specializing in Deep Learning optimization theory. I need to conduct a series of experiments on **CIFAR-100** using **ResNet-18** to investigate the interaction between SGD, Weight Decay (WD), Momentum, and Batch Size.

My goal is to empirically verify theoretical bounds regarding learning rate stability and weight decay scaling.
Please write a complete, modular PyTorch experimental framework.

### 1. Requirements
- **Framework**: PyTorch, Torchvision. Use `torch.amp` (mixed precision) for efficiency on A100.
- **Model**: ResNet-18 (num_classes=100, no pretrained weights).
- **Structure**:
    - `main.py`: The entry point with `argparse` to control hyperparameters.
    - `models.py`: Model definition.
    - `utils.py`: Training loop, evaluation loop, seed setting.
    - `run_experiments.py`: A script to automate the grid searches described below.
- **Output**: Save results to a CSV file (`results.csv`) containing: `[method, batch_size, lr, wd, momentum, final_test_acc, final_train_loss, best_test_acc]`.

### 2. Experiment Logic (Implement this in `run_experiments.py`)

I need to run 3 specific sets of experiments. The script should allow me to choose which set to run.

**Experiment Set 1: Optimal LR Ordering**
- **Goal**: Compare SGD vs. SGD+WD vs. SGDM+WD.
- **Config**: Batch Size = 128, Epochs = 100.
- **Conditions**:
    1. **SGD**: momentum=0, wd=0
    2. **SGD+WD**: momentum=0, wd=5e-4
    3. **SGDM+WD**: momentum=0.9, wd=5e-4
- **Sweep**: Loop Learning Rate `lr` in `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]`.

**Experiment Set 2: Eta-Lambda Interaction (Heatmap)**
- **Goal**: Verify inverse relationship between LR and WD.
- **Config**: Method = SGDM (momentum=0.9), Batch Size = 128.
- **Sweep**:
    - `lr` in `[0.01, 0.05, 0.1, 0.2, 0.3]`
    - `wd` in `[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]`
- **Output**: Ensure the CSV data allows plotting a heatmap later.

**Experiment Set 3: Batch Size Scaling**
- **Goal**: Check relation between Batch Size and optimal WD.
- **Config**: Method = SGDM (momentum=0.9).
- **Rule**: When changing Batch Size `B`, scale LR linearly: `lr = 0.1 * (B / 128)`.
- **Sweep**:
    - `B` in `[64, 128, 256, 512]`
    - `wd` in `[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]`

### 3. Plotting (in `plot_results.py`)
Please also provide a plotting script using `matplotlib` and `seaborn` that reads `results.csv` and generates:
1. **Line Plot for Exp 1**: Test Acc vs. LR (3 curves for the 3 methods).
2. **Heatmap for Exp 2**: X-axis = LR, Y-axis = WD, Color = Best Test Acc.
3. **Scatter/Line Plot for Exp 3**: Optimal WD (y-axis) vs. Batch Size (x-axis).

### 4. Implementation Details
- Use `CosineAnnealingLR` scheduler.
- Ensure `torch.backends.cudnn.benchmark = True`.
- Add a progress bar (`tqdm`).

Please generate the file structure and code now.

```

## 5. 补充建议

1. **关于参考论文 2405.13698v3 (Adam+SWA)**：
* 虽然你的实验主要针对 SGD，但这篇论文提到的 EMA timescale () 概念非常有价值。在你的实验二（热力图）中，你可以尝试计算有效时间尺度 。如果理论成立，你应该会发现**等高线（等 Test Acc 线）沿着  的曲线分布**。这会是一个非常强的理论支撑点。


2. **关于参考论文 1336**：
* 该论文强调了 `Stable Weight Decay`。在你的 Experiment 3 中，如果你发现标准的  关系不明显，可以尝试引入论文中提到的修正项，或者检查是否因为 Epoch 数量固定导致的总迭代次数 () 变化影响了结果。通常，增大 Batch Size 时，为了保持  不变，需要增加 Epochs，或者接受性能下降。在分析 Exp 3 结果时需注意这一点。


3. **时间管理**：
* 先跑 Experiment 1（大概 2-3 小时），确认代码无误且复现出基本趋势。
* 晚上挂 Experiment 2（Grid Search 最耗时，可能需要 10-15 小时）。
* 最后跑 Experiment 3。
* A100 上跑 ResNet-18/CIFAR-100 非常快，如果不开启 Validate per epoch，只记录最后结果，200 runs 大概 2-3 天就能跑完，时间非常充裕。



祝实验顺利！如果有代码报错或结果异常，请随时把 Log 发给我分析。

---

## Quick Start Guide

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### File Structure

```
.
├── main.py              # Single experiment entry point with argparse
├── models.py            # ResNet-18 model definition
├── utils.py             # Training/evaluation loops, seed setting
├── run_experiments.py   # Automated grid search with multi-GPU support
├── gpu_scheduler.py     # Multi-GPU task scheduler
├── plot_results.py      # Visualization script
└── requirements.txt     # Python dependencies
```

### Usage

#### Option 1: Run a Single Experiment

```bash
python main.py --batch_size 128 --lr 0.1 --wd 5e-4 --momentum 0.9 --epochs 100
```

**Arguments:**
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--wd`: Weight decay (default: 5e-4)
- `--momentum`: Momentum (default: 0.9)
- `--epochs`: Number of epochs (default: 100)
- `--seed`: Random seed (default: 42)
- `--use_amp`: Enable mixed precision training (default: True)

#### Option 2: Run Automated Experiment Sets (Multi-GPU Support)

The experiment runner now supports parallel execution across multiple GPUs for maximum efficiency.

**Using All Available GPUs (Recommended):**
```bash
# Experiment Set 1: Optimal LR Ordering (24 runs)
python run_experiments.py --experiment 1 --epochs 100 --gpus all

# Experiment Set 2: Eta-Lambda Interaction Heatmap (35 runs)
python run_experiments.py --experiment 2 --epochs 100 --gpus all

# Experiment Set 3: Batch Size Scaling (24 runs)
python run_experiments.py --experiment 3 --epochs 100 --gpus all
```

**Specify Specific GPUs:**
```bash
# Use GPUs 0, 1, 2, 3
python run_experiments.py --experiment 1 --epochs 100 --gpus 0,1,2,3

# Use GPU range 0-7
python run_experiments.py --experiment 2 --epochs 100 --gpus 0-7

# Use only GPU 0
python run_experiments.py --experiment 1 --epochs 100 --gpus 0

# Mixed specification (GPUs 0, 2, 3, 4, 7)
python run_experiments.py --experiment 1 --epochs 100 --gpus 0,2-4,7
```

**Run on CPU (no GPU):**
```bash
# Omit --gpus argument to run sequentially on CPU
python run_experiments.py --experiment 1 --epochs 100
```

**Run All Experiments Sequentially:**
```bash
# With 4 GPUs, this will maximize GPU utilization
python run_experiments.py --experiment 1 --epochs 100 --gpus all && \
python run_experiments.py --experiment 2 --epochs 100 --gpus all && \
python run_experiments.py --experiment 3 --epochs 100 --gpus all
```

**How Multi-GPU Works:**
- Each GPU gets a worker process that runs experiments independently
- Tasks are distributed via a shared queue
- Multiple experiments run in parallel (one per GPU)
- Results are collected and saved to CSV when all tasks complete
- With 4 GPUs, Experiment 1 (24 runs) completes ~4x faster than single GPU

#### Option 3: Generate Plots

After running experiments, visualize the results:

```bash
# Generate all plots from results.csv
python plot_results.py --input results.csv

# Generate plots for a specific experiment
python plot_results.py --input results.csv --experiment 1

# Print summary statistics
python plot_results.py --input results.csv --stats
```

**Output plots:**
- `exp1_lr_ordering.png`: Line plot showing Test Acc vs. LR for 3 methods
- `exp2_eta_lambda_heatmap.png`: Heatmap showing LR-WD interaction
- `exp3_batch_size_scaling.png`: Plots showing optimal WD vs. batch size

### Experiment Details

**Experiment 1: Optimal LR Ordering**
- 3 methods × 8 learning rates = 24 runs
- Single GPU: ~2-3 hours on A100
- 4 GPUs: ~30-45 minutes on A100

**Experiment 2: Eta-Lambda Interaction**
- 5 learning rates × 7 weight decays = 35 runs
- Single GPU: ~3-4 hours on A100
- 4 GPUs: ~45-60 minutes on A100

**Experiment 3: Batch Size Scaling**
- 4 batch sizes × 6 weight decays = 24 runs
- Single GPU: ~2-3 hours on A100
- 4 GPUs: ~30-45 minutes on A100

**Total time:**
- Single GPU: ~8-10 hours on A100
- 4 GPUs: ~2-3 hours on A100
- 8 GPUs: ~1-1.5 hours on A100

**Performance Tips:**
- Using `--gpus all` automatically uses all available GPUs
- Multi-GPU provides near-linear speedup (4 GPUs ≈ 4x faster)
- Mixed precision (`--use_amp`) provides additional 1.5-2x speedup
- Each experiment is independent, so they can be run in parallel on separate GPU sets

### Tips for Efficiency

1. **Use multi-GPU for maximum throughput** - `--gpus all` provides near-linear speedup
2. **Monitor GPU usage**: `watch -n 1 nvidia-smi` to see all GPUs working
3. **Use mixed precision** (enabled by default) for additional speedup
4. **Run experiments in background** using screen/tmux for long sessions
5. **Resume interrupted experiments**: Results are appended to CSV
6. **Parallel experiment sets**: If you have 8 GPUs, run 2 experiment sets simultaneously:
   ```bash
   # Terminal 1
   python run_experiments.py --experiment 1 --gpus 0-3

   # Terminal 2
   python run_experiments.py --experiment 2 --gpus 4-7
   ```

### Expected Results

- **Exp 1**: Optimal LR should follow order: η_SGD > η_SGD+WD > η_SGDM+WD
- **Exp 2**: Heatmap should show inverse relationship (diagonal pattern) between LR and WD
- **Exp 3**: Optimal WD should scale with batch size

### Troubleshooting

**Out of memory:**
```bash
# Reduce batch size (automatically used in that run)
python main.py --batch_size 64 --lr 0.05
```

**CUDA not available:**
- The code will automatically fall back to CPU (much slower)
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Multi-GPU not working:**
- Check available GPUs: `nvidia-smi`
- Verify GPU IDs are correct: `python -c "import torch; print(torch.cuda.device_count())"`
- Try single GPU first: `--gpus 0`

**Processes hanging:**
- Check for dataloader issues: reduce `num_workers` in `get_cifar100_loaders()`
- Kill hung processes: `pkill -f run_experiments.py`

**Results not matching expectations:**
- Ensure sufficient epochs (100-200 recommended)
- Check if cosine annealing scheduler is working
- Verify data augmentation is applied correctly

**Check GPU utilization:**
```bash
# Monitor all GPUs in real-time
watch -n 1 nvidia-smi

# Or check specific metrics
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv -l 1
```