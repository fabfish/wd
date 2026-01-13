# Quick Start Guide

## 🚀 Run All Experiments (One Command)

For GPUs 0-7, simply run:

```bash
./run_all_experiments.sh
```

Or with Python:

```bash
python run_all_experiments.py
```

**That's it!** The script will:
- ✅ Check your environment
- ✅ Run all 3 experiment sets (83 runs total)
- ✅ Use all 8 GPUs efficiently
- ✅ Generate plots automatically
- ✅ Save results to `outputs/results/results.csv`
- ✅ Print summary with best results

**Estimated time: ~1.5-2 hours on 8×A100**

---

## ⚡ Parallel Mode (Faster!)

Run all experiments simultaneously:

```bash
./run_all_experiments.sh --parallel
```

**Estimated time: ~45-60 minutes on 8×A100**

GPU allocation:
- Experiment 1 (24 runs) → GPUs 0-2
- Experiment 2 (35 runs) → GPUs 3-5  
- Experiment 3 (24 runs) → GPUs 6-7

---

## 💾 Save Model Checkpoints

```bash
./run_all_experiments.sh --save-checkpoints
```

Checkpoints will be saved to `outputs/checkpoints/`

---

## 📊 Output Files

After running, you'll find:

```
outputs/
├── logs/              # Detailed logs
├── results/           
│   └── results.csv    # All experiment results
├── checkpoints/       # Model weights (if --save-checkpoints)
└── plots/             # Visualizations
    ├── exp1_lr_ordering.png
    ├── exp2_eta_lambda_heatmap.png
    └── exp3_batch_size_scaling.png
```

---

## 🔍 View Results

```bash
# View plots
eog outputs/plots/*.png           # Linux
open outputs/plots/*.png          # macOS

# Analyze results
python plot_results.py --stats

# Check logs
cat outputs/logs/exp*.log
```

---

## ⚙️ Advanced Options

```bash
# Custom epochs
./run_all_experiments.sh --epochs 200

# Different GPUs
./run_all_experiments.sh --gpus 0-3

# All options
./run_all_experiments.sh --help
```

---

## 🧪 Test Your Setup

Before running experiments, test your environment:

```bash
python test_multi_gpu.py
```

This checks:
- GPU availability
- Python packages
- Multi-GPU scheduler

---

## 🐛 Troubleshooting

**No GPUs detected?**
```bash
nvidia-smi  # Check GPU status
python -c "import torch; print(torch.cuda.device_count())"
```

**Out of memory?**
- Reduce batch size in the experiment configurations
- Use fewer GPUs: `--gpus 0-3`

**Script won't run?**
```bash
chmod +x run_all_experiments.sh  # Make executable
```

---

## 📚 Full Documentation

See [README.md](README.md) for complete documentation.

---

## 💡 Tips

1. **Monitor Progress**: Open another terminal and run `watch -n 1 nvidia-smi`
2. **Background Execution**: Use `screen` or `tmux` for long runs
3. **Resume Failed Runs**: Results are appended, just re-run the script
4. **Save Disk Space**: Skip checkpoints unless you need them

---

Happy Experimenting! 🎉
