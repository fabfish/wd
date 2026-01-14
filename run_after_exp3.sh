#!/bin/bash
# 监控 exp3 完成后自动运行改进实验

echo "=========================================="
echo "监控 exp3 运行状态..."
echo "=========================================="

# 等待 exp3 完成
while pgrep -f "run_experiments.py --experiment 3" > /dev/null; do
    LINES=$(wc -l < /home/yzy/GitHub/wd/outputs/results/results.csv)
    echo "[$(date '+%H:%M:%S')] exp3 运行中... results.csv 行数: $LINES"
    sleep 30
done

echo ""
echo "=========================================="
echo "exp3 已完成! 开始运行改进实验..."
echo "=========================================="

# 检查最终结果
FINAL_LINES=$(wc -l < /home/yzy/GitHub/wd/outputs/results/results.csv)
echo "results.csv 最终行数: $FINAL_LINES (预期: 84 = 1 header + 24 exp1 + 35 exp2 + 24 exp3)"

# 运行改进实验，结果存到新文件
echo ""
echo "=========================================="
echo "[$(date '+%H:%M:%S')] 开始实验一改进: 细粒度LR搜索 (45 runs)"
echo "=========================================="
cd /home/yzy/GitHub/wd
python run_experiments_v2.py --experiment 1 --gpus all --epochs 100 --output outputs/results/results_v2.csv

echo ""
echo "=========================================="
echo "[$(date '+%H:%M:%S')] 开始实验二: η-λ 反比例验证 (56 runs)"
echo "=========================================="
python run_experiments_v2.py --experiment 2 --gpus all --epochs 100 --output outputs/results/results_v2.csv

echo ""
echo "=========================================="
echo "[$(date '+%H:%M:%S')] 开始实验三: B-λ 正比例验证 (40 runs)"
echo "=========================================="
python run_experiments_v2.py --experiment 3 --gpus all --epochs 100 --output outputs/results/results_v2.csv

echo ""
echo "=========================================="
echo "[$(date '+%H:%M:%S')] 所有实验完成! 生成可视化..."
echo "=========================================="
python plot_results_v2.py --input outputs/results/results_v2.csv --output_dir outputs/plots

echo ""
echo "=========================================="
echo "完成! 结果保存在:"
echo "  - outputs/results/results_v2.csv"
echo "  - outputs/plots/"
echo "=========================================="
