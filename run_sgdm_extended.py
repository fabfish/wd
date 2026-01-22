"""
扩展 SGDM+WD Grid Search 实验
测试不同 WD 值来找到正确的最优 LR 顺序: SGD > SGD+WD > SGDM+WD
"""
import argparse
import csv
import os
import time
from pathlib import Path

import torch

from run_three_methods_comparison import run_single_experiment_worker
from gpu_scheduler import GPUScheduler, parse_gpu_ids


def main():
    parser = argparse.ArgumentParser(description='扩展 SGDM+WD Grid Search')
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU IDs, e.g., "0,1"')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=str, default='outputs/results/sgdm_extended.csv')
    args = parser.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)
    
    # 参数网格
    sgdm_wds = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    sgdm_lrs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    momentum = 0.9
    batch_size = 128
    seed = 42
    use_amp = True

    # 构建任务列表
    tasks = []
    for wd in sgdm_wds:
        for lr in sgdm_lrs:
            task = ("SGDM+WD", batch_size, lr, wd, momentum, args.epochs, seed, use_amp, False)
            tasks.append(task)

    print(f"总实验数: {len(tasks)}")
    print(f"WD values: {sgdm_wds}")
    print(f"LR values: {sgdm_lrs}")
    print(f"GPUs: {gpu_ids}")

    # 使用 GPU 调度器
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)

    start_time = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker)
    elapsed_time = time.time() - start_time

    print(f"\n完成 {len(tasks)} 实验，耗时 {elapsed_time/60:.2f} 分钟")

    # 保存结果
    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc']

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if result is not None:
                writer.writerow(result)

    print(f"结果保存到: {args.output}")

    # 分析结果: 找每个 WD 的最优 LR
    print("\n" + "="*60)
    print("各 WD 值的最优配置:")
    print("="*60)
    for wd in sgdm_wds:
        wd_results = [r for r in results if r and r['wd'] == wd]
        if wd_results:
            best = max(wd_results, key=lambda x: x['best_test_acc'])
            print(f"WD={wd:.0e}: 最优LR={best['lr']:.3f}, Acc={best['best_test_acc']:.2f}%")


if __name__ == '__main__':
    main()
