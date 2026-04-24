#!/usr/bin/env python3
"""
Generate plots from MNIST WD experiment results
"""
import json
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = "/mnt/afs/visitor13/mnist_wd_results"
OUTPUT_DIR = Path(RESULTS_DIR)

def load_result(name):
    """Load experiment result JSON"""
    path = OUTPUT_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def plot_experiment1():
    """Plot Experiment 1: Effect of Weight Decay"""
    configs = [
        ('exp1_no_wd', 'No WD'),
        ('exp1_wd_1e4', 'WD=1e-4'),
        ('exp1_wd_1e3', 'WD=1e-3'),
        ('exp1_wd_1e2', 'WD=1e-2'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, label in configs:
        data = load_result(name)
        if data:
            epochs = range(1, len(data['train_loss']) + 1)
            ax1.plot(epochs, data['train_loss'], marker='o', label=label)
            ax2.plot(epochs, data['test_acc'], marker='o', label=label)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Effect of Weight Decay - Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Effect of Weight Decay - Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'experiment1_wd_effect.png', dpi=150)
    print(f"   Saved: experiment1_wd_effect.png")
    plt.close()

def plot_experiment2():
    """Plot Experiment 2: Grid Search LR × WD"""
    lrs = [0.001, 0.01, 0.1]
    wds = [0.0, 1e-5, 1e-4, 1e-3]
    
    # Create accuracy heatmap
    accuracy_matrix = np.zeros((len(lrs), len(wds)))
    
    for i, lr in enumerate(lrs):
        for j, wd in enumerate(wds):
            wd_str = f"{wd}" if wd >= 0.0001 else f"{wd:.0e}"
            name = f"exp2_lr{lr}_wd{wd_str}"
            data = load_result(name)
            if data:
                accuracy_matrix[i, j] = data['final_test_acc']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(accuracy_matrix, cmap='viridis', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(wds)))
    ax.set_xticklabels([f'{w}' for w in wds])
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f'{l}' for l in lrs])
    
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Test Accuracy: LR × WD Grid Search')
    
    # Add text annotations
    for i in range(len(lrs)):
        for j in range(len(wds)):
            text = ax.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                         ha="center", va="center", color="white", fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Test Accuracy')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'experiment2_lr_wd_grid.png', dpi=150)
    print(f"   Saved: experiment2_lr_wd_grid.png")
    plt.close()
    
    # Also plot learning curves for best config
    best_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_lr = lrs[best_idx[0]]
    best_wd = wds[best_idx[1]]
    best_name = f"exp2_lr{best_lr}_wd{best_wd}"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot all configs with lr=0.1 (highest performing)
    for wd in wds:
        wd_str = f"{wd}" if wd >= 0.0001 else f"{wd:.0e}"
        name = f"exp2_lr0.1_wd{wd_str}"
        data = load_result(name)
        if data:
            epochs = range(1, len(data['train_loss']) + 1)
            ax1.plot(epochs, data['train_loss'], marker='o', label=f'WD={wd}')
            ax2.plot(epochs, data['test_acc'], marker='o', label=f'WD={wd}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('LR=0.1: Training Loss vs WD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('LR=0.1: Test Accuracy vs WD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'experiment2_lr01_comparison.png', dpi=150)
    print(f"   Saved: experiment2_lr01_comparison.png")
    plt.close()

def plot_experiment3():
    """Plot Experiment 3: Momentum + WD"""
    configs = [
        ('exp3_sgd', 'SGD'),
        ('exp3_sgd_wd', 'SGD + WD'),
        ('exp3_sgdm', 'SGD + Momentum'),
        ('exp3_sgdm_wd', 'SGD + Momentum + WD'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, label in configs:
        data = load_result(name)
        if data:
            epochs = range(1, len(data['train_loss']) + 1)
            ax1.plot(epochs, data['train_loss'], marker='o', label=label)
            ax2.plot(epochs, data['test_acc'], marker='o', label=label)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Momentum + Weight Decay - Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Momentum + Weight Decay - Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'experiment3_momentum_wd.png', dpi=150)
    print(f"   Saved: experiment3_momentum_wd.png")
    plt.close()

def plot_summary_bar():
    """Summary bar chart of all final accuracies"""
    # Experiment 1
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Exp 1
    names1 = ['No WD', 'WD=1e-4', 'WD=1e-3', 'WD=1e-2']
    accs1 = []
    for name in ['exp1_no_wd', 'exp1_wd_1e4', 'exp1_wd_1e3', 'exp1_wd_1e2']:
        data = load_result(name)
        accs1.append(data['final_test_acc'] if data else 0)
    
    axes[0].bar(range(len(names1)), accs1, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_xticks(range(len(names1)))
    axes[0].set_xticklabels(names1, rotation=15)
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Exp 1: Weight Decay Effect')
    axes[0].set_ylim([0.90, 0.94])
    for i, v in enumerate(accs1):
        axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
    
    # Exp 2 - Best per LR
    names2 = ['LR=0.001', 'LR=0.01', 'LR=0.1']
    accs2 = [0.8269, 0.9326, 0.9787]  # Best WD for each LR
    
    axes[1].bar(range(len(names2)), accs2, color=['#9467bd', '#8c564b', '#e377c2'])
    axes[1].set_xticks(range(len(names2)))
    axes[1].set_xticklabels(names2)
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Exp 2: Best Accuracy per LR')
    axes[1].set_ylim([0.75, 1.0])
    for i, v in enumerate(accs2):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
    
    # Exp 3
    names3 = ['SGD', 'SGD+WD', 'SGD+M', 'SGD+M+WD']
    accs3 = []
    for name in ['exp3_sgd', 'exp3_sgd_wd', 'exp3_sgdm', 'exp3_sgdm_wd']:
        data = load_result(name)
        accs3.append(data['final_test_acc'] if data else 0)
    
    axes[2].bar(range(len(names3)), accs3, color=['#17becf', '#bcbd22', '#7f7f7f', '#aec7e8'])
    axes[2].set_xticks(range(len(names3)))
    axes[2].set_xticklabels(names3, rotation=15)
    axes[2].set_ylabel('Test Accuracy')
    axes[2].set_title('Exp 3: Momentum + WD')
    axes[2].set_ylim([0.90, 1.0])
    for i, v in enumerate(accs3):
        axes[2].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_all_experiments.png', dpi=150)
    print(f"   Saved: summary_all_experiments.png")
    plt.close()

def main():
    print("=" * 60)
    print("Generating plots from MNIST WD experiment results")
    print("=" * 60)
    
    print("\n[1/4] Experiment 1: WD Effect...")
    plot_experiment1()
    
    print("\n[2/4] Experiment 2: LR × WD Grid...")
    plot_experiment2()
    
    print("\n[3/4] Experiment 3: Momentum + WD...")
    plot_experiment3()
    
    print("\n[4/4] Summary bar chart...")
    plot_summary_bar()
    
    print("\n" + "=" * 60)
    print("All plots generated!")
    print(f"Location: {RESULTS_DIR}")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
