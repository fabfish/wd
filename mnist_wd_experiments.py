#!/usr/bin/env python3
"""
MLP + MNIST Weight Decay Experiments
2-3 layer MLP on MNIST with grid search
Demonstrates three key WD effects
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Config
RESULTS_DIR = "/mnt/afs/visitor13/mnist_wd_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Simple 2-layer MLP
class MLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)

def get_mnist_loaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    
    return total_loss / total, correct / total

def run_experiment(name, lr, wd, momentum, epochs=10, batch_size=256):
    """Run single experiment and save results"""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  lr={lr}, wd={wd}, momentum={momentum}, epochs={epochs}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_loader, test_loader = get_mnist_loaders(batch_size)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    
    history = {
        'name': name,
        'config': {'lr': lr, 'wd': wd, 'momentum': momentum, 'epochs': epochs, 'batch_size': batch_size},
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['test_loss'].append(float(test_loss))
        history['test_acc'].append(float(test_acc))
        history['epoch_time'].append(epoch_time)
        
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, "
              f"time={epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    history['total_time'] = total_time
    history['final_test_acc'] = float(history['test_acc'][-1])
    
    # Save results
    result_file = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(result_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model checkpoint
    checkpoint_file = os.path.join(RESULTS_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), checkpoint_file)
    
    print(f"\n  Results saved to {result_file}")
    print(f"  Final test accuracy: {history['final_test_acc']:.4f}")
    print(f"  Total time: {total_time:.2f}s")
    
    return history

def main():
    print("="*60)
    print("MLP + MNIST Weight Decay Experiments")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Experiment 1: Effect of Weight Decay (baseline comparison)
    # SGD vs SGD+WD at fixed lr
    print("\n\n" + "="*60)
    print("EXPERIMENT 1: Effect of Weight Decay (fixed lr=0.01)")
    print("="*60)
    
    exp1_configs = [
        ('exp1_no_wd', 0.01, 0.0, 0.0),
        ('exp1_wd_1e4', 0.01, 1e-4, 0.0),
        ('exp1_wd_1e3', 0.01, 1e-3, 0.0),
        ('exp1_wd_1e2', 0.01, 1e-2, 0.0),
    ]
    
    exp1_results = []
    for name, lr, wd, mom in exp1_configs:
        result = run_experiment(name, lr, wd, mom, epochs=10)
        exp1_results.append(result)
    
    # Experiment 2: Grid Search λ × η (WD × LR interaction)
    print("\n\n" + "="*60)
    print("EXPERIMENT 2: Grid Search WD × LR")
    print("="*60)
    
    lrs = [0.001, 0.01, 0.1]
    wds = [0.0, 1e-5, 1e-4, 1e-3]
    
    exp2_results = []
    for lr in lrs:
        for wd in wds:
            name = f"exp2_lr{lr}_wd{wd}"
            result = run_experiment(name, lr, wd, 0.0, epochs=10)
            exp2_results.append(result)
    
    # Experiment 3: Momentum + WD combined effect
    print("\n\n" + "="*60)
    print("EXPERIMENT 3: Momentum + Weight Decay")
    print("="*60)
    
    exp3_configs = [
        ('exp3_sgd', 0.01, 0.0, 0.0),
        ('exp3_sgd_wd', 0.01, 1e-4, 0.0),
        ('exp3_sgdm', 0.01, 0.0, 0.9),
        ('exp3_sgdm_wd', 0.01, 1e-4, 0.9),
    ]
    
    exp3_results = []
    for name, lr, wd, mom in exp3_configs:
        result = run_experiment(name, lr, wd, mom, epochs=10)
        exp3_results.append(result)
    
    # Save summary
    summary = {
        'experiment1': {r['name']: r['final_test_acc'] for r in exp1_results},
        'experiment2': {r['name']: r['final_test_acc'] for r in exp2_results},
        'experiment3': {r['name']: r['final_test_acc'] for r in exp3_results},
    }
    
    summary_file = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Summary: {summary_file}")
    
    # Print summary table
    print("\n--- Experiment 1: WD Effect ---")
    for name, acc in summary['experiment1'].items():
        print(f"  {name}: {acc:.4f}")
    
    print("\n--- Experiment 2: LR × WD Grid ---")
    for name, acc in summary['experiment2'].items():
        print(f"  {name}: {acc:.4f}")
    
    print("\n--- Experiment 3: Momentum + WD ---")
    for name, acc in summary['experiment3'].items():
        print(f"  {name}: {acc:.4f}")

if __name__ == "__main__":
    main()
