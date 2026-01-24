"""
Supplement SGDM+WD (wd=0.0015, m=0.8) at LR=0.4 and 0.5.
To match SGD and SGD+WD which already have these points.
"""
import argparse
import csv
import os
import threading
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import resnet18
from utils import set_seed, train_model
from gpu_scheduler import GPUScheduler, parse_gpu_ids
from logger import get_logger
from pathlib import Path

csv_lock = threading.Lock()

def get_cifar100_loaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def run_single_experiment_worker(method, batch_size, lr, wd, momentum, epochs, seed, use_amp, log_interval=10):
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar100_loaders(batch_size)
    model = resnet18(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_test_acc, final_test_acc, final_train_loss = train_model(
        model, train_loader, test_loader, optimizer, scheduler,
        device, epochs=epochs, use_amp=use_amp, log_interval=log_interval
    )
    return {'method': method, 'batch_size': batch_size, 'lr': lr, 'wd': wd, 'momentum': momentum, 
            'final_test_acc': final_test_acc, 'final_train_loss': final_train_loss, 'best_test_acc': best_test_acc}

def save_single_result(result, output_file):
    if not result: return
    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum', 'final_test_acc', 'final_train_loss', 'best_test_acc']
    with csv_lock:
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            writer.writerow(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    
    # Check existing data to avoid re-run if present (optional, but good practice)
    # We assume we need to run it.
    
    tasks = []
    # SGDM+WD (wd=0.0015, mom=0.8) at 0.4 and 0.5
    for lr in [0.4, 0.5]:
        tasks.append(("SGDM+WD", 128, lr, 0.0015, 0.8, 100, 42, True, 25))
        
    scheduler = GPUScheduler(parse_gpu_ids(args.gpus), verbose=True, workers_per_gpu=2)
    # Append to lr_extension.csv directly so plotting script picks it up
    output_file = 'outputs/results/lr_extension.csv'
    
    def on_complete(res):
        save_single_result(res, output_file)
        print(f"Finished LR={res['lr']} Acc={res['best_test_acc']:.2f}%")

    scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=on_complete)

if __name__ == '__main__':
    main()
