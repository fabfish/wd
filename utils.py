"""
Utility functions for training, evaluation, and seed setting.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, scheduler, device, use_amp=True):
    """
    Train for one epoch with mixed precision.

    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None

    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate(model, test_loader, device):
    """
    Evaluate model on test set.

    Returns:
        accuracy: Test accuracy (0-100)
        avg_loss: Average test loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return accuracy, avg_loss


def train_model(model, train_loader, test_loader, optimizer, scheduler,
                device, epochs=100, use_amp=True):
    """
    Complete training loop.

    Returns:
        best_test_acc: Best test accuracy achieved
        final_test_acc: Final test accuracy
        final_train_loss: Final training loss
    """
    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp)
        test_acc, test_loss = evaluate(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if epoch == epochs - 1:
            final_test_acc = test_acc
            final_train_loss = train_loss

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_test_acc:.2f}%")

    return best_test_acc, final_test_acc, final_train_loss
