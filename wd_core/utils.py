"""
Utility functions for training, evaluation, and seed setting.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path


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

    pbar = tqdm(train_loader, desc="Training", leave=False, disable=True)  # Disabled for speed
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
                device, epochs=100, use_amp=True, log_interval=10):
    """
    Complete training loop.

    Args:
        log_interval: Log every N epochs (default 10). Set to 1 for verbose logging.

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

        # Only log at intervals or at the end
        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_test_acc:.2f}%")

    return best_test_acc, final_test_acc, final_train_loss


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics to save
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)

    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def train_model_with_checkpoints(model, train_loader, test_loader, optimizer, scheduler,
                                   device, epochs=100, use_amp=True, save_best=True,
                                   checkpoint_dir=None, run_id="run", logger=None, log_interval=10):
    """
    Complete training loop with checkpoint saving and logging support.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of training epochs
        use_amp: Use automatic mixed precision
        save_best: Save checkpoint when best accuracy is achieved
        checkpoint_dir: Directory to save checkpoints (default: outputs/checkpoints)
        run_id: Identifier for this run
        logger: Optional logger instance
        log_interval: Log every N epochs (default 10). Set to 1 for verbose logging.

    Returns:
        best_test_acc: Best test accuracy achieved
        final_test_acc: Final test accuracy
        final_train_loss: Final training loss
    """
    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

    # Set default checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path("outputs/checkpoints")
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp)
        test_acc, test_loss = evaluate(model, test_loader, device)

        # Log metrics at intervals or at the end
        should_log = (epoch + 1) % log_interval == 0 or epoch == epochs - 1
        if should_log:
            if logger:
                logger.log_metrics(epoch + 1, {
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'best_acc': best_test_acc
                })
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_test_acc:.2f}%")

        # Save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc

            if save_best:
                metrics = {
                    'best_test_acc': best_test_acc,
                    'test_acc': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'epoch': epoch + 1
                }

                checkpoint_path = checkpoint_dir / f"{run_id}_best.pth"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, checkpoint_path)

                if logger:
                    logger.info(f"  Saved best checkpoint: {checkpoint_path}")

        if epoch == epochs - 1:
            final_test_acc = test_acc
            final_train_loss = train_loss

            # Save final checkpoint
            if save_best:
                metrics = {
                    'final_test_acc': final_test_acc,
                    'final_train_loss': final_train_loss,
                    'best_test_acc': best_test_acc,
                    'epoch': epoch + 1
                }

                checkpoint_path = checkpoint_dir / f"{run_id}_final.pth"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, checkpoint_path)

    return best_test_acc, final_test_acc, final_train_loss

