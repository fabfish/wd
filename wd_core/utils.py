"""
Utility functions for training, evaluation, and seed setting.
"""
import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path


def make_grad_scaler(enabled=True):
    """
    Build a GradScaler, preferring the non-deprecated torch.amp entry point.

    A scaler built here is meant to live for the whole run: rebuilding it every
    epoch resets the loss scale to its initial value, which throws away a few
    steps at the start of each epoch while the scale re-settles.
    """
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return GradScaler(enabled=enabled)


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, scheduler, device, use_amp=True,
                scaler=None, return_acc=False):
    """
    Train for one epoch with mixed precision.

    Args:
        scaler: Optional persistent GradScaler. When omitted a fresh scaler is
            built for this epoch, which is the historical behaviour that all the
            already-collected results were produced with.
        return_acc: When True, also return the running training accuracy.

    Returns:
        avg_loss, or (avg_loss, train_acc) when return_acc is set.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    if use_amp and scaler is None:
        scaler = GradScaler()

    total_loss = 0.0
    total_samples = 0
    correct = 0

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
        if return_acc:
            correct += outputs.detach().argmax(1).eq(targets).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total_samples
    if return_acc:
        return avg_loss, 100.0 * correct / total_samples
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
    final_test_loss = 0.0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp)
        test_acc, test_loss = evaluate(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if epoch == epochs - 1:
            final_test_acc = test_acc
            final_train_loss = train_loss
            final_test_loss = test_loss

        # Only log at intervals or at the end
        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_test_acc:.2f}%")

    return best_test_acc, final_test_acc, final_train_loss, final_test_loss


def train_model_ext(model, train_loader, test_loader, optimizer, scheduler,
                    device, epochs=100, use_amp=True, log_interval=10,
                    divergence_loss_threshold=None, divergence_check_epoch=3,
                    probe_fn=None, keep_history=False, tag="",
                    wd_schedule_fn=None):
    """
    Training loop for the coupling experiments.

    Differs from train_model in four ways that the analysis depends on:
      * reports final training accuracy, so the generalization gap is measurable
      * aborts early on divergence, which makes learning-rate boundary searches
        cheap (a diverged cell costs seconds rather than a full run)
      * accumulates sum_lr = sum_t eta_t over optimizer steps, which is the
        quantity the coupling law is actually stated in (it equals eta*T for a
        constant schedule but eta*T/2 for cosine-to-zero)
      * exposes a probe hook for per-epoch diagnostics such as weight norms

    wd_schedule_fn:
        Optional callable ``wd_schedule_fn(epoch) -> float`` (0-based epoch).
        When set, every epoch begins by writing that value into every
        ``optimizer.param_groups[*]['weight_decay']``. Learning-rate scheduling
        is unchanged (pass ``scheduler=None`` for a fixed learning rate).

    Returns:
        dict of summary metrics (plus 'history' when keep_history is set).
    """
    scaler = make_grad_scaler(enabled=use_amp) if use_amp else None
    steps_per_epoch = len(train_loader)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = float('nan')
    final_train_acc = float('nan')
    final_test_loss = float('nan')
    sum_lr = 0.0
    diverged = False
    epochs_run = 0
    history = []

    for epoch in range(epochs):
        if wd_schedule_fn is not None:
            wd_now = float(wd_schedule_fn(epoch))
            for group in optimizer.param_groups:
                group['weight_decay'] = wd_now

        lr_this_epoch = optimizer.param_groups[0]['lr']
        sum_lr += lr_this_epoch * steps_per_epoch

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, use_amp,
            scaler=scaler, return_acc=True
        )
        epochs_run = epoch + 1

        blew_up = math.isnan(train_loss) or math.isinf(train_loss)
        if not blew_up and divergence_loss_threshold is not None:
            if epoch + 1 >= divergence_check_epoch and train_loss > divergence_loss_threshold:
                blew_up = True

        if blew_up:
            diverged = True
            final_train_loss = train_loss
            final_train_acc = train_acc
            test_acc, test_loss = evaluate(model, test_loader, device)
            final_test_acc, final_test_loss = test_acc, test_loss
            best_test_acc = max(best_test_acc, test_acc)
            print(f"{tag}DIVERGED at epoch {epochs_run}/{epochs} (train loss {train_loss:.4f})")
            break

        test_acc, test_loss = evaluate(model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        wd_this_epoch = optimizer.param_groups[0].get('weight_decay', 0.0)
        if probe_fn is not None:
            probe = probe_fn(epoch, model)
            if probe:
                history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                                'train_acc': train_acc, 'test_acc': test_acc,
                                'test_loss': test_loss, 'lr': lr_this_epoch,
                                'wd': wd_this_epoch, **probe})
        elif keep_history:
            history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                            'train_acc': train_acc, 'test_acc': test_acc,
                            'test_loss': test_loss, 'lr': lr_this_epoch,
                            'wd': wd_this_epoch})

        if epoch == epochs - 1:
            final_test_acc = test_acc
            final_train_loss = train_loss
            final_train_acc = train_acc
            final_test_loss = test_loss

        if log_interval and ((epoch + 1) % log_interval == 0 or epoch == epochs - 1):
            print(f"{tag}Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
                  f"Best: {best_test_acc:.2f}%")

    result = {
        'best_test_acc': best_test_acc,
        'final_test_acc': final_test_acc,
        'final_train_loss': final_train_loss,
        'final_train_acc': final_train_acc,
        'final_test_loss': final_test_loss,
        'sum_lr': sum_lr,
        'diverged': int(diverged),
        'epochs_run': epochs_run,
    }
    if keep_history or probe_fn is not None:
        result['history'] = history
    return result


def weight_norm_probe(model):
    """
    Per-layer weight statistics used for the rotational-equilibrium diagnostic.

    Reports the global L2 norm plus the norm of every weight matrix/kernel that
    weight decay actually acts on (biases and normalization parameters are
    excluded because their scale behaves differently).
    """
    total_sq = 0.0
    per_layer = {}
    for name, p in model.named_parameters():
        if p.ndim < 2:
            continue
        n = float(p.detach().norm().item())
        per_layer[name] = n
        total_sq += n ** 2
    return {'weight_norm_total': math.sqrt(total_sq), 'per_layer': per_layer}


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

