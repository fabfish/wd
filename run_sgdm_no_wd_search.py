"""
SGDM (no WD) Momentum Search Experiment
Goal: Find optimal Momentum + LR configuration for SGDM (no WD)
to fit between SGD (LR~0.3) and SGD+WD (LR~0.1).
Run on GPU 1 with 10 workers.
"""
import argparse
import csv
import os
import time
from itertools import product
import torch
from run_three_methods_comparison import run_single_experiment_worker
from gpu_scheduler import GPUScheduler, parse_gpu_ids
from logger import get_logger

def run_sgdm_search(gpu_ids, epochs=100, seed=42, use_amp=True, logger=None):
    if logger:
        logger.info("\n" + "="*80)
        logger.info("SGDM (no WD) Momentum Search")
        logger.info("="*80)

    batch_size = 128
    tasks = []

    # Search Space
    momentums = [0.5, 0.6, 0.7, 0.8]
    lrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    
    # Create tasks
    for mom, lr in product(momentums, lrs):
        # method, batch_size, lr, wd, momentum, epochs, seed, use_amp, save_checkpoints
        task = ("SGDM", batch_size, lr, 0.0, mom, epochs, seed, use_amp, False)
        tasks.append(task)

    total_runs = len(tasks)
    if logger:
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Momentums: {momentums}")
        logger.info(f"LRs: {lrs}")

    # Use GPUScheduler with high concurrency
    # workers_per_gpu=10 as requested
    scheduler = GPUScheduler(gpu_ids=gpu_ids, workers_per_gpu=10, verbose=True)
    
    start_time = time.time()
    
    # Define callback to save incrementally (optional but good)
    # But GPUScheduler returns all results at end.
    # We can rely on final save or hack scheduler? 
    # The scheduler doesn't support incremental callback in its current simple form easily 
    # (actually it does, looking at code: `on_complete` arg is there!)
    
    results = []
    
    def on_task_complete(result):
        if result:
             save_result_row(result, "outputs/results/sgdm_no_wd_search.csv")
             if logger:
                 logger.info(f"Saved: Mom={result['momentum']}, LR={result['lr']}, Acc={result['best_test_acc']:.2f}%")
        results.append(result)

    scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=on_task_complete)
    elapsed_time = time.time() - start_time

    if logger:
        logger.info(f"\nCompleted {total_runs} experiments in {elapsed_time/60:.2f} minutes")

    return results

def save_result_row(result, output_file):
    """Save a single result row to CSV immediately."""
    if result is None:
        return

    fieldnames = ['method', 'batch_size', 'lr', 'wd', 'momentum',
                  'final_test_acc', 'final_train_loss', 'best_test_acc']

    file_exists = os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def main():
    parser = argparse.ArgumentParser(description='SGDM No WD Momentum Search')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/results/sgdm_no_wd_search.csv')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpus', type=str, default='1', help='GPU ID to use, e.g., "1"')
    parser.add_argument('--no_log', action='store_true')
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus == 'all':
        gpu_ids = None
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
    else:
        gpu_ids = []

    exp_name = "sgdm_no_wd_search"
    logger = get_logger(exp_name, log_to_file=not args.no_log, log_to_console=True)

    logger.log_config({
        'Experiment': 'SGDM No WD Search',
        'Epochs': args.epochs,
        'Seed': args.seed,
        'GPUs': gpu_ids
    })

    # Clear output file if exists to avoid mixing? Or append?
    # Append is safer, but if we want fresh start:
    # We can check if file exists and maybe rename it?
    # User didn't specify, but "search" usually implies a new set.
    # Let's keep append but maybe add a header check is done in save_result_row.
    
    run_sgdm_search(gpu_ids, args.epochs, args.seed, args.use_amp, logger)

if __name__ == '__main__':
    main()
