#!/usr/bin/env python3
"""
One-Click Experiment Runner for CIFAR-100 Optimization Experiments

This script runs all 3 experiment sets with optimal GPU utilization.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class Colors:
    """Terminal colors for pretty output"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


def print_header(message):
    """Print a colored header"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{message}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")


def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.NC}")


def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.NC}")


def print_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.NC}")


def check_environment():
    """Check if the environment is properly set up"""
    print_header("CHECKING ENVIRONMENT")

    # Check Python packages
    try:
        import torch
        print_success("PyTorch installed")

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print_success(f"{num_gpus} GPU(s) detected")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
        else:
            print_warning("No GPUs detected. Will run on CPU (slow).")
    except ImportError:
        print_error("PyTorch not found. Please install: pip install -r requirements.txt")
        sys.exit(1)

    # Check required files
    required_files = [
        'run_experiments.py',
        'models.py',
        'utils.py',
        'gpu_scheduler.py',
        'logger.py',
        'plot_results.py'
    ]

    for file in required_files:
        if not Path(file).exists():
            print_error(f"Required file not found: {file}")
            sys.exit(1)

    print_success("All required files found")


def run_experiment(exp_num, gpus, epochs, save_checkpoints, background=False):
    """Run a single experiment set"""
    print_header(f"EXPERIMENT SET {exp_num}")

    cmd = [
        'python', 'run_experiments.py',
        '--experiment', str(exp_num),
        '--epochs', str(epochs),
        '--gpus', gpus
    ]

    if save_checkpoints:
        cmd.append('--save_checkpoints')

    print(f"Command: {' '.join(cmd)}")

    if background:
        # Run in background and save output
        log_file = Path(f"outputs/logs/exp{exp_num}_parallel.txt")
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        print(f"Running in background (PID: {process.pid})")
        print(f"Log file: {log_file}")
        return process
    else:
        # Run in foreground
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print_success(f"Experiment {exp_num} completed successfully")
            return True
        else:
            print_error(f"Experiment {exp_num} failed")
            return False


def run_sequential(gpus, epochs, save_checkpoints):
    """Run all experiments sequentially"""
    print_header("SEQUENTIAL MODE: Running experiments one by one")
    print_warning(f"Each experiment will use GPUs: {gpus}")

    for exp_num in [1, 2, 3]:
        success = run_experiment(exp_num, gpus, epochs, save_checkpoints, background=False)
        if not success:
            return False

    return True


def run_parallel(epochs, save_checkpoints):
    """Run all experiments in parallel with different GPU subsets"""
    print_header("PARALLEL MODE: Running all experiments simultaneously")
    print_warning("This will use different GPU subsets for each experiment")

    print("\nGPU Allocation:")
    print("  Experiment 1 (24 runs): GPUs 0-2 (3 GPUs)")
    print("  Experiment 2 (35 runs): GPUs 3-5 (3 GPUs)")
    print("  Experiment 3 (24 runs): GPUs 6-7 (2 GPUs)")
    print()

    # Launch all experiments
    gpu_allocation = {
        1: "0-2",
        2: "3-5",
        3: "6-7"
    }

    processes = {}
    for exp_num, gpus in gpu_allocation.items():
        proc = run_experiment(exp_num, gpus, epochs, save_checkpoints, background=True)
        processes[exp_num] = proc

    print_success("All experiments launched in parallel")
    print("\nMonitoring progress...")
    print("Logs available in outputs/logs/")
    print()

    # Wait for all to complete
    all_success = True
    for exp_num, proc in processes.items():
        proc.wait()
        if proc.returncode == 0:
            print_success(f"Experiment {exp_num} completed")
        else:
            print_error(f"Experiment {exp_num} failed")
            all_success = False

    return all_success


def generate_plots():
    """Generate plots from results"""
    print_header("GENERATING PLOTS")

    result = subprocess.run(['python', 'plot_results.py', '--stats'])

    if result.returncode == 0:
        print_success("Plots generated successfully")
        print("  - outputs/plots/exp1_lr_ordering.png")
        print("  - outputs/plots/exp2_eta_lambda_heatmap.png")
        print("  - outputs/plots/exp3_batch_size_scaling.png")
        return True
    else:
        print_warning("Failed to generate plots. Run manually: python plot_results.py")
        return False


def print_summary(start_time, save_checkpoints):
    """Print experiment summary"""
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print_header("ALL EXPERIMENTS COMPLETED")
    print_success(f"Total time: {hours}h {minutes}m {seconds}s")

    # Analyze results
    results_file = Path("outputs/results/results.csv")
    if results_file.exists():
        try:
            import pandas as pd

            df = pd.read_csv(results_file)
            total_runs = len(df)

            print_header("SUMMARY")
            print(f"Results saved to: {results_file}")
            print(f"Logs saved to: outputs/logs/")
            if save_checkpoints:
                print(f"Checkpoints saved to: outputs/checkpoints/")
            print(f"Plots saved to: outputs/plots/")
            print()
            print_success(f"Total experiment runs: {total_runs}")

            # Best results
            print("\nBest Results:")
            exp1_methods = ['SGD', 'SGD+WD', 'SGDM+WD']
            for method in exp1_methods:
                df_method = df[df['method'] == method]
                if not df_method.empty:
                    best_row = df_method.loc[df_method['best_test_acc'].idxmax()]
                    print(f"  {method:10s}: {best_row['best_test_acc']:.2f}% "
                          f"(lr={best_row['lr']}, wd={best_row['wd']})")

            # SGDM best overall
            df_sgdm = df[df['method'] == 'SGDM']
            if not df_sgdm.empty:
                best_row = df_sgdm.loc[df_sgdm['best_test_acc'].idxmax()]
                print(f"  SGDM Best : {best_row['best_test_acc']:.2f}% "
                      f"(lr={best_row['lr']}, wd={best_row['wd']})")

        except Exception as e:
            print_warning(f"Could not analyze results: {e}")

    print_success("\nAll done!")
    print("\nTo view plots:")
    print("  eog outputs/plots/*.png    # Linux")
    print("  open outputs/plots/*.png   # macOS")
    print("\nTo analyze results:")
    print("  python plot_results.py --stats")


def main():
    parser = argparse.ArgumentParser(
        description='One-click runner for all CIFAR-100 optimization experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py                    # Run sequentially on GPUs 0-7
  python run_all_experiments.py --parallel         # Run in parallel
  python run_all_experiments.py --epochs 200       # Run with 200 epochs
  python run_all_experiments.py --save-checkpoints # Save model checkpoints
  python run_all_experiments.py --gpus 0-3         # Use only GPUs 0-3
        """
    )

    parser.add_argument('--parallel', action='store_true',
                        help='Run experiment sets in parallel (uses separate GPUs)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--save-checkpoints', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--gpus', type=str, default='0-7',
                        help='GPU IDs to use (default: 0-7)')

    args = parser.parse_args()

    # Print configuration
    print_header("EXPERIMENT CONFIGURATION")
    print(f"GPU Configuration: {args.gpus}")
    print(f"Epochs: {args.epochs}")
    print(f"Save Checkpoints: {args.save_checkpoints}")
    print(f"Parallel Execution: {args.parallel}")

    # Check environment
    check_environment()

    # Backup existing results
    results_file = Path("outputs/results/results.csv")
    if results_file.exists():
        backup_file = results_file.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        print_warning(f"Backing up existing results to: {backup_file}")
        import shutil
        shutil.copy(results_file, backup_file)

    # Record start time
    start_time = time.time()
    start_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print_header("STARTING EXPERIMENTS")
    print(f"Start time: {start_date}")

    # Run experiments
    if args.parallel:
        success = run_parallel(args.epochs, args.save_checkpoints)
    else:
        success = run_sequential(args.gpus, args.epochs, args.save_checkpoints)

    if not success:
        print_error("Some experiments failed. Check logs in outputs/logs/")
        sys.exit(1)

    # Generate plots
    generate_plots()

    # Print summary
    print_summary(start_time, args.save_checkpoints)


if __name__ == '__main__':
    main()
