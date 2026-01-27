"""
Logging utility for managing experiment logs and output directories.
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    Manages logging for experiments with organized directory structure.

    Directory structure:
        outputs/
        ├── logs/               # Log files
        │   ├── exp1_20260113_143022.log
        │   └── ...
        ├── results/            # CSV results
        │   ├── results.csv
        │   └── ...
        ├── checkpoints/        # Model checkpoints
        │   ├── exp1_run0_best.pth
        │   └── ...
        └── plots/              # Generated plots
            ├── exp1_lr_ordering.png
            └── ...
    """

    def __init__(self, experiment_name="experiment", log_to_file=True, log_to_console=True):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment (e.g., "exp1", "exp2")
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory structure
        self.base_dir = Path("outputs")
        self.log_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.plots_dir = self.base_dir / "plots"

        self._create_directories()

        # Set up logger
        self.logger = self._setup_logger(log_to_file, log_to_console)

    def _create_directories(self):
        """Create all necessary directories"""
        for directory in [self.log_dir, self.results_dir, self.checkpoint_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self, log_to_file, log_to_console):
        """Set up logger with file and console handlers"""
        logger = logging.getLogger(f"{self.experiment_name}_{self.timestamp}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Clear existing handlers
        logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        if log_to_file:
            log_file = self.log_dir / f"{self.experiment_name}_{self.timestamp}.log"
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def info(self, message):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

    def get_results_path(self, filename="results.csv"):
        """Get path for results file"""
        return self.results_dir / filename

    def get_checkpoint_path(self, run_id, checkpoint_type="best"):
        """
        Get path for checkpoint file.

        Args:
            run_id: Identifier for this run (e.g., "exp1_run0", "SGD_lr0.1")
            checkpoint_type: Type of checkpoint ("best", "latest", "final")

        Returns:
            Path to checkpoint file
        """
        filename = f"{self.experiment_name}_{run_id}_{checkpoint_type}.pth"
        return self.checkpoint_dir / filename

    def get_plot_path(self, plot_name):
        """Get path for plot file"""
        return self.plots_dir / plot_name

    def log_config(self, config_dict):
        """Log configuration dictionary"""
        self.info("="*80)
        self.info("EXPERIMENT CONFIGURATION")
        self.info("="*80)
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")
        self.info("="*80)

    def log_metrics(self, epoch, metrics_dict):
        """Log metrics for an epoch"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in metrics_dict.items()])
        self.info(f"Epoch {epoch} | {metrics_str}")

    def log_experiment_start(self, total_runs):
        """Log experiment start"""
        self.info("\n" + "="*80)
        self.info(f"STARTING EXPERIMENT: {self.experiment_name}")
        self.info(f"Total runs: {total_runs}")
        self.info(f"Timestamp: {self.timestamp}")
        self.info("="*80 + "\n")

    def log_experiment_end(self, successful_runs, total_runs, elapsed_time):
        """Log experiment end"""
        self.info("\n" + "="*80)
        self.info(f"EXPERIMENT COMPLETED: {self.experiment_name}")
        self.info(f"Successful runs: {successful_runs}/{total_runs}")
        self.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        self.info("="*80 + "\n")

    def log_run_start(self, run_id, run_config):
        """Log individual run start"""
        config_str = " | ".join([f"{k}={v}" for k, v in run_config.items()])
        self.info(f"\n>>> Starting run: {run_id} | {config_str}")

    def log_run_end(self, run_id, results):
        """Log individual run end"""
        self.info(f">>> Completed run: {run_id}")
        self.info(f"    Best Test Acc: {results.get('best_test_acc', 'N/A'):.2f}%")
        self.info(f"    Final Test Acc: {results.get('final_test_acc', 'N/A'):.2f}%")


def get_logger(experiment_name="experiment", log_to_file=True, log_to_console=True):
    """
    Factory function to create an experiment logger.

    Args:
        experiment_name: Name of the experiment
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, log_to_file, log_to_console)


def setup_directories():
    """
    Set up all output directories.
    This is a convenience function that can be called at the start of any script.
    """
    base_dir = Path("outputs")
    directories = [
        base_dir / "logs",
        base_dir / "results",
        base_dir / "checkpoints",
        base_dir / "plots",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        'base': base_dir,
        'logs': base_dir / "logs",
        'results': base_dir / "results",
        'checkpoints': base_dir / "checkpoints",
        'plots': base_dir / "plots",
    }


if __name__ == '__main__':
    # Test the logger
    logger = get_logger("test_exp")

    logger.log_experiment_start(10)

    logger.log_config({
        'batch_size': 128,
        'learning_rate': 0.1,
        'epochs': 100,
        'gpu': 0
    })

    logger.log_run_start("run_0", {'lr': 0.1, 'wd': 5e-4})

    for epoch in range(1, 4):
        logger.log_metrics(epoch, {
            'train_loss': 2.5 - epoch * 0.1,
            'test_acc': 50.0 + epoch * 5.0,
            'lr': 0.1
        })

    logger.log_run_end("run_0", {'best_test_acc': 65.0, 'final_test_acc': 63.5})

    logger.log_experiment_end(10, 10, 3600.0)

    print(f"\nLog saved to: {logger.log_dir / f'{logger.experiment_name}_{logger.timestamp}.log'}")
    print(f"Results dir: {logger.results_dir}")
    print(f"Checkpoint dir: {logger.checkpoint_dir}")
    print(f"Plots dir: {logger.plots_dir}")
