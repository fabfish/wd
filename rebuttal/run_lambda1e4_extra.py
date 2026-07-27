"""
One-shot follow-ups for λ=0.0001:
  (a) extend right tail with η=10 (η×λ=1e-3) — likely to diverge
  (b) re-run η=2.5 (η×λ=2.5e-4) with a different seed (43) to verify
      the lone seed=42 point that looks anomalous on the test-loss curve

Same training recipe as run_rebuttal.py (SGDM, mom=0.9, BS=128, ResNet-18,
CIFAR-100, 100 epochs). Appends to results_resnet18_seed42_exp2_fill.csv.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wd_core.gpu_scheduler import GPUScheduler, parse_gpu_ids  # noqa: E402
from wd_core.logger import get_logger  # noqa: E402
from rebuttal.run_rebuttal import run_single_experiment_worker  # noqa: E402
from rebuttal.run_exp2_fill import append_one_row  # noqa: E402


# (lr, wd, seed)
CELLS = [
    (10.0, 1e-4, 42),  # η×λ = 1e-3 (right-tail extension)
    (2.5,  1e-4, 43),  # η×λ = 2.5e-4 (verification of suspicious point)
]
MODEL = 'resnet18'
METHOD = 'SGDM+WD'
BATCH = 128
MOMENTUM = 0.9
EPOCHS = 100
USE_AMP = True
GPUS = 'all'
WORKERS_PER_GPU = 2
OUTPUT = ROOT / 'rebuttal/results/results_resnet18_seed42_exp2_fill.csv'


def main():
    logger = get_logger('lambda1e4_extra')
    tasks = [(MODEL, METHOD, BATCH, lr, wd, MOMENTUM, EPOCHS, seed, USE_AMP)
             for lr, wd, seed in CELLS]
    logger.info(f"running {len(tasks)} cell(s) for λ=1e-4: "
                f"{[(lr, wd, lr*wd, seed) for lr, wd, seed in CELLS]}")

    gpu_ids = None if GPUS == 'all' else parse_gpu_ids(GPUS)
    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True,
                             workers_per_gpu=WORKERS_PER_GPU)

    done = {'n': 0}

    def on_complete(result):
        if result is None:
            return
        done['n'] += 1
        append_one_row(OUTPUT, result)
        logger.info(
            f"[{done['n']}/{len(tasks)}] saved lr={result['lr']:g} "
            f"wd={result['wd']:g} -> "
            f"train_loss={result.get('final_train_loss', float('nan')):.4f} "
            f"test_loss={result.get('final_test_loss', float('nan')):.4f} "
            f"test_acc={result.get('final_test_acc', float('nan')):.2f}"
        )

    start = time.time()
    results = scheduler.run_tasks(tasks, run_single_experiment_worker, on_complete=on_complete)
    elapsed = (time.time() - start) / 60
    ok = sum(1 for r in results if r)
    logger.info(f"finished {ok}/{len(results)} cell(s) in {elapsed:.1f} min")


if __name__ == '__main__':
    main()
