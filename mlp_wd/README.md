# mlp_wd: MLP weight-decay experiments (CIFAR-10 / MNIST)

Lightweight reproduction of the three weight-decay / learning-rate experiments
from the parent repo, but on a 3-layer MLP + CIFAR-10 instead of ResNet-18 +
CIFAR-100. A single run takes 1-3 minutes on an A100, so a full grid finishes
in under an hour with 4x A100 + many workers per GPU.

The headline figure is **Exp2 - "bunch of spoons"**: final test loss vs
`eta * lambda`, one curve per `lambda`. If the eta x lambda scaling law holds
under cosine annealing, every curve descends from a similar starting loss,
bottoms out near a *common* `eta * lambda`, and rises again as training
becomes unstable.

## 1. Why MLP + CIFAR-10 (and not MNIST or ResNet)

- **Not ResNet/CIFAR-100**: the depth, BatchNorm, and CIFAR-100 difficulty
  hide weight-decay/lr dynamics behind generic optimization noise. Each run
  needs ~100 epochs to be informative -> grid sweeps cost hours.
- **Not MNIST**: a 3-layer MLP hits 98% with virtually any sane `eta`, `lambda`,
  so the spoon's right arm (instability/over-regularization) doesn't appear
  cleanly in any reasonable grid. Loss saturates near zero.
- **MLP + CIFAR-10**: best test accuracy lands around 50-58%, and the network
  is *visibly* sensitive to `eta` and `lambda`. With cosine annealing over
  30 epochs we reproducibly see (i) WD-induced LR shift in Exp1, (ii) a
  shared `eta x lambda` minimum in Exp2, and (iii) a clear linear-LR-rule
  dependence on batch size in Exp3.
- **3 layers, not 2 or 4**: 2 layers is essentially a logistic regression and
  the optimizer dynamics flatten out. 4 layers slows training by ~25% with no
  added scientific signal. Width is fixed to 512; that, plus 3 layers, gives
  ~1.84M parameters which is enough for the dynamics we want.

Architecture: `Flatten -> Linear(3072, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)`.
No BatchNorm, no Dropout, no augmentation - only `ToTensor + Normalize`.

## 2. Layout

```text
mlp_wd/
  mlp_core/                # importable library
    models.py              # MLP, build_mlp_for_dataset
    datasets.py            # get_cifar10_loaders, get_mnist_loaders, get_loaders
    utils.py               # set_seed + train_model_with_history (records final_test_loss & best_test_loss)
    runner.py              # run_single_experiment (one CSV row + history JSON per call)
    grid.py                # build_tasks / run_grid (resume + filelock + multi-GPU)
    gpu_scheduler.py       # multiprocess GPU dispatcher (copied from wd_core)
    logger.py              # ExperimentLogger (copied from wd_core)
    io.py                  # CSV append + resume helpers
  scripts/                 # CLIs
    train_one.py           # single-run sanity check
    run_pilot_exp2.py      # 4x4 mini-grid before committing to full Exp2
    run_exp1_lr_ordering.py
    run_exp2_eta_lambda.py
    run_exp3_batchsize.py
  analysis/                # plotting CLIs
    plot_exp1_lr_ordering.py
    plot_exp2_loss_spoons.py     # the headline plot
    plot_exp2_heatmap.py
    plot_exp3_batchsize.py
  outputs/
    results/<exp>.csv      # one CSV per experiment, schema below
    history/<exp>/*.json   # per-run per-epoch history (for re-plots)
    plots/                 # PNGs
    logs/                  # ExperimentLogger output
```

CSV schema (every CSV produced by `mlp_wd` shares this; see
`mlp_core/runner.py:CSV_FIELDS`):

```text
method, dataset, num_layers, hidden_dim,
batch_size, lr, wd, momentum, epochs, seed,
final_train_loss, final_train_acc,
final_test_loss,  final_test_acc,
best_test_loss,   best_test_acc,
diverged, epochs_run
```

## 3. Environment

The repo's existing conda env `trace` already has everything (`torch 2.9.1+cu128`,
`torchvision 0.24.1+cu128`, `pandas 2.1.1`, `filelock 3.20.0`).

```bash
# from repo root
PY=~/.conda/envs/trace/bin/python
$PY -c "import torch, torchvision, pandas, filelock; print(torch.cuda.device_count(), 'GPUs')"
```

If you need a fresh env: `pip install -r requirements.txt && pip install filelock`.

GPU note: GPU 0 currently hosts another long-running job, so default
`--gpus 1,2,3`. The scripts also accept `--gpus all` or `0-3`.

## 4. End-to-end recipe

All commands are run from the repo root (`/home/yzy/GitHub/wd`) so that
`python -m mlp_wd.scripts.<name>` resolves the `mlp_wd` package.

### Step 0 - sanity (~1 minute on A100, downloads CIFAR-10 first time)

```bash
PY=~/.conda/envs/trace/bin/python
CUDA_VISIBLE_DEVICES=3 $PY -m mlp_wd.scripts.train_one \
  --epochs 5 --lr 0.05 --wd 1e-3 --momentum 0.9 \
  --num_layers 3 --hidden_dim 512 --dataset cifar10
```

Expected: `best_test_acc` between 50% and 60% after 5 epochs.

### Step 1 - Exp2 pilot (~5-10 min on 3 GPUs)

This is the gate before committing to the full 36-run grid: confirm spoons
exist for the CIFAR-10 / 3-layer MLP setting.

```bash
PY=~/.conda/envs/trace/bin/python
$PY -m mlp_wd.scripts.run_pilot_exp2 --gpus 1,2,3 --workers_per_gpu 8 --epochs 20
$PY -m mlp_wd.analysis.plot_exp2_loss_spoons \
  --results mlp_wd/outputs/results/exp2_pilot.csv \
  --output  mlp_wd/outputs/plots/exp2_pilot_loss_spoons.png
```

Then **inspect** `outputs/plots/exp2_pilot_loss_spoons.png`. Decision rules:

- If every lambda curve has both a clean descent **and** a clean rise, lock
  the grid and run Exp2 in full.
- If only the largest lambda curves rise on the right, extend the LR grid
  upward (e.g. add `eta=3.0`).
- If curves don't descend (best is at the smallest `eta * lambda`), extend
  the LR grid downward (e.g. add `eta=1e-3`) and/or shrink lambda.
- If the optimum spread (max/min of per-lambda minima) is > 5x in
  `eta x lambda`, the scaling law is **not** holding cleanly here -> consider
  lengthening training (`--epochs 50`) or switching to a denser grid.

### Step 2 - full Exp2 (~15-25 min on 3 GPUs)

Default: 6x6 = 36 runs, SGDM, BS=128, 30 epochs.

```bash
PY=~/.conda/envs/trace/bin/python
$PY -m mlp_wd.scripts.run_exp2_eta_lambda --gpus 1,2,3 --workers_per_gpu 10 --epochs 30
$PY -m mlp_wd.analysis.plot_exp2_loss_spoons \
  --results mlp_wd/outputs/results/exp2.csv \
  --output  mlp_wd/outputs/plots/exp2_loss_spoons.png
$PY -m mlp_wd.analysis.plot_exp2_heatmap \
  --results mlp_wd/outputs/results/exp2.csv \
  --output  mlp_wd/outputs/plots/exp2_heatmap.png \
  --metric  final_test_loss
```

Override the grid via `--lrs 1e-3,3e-3,1e-2,...` and `--wds 1e-4,3e-4,...`.

### Step 3 - Exp1 (~10 min on 3 GPUs)

Two paired ablations, ~32 runs total. Each pair shares an LR grid so the
WD-induced LR shift is read off directly.

```bash
PY=~/.conda/envs/trace/bin/python
$PY -m mlp_wd.scripts.run_exp1_lr_ordering --gpus 1,2,3 --workers_per_gpu 10 --epochs 30
$PY -m mlp_wd.analysis.plot_exp1_lr_ordering \
  --results mlp_wd/outputs/results/exp1.csv \
  --output  mlp_wd/outputs/plots/exp1_lr_ordering.png
```

### Step 4 - Exp3 (~10 min on 3 GPUs)

5 batch sizes x 5 lambdas, linear LR rule `eta = base_lr * (B / base_bs)`.

```bash
PY=~/.conda/envs/trace/bin/python
$PY -m mlp_wd.scripts.run_exp3_batchsize --gpus 1,2,3 --workers_per_gpu 10 --epochs 30
$PY -m mlp_wd.analysis.plot_exp3_batchsize \
  --results mlp_wd/outputs/results/exp3.csv \
  --output  mlp_wd/outputs/plots/exp3_batchsize.png
```

## 5. Default grids (one place to edit)

| experiment | grid | runs |
|---|---|---|
| Exp1 group A (mom=0)   | SGD vs SGD+WD over `eta in {0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}`, `lambda_a = 1e-3` | 16 |
| Exp1 group B (mom=0.9) | SGDM vs SGDM+WD over `eta in {0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3}`, `lambda_b = 1e-3` | 16 |
| Exp2 (SGDM, BS=128)    | `eta in {3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0}` x `lambda in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}` | 36 |
| Exp3 (SGDM)            | `B in {32, 64, 128, 256, 512}` x `lambda in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`, `eta = 0.05 * (B/128)` | 25 |

Total: 93 runs, ~30-50 min wall-clock with `workers_per_gpu=8-10` on 3 GPUs.

## 6. Concurrency / GPU notes

- A 3-layer MLP on CIFAR-10 uses ~600 MB GPU RAM at BS=128, so an A100 (80 GB)
  can comfortably host **8-12 concurrent workers**. Default in scripts is 8;
  bump to 10 if the GPUs look under-utilized in `nvidia-smi`.
- The scheduler uses `multiprocessing.spawn`, so each worker is a fresh
  Python process with `CUDA_VISIBLE_DEVICES` masked to its assigned GPU.
- DataLoader `num_workers=2` is plenty for these tiny inputs; do **not**
  raise it to fight CPU-side queue contention with the scheduler.

## 7. Resume / dedup

All `run_*` scripts write the CSV row only after a run completes (with a
`filelock` around appends). On the next launch with the same `--output`
path, `mlp_core/grid.py:filter_completed` rebuilds the dedup keys and skips
any (`method, dataset, num_layers, hidden_dim, batch_size, lr, wd, momentum,
epochs, seed`) tuple that already exists. Add new grid points by simply
re-running the script with extended `--lrs` / `--wds`.

## 8. Re-plots without re-training

Every run also dumps a per-epoch history JSON to `outputs/history/<exp>/`,
so you can later draw additional figures (e.g. loss-trajectory subplots)
without retraining. The CSV is the source of truth for the four canonical
plots above.

## 9. Quick troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `ModuleNotFoundError: filelock` | wrong env | `PY=~/.conda/envs/trace/bin/python` |
| All Exp2 losses identical (NaN/Inf) | divergence at extreme `eta x lambda` | check `diverged` column in CSV; you can ignore those rows when plotting |
| Spoons have a flat right side | `eta` grid doesn't reach instability | extend `--lrs` upward (e.g. `1e-2,...,3.0`) |
| Spoons have no clear descent | LR grid too low | extend `--lrs` downward |
| GPU 0 OOM | another job is using GPU 0 | use `--gpus 1,2,3` (default) |
