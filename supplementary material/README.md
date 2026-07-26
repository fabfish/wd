Anonymized Supplementary Material
=================================

This folder contains anonymized code for reproducing the main CIFAR-100 vision experiments in the paper.
The code is intentionally compact and self-contained for review.

Environment
-----------

```bash
pip install -r requirements.txt
```

Single Run
----------

```bash
python train_cifar100.py --model resnet18 --batch_size 128 --lr 0.1 --wd 5e-4 --momentum 0.9 --epochs 100 --seed 42
```

Grid Experiments
----------------

```bash
# Exp. 1: optimizer comparison and optimal learning-rate ordering
python run_grid_experiments.py --experiment 1 --epochs 100 --gpus all

# Exp. 2: eta-lambda interaction at fixed batch size
python run_grid_experiments.py --experiment 2 --epochs 100 --gpus all

# Exp. 3: batch-size scaling under the linear learning-rate rule
python run_grid_experiments.py --experiment 3 --epochs 100 --gpus all
```

Results are appended to `outputs/results/results.csv`.
CIFAR-100 is downloaded automatically by torchvision into `data/`.

Plotting
--------

```bash
python plot_results.py --input outputs/results/results.csv --output_dir outputs/plots
```

Notes
-----

- The released code contains no author or institution information.
- The vision experiments use CIFAR-100, standard augmentation, ResNet/VGG models adapted for 32x32 images, SGD/SGDM with weight decay, cosine learning-rate scheduling, and AMP by default.
- Exact numbers can vary slightly by GPU, PyTorch/CUDA version, and nondeterministic kernels.
