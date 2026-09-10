E12 full ladder: cosine LR, B=128, eta0=0.1, T=100, seed 42.
Budget in setting-local C units (C = fixed-WD oracle lambda_ref of
that setting/phase). R18/C100 rows come from the E11 CSV (read-only);
R50/C100 rows appear as they land.

| setting | phase | wd_sched | lambda0 | budget_C | best_acc | diverged | src |
|---|---|---|---:|---:|---:|---:|---|
| mlp/cifar10 | SGD | fixed | 0.001 | 0.10 | 57.55 | 0 | e12 |
| mlp/cifar10 | SGD | fixed | 0.002 | 0.20 | 57.78 | 0 | e12 |
| mlp/cifar10 | SGD | fixed | 0.003 | 0.30 | 58.00 | 0 | e12 |
| mlp/cifar10 | SGD | fixed | 0.005 | 0.50 | 58.13 | 0 | e12 |
| mlp/cifar10 | SGD | fixed | 0.0075 | 0.75 | 57.91 | 0 | e12 |
| mlp/cifar10 | SGD | fixed | 0.01 | 1.00 | 58.15 | 0 | e12 |
| mlp/cifar10 | SGD | fixed | 0.02 | 2.00 | 55.59 | 0 | e12 |
| mlp/cifar10 | SGD | iso up | 0.005809 | 1.00 | 55.27 | 0 | e12 |
| mlp/cifar10 | SGD | iso up | 0.01162 | 2.00 | 52.06 | 0 | e12 |
| mlp/cifar10 | SGD | iso up | 0.01743 | 3.00 | 50.02 | 0 | e12 |
| mlp/cifar10 | SGD | iso up | 0.02324 | 4.00 | 48.96 | 0 | e12 |
| mlp/cifar10 | SGD | iso up | 0.03485 | 6.00 | 46.16 | 0 | e12 |
| mlp/cifar10 | SGD | iso up | 0.05228 | 9.00 | 42.97 | 0 | e12 |
| mlp/cifar10 | SGD | iso up | 0.08713 | 15.00 | 36.98 | 0 | e12 |
| mlp/cifar10 | SGD | linear down | 0.01417 | 1.00 | 56.84 | 0 | e12 |
| mlp/cifar10 | SGD | linear down | 0.02834 | 2.00 | 56.78 | 0 | e12 |
| mlp/cifar10 | SGD | linear down | 0.04252 | 3.00 | 56.29 | 0 | e12 |
| mlp/cifar10 | SGD | linear down | 0.05669 | 4.00 | 55.03 | 0 | e12 |
| mlp/cifar10 | SGD | linear down | 0.08503 | 6.00 | 53.81 | 0 | e12 |
| mlp/cifar10 | SGD | linear down | 0.1276 | 9.00 | 50.89 | 0 | e12 |
| mlp/cifar10 | SGD | linear down | 0.2126 | 15.00 | 44.98 | 0 | e12 |
| mlp/cifar10 | SGD | linear up | 0.03397 | 1.00 | 54.86 | 0 | e12 |
| mlp/cifar10 | SGD | linear up | 0.06794 | 2.00 | 53.49 | 0 | e12 |
| mlp/cifar10 | SGD | linear up | 0.1019 | 3.00 | 52.66 | 0 | e12 |
| mlp/cifar10 | SGD | linear up | 0.1359 | 4.00 | 52.06 | 0 | e12 |
| mlp/cifar10 | SGD | linear up | 0.2038 | 6.00 | 51.80 | 0 | e12 |
| mlp/cifar10 | SGD | linear up | 0.3057 | 9.00 | 51.14 | 0 | e12 |
| mlp/cifar10 | SGD | linear up | 0.5095 | 15.00 | 50.25 | 0 | e12 |
| mlp/cifar10 | SGDM | fixed | 0.0001 | 0.10 | 51.90 | 0 | e12 |
| mlp/cifar10 | SGDM | fixed | 0.0003 | 0.30 | 55.33 | 0 | e12 |
| mlp/cifar10 | SGDM | fixed | 0.0006 | 0.60 | 56.02 | 0 | e12 |
| mlp/cifar10 | SGDM | fixed | 0.001 | 1.00 | 56.86 | 0 | e12 |
| mlp/cifar10 | SGDM | fixed | 0.002 | 2.00 | 56.42 | 0 | e12 |
| mlp/cifar10 | SGDM | fixed | 0.003 | 3.00 | 56.15 | 0 | e12 |
| mlp/cifar10 | SGDM | fixed | 0.005 | 5.00 | 52.20 | 0 | e12 |
| mlp/cifar10 | SGDM | iso up | 0.0001936 | 0.33 | 57.01 | 0 | e12 |
| mlp/cifar10 | SGDM | iso up | 0.0005809 | 1.00 | 58.67 | 0 | e12 |
| mlp/cifar10 | SGDM | iso up | 0.0008713 | 1.50 | 58.46 | 0 | e12 |
| mlp/cifar10 | SGDM | iso up | 0.001162 | 2.00 | 57.57 | 0 | e12 |
| mlp/cifar10 | SGDM | iso up | 0.001452 | 2.50 | 56.42 | 0 | e12 |
| mlp/cifar10 | SGDM | iso up | 0.001743 | 3.00 | 55.75 | 0 | e12 |
| mlp/cifar10 | SGDM | iso up | 0.002324 | 4.00 | 53.71 | 0 | e12 |
| mlp/cifar10 | SGDM | linear down | 0.0004724 | 0.33 | 54.71 | 0 | e12 |
| mlp/cifar10 | SGDM | linear down | 0.001417 | 1.00 | 55.07 | 0 | e12 |
| mlp/cifar10 | SGDM | linear down | 0.002126 | 1.50 | 55.71 | 0 | e12 |
| mlp/cifar10 | SGDM | linear down | 0.002834 | 2.00 | 54.55 | 0 | e12 |
| mlp/cifar10 | SGDM | linear down | 0.003543 | 2.50 | 55.18 | 0 | e12 |
| mlp/cifar10 | SGDM | linear down | 0.004252 | 3.00 | 53.96 | 0 | e12 |
| mlp/cifar10 | SGDM | linear down | 0.005669 | 4.00 | 50.42 | 0 | e12 |
| mlp/cifar10 | SGDM | linear up | 0.001132 | 0.33 | 57.31 | 0 | e12 |
| mlp/cifar10 | SGDM | linear up | 0.003397 | 1.00 | 58.66 | 0 | e12 |
| mlp/cifar10 | SGDM | linear up | 0.005095 | 1.50 | 58.80 | 0 | e12 |
| mlp/cifar10 | SGDM | linear up | 0.006794 | 2.00 | 57.99 | 0 | e12 |
| mlp/cifar10 | SGDM | linear up | 0.008492 | 2.50 | 58.01 | 0 | e12 |
| mlp/cifar10 | SGDM | linear up | 0.01019 | 3.00 | 57.01 | 0 | e12 |
| mlp/cifar10 | SGDM | linear up | 0.01359 | 4.00 | 56.54 | 0 | e12 |
| mlp/mnist | SGD | fixed | 0.001 | 0.50 | 98.38 | 0 | e12 |
| mlp/mnist | SGD | fixed | 0.002 | 1.00 | 98.38 | 0 | e12 |
| mlp/mnist | SGD | fixed | 0.003 | 1.50 | 98.25 | 0 | e12 |
| mlp/mnist | SGD | fixed | 0.005 | 2.50 | 98.05 | 0 | e12 |
| mlp/mnist | SGD | fixed | 0.0075 | 3.75 | 97.73 | 0 | e12 |
| mlp/mnist | SGD | fixed | 0.01 | 5.00 | 97.38 | 0 | e12 |
| mlp/mnist | SGD | fixed | 0.02 | 10.00 | 95.82 | 0 | e12 |
| mlp/mnist | SGD | iso up | 0.001162 | 1.00 | 98.32 | 0 | e12 |
| mlp/mnist | SGD | iso up | 0.002324 | 2.00 | 98.18 | 0 | e12 |
| mlp/mnist | SGD | iso up | 0.003485 | 3.00 | 97.97 | 0 | e12 |
| mlp/mnist | SGD | iso up | 0.004647 | 4.00 | 97.78 | 0 | e12 |
| mlp/mnist | SGD | iso up | 0.006971 | 6.00 | 97.45 | 0 | e12 |
| mlp/mnist | SGD | iso up | 0.01046 | 9.00 | 96.93 | 0 | e12 |
| mlp/mnist | SGD | iso up | 0.01743 | 15.00 | 95.68 | 0 | e12 |
| mlp/mnist | SGD | linear down | 0.002834 | 1.00 | 98.47 | 0 | e12 |
| mlp/mnist | SGD | linear down | 0.005669 | 2.00 | 98.36 | 0 | e12 |
| mlp/mnist | SGD | linear down | 0.008503 | 3.00 | 98.29 | 0 | e12 |
| mlp/mnist | SGD | linear down | 0.01134 | 4.00 | 98.19 | 0 | e12 |
| mlp/mnist | SGD | linear down | 0.01701 | 6.00 | 97.97 | 0 | e12 |
| mlp/mnist | SGD | linear down | 0.02551 | 9.00 | 97.69 | 0 | e12 |
| mlp/mnist | SGD | linear down | 0.04252 | 15.00 | 96.87 | 0 | e12 |
| mlp/mnist | SGD | linear up | 0.006794 | 1.00 | 98.34 | 0 | e12 |
| mlp/mnist | SGD | linear up | 0.01359 | 2.00 | 98.17 | 0 | e12 |
| mlp/mnist | SGD | linear up | 0.02038 | 3.00 | 98.12 | 0 | e12 |
| mlp/mnist | SGD | linear up | 0.02717 | 4.00 | 98.05 | 0 | e12 |
| mlp/mnist | SGD | linear up | 0.04076 | 6.00 | 97.88 | 0 | e12 |
| mlp/mnist | SGD | linear up | 0.06114 | 9.00 | 97.67 | 0 | e12 |
| mlp/mnist | SGD | linear up | 0.1019 | 15.00 | 97.40 | 0 | e12 |
| mlp/mnist | SGDM | fixed | 0.0001 | 0.33 | 98.72 | 0 | e12 |
| mlp/mnist | SGDM | fixed | 0.0003 | 1.00 | 98.76 | 0 | e12 |
| mlp/mnist | SGDM | fixed | 0.0006 | 2.00 | 98.70 | 0 | e12 |
| mlp/mnist | SGDM | fixed | 0.001 | 3.33 | 98.62 | 0 | e12 |
| mlp/mnist | SGDM | fixed | 0.002 | 6.67 | 98.58 | 0 | e12 |
| mlp/mnist | SGDM | fixed | 0.003 | 10.00 | 98.44 | 0 | e12 |
| mlp/mnist | SGDM | fixed | 0.005 | 16.67 | 98.02 | 0 | e12 |
| mlp/mnist | SGDM | iso up | 5.809e-05 | 0.33 | 98.67 | 0 | e12 |
| mlp/mnist | SGDM | iso up | 0.0001743 | 1.00 | 98.66 | 0 | e12 |
| mlp/mnist | SGDM | iso up | 0.0002614 | 1.50 | 98.60 | 0 | e12 |
| mlp/mnist | SGDM | iso up | 0.0003485 | 2.00 | 98.58 | 0 | e12 |
| mlp/mnist | SGDM | iso up | 0.0004357 | 2.50 | 98.39 | 0 | e12 |
| mlp/mnist | SGDM | iso up | 0.0005228 | 3.00 | 98.41 | 0 | e12 |
| mlp/mnist | SGDM | iso up | 0.0006971 | 4.00 | 98.25 | 0 | e12 |
| mlp/mnist | SGDM | linear down | 0.0001417 | 0.33 | 98.74 | 0 | e12 |
| mlp/mnist | SGDM | linear down | 0.0004252 | 1.00 | 98.77 | 0 | e12 |
| mlp/mnist | SGDM | linear down | 0.0006378 | 1.50 | 98.76 | 0 | e12 |
| mlp/mnist | SGDM | linear down | 0.0008503 | 2.00 | 98.71 | 0 | e12 |
| mlp/mnist | SGDM | linear down | 0.001063 | 2.50 | 98.71 | 0 | e12 |
| mlp/mnist | SGDM | linear down | 0.001276 | 3.00 | 98.68 | 0 | e12 |
| mlp/mnist | SGDM | linear down | 0.001701 | 4.00 | 98.76 | 0 | e12 |
| mlp/mnist | SGDM | linear up | 0.0003397 | 0.33 | 98.68 | 0 | e12 |
| mlp/mnist | SGDM | linear up | 0.001019 | 1.00 | 98.73 | 0 | e12 |
| mlp/mnist | SGDM | linear up | 0.001529 | 1.50 | 98.60 | 0 | e12 |
| mlp/mnist | SGDM | linear up | 0.002038 | 2.00 | 98.57 | 0 | e12 |
| mlp/mnist | SGDM | linear up | 0.002548 | 2.50 | 98.57 | 0 | e12 |
| mlp/mnist | SGDM | linear up | 0.003057 | 3.00 | 98.53 | 0 | e12 |
| mlp/mnist | SGDM | linear up | 0.004076 | 4.00 | 98.36 | 0 | e12 |
| resnet18/cifar100 | SGD | fixed | 0.0001 | 0.02 | 73.31 | 0 | e11 |
| resnet18/cifar100 | SGD | fixed | 0.0003 | 0.06 | 74.46 | 0 | e11 |
| resnet18/cifar100 | SGD | fixed | 0.0005 | 0.10 | 75.04 | 0 | e11 |
| resnet18/cifar100 | SGD | fixed | 0.0005982 | 0.12 | 75.33 | 0 | e11 |
| resnet18/cifar100 | SGD | fixed | 0.001 | 0.20 | 75.87 | 0 | e11 |
| resnet18/cifar100 | SGD | fixed | 0.002 | 0.40 | 77.47 | 0 | e11 |
| resnet18/cifar100 | SGD | fixed | 0.003 | 0.60 | 77.64 | 0 | e11 |
| resnet18/cifar100 | SGD | fixed | 0.005 | 1.00 | 77.75 | 0 | e12 |
| resnet18/cifar100 | SGD | fixed | 0.005 | 1.00 | 77.75 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.0001158 | 0.04 | 74.18 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.0003475 | 0.12 | 75.16 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.0005212 | 0.18 | 75.73 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.000695 | 0.24 | 75.78 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.0008687 | 0.30 | 76.14 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.001042 | 0.36 | 76.69 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.00139 | 0.48 | 77.31 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.002085 | 0.72 | 77.68 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.003127 | 1.08 | 77.83 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.005212 | 1.79 | 78.01 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.008687 | 2.99 | 57.74 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.0139 | 4.79 | 39.63 | 0 | e11 |
| resnet18/cifar100 | SGD | iso up | 0.02085 | 7.18 | 31.43 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.0001 | 0.01 | 73.46 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.0005 | 0.03 | 74.00 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.0006773 | 0.04 | 74.09 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.001 | 0.06 | 74.36 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.002 | 0.12 | 75.32 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.002032 | 0.12 | 75.32 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.003048 | 0.18 | 75.69 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.004064 | 0.24 | 75.94 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.005 | 0.29 | 76.46 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.00508 | 0.30 | 76.22 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.006096 | 0.36 | 76.98 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.008128 | 0.48 | 77.53 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.01 | 0.59 | 77.60 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.01219 | 0.72 | 77.91 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.01829 | 1.08 | 77.83 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.02 | 1.18 | 77.94 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.03048 | 1.79 | 78.08 | 0 | e12 |
| resnet18/cifar100 | SGD | linear up | 0.03048 | 1.79 | 78.08 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.05 | 2.94 | 76.32 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.0508 | 2.99 | 76.59 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.08128 | 4.79 | 59.50 | 0 | e11 |
| resnet18/cifar100 | SGD | linear up | 0.1219 | 7.18 | 40.49 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 5.982e-05 | 0.10 | 73.01 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 9.727e-05 | 0.16 | 72.42 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0001 | 0.17 | 74.83 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0001994 | 0.33 | 74.62 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0002 | 0.33 | 75.56 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0003 | 0.50 | 76.46 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0003242 | 0.54 | 76.17 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0004 | 0.67 | 75.76 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0005 | 0.83 | 76.73 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0005982 | 1.00 | 76.72 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0006 | 1.00 | 77.45 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.00075 | 1.25 | 76.68 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0008 | 1.33 | 77.41 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.001 | 1.67 | 77.28 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0012 | 2.00 | 77.13 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0015 | 2.50 | 77.37 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.001795 | 2.99 | 76.19 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.002 | 3.33 | 76.06 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.00225 | 3.75 | 75.50 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.0025 | 4.17 | 74.94 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.002918 | 4.86 | 74.24 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.003 | 5.00 | 73.57 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.005 | 8.33 | 68.28 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.005982 | 9.97 | 63.91 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.009727 | 16.21 | 44.97 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.01 | 16.67 | 42.07 | 0 | e11 |
| resnet18/cifar100 | SGDM | fixed | 0.02 | 33.33 | 3.80 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.0001 | 0.29 | 75.56 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.0001158 | 0.33 | 75.86 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.0003475 | 1.00 | 78.22 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.0005 | 1.43 | 77.98 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.0005212 | 1.50 | 77.93 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.000695 | 1.99 | 78.19 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.0008687 | 2.49 | 77.62 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.001 | 2.87 | 77.98 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.001042 | 2.99 | 77.50 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.00139 | 3.99 | 75.42 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.002 | 5.74 | 70.36 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.002085 | 5.98 | 70.04 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.003127 | 8.97 | 57.58 | 0 | e11 |
| resnet18/cifar100 | SGDM | iso up | 0.005 | 14.35 | 22.13 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear down | 0.0002826 | 0.33 | 74.43 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear down | 0.0008478 | 1.00 | 76.44 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear down | 0.002543 | 2.99 | 75.53 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.0001 | 0.05 | 72.07 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.0005 | 0.25 | 74.92 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.0006773 | 0.33 | 75.84 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.001 | 0.49 | 76.50 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.002 | 0.98 | 77.98 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.002032 | 1.00 | 77.81 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.003048 | 1.50 | 78.44 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.004064 | 1.99 | 78.45 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.005 | 2.45 | 78.28 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.00508 | 2.49 | 77.77 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.006096 | 2.99 | 77.44 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.008128 | 3.99 | 76.18 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.01 | 4.91 | 74.13 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.01219 | 5.98 | 72.53 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.01829 | 8.97 | 64.49 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.02 | 9.81 | 61.38 | 0 | e11 |
| resnet18/cifar100 | SGDM | linear up | 0.05 | 24.53 | 32.71 | 0 | e11 |
| resnet50/cifar100 | SGD | fixed | 0.001 | 0.20 | 77.63 | 0 | e12 |
| resnet50/cifar100 | SGD | fixed | 0.002 | 0.40 | 78.64 | 0 | e12 |
| resnet50/cifar100 | SGD | fixed | 0.003 | 0.60 | 78.94 | 0 | e12 |
| resnet50/cifar100 | SGD | fixed | 0.005 | 1.00 | 79.21 | 0 | e12 |
| resnet50/cifar100 | SGD | fixed | 0.0075 | 1.50 | 78.98 | 0 | e12 |
| resnet50/cifar100 | SGD | fixed | 0.01 | 2.00 | 78.65 | 0 | e12 |
| resnet50/cifar100 | SGD | fixed | 0.02 | 4.00 | 73.40 | 0 | e12 |
| resnet50/cifar100 | SGDM | fixed | 0.0001 | 0.10 | 72.66 | 0 | e12 |
| resnet50/cifar100 | SGDM | fixed | 0.000256 | 0.27 | 76.19 | 0 | e11 |
| resnet50/cifar100 | SGDM | fixed | 0.0003 | 0.31 | 74.94 | 0 | e12 |
| resnet50/cifar100 | SGDM | fixed | 0.000598 | 0.62 | 76.86 | 0 | e11 |
| resnet50/cifar100 | SGDM | fixed | 0.0006 | 0.62 | 76.07 | 0 | e12 |
| resnet50/cifar100 | SGDM | fixed | 0.000962 | 1.00 | 78.20 | 0 | e11 |
| resnet50/cifar100 | SGDM | fixed | 0.001 | 1.04 | 75.66 | 0 | e12 |
| resnet50/cifar100 | SGDM | fixed | 0.002 | 2.08 | 72.34 | 0 | e12 |
| resnet50/cifar100 | SGDM | fixed | 0.003 | 3.12 | 64.52 | 0 | e12 |
| resnet50/cifar100 | SGDM | fixed | 0.005 | 5.20 | 55.48 | 0 | e12 |
| vgg16/cifar100 | SGD | fixed | 0.001 | 0.10 | 72.35 | 0 | e12 |
| vgg16/cifar100 | SGD | fixed | 0.002 | 0.20 | 73.67 | 0 | e12 |
| vgg16/cifar100 | SGD | fixed | 0.003 | 0.30 | 74.20 | 0 | e12 |
| vgg16/cifar100 | SGD | fixed | 0.005 | 0.50 | 74.83 | 0 | e12 |
| vgg16/cifar100 | SGD | fixed | 0.0075 | 0.75 | 74.95 | 0 | e12 |
| vgg16/cifar100 | SGD | fixed | 0.01 | 1.00 | 75.16 | 0 | e12 |
| vgg16/cifar100 | SGD | fixed | 0.02 | 2.00 | 74.10 | 0 | e12 |
| vgg16/cifar100 | SGD | iso up | 0.005809 | 1.00 | 75.89 | 0 | e12 |
| vgg16/cifar100 | SGD | iso up | 0.01162 | 2.00 | 47.34 | 0 | e12 |
| vgg16/cifar100 | SGD | iso up | 0.01743 | 3.00 | 31.69 | 0 | e12 |
| vgg16/cifar100 | SGD | iso up | 0.02324 | 4.00 | 22.46 | 0 | e12 |
| vgg16/cifar100 | SGD | iso up | 0.03485 | 6.00 | 11.38 | 0 | e12 |
| vgg16/cifar100 | SGD | iso up | 0.05228 | 9.00 | 4.31 | 0 | e12 |
| vgg16/cifar100 | SGD | iso up | 0.08713 | 15.00 | 3.83 | 0 | e12 |
| vgg16/cifar100 | SGD | linear down | 0.01417 | 1.00 | 74.29 | 0 | e12 |
| vgg16/cifar100 | SGD | linear down | 0.02834 | 2.00 | 73.44 | 0 | e12 |
| vgg16/cifar100 | SGD | linear down | 0.04252 | 3.00 | 66.26 | 0 | e12 |
| vgg16/cifar100 | SGD | linear down | 0.05669 | 4.00 | 4.21 | 0 | e12 |
| vgg16/cifar100 | SGD | linear down | 0.08503 | 6.00 | 4.04 | 0 | e12 |
| vgg16/cifar100 | SGD | linear down | 0.1276 | 9.00 | 2.52 | 0 | e12 |
| vgg16/cifar100 | SGD | linear down | 0.2126 | 15.00 | 1.00 | 0 | e12 |
| vgg16/cifar100 | SGD | linear up | 0.03397 | 1.00 | 75.43 | 0 | e12 |
| vgg16/cifar100 | SGD | linear up | 0.06794 | 2.00 | 70.48 | 0 | e12 |
| vgg16/cifar100 | SGD | linear up | 0.1019 | 3.00 | 45.21 | 0 | e12 |
| vgg16/cifar100 | SGD | linear up | 0.1359 | 4.00 | 35.91 | 0 | e12 |
| vgg16/cifar100 | SGD | linear up | 0.2038 | 6.00 | 28.99 | 0 | e12 |
| vgg16/cifar100 | SGD | linear up | 0.3057 | 9.00 | 25.53 | 0 | e12 |
| vgg16/cifar100 | SGD | linear up | 0.5095 | 15.00 | 18.47 | 0 | e12 |
| vgg16/cifar100 | SGDM | fixed | 0.0001 | 0.10 | 69.29 | 0 | e12 |
| vgg16/cifar100 | SGDM | fixed | 0.000256 | 0.27 | 72.05 | 0 | e11 |
| vgg16/cifar100 | SGDM | fixed | 0.0003 | 0.31 | 72.27 | 0 | e12 |
| vgg16/cifar100 | SGDM | fixed | 0.000598 | 0.62 | 72.67 | 0 | e11 |
| vgg16/cifar100 | SGDM | fixed | 0.0006 | 0.62 | 72.95 | 0 | e12 |
| vgg16/cifar100 | SGDM | fixed | 0.000962 | 1.00 | 73.43 | 0 | e11 |
| vgg16/cifar100 | SGDM | fixed | 0.001 | 1.04 | 73.02 | 0 | e12 |
| vgg16/cifar100 | SGDM | fixed | 0.002 | 2.08 | 72.39 | 0 | e12 |
| vgg16/cifar100 | SGDM | fixed | 0.003 | 3.12 | 71.16 | 0 | e12 |
| vgg16/cifar100 | SGDM | fixed | 0.005 | 5.20 | 67.50 | 0 | e12 |
| vgg16/cifar100 | SGDM | iso up | 0.0001936 | 0.35 | 72.86 | 0 | e12 |
| vgg16/cifar100 | SGDM | iso up | 0.0005809 | 1.04 | 74.24 | 0 | e12 |
| vgg16/cifar100 | SGDM | iso up | 0.0008713 | 1.56 | 73.68 | 0 | e12 |
| vgg16/cifar100 | SGDM | iso up | 0.001162 | 2.08 | 72.35 | 0 | e12 |
| vgg16/cifar100 | SGDM | iso up | 0.001452 | 2.60 | 71.86 | 0 | e12 |
| vgg16/cifar100 | SGDM | iso up | 0.001743 | 3.12 | 69.77 | 0 | e12 |
| vgg16/cifar100 | SGDM | iso up | 0.002324 | 4.16 | 66.57 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear down | 0.0004724 | 0.35 | 72.24 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear down | 0.001417 | 1.04 | 72.46 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear down | 0.002126 | 1.56 | 72.16 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear down | 0.002834 | 2.08 | 71.57 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear down | 0.003543 | 2.60 | 71.40 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear down | 0.004252 | 3.12 | 70.06 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear down | 0.005669 | 4.16 | 68.02 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear up | 0.001132 | 0.35 | 73.07 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear up | 0.003397 | 1.04 | 73.84 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear up | 0.005095 | 1.56 | 73.11 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear up | 0.006794 | 2.08 | 72.47 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear up | 0.008492 | 2.60 | 71.79 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear up | 0.01019 | 3.12 | 70.72 | 0 | e12 |
| vgg16/cifar100 | SGDM | linear up | 0.01359 | 4.16 | 67.79 | 0 | e12 |
