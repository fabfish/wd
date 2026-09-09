E12 per-setting peaks: cosine LR, B=128, eta0=0.1, T=100.
Budget in setting-local C units (lambda_ref = fixed oracle of
that setting/phase, seed 42). n_runs counts seed-42 rows.

| setting | phase | wd_sched | peak_acc | peak_budget_C | peak_lambda0 | n_runs |
|---|---|---|---:|---:|---:|---:|
| mlp/cifar10/SGDM | SGDM | fixed | 56.86 | 1.00C | 0.001 | 7 |
| mlp/cifar10/SGDM | SGDM | linear_up | 58.80 | 1.50C | 0.005095 | 7 |
| mlp/cifar10/SGDM | SGDM | linear | 55.71 | 1.50C | 0.002126 | 7 |
| mlp/cifar10/SGDM | SGDM | iso_product | 58.67 | 1.00C | 0.0005809 | 7 |
| mlp/cifar10/SGD | SGD | fixed | 58.15 | 1.00C | 0.01 | 7 |
| mlp/cifar10/SGD | SGD | linear_up | 54.86 | 1.00C | 0.03397 | 7 |
| mlp/cifar10/SGD | SGD | linear | 56.84 | 1.00C | 0.01417 | 7 |
| mlp/cifar10/SGD | SGD | iso_product | 55.27 | 1.00C | 0.005809 | 7 |
| resnet18/cifar100/SGDM | SGDM | fixed | 76.72 | 1.00C | 0.0005982 | 3 |
| resnet18/cifar100/SGDM | SGDM | linear_up | 78.45 | 1.99C | 0.004064 | 19 |
| resnet18/cifar100/SGDM | SGDM | linear | 76.44 | 1.00C | 0.0008478 | 3 |
| resnet18/cifar100/SGDM | SGDM | iso_product | 78.22 | 1.00C | 0.0003475 | 14 |
| resnet18/cifar100/SGD | SGD | fixed | 77.75 | 1.00C | 0.005 | 5 |
| resnet18/cifar100/SGD | SGD | linear_up | 78.08 | 1.79C | 0.03048 | 21 |
| resnet18/cifar100/SGD | SGD | iso_product | 78.01 | 1.79C | 0.005212 | 13 |
