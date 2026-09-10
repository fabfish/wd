E12 per-setting peaks: cosine LR, B=128, eta0=0.1, T=100.
Budget in setting-local C units (lambda_ref = fixed oracle of
that setting/phase, seed 42). n_runs counts seed-42 rows.

| setting | phase | wd_sched | peak_acc | peak_budget_C | peak_lambda0 | n_runs |
|---|---|---|---:|---:|---:|---:|
| mlp/cifar10/SGDM | SGDM | fixed | 56.86 | 1.00C | 0.001 | 9 |
| mlp/cifar10/SGDM | SGDM | linear_up | 58.80 | 1.50C | 0.005095 | 9 |
| mlp/cifar10/SGDM | SGDM | linear | 55.71 | 1.50C | 0.002126 | 7 |
| mlp/cifar10/SGDM | SGDM | iso_product | 58.67 | 1.00C | 0.0005809 | 7 |
| mlp/cifar10/SGD | SGD | fixed | 58.15 | 1.00C | 0.01 | 9 |
| mlp/cifar10/SGD | SGD | linear_up | 54.86 | 1.00C | 0.03397 | 11 |
| mlp/cifar10/SGD | SGD | linear | 56.84 | 1.00C | 0.01417 | 9 |
| mlp/cifar10/SGD | SGD | iso_product | 55.27 | 1.00C | 0.005809 | 9 |
| mlp/mnist/SGDM | SGDM | fixed | 98.76 | 1.00C | 0.0003 | 9 |
| mlp/mnist/SGDM | SGDM | linear_up | 98.73 | 1.00C | 0.001019 | 7 |
| mlp/mnist/SGDM | SGDM | linear | 98.77 | 1.00C | 0.0004252 | 9 |
| mlp/mnist/SGDM | SGDM | iso_product | 98.67 | 0.33C | 5.809e-05 | 7 |
| mlp/mnist/SGD | SGD | fixed | 98.38 | 1.00C | 0.002 | 9 |
| mlp/mnist/SGD | SGD | linear_up | 98.34 | 1.00C | 0.006794 | 7 |
| mlp/mnist/SGD | SGD | linear | 98.47 | 1.00C | 0.002834 | 9 |
| mlp/mnist/SGD | SGD | iso_product | 98.32 | 1.00C | 0.001162 | 7 |
| resnet50/cifar100/SGDM | SGDM | fixed | 78.20 | 1.00C | 0.000962 | 13 |
| resnet50/cifar100/SGDM | SGDM | linear_up | 78.72 | 0.94C | 0.003057 | 7 |
| resnet50/cifar100/SGDM | SGDM | linear | 77.01 | 0.62C | 0.0008503 | 7 |
| resnet50/cifar100/SGDM | SGDM | iso_product | 78.51 | 0.62C | 0.0003485 | 7 |
| resnet50/cifar100/SGD | SGD | fixed | 79.21 | 1.00C | 0.005 | 7 |
| resnet50/cifar100/SGD | SGD | linear_up | 79.55 | 1.00C | 0.01698 | 7 |
| resnet50/cifar100/SGD | SGD | linear | 79.25 | 1.00C | 0.007086 | 7 |
| resnet50/cifar100/SGD | SGD | iso_product | 79.31 | 1.00C | 0.002904 | 7 |
| vgg16/cifar100/SGDM | SGDM | fixed | 73.43 | 1.00C | 0.000962 | 12 |
| vgg16/cifar100/SGDM | SGDM | linear_up | 73.84 | 1.04C | 0.003397 | 8 |
| vgg16/cifar100/SGDM | SGDM | linear | 72.46 | 1.04C | 0.001417 | 7 |
| vgg16/cifar100/SGDM | SGDM | iso_product | 74.24 | 1.04C | 0.0005809 | 10 |
| vgg16/cifar100/SGD | SGD | fixed | 75.16 | 1.00C | 0.01 | 9 |
| vgg16/cifar100/SGD | SGD | linear_up | 75.43 | 1.00C | 0.03397 | 9 |
| vgg16/cifar100/SGD | SGD | linear | 74.29 | 1.00C | 0.01417 | 7 |
| vgg16/cifar100/SGD | SGD | iso_product | 75.89 | 1.00C | 0.005809 | 11 |
| resnet18/cifar100/SGDM | SGDM | fixed | 77.45 | 1.00C | 0.0006 | 34 |
| resnet18/cifar100/SGDM | SGDM | linear_up | 78.45 | 1.99C | 0.004064 | 19 |
| resnet18/cifar100/SGDM | SGDM | linear | 76.44 | 1.00C | 0.0008478 | 3 |
| resnet18/cifar100/SGDM | SGDM | iso_product | 78.22 | 1.00C | 0.0003475 | 14 |
| resnet18/cifar100/SGD | SGD | fixed | 77.75 | 1.00C | 0.005 | 8 |
| resnet18/cifar100/SGD | SGD | linear_up | 78.08 | 1.79C | 0.03048 | 21 |
| resnet18/cifar100/SGD | SGD | iso_product | 78.01 | 1.79C | 0.005212 | 13 |
