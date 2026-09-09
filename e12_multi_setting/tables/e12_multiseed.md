E12 multi-seed: peak configs replicated across seeds.

| setting                | wd_sched   |   lambda0 |   mean_acc |   std_acc |   n_seeds | seeds    |
|:-----------------------|:-----------|----------:|-----------:|----------:|----------:|:---------|
| resnet18/cifar100/SGDM | cosine_up  |    0.005  |     77.845 |     0.135 |         2 | 123,2024 |
| resnet18/cifar100/SGDM | fixed      |    0.0006 |     77.455 |     0.385 |         2 | 123,2024 |
| resnet18/cifar100/SGDM | linear_up  |    0.005  |     77.935 |     0.025 |         2 | 123,2024 |
| resnet18/cifar100/SGDM | step_up    |    0.01   |     77.845 |     0.125 |         2 | 123,2024 |
