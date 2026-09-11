E12 multi-seed: peak configs replicated across seeds.

| setting                | wd_sched    |   lambda0 |   mean_acc |   std_acc |   n_seeds | seeds       |
|:-----------------------|:------------|----------:|-----------:|----------:|----------:|:------------|
| mlp/cifar10/SGD        | fixed       | 0.01      |    58.03   | 0.14      |         2 | 123,2024    |
| mlp/cifar10/SGD        | linear_up   | 0.03397   |    55.275  | 0.185     |         2 | 123,2024    |
| mlp/cifar10/SGDM       | fixed       | 0.001     |    56.355  | 0.115     |         2 | 123,2024    |
| mlp/cifar10/SGDM       | linear_up   | 0.005095  |    58.095  | 0.075     |         2 | 123,2024    |
| mlp/mnist/SGD          | fixed       | 0.002     |    98.41   | 0.02      |         2 | 123,2024    |
| mlp/mnist/SGD          | linear      | 0.002834  |    98.46   | 0.01      |         2 | 123,2024    |
| mlp/mnist/SGDM         | fixed       | 0.0003    |    98.785  | 0.005     |         2 | 123,2024    |
| mlp/mnist/SGDM         | linear      | 0.0004252 |    98.805  | 0.015     |         2 | 123,2024    |
| resnet18/cifar100/SGD  | fixed       | 0.005     |    77.6533 | 0.158395  |         3 | 42,123,2024 |
| resnet18/cifar100/SGD  | linear      | 0.01063   |    77.545  | 0.225     |         2 | 123,2024    |
| resnet18/cifar100/SGD  | linear_up   | 0.03048   |    78.1733 | 0.0659966 |         3 | 42,123,2024 |
| resnet18/cifar100/SGDM | cosine_up   | 0.005     |    77.845  | 0.135     |         2 | 123,2024    |
| resnet18/cifar100/SGDM | fixed       | 0.0006    |    77.455  | 0.385     |         2 | 123,2024    |
| resnet18/cifar100/SGDM | linear_up   | 0.005     |    77.935  | 0.025     |         2 | 123,2024    |
| resnet18/cifar100/SGDM | step_up     | 0.01      |    77.845  | 0.125     |         2 | 123,2024    |
| resnet50/cifar100/SGD  | fixed       | 0.005     |    79.26   | 0.15      |         2 | 123,2024    |
| resnet50/cifar100/SGD  | linear_up   | 0.01698   |    79.675  | 0.065     |         2 | 123,2024    |
| resnet50/cifar100/SGDM | fixed       | 0.000962  |    76.92   | 0.845813  |         3 | 42,123,2024 |
| resnet50/cifar100/SGDM | linear_up   | 0.003057  |    78.095  | 0.205     |         2 | 123,2024    |
| vgg16/cifar100/SGD     | fixed       | 0.01      |    74.885  | 0.215     |         2 | 123,2024    |
| vgg16/cifar100/SGD     | iso_product | 0.005809  |    76.105  | 0.325     |         2 | 123,2024    |
| vgg16/cifar100/SGDM    | fixed       | 0.001     |    73.125  | 0.025     |         2 | 123,2024    |
| vgg16/cifar100/SGDM    | iso_product | 0.0005809 |    73.61   | 0.23      |         2 | 123,2024    |
