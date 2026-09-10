E12 cross-setting summary: optimal budget C* per shape
(columns "wd_sched_C" are peak budgets in local C units;
"wd_sched_acc" the corresponding accuracy).

|                                    |   fixed_C |   iso_product_C |   linear_C |   linear_up_C |   fixed_acc |   iso_product_acc |   linear_acc |   linear_up_acc |
|:-----------------------------------|----------:|----------------:|-----------:|--------------:|------------:|------------------:|-------------:|----------------:|
| ('mlp/cifar10/SGD', 'SGD')         |         1 |            1    |       1    |          1    |       58.15 |             55.27 |        56.84 |           54.86 |
| ('mlp/cifar10/SGDM', 'SGDM')       |         1 |            1    |       1.5  |          1.5  |       56.86 |             58.67 |        55.71 |           58.8  |
| ('mlp/mnist/SGD', 'SGD')           |         1 |            1    |       1    |          1    |       98.38 |             98.32 |        98.47 |           98.34 |
| ('mlp/mnist/SGDM', 'SGDM')         |         1 |            0.33 |       1    |          1    |       98.76 |             98.67 |        98.77 |           98.73 |
| ('resnet18/cifar100/SGD', 'SGD')   |         1 |            1.79 |     nan    |          1.79 |       77.75 |             78.01 |       nan    |           78.08 |
| ('resnet18/cifar100/SGDM', 'SGDM') |         1 |            1    |       1    |          1.99 |       77.45 |             78.22 |        76.44 |           78.45 |
| ('resnet50/cifar100/SGD', 'SGD')   |         1 |            1    |       1    |          1    |       79.21 |             79.31 |        79.25 |           79.55 |
| ('resnet50/cifar100/SGDM', 'SGDM') |         1 |            0.62 |       0.62 |          0.94 |       78.2  |             78.51 |        77.01 |           78.72 |
| ('vgg16/cifar100/SGD', 'SGD')      |         1 |            1    |       1    |          1    |       75.16 |             75.89 |        74.29 |           75.43 |
| ('vgg16/cifar100/SGDM', 'SGDM')    |         1 |            1.04 |       1.04 |          1.04 |       73.43 |             74.24 |        72.46 |           73.84 |

