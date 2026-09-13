E12 cross-setting summary: optimal budget C* per shape
(columns "wd_sched_C" are peak budgets in local C units;
"wd_sched_acc" the corresponding accuracy).

|                                    |   fixed_C |   iso_product_C |   linear_C |   linear_up_C |   fixed_acc |   iso_product_acc |   linear_acc |   linear_up_acc |
|:-----------------------------------|----------:|----------------:|-----------:|--------------:|------------:|------------------:|-------------:|----------------:|
| ('mlp/cifar10/SGD', 'SGD')         |       1   |            1    |        1   |           1   |       58.15 |             55.27 |        56.84 |           54.86 |
| ('mlp/cifar10/SGDM', 'SGDM')       |       1   |            1.5  |        2   |           2   |       57.18 |             58.67 |        55.71 |           58.8  |
| ('mlp/mnist/SGD', 'SGD')           |       1   |            1    |        1   |           1   |       98.38 |             98.32 |        98.47 |           98.34 |
| ('mlp/mnist/SGDM', 'SGDM')         |       1   |            0.33 |        1   |           1   |       98.76 |             98.67 |        98.77 |           98.73 |
| ('mlp_bn/cifar10/SGD', 'SGD')      |       1   |            1    |        1   |           1   |       56.74 |             52.5  |        55.68 |           56.75 |
| ('mlp_bn/cifar10/SGDM', 'SGDM')    |       4   |            1    |        3   |           2   |       58.67 |             57.99 |        58.91 |           57.9  |
| ('resnet18/cifar100/SGD', 'SGD')   |       1   |            2    |      nan   |           2   |       77.75 |             78.01 |       nan    |           78.08 |
| ('resnet18/cifar100/SGDM', 'SGDM') |       1   |            1    |        1   |           2   |       77.45 |             78.22 |        76.44 |           78.45 |
| ('resnet34/cifar100/SGD', 'SGD')   |       1   |            1    |        1   |           1   |       78.93 |             79    |        78.52 |           79.3  |
| ('resnet34/cifar100/SGDM', 'SGDM') |       1   |            1    |        1   |           1   |       78.41 |             78.18 |        76.97 |           78.16 |
| ('resnet50/cifar100/SGD', 'SGD')   |       1   |            1    |        1   |           1   |       79.4  |             79.31 |        79.25 |           79.55 |
| ('resnet50/cifar100/SGDM', 'SGDM') |       1   |            0.5  |        0.5 |           1   |       77.72 |             78.51 |        77.01 |           78.72 |
| ('vgg13/cifar100/SGD', 'SGD')      |       1.5 |            1    |        1   |           1   |       76.59 |             76.76 |        75.24 |           76.41 |
| ('vgg13/cifar100/SGDM', 'SGDM')    |       2   |            2    |        1.5 |           1.5 |       74.65 |             75.31 |        73.94 |           74.81 |
| ('vgg16/cifar100/SGD', 'SGD')      |       1   |            1    |        1   |           1   |       75.16 |             75.89 |        74.29 |           75.43 |
| ('vgg16/cifar100/SGDM', 'SGDM')    |       1   |            1    |        1   |           1   |       73.47 |             74.24 |        72.46 |           73.84 |

