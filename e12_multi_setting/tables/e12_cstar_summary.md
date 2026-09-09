E12 cross-setting summary: optimal budget C* per shape
(columns "wd_sched_C" are peak budgets in local C units;
"wd_sched_acc" the corresponding accuracy).

|                                    |   fixed_C |   iso_product_C |   linear_C |   linear_up_C |   fixed_acc |   iso_product_acc |   linear_acc |   linear_up_acc |
|:-----------------------------------|----------:|----------------:|-----------:|--------------:|------------:|------------------:|-------------:|----------------:|
| ('mlp/cifar10/SGD', 'SGD')         |         1 |            1    |        1   |          1    |       58.15 |             55.27 |        56.84 |           54.86 |
| ('mlp/cifar10/SGDM', 'SGDM')       |         1 |            1    |        1.5 |          1.5  |       56.86 |             58.67 |        55.71 |           58.8  |
| ('resnet18/cifar100/SGD', 'SGD')   |         1 |            1.79 |      nan   |          1.79 |       77.75 |             78.01 |       nan    |           78.08 |
| ('resnet18/cifar100/SGDM', 'SGDM') |         1 |            1    |        1   |          1.99 |       76.72 |             78.22 |        76.44 |           78.45 |

