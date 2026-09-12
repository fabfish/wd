# E4 blind-prediction points (historical, excluded from E12 main tables)

E4 predicted lambda with no tuning at the target setting: five
strategies calibrated once on the R18 reference, applied blind to
held-out settings. These rows are deliberately NOT part of the
grid-search narrative of the main tables: 1C there means the best
GRID-SEARCHED const WD, and these predicted points (e.g. 9.62e-4 on
R50/VGG16) would otherwise pollute that definition.

Protocol: cosine LR, momentum 0.9, seed 42.

## Protocol-matched rows (B=128, lr=0.1, T=100)

| model | lambda | best acc | diverged |
|---|---:|---:|---:|
| resnet50 | 0 | 63.90 | 0 |
| resnet50 | 0.000256 | 76.19 | 0 |
| resnet50 | 0.000598 | 76.86 | 0 |
| resnet50 | 0.000962 | 78.20 | 0 |
| vgg16 | 0 | 65.95 | 0 |
| vgg16 | 0.000256 | 72.05 | 0 |
| vgg16 | 0.000598 | 72.67 | 0 |
| vgg16 | 0.000962 | 73.43 | 0 |

The 9.62e-4 rows beat the grid oracle on R50 (grid 1.1e-3 = 77.72)
and VGG16 (grid 1e-3 = 73.02) but are a prediction, not a search
result, so they are kept out of the main comparison. The 2.56e-4
rows are the wang strategy 1/(eta*T_steps); 5.98e-4 is ours/kosson
under the cosine-calibrated C.

## All E4 rows (including R18 protocol variants)

| model | B | lr | T | lambda | best acc | diverged |
|---|---:|---:|---:|---:|---:|---:|
| resnet18 | 32 | 0.025 | 100 | 0 | 73.56 | 0 |
| resnet18 | 32 | 0.025 | 100 | 0.000256 | 77.01 | 0 |
| resnet18 | 32 | 0.025 | 100 | 0.000599 | 77.91 | 0 |
| resnet18 | 32 | 0.025 | 100 | 0.000963 | 77.54 | 0 |
| resnet18 | 32 | 0.025 | 100 | 0.00239 | 76.65 | 0 |
| resnet18 | 32 | 0.025 | 100 | 0.00385 | 74.46 | 0 |
| resnet18 | 128 | 0.1 | 25 | 0 | 67.53 | 0 |
| resnet18 | 128 | 0.1 | 25 | 0.000598 | 72.44 | 0 |
| resnet18 | 128 | 0.1 | 25 | 0.000962 | 74.14 | 0 |
| resnet18 | 128 | 0.1 | 25 | 0.00102 | 74.41 | 0 |
| resnet18 | 128 | 0.1 | 25 | 0.00232 | 73.44 | 0 |
| resnet18 | 128 | 0.1 | 25 | 0.00374 | 70.04 | 0 |
| resnet18 | 128 | 0.1 | 200 | 0 | 70.02 | 0 |
| resnet18 | 128 | 0.1 | 200 | 0.000128 | 75.68 | 0 |
| resnet18 | 128 | 0.1 | 200 | 0.000301 | 77.35 | 0 |
| resnet18 | 128 | 0.1 | 200 | 0.000483 | 77.89 | 0 |
| resnet18 | 128 | 0.1 | 200 | 0.000598 | 78.30 | 0 |
| resnet18 | 128 | 0.1 | 200 | 0.000962 | 78.93 | 0 |
| resnet18 | 512 | 0.4 | 100 | 0 | 64.56 | 0 |
| resnet18 | 512 | 0.4 | 100 | 0.00015 | 71.71 | 0 |
| resnet18 | 512 | 0.4 | 100 | 0.000241 | 73.06 | 0 |
| resnet18 | 512 | 0.4 | 100 | 0.000255 | 74.56 | 0 |
| resnet18 | 512 | 0.4 | 100 | 0.000597 | 76.04 | 0 |
| resnet18 | 512 | 0.4 | 100 | 0.00096 | 75.29 | 0 |
| resnet50 | 128 | 0.1 | 100 | 0 | 63.90 | 0 |
| resnet50 | 128 | 0.1 | 100 | 0.000256 | 76.19 | 0 |
| resnet50 | 128 | 0.1 | 100 | 0.000598 | 76.86 | 0 |
| resnet50 | 128 | 0.1 | 100 | 0.000962 | 78.20 | 0 |
| vgg16 | 128 | 0.1 | 100 | 0 | 65.95 | 0 |
| vgg16 | 128 | 0.1 | 100 | 0.000256 | 72.05 | 0 |
| vgg16 | 128 | 0.1 | 100 | 0.000598 | 72.67 | 0 |
| vgg16 | 128 | 0.1 | 100 | 0.000962 | 73.43 | 0 |

