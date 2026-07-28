Peak best-test accuracy under constant LR (η=0.1, T=100, B=128, R18/CIFAR-100).

| optimizer | wd_sched | peak_acc | peak_λ0 | Δ vs fixed | n |
|---|---|---:|---:|---:|---:|
| SGD | fixed | 73.20 | 0.0001 | +0.00 | 5 |
| SGD | cosine | 73.50 | 0.0005 | +0.30 | 5 |
| SGD | linear | 73.22 | 0.0005 | +0.02 | 5 |
| SGD | step | 74.24 | 0.001 | +1.04 | 5 |
| SGD | cosine_restarts | 73.43 | 0.0001 | +0.23 | 5 |
| SGDM | fixed | 66.67 | 0.0001 | +0.00 | 5 |
| SGDM | cosine | 71.34 | 0.0005 | +4.67 | 5 |
| SGDM | linear | 70.32 | 0.0001 | +3.65 | 5 |
| SGDM | step | 73.10 | 0.0005 | +6.43 | 5 |
| SGDM | cosine_restarts | 70.13 | 0.0001 | +3.46 | 5 |
