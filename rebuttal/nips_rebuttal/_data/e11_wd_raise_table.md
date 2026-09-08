E11: raise-up lambda under cosine LR (R18/CIFAR-100, B=128, eta0=0.1, T=100).
Raise-up grid runs are seed 42; lambda0 is the end-of-run peak value.

| phase | tag | wd_sched | peak_acc | peak_lambda0 | delta vs fixed(seed42) | n | note |
|---|---|---|---:|---:|---:|---:|---|
| SGDM | fixed_seed42 | fixed | 77.45 | 0.0006 | +0.00 | 27 | oracle over available lambdas, seed 42 (control) |
| SGDM | fixed_allseeds | fixed | 77.84 | 0.0006 | +0.39 | 34 | oracle over available lambdas, all seeds |
| SGDM | e11_raise | linear_up | 78.28 | 0.005 | +0.83 | 8 |  |
| SGDM | e11_raise | cosine_up | 78.09 | 0.005 | +0.64 | 8 |  |
| SGDM | e11_raise | step_up | 77.39 | 0.01 | -0.06 | 8 |  |
| SGDM | e9_iso | iso_product | 78.22 | 0.0003475 | +0.77 | 14 | analytic raise-up lambda0*eta0/eta_t (reference) |
| SGD | fixed_seed42 | fixed | 77.75 | 0.005 | +0.00 | 8 | oracle over available lambdas, seed 42 (control) |
| SGD | fixed_allseeds | fixed | 77.75 | 0.005 | +0.00 | 8 | oracle over available lambdas, all seeds |
| SGD | e11_raise | linear_up | 77.94 | 0.02 | +0.19 | 8 |  |
| SGD | e11_raise | cosine_up | 77.91 | 0.02 | +0.16 | 8 |  |
| SGD | e11_raise | step_up | 76.73 | 0.05 | -1.02 | 8 |  |
| SGD | e9_iso | iso_product | 78.01 | 0.005212 | +0.26 | 13 | analytic raise-up lambda0*eta0/eta_t (reference) |
