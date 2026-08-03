# E9: schedules that preserve the coupling, and a matched-contraction comparison

ResNet-18 / CIFAR-100, `B = 128`, `eta_0 = 0.1`, `T = 100`, SGDM (`beta = 0.9`), seed 42, cosine learning rate throughout — so these are directly comparable to the main protocol rather than to the constant-LR E8 arms.

Contraction budget unit: `C = 1.181`, i.e. `lambda_ref * sum_t eta_t` at the reference setting (`lambda_ref = 5.982e-4`, the value our rule already predicts there).

## (a) Matched cumulative contraction

Every shape `m_lambda(t)` is rescaled so all methods spend the same `sum_t eta_t*lambda_t`. `iso_product` is the reviewer's `lambda_t = lambda_0*eta_0/eta_t` (cap: the multiplier is limited to 10x, i.e. eta is floored at eta_0/10 inside the lambda formula).

| budget | shape | lambda_0 | realized sum eta*lambda | best acc | train acc |
|---:|---|---:|---:|---:|---:|
| 0.333 C | cosine | 0.000265 | 0.394 (0.33 C) | 74.34 | 100.0 |
| 0.333 C | fixed | 0.0001994 | 0.394 (0.33 C) | 74.62 | 100.0 |
| 0.333 C | iso_product | 0.0001158 | 0.394 (0.33 C) | 75.86 | 100.0 |
| 0.333 C | linear | 0.0002826 | 0.394 (0.33 C) | 74.43 | 100.0 |
| 0.333 C | step | 0.0002399 | 0.394 (0.33 C) | 74.16 | 100.0 |
| 1 C | cosine | 0.000795 | 1.18 (1.00 C) | 75.50 | 100.0 |
| 1 C | fixed | 0.0005982 | 1.18 (1.00 C) | 76.72 | 100.0 |
| 1 C | iso_product | 0.0003475 | 1.18 (1.00 C) | 78.22 | 99.9 |
| 1 C | linear | 0.0008478 | 1.18 (1.00 C) | 76.44 | 100.0 |
| 1 C | step | 0.0007196 | 1.18 (1.00 C) | 75.85 | 99.9 |
| 3 C | cosine | 0.002385 | 3.54 (3.00 C) | 75.03 | 99.8 |
| 3 C | fixed | 0.001795 | 3.54 (3.00 C) | 76.19 | 99.6 |
| 3 C | iso_product | 0.001042 | 3.54 (3.00 C) | 77.50 | 95.5 |
| 3 C | linear | 0.002543 | 3.54 (3.00 C) | 75.53 | 99.8 |
| 3 C | step | 0.002159 | 3.54 (3.00 C) | 73.97 | 99.9 |

Spread across shapes at fixed budget:

- budget 0.333 C: 74.16 to 75.86 (spread 1.70 pp, n=5)
- budget 1 C: 75.50 to 78.22 (spread 2.72 pp, n=5)
- budget 3 C: 73.97 to 77.50 (spread 3.53 pp, n=5)

Spread across budgets at fixed shape:

- cosine: 74.34 to 75.50 (spread 1.16 pp, n=3)
- fixed: 74.62 to 76.72 (spread 2.10 pp, n=3)
- iso_product: 75.86 to 78.22 (spread 2.36 pp, n=3)
- linear: 74.43 to 76.44 (spread 2.01 pp, n=3)
- step: 73.97 to 75.85 (spread 1.88 pp, n=3)

## (b) Iso-product arm on the standard lambda_0 ladder

| lambda_0 | realized sum eta*lambda | best acc | train acc | diverged |
|---:|---:|---:|---:|---:|
| 0.0001 | 0.34 (0.29 C) | 75.56 | 100.0 | 0 |
| 0.0005 | 1.7 (1.44 C) | 77.98 | 99.5 | 0 |
| 0.001 | 3.4 (2.88 C) | 77.98 | 96.1 | 0 |
| 0.002 | 6.8 (5.76 C) | 70.36 | 79.5 | 0 |
| 0.005 | 17 (14.39 C) | 22.13 | 21.1 | 0 |

## Reference points at the same eta_0, T, optimizer

| method | best acc |
|---|---:|
| cosine LR + constant lambda (oracle over lambda_0 grid) | 77.28 |
| cosine LR + constant lambda (ours, C/sum_eta) | 76.72 |
| cosine LR + constant lambda (default 5e-4) | 76.73 |
| joint m(t) on eta and lambda (best shape: cosine) | 76.42 |
| constant LR + scheduled lambda (best shape: step) | 73.10 |
| constant LR + constant lambda | 66.67 |

Figure: `outputs/plots/nips26/e9_iso_matched.png`.
