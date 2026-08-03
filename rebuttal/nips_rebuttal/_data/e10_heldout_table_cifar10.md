# E10 (cifar10): held-out test of the constant C across MLP widths

3-layer ReLU MLP, no normalization, CIFAR10, `B = 128`, `T = 30` epochs, cosine learning rate, seed 42; momenta [0.9], learning rates [0.03, 0.1, 0.3], weight-decay ladder [0.0001, 0.0003, 0.001, 0.003, 0.01].

Calibration widths **[256, 512]**, held out **[1024]**. Predictions were written to `e10_predictions_cifar10.json` at 2026-08-04T02:35:37 (blind).

## (a) C across the calibration ladder

| momentum | width | C | n eta cells |
|---:|---:|---:|---:|
| 0.9 | 256 | 2.800 | 2 |
| 0.9 | 512 | 2.672 | 2 |

Regression of `log C` on `log width`:

- momentum 0.9: slope **-0.068** [-0.068, -0.068] over widths [256, 512]; C = 2.800, 2.672

Extrapolated to the held-out widths:

| momentum | width | C predicted | C measured | ratio |
|---:|---:|---:|---:|---:|
| 0.9 | 1024 | 2.549 | 2.479 | 1.03x |

## (b) Zero-tuning rules at the held-out widths

Every rule is applied blind. The oracle for each cell is the best weight decay measured anywhere in that cell — the 5-point ladder plus every rule's own lambda — so no rule can appear to beat tuning just because the ladder is coarse. Tuning cost: 5+ training runs per (width, momentum, eta) cell for the oracle, one run for any rule, zero after a one-time calibration.

Gaps are averaged over the cells that trained; diverged cells are counted separately rather than imputed.

| rule | mean acc gap (pp) | worst acc gap (pp) | mean loss gap | worst loss gap | lambda_rule/lambda_oracle (geo) | cells | diverged |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed default 5e-4 | 0.84 | 1.19 | 0.1576 | 0.2256 | 0.13x | 3/3 | 0 |
| 1/(eta*T) | 0.52 | 1.11 | 0.0792 | 0.1542 | 0.23x | 3/3 | 0 |
| constant eta*lambda | 1.49 | 3.73 | 0.0171 | 0.0382 | 1.08x | 3/3 | 0 |
| ours: C_pred(h)/sum_t eta_t | 2.01 | 3.79 | 0.0233 | 0.0465 | 1.14x | 2/3 | 1 |

Per-cell detail:

| width | momentum | eta | rule | lambda | lambda oracle | acc | oracle acc | acc gap |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1024 | 0.9 | 0.03 | default | 0.0005 | 0.01 | 58.76 | 59.77 | +1.01 |
| 1024 | 0.9 | 0.03 | kosson | 0.01322 | 0.01 | 56.04 | 59.77 | +3.73 |
| 1024 | 0.9 | 0.03 | ours | 0.01402 | 0.01 | 55.98 | 59.77 | +3.79 |
| 1024 | 0.9 | 0.03 | wang | 0.002842 | 0.01 | 59.33 | 59.77 | +0.44 |
| 1024 | 0.9 | 0.1 | default | 0.0005 | 0.004206 | 56.04 | 57.23 | +1.19 |
| 1024 | 0.9 | 0.1 | kosson | 0.003967 | 0.004206 | 56.50 | 57.23 | +0.73 |
| 1024 | 0.9 | 0.1 | ours | 0.004206 | 0.004206 | 57.01 | 57.23 | +0.22 |
| 1024 | 0.9 | 0.1 | wang | 0.0008525 | 0.004206 | 57.21 | 57.23 | +0.02 |
| 1024 | 0.9 | 0.3 | default | 0.0005 | 0.001322 | 55.03 | 55.34 | +0.31 |
| 1024 | 0.9 | 0.3 | kosson | 0.001322 | 0.001322 | 55.34 | 55.34 | +0.00 |
| 1024 | 0.9 | 0.3 | ours | 0.001402 | 0.001322 | diverged | 55.34 | - |
| 1024 | 0.9 | 0.3 | wang | 0.0002842 | 0.001322 | 54.23 | 55.34 | +1.11 |

Figure: `outputs/plots/nips26/e10_c_width_cifar10.png`.
