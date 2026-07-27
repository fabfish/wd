# Wave 0: what the existing runs already answer

Built from 1318 previously collected runs, no new training.


## E2a  accuracy envelope over lambda

Across the eight learning rates spanned by the grid, the envelope `max_lambda acc` varies by only 2.8 points, so a correctly coupled weight decay keeps accuracy nearly flat over two decades of learning rate.

Holding the common default `lambda = 5e-4` gives up 1.44 points on average and 3.77 points at its worst learning rate. The best possible *fixed* choice is `lambda = 0.0005`, and even that still gives up 3.77 points somewhere in the range.

The optimum itself moves: fitting `log lambda*` against `log eta` gives slope -0.87 with 95% bootstrap interval [-0.94, -0.75], consistent with the inverse coupling and inconsistent with lambda being a constant.

Worst-case shortfall of each fixed weight decay (percentage points):

- `lambda = 0.0005`: mean 1.44, worst 3.77
- `lambda = 0.0002`: mean 2.14, worst 4.12
- `lambda = 0.0001`: mean 2.92, worst 4.37
- `lambda = 0.001`: mean 2.08, worst 8.50
- `lambda = 0.002`: mean 7.27, worst 37.79
- `lambda = 0.005`: mean 21.90, worst 73.47
- `lambda = 0.01`: mean 32.84, worst 74.31
- `lambda = 0.02`: mean 42.80, worst 74.62
- `lambda = 0.05`: mean 66.43, worst 75.81

## E5a  how stable is the constant C

Fitting `C = lambda* * sum_t eta_t` independently in 65 settings (of 72 weight-decay sweeps available; the rest peak on the edge of their swept range and only bound C) gives a geometric mean of C = 1.48, a multiplicative standard deviation of x/1.70, and a full range of 0.17 to 2.99. The settings span architectures ['resnet18', 'resnet50', 'vgg16'], batch sizes [32, 64, 128, 256, 512], learning rates 0.001 to 0.5, and two seeds.

Geometric mean of C by architecture:

- resnet18: 1.42
- resnet50: 1.42
- vgg16: 1.72

Geometric mean of C by batch size:

- B = 32: 1.38
- B = 64: 1.89
- B = 128: 1.40
- B = 256: 1.60
- B = 512: 1.71

The residual batch-size trend in panel (c) is the part of the batch size dependence that the 1/sum_lr factor does not already absorb, and is the honest version of the paper's claim that the optimal product grows with B.

## E6a  momentum, from the existing sweeps

- $\lambda=0$: `log eta*` against `log(1-beta)` has slope 1.24 with 95% interval [0.80, 2.32] over 9 momentum values (the effective-step argument predicts 1).
- $\lambda=2\times10^{-3}$: `log eta*` against `log(1-beta)` has slope 0.72 with 95% interval [0.58, 0.93] over 7 momentum values (the effective-step argument predicts 1).

Peak accuracy at each momentum, after retuning the learning rate:

- $\lambda=0$: beta=0: 73.61%, beta=0.1: 73.59%, beta=0.2: 73.66%, beta=0.3: 73.53%, beta=0.4: 73.60%, beta=0.5: 73.45%, beta=0.6: 73.48%, beta=0.7: 73.62%, beta=0.8: 73.41%
- $\lambda=2\times10^{-3}$: beta=0: 78.95%, beta=0.5: 77.57%, beta=0.7: 78.72%, beta=0.8: 78.48%, beta=0.9: 78.56%, beta=0.95: 78.36%, beta=0.99: 73.68%

- With $\lambda=0$, momentum from 0 to 0.95 changes peak accuracy by only 0.25 points once the learning rate is retuned.
- With $\lambda=2\times10^{-3}$, momentum from 0 to 0.95 changes peak accuracy by only 1.38 points once the learning rate is retuned. At beta = 0.99 accuracy falls to 73.68%, which is the effective step size eta/(1-beta) running past the stability boundary rather than a generalization effect.

This is the direct answer to the question of whether momentum generalizes better: at its own optimal learning rate it does not, in either arm. What momentum changes is where that optimum sits, which is exactly what the momentum factor in the stability bound describes. Weight decay, by contrast, moves peak accuracy by about five points at every momentum value.
