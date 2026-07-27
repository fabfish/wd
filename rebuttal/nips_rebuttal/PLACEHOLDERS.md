# Placeholder registry

Every `[[TOKEN]]` in the reviewer responses is a number that is not measured
yet. Fill them from the experiment listed here, then delete the row.

Run `python -m analysis.nips26_report` after each wave: it prints the current
value of every token that can already be resolved.

| Token | Filled by | Meaning | Status |
|---|---|---|---|
| `[[E1-T-SLOPE]]` | **E1-fine** (not prelim) | slope of `log lambda*` vs `log T` at eta=0.1 | queued |
| `[[E1-T-CI]]` | E1-fine | 95% bootstrap interval for the above | queued |
| `[[E1-T-LAMBDA-25]]` | E1-fine | lambda* at T=25 | queued |
| `[[E1-T-LAMBDA-100]]` | E1-fine | lambda* at T=100 | queued |
| `[[E1-T-LAMBDA-200]]` | E1-fine | lambda* at T=200 | queued |
| `[[E1-PRODUCT-DRIFT]]` | E1-fine | factor by which the optimal `eta*lambda` moves from T=25 to T=200 | queued |
| `[[E1-LOWLR-SLOPE]]` | E1-full | same slope measured at eta=0.02 | pending |
| `[[E1-ETAT-COLLAPSE]]` | E1-full | residual spread of C after collapsing both eta arms onto `sum_lr` | pending |
| `[[E1-SCHED-RATIO]]` | E1-full | ratio of lambda* between constant-LR and cosine at matched eta, T (prediction: 0.5) | pending |
| `[[E2B-ISO-DROP]]` | E2b | accuracy lost at the ends of the iso-product line relative to its peak | pending |
| `[[E2B-ISO-RANGE]]` | E2b | eta range over which the iso-product line stays within 1 point of its peak | pending |
| `[[E3-SLOPE]]` | E3 | fitted slope of `1/eta_max` against lambda (theory: 1) | pending |
| `[[E3-INTERCEPT]]` | E3 | intercept of that fit, i.e. the implied `L/2` | pending |
| `[[E3-LMAX]]` | E3 | top Hessian eigenvalue measured by power iteration | pending |
| `[[E3-MOM-RATIO]]` | E3 | ratio of eta_max between beta=0 and beta=0.9 (theory: 1-beta = 0.1) | pending |
| `[[E4-TABLE]]` | E4 | full accuracy-gap-to-oracle table over the five strategies | pending |
| `[[E4-OURS-MEAN]]` | E4 | mean gap to oracle, our rule | pending |
| `[[E4-OURS-WORST]]` | E4 | worst gap to oracle, our rule | pending |
| `[[E4-DEFAULT-MEAN]]` | E4 | mean gap to oracle, fixed lambda=5e-4 | pending |
| `[[E4-KOSSON-MEAN]]` | E4 | mean gap to oracle, constant `eta*lambda` | pending |
| `[[E4-WANG-MEAN]]` | E4 | mean gap to oracle, `1/(eta*T)` | pending |
| `[[E5B-3X]]` | E5b | accuracy lost when C is wrong by a factor of 3 | pending |
| `[[E5B-10X]]` | E5b | accuracy lost when C is wrong by a factor of 10 | pending |
| `[[E6B-LAMBDA-SLOPE]]` | E6b | slope of `log lambda*` against `log(1-beta)` (prediction: 1) | pending |
| `[[E6B-GAP-SGD]]` | E6b | train-test accuracy gap, SGD+WD at its optimum | pending |
| `[[E6B-GAP-SGDM]]` | E6b | train-test accuracy gap, SGDM+WD at its optimum | pending |
| `[[E7-DIVERGENCE-RATIO]]` | E7a | ratio of `\|\|theta_T - theta'_T\|\|` between lambda=0 and lambda>0 | pending |
| `[[E7-PLATEAU]]` | E7a | whether the weight-decay run plateaus, and at what step | pending |
| `[[E7-BN]]` | E7c | whether the coupling survives without normalization layers | pending |

## Numbers that are already measured

These are in the drafts as literal values, from Wave 0 (no new training):

- envelope varies by 2.8 points over two decades of learning rate
- fixed `lambda = 5e-4` gives up 1.44 points on average, 3.77 at worst
- no fixed `lambda` does better than 3.77 points worst-case
- `log lambda*` vs `log eta` slope: -0.87, 95% interval [-0.94, -0.75]
- `C = lambda* * sum_lr`: geometric mean 1.48, spread x/1.70, 65 settings
- C by architecture: ResNet-18 1.42, ResNet-50 1.42, VGG-16 1.72
- momentum at its own optimal eta changes accuracy by 0.25 points
  (`lambda=0`, beta in [0, 0.8]) and 1.38 points (`lambda=2e-3`, beta in [0, 0.95])
- `log eta*` vs `log(1-beta)` slope: 1.24 [0.80, 2.32] without weight decay,
  0.72 [0.58, 0.93] with it
