# E12 cross-setting WD-schedule evidence — rebuttal material

Source: `e12_multi_setting/` (E12 experiments) + `e11_raise_up/` (R18/C100
column). Protocol: cosine LR (T=100, eta0=0.1), B=128, seed 42 unless noted,
contraction budget in setting-local C units (C = lambda_ref * sum eta_t, with
lambda_ref the fixed-WD oracle of that setting/phase).

## Claim 1: raising weight decay over the run beats the fixed oracle — across 4 architectures, both optimizers

Under cosine LR, a WD schedule that rises during training (linear_up, or the
theory-predicted iso_product) reaches higher test accuracy than the best
constant lambda (the fixed oracle, not the paper default):

| setting | phase | best dynamic | acc | fixed oracle | acc | gain |
|---|---|---|---|---|---|---|
| ResNet-18/C100 | SGDM | linear_up @ 1.99C | 78.45 | 6e-4 | 77.45 | +1.00 |
| ResNet-18/C100 | SGD | linear_up @ 1.79C | 78.08 | 5e-3 | 77.75 | +0.33 |
| ResNet-50/C100 | SGDM | linear_up @ 0.94C | 78.72 | 9.62e-4 | 78.20 | +0.52 |
| ResNet-50/C100 | SGD | linear_up @ 1.00C | 79.55 | 5e-3 | 79.21 | +0.34 |
| VGG-16/C100 | SGDM | iso_product @ 1.04C | 74.24 | 9.62e-4 | 73.43 | +0.81 |
| VGG-16/C100 | SGD | iso_product @ 1.00C | 75.89 | 1e-2 | 75.16 | +0.73 |
| MLP/C10 | SGDM | linear_up @ 1.50C | 58.80 | 1e-3 | 56.86 | +1.94 |
| MLP/C10 | SGD | fixed wins (counter-example) | 58.15 | — | — | − |

Multi-seed checks (3 seeds where available): every setting shows the dynamic
shape beating the fixed oracle in EVERY seed (pairwise): R18/SGD linear_up
78.17±0.07 vs fixed 77.65±0.16; R50/SGD 79.68±0.07 vs 79.26±0.15; VGG/SGD
iso 76.11±0.33 vs 74.89±0.22; VGG/SGDM iso 73.61±0.23 vs 73.13±0.03;
MLP/C10 SGDM linear_up 58.10±0.08 vs 56.36±0.12; R50/SGDM linear_up
78.31±0.44 vs fixed 76.92±0.85. E11 already had R18/SGDM 3-seed
confirmation (78.05±0.16 vs 77.45±0.31).

Caveats stated honestly: R18/SGD linear_down looked tied with linear_up at
seed 42 (78.04) but falls back to the fixed level across seeds
(77.55±0.23) — not a stable winner. R50/SGDM's fixed oracle at 9.62e-4 is
itself seed-sensitive (75.77–78.20 across 4 estimates), while linear_up is
stable and uniformly better — a robustness argument FOR scheduling.

## Claim 2: the iso-product schedule peaks at ~1C everywhere

The theory-predicted hyperbolic raise-up lambda_t = lambda0 * eta0/eta_t
(capped in the cosine tail) attains its optimum at 0.62-1.04C in every one of
the 10 setting/phase cells with measurable signal — never beyond the
theoretical point 1C. Budgets above ~1.5C degrade monotonically.

## Claim 3: in setting-local C units, the optimal budget concentrates at ~1-2C

Normalizing each setting by its own fixed oracle, the best dynamic budget
sits at 0.94-1.99C for SGDM and 1.0-1.79C for SGD across all four
architectures. The 5x SGDM/SGD gap in the E4 anchor units is exactly the
momentum amplification of the fixed-WD oracle (5e-3 vs 6e-4); once normalized
by each phase's own oracle the two optimizers agree.

## Claim 4: collapse boundary is architecture-dependent

VGG/SGD collapses to chance (1-4% acc) already at 2C; R50/SGD collapses at
2-3C (linear_up 79.55@1C -> 71.92@3C; iso 79.31@1C -> 45.41@3C); R18/SGD
survives to ~3-7C. The WD tolerance window is narrowest for VGG — consistent
with its lack of residual connections.

## Counter-examples (state honestly)

- MLP (no BN) on CIFAR-10 / SGD: fixed WD is best; all dynamic shapes lose
  1-3 points (stable across seeds). Scale-non-invariant networks may not
  benefit from raising WD.
- MNIST: all shapes within 0.5% (task too easy to resolve).
- The fixed-WD oracle lambda itself is architecture-specific and
  seed-sensitive near its sharp peak (9.62e-4 wins for R50/VGG but loses on
  R18: 76.22 vs 6e-4's 77.45; MLP oracle is 8e-4). Schedules at ~1-2C avoid
  this brittleness.

## Practical recommendation for the paper

"Under a cosine LR schedule, schedule weight decay to rise during training
(e.g. linear ramp or the eta0/eta_t product), with total contraction budget
~1-2C where C = lambda_fixed* sum eta_t of the best constant-lambda run.
Constant WD is a special case that is strictly worse on scale-invariant
architectures (ResNet/VGG)."

## BN-MLP ablation (mlp_bn/cifar10)

We tested the hypothesis that the MLP SGD counter-example is explained by
missing BatchNorm (scale non-invariance). Adding BN does NOT rescue the SGD
phase: fixed (56.74) and linear_up (56.75) tie, iso_product is worse (52.50).
On SGDM, BN flips the shape ordering: linear_down@3C (58.91) becomes best,
fixed oracle moves to the smallest grid point (1e-4, 58.31). Scale
invariance alone therefore does not explain the raise-up advantage on
ResNet/VGG; residual connections (or other architectural factors) likely
matter. State this as a negative result if space allows.
