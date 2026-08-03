# F1: `lambda* ∝ B` at fixed eta, `eta*lambda* ∝ B` under linear scaling

Reanalysis of the optima already fitted for E5a (`e5a_C_per_setting.csv`, 65 interior settings, T=100, SGDM, three architectures, two seeds). No new training.

One law, two conditional readings. With `sum_t eta_t = eta * ceil(n/B) * (T+1)/2`:

| regime | `sum_t eta_t` vs B | predicted `lambda*` | predicted `eta*lambda*` |
|---|---|---|---|
| eta fixed | `∝ 1/B` | `∝ B` | `∝ B` |
| eta ∝ B (Exp. 3) | invariant | flat | `∝ B` |

## (a) At fixed eta: measured slope of `lambda*` in B

Within-(model, eta) pooled slope of `log lambda*` on `log B`: **+1.02** [+0.88, +1.15] (24 points from 11 cells spanning more than one batch size), against a predicted **+1**.

| model | eta | batch sizes | lambda* | slope |
|---|---:|---|---|---:|
| resnet18 | 0.005 | 32,128 | 0.00117, 0.00958 | +1.52 |
| resnet18 | 0.01 | 32,128 | 0.000929, 0.00603 | +1.35 |
| resnet18 | 0.02 | 32,128 | 0.000735, 0.00208 | +0.75 |
| resnet18 | 0.05 | 32,64,128 | 0.00071, 0.00137, 0.00189 | +0.71 |
| resnet18 | 0.1 | 32,128 | 0.000346, 0.00119 | +0.89 |
| resnet18 | 0.2 | 32,128,256 | 9.7e-05, 0.000615, 0.000841 | +1.08 |
| resnet18 | 0.3 | 32,128 | 7.11e-05, 0.000301 | +1.04 |
| resnet50 | 0.05 | 64,128 | 0.000709, 0.00112 | +0.66 |
| resnet50 | 0.2 | 128,256 | 0.000378, 0.00067 | +0.82 |
| vgg16 | 0.05 | 64,128 | 0.000903, 0.0012 | +0.41 |
| vgg16 | 0.2 | 128,256 | 0.000461, 0.000942 | +1.03 |

Ignoring eta and regressing over all 65 settings instead gives +0.23 [-0.05, +0.55]: the raw scatter shows almost nothing, because in these sweeps the two dependencies partly cancel. The conditioning is the whole content of the claim.

## (b) Along Exp. 3's line eta ∝ B: `lambda*` is flat, the product grows

These are the Exp. 3 configurations themselves (the same (B, eta) pairs the E4 transfer test uses).

| B | eta | `sum_t eta_t` | lambda* (geo) | eta*lambda* (geo) | C (geo) | n | models |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 0.025 | 1973 | 0.000873 | 2.18e-05 | 1.72 | 1 | resnet18 |
| 64 | 0.05 | 1975 | 0.000956 | 4.78e-05 | 1.89 | 6 | resnet18,resnet50,vgg16 |
| 128 | 0.1 | 1975 | 0.00121 | 0.000121 | 2.39 | 6 | resnet18,resnet50,vgg16 |
| 256 | 0.2 | 1980 | 0.00081 | 0.000162 | 1.60 | 6 | resnet18,resnet50,vgg16 |
| 512 | 0.4 | 1980 | 0.000865 | 0.000346 | 1.71 | 6 | resnet18,resnet50,vgg16 |

- `lambda*` slope in B: **-0.03** [-0.28, +0.24], total spread 1.50x over a 16x range of B — flat, as predicted, and inside the residual C spread reported in (c).
- `eta*lambda*` slope in B: **+0.97** [+0.72, +1.24], total spread 15.9x — this is the growing quantity.
- `sum_t eta_t` is B-invariant along this line by construction (1973-1980), which is why `lambda*` has nothing to respond to.

## (c) Residual drift of C across B

Geometric mean of C by batch size (all settings, not just the linear rule line): 1.38, 1.89, 1.40, 1.60, 1.71 for B = 32, 64, 128, 256, 512; spread 1.37x.

This is the part of the batch-size dependence that `1/sum_t eta_t` does not absorb. It is also the noise floor against which the flatness in (b) has to be read.

## What we will state in the paper

1. The reviewer is right that under Exp. 3's constraint the prediction is a roughly constant `lambda*` with a product that grows like B. Our own data agrees: `lambda*` slope -0.03, product slope +0.97.
2. We will therefore drop the sentence in Exp. 3 that says the optimal `lambda` grows with batch size, and state instead that the *product* grows with B while `lambda*` stays put, because `sum_t eta_t` is held fixed by linear learning-rate scaling.
3. The `lambda* ∝ B` reading is the *fixed-eta* one, and we now measure it separately: +1.02 [+0.88, +1.15] against a predicted +1. Stating which quantity is held fixed removes the apparent conflict between Eq. (17) and Table 4.
