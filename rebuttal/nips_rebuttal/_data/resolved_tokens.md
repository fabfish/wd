# Resolved placeholder values

From 1116 runs (315 from this round).

- `[[E1-T-SLOPE]]` = -0.226
- `[[E1-T-CI]]` = [-0.28, -0.17]
- `[[E1-T-LAMBDA-25]]` = 0.0012
- `[[E1-T-LAMBDA-100]]` = 0.000877
- `[[E1-T-LAMBDA-200]]` = 0.000737
- `[[E1-PRODUCT-DRIFT]]` = 1.62x in eta*lambda (T=25 to T=200; prediction ours=8x, equilibrium=1x)
- `[[E1-LOWLR-SLOPE]]` = -0.52 [-1.03, 0.10]
- `[[E1-ETAT-COLLAPSE]]` = low-eta C x/1.57; joint x/1.73 over 8 points
- `[[E1-SCHED-RATIO]]` = PENDING
- `[[E2B-ISO-DROP]]` = 10.3
- `[[E2B-ISO-RANGE]]` = a factor of 5 in eta
- `[[E3-SLOPE]]` = 0.67
- `[[E3-INTERCEPT]]` = 0.08 (implies L = 0.2)
- `[[E3-LMAX]]` = PENDING
- `[[E3-MOM-RATIO]]` = 0.23
- `[[E4-TABLE]]` = PENDING
- `[[E4-OURS-MEAN]]` = PENDING
- `[[E4-OURS-WORST]]` = PENDING
- `[[E4-DEFAULT-MEAN]]` = PENDING
- `[[E4-KOSSON-MEAN]]` = PENDING
- `[[E4-WANG-MEAN]]` = PENDING
- `[[E4-ZERO-MEAN]]` = PENDING
- `[[E5B-3X]]` = PENDING
- `[[E5B-10X]]` = PENDING
- `[[E6B-LAMBDA-SLOPE]]` = PENDING
- `[[E6B-GAP-SGD]]` = 24.3 points
- `[[E6B-GAP-SGDM]]` = 23.8 points
- `[[E7-DIVERGENCE-RATIO]]` = PENDING
- `[[E7-PLATEAU]]` = PENDING
- `[[E7-BN]]` = PENDING

15 of 30 resolved.


- `[[E5C-C-SGD]]` = 0.44
- `[[E5C-C-SGDM]]` = 0.32
- `[[E5C-C-RATIO]]` = 1.38x
- `[[E5C-3X]]` = 0.018 test-loss
- `[[E5C-10X]]` = 0.083 test-loss
- `[[E5C-FIG]]` = outputs/plots/nips26/e5c_mnist_mlp_C.png

## Notes

- E1 soft-peak slope -0.280 [-0.39, -0.14]; values {25: 0.001304, 50: 0.000993, 100: 0.000806, 200: 0.000732}
- E1 headline eta=0.1: interp slope -0.226 [-0.28, -0.17]; grid argmax identical at [0.001].
- E1 RESCUE: eta=0.02 slope -0.523 [-1.03, 0.10] over T in [25, 50, 100, 200]; argmaxes [0.005, 0.005, 0.003, 0.002]. Two-constraint reading: timescale binds at small eta.
- E1-SCHED-RATIO withheld: const optima on ladder edge (raw ratio 0.01); wait for e1_rescue.
- E1: 4 training lengths (25, 50, 100, 200), slope -0.226 [-0.28, -0.17]. Interior optima: 4/4.
- E2b: 13 runs over 2 product levels, largest end-to-peak drop 10.29 points
- E3: NaN-divergence brackets for 14 (beta, lambda) pairs; median tightness 2.88x; median eta*lambda at boundary 0.431
- E4: 0/6 settings complete; headline tokens left PENDING (partial table in _data/e4_transfer_table.md)
- E5b: 8 mis-specified runs analysed
- E6b: 20 runs
- E5b: factors landed near {0.16, 0.54, 5, 16} (approx S); do not fill 3x/10x tokens until exact sum_lr re-run. Undershoot cheap; overshoot asymmetric.
- E5c MNIST-MLP: C_sgd=0.44, C_sgdm=0.32 (ratio 1.38x); wrong-C costs 0.018/0.083 test-loss at 3x/10x.
