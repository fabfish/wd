# Resolved placeholder values

From 1033 runs (229 from this round).

- `[[E1-T-SLOPE]]` = -0.226
- `[[E1-T-CI]]` = [-0.28, -0.17]
- `[[E1-T-LAMBDA-25]]` = 0.0012
- `[[E1-T-LAMBDA-100]]` = 0.000877
- `[[E1-T-LAMBDA-200]]` = 0.000737
- `[[E1-PRODUCT-DRIFT]]` = 1.62x in eta*lambda (T=25 to T=200; prediction ours=8x, equilibrium=1x)
- `[[E1-LOWLR-SLOPE]]` = -0.61 [-1.34, -0.32]
- `[[E1-ETAT-COLLAPSE]]` = low-eta C x/1.61; joint x/1.81 over 7 points
- `[[E1-SCHED-RATIO]]` = 0.10 (prediction 0.5, n=2)
- `[[E2B-ISO-DROP]]` = 10.3
- `[[E2B-ISO-RANGE]]` = a factor of 5 in eta
- `[[E3-SLOPE]]` = 0.67
- `[[E3-INTERCEPT]]` = 0.08 (implies L = 0.2)
- `[[E3-LMAX]]` = PENDING
- `[[E3-MOM-RATIO]]` = 0.23
- `[[E4-TABLE]]` = see _data/e4_transfer_table.md
- `[[E4-OURS-MEAN]]` = 0.79
- `[[E4-OURS-WORST]]` = 4.38
- `[[E4-DEFAULT-MEAN]]` = 0.54
- `[[E4-KOSSON-MEAN]]` = 1.01
- `[[E4-WANG-MEAN]]` = 1.45
- `[[E4-ZERO-MEAN]]` = 8.84
- `[[E5B-3X]]` = 35.53
- `[[E5B-10X]]` = 72.18
- `[[E6B-LAMBDA-SLOPE]]` = PENDING
- `[[E6B-GAP-SGD]]` = PENDING
- `[[E6B-GAP-SGDM]]` = PENDING
- `[[E7-DIVERGENCE-RATIO]]` = PENDING
- `[[E7-PLATEAU]]` = PENDING
- `[[E7-BN]]` = PENDING

23 of 30 resolved.

## Notes

- E1 soft-peak slope -0.280 [-0.39, -0.14]; values {25: 0.001304, 50: 0.000993, 100: 0.000806, 200: 0.000732}
- E1 headline eta=0.1: interp slope -0.226 [-0.28, -0.17]; grid argmax identical at [0.001].
- E1 RESCUE: eta=0.02 slope -0.609 [-1.34, -0.32] over T in [25, 100, 200]; argmaxes [0.005, 0.005, 0.001]. Two-constraint reading: timescale binds at small eta.
- E1: low-eta arm is compatible with lambda ∝ 1/T (and incompatible with slope 0). Do NOT call E1 a negative result.
- E1: 4 training lengths (25, 50, 100, 200), slope -0.226 [-0.28, -0.17]. Interior optima: 4/4.
- E2b: 13 runs over 2 product levels, largest end-to-peak drop 10.29 points
- E3: NaN-divergence brackets for 14 (beta, lambda) pairs; median tightness 2.88x; median eta*lambda at boundary 0.431
- E4: all 6 settings complete
- E5b: 8 mis-specified runs analysed
- E6b: no runs yet