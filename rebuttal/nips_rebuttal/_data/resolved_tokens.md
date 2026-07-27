# Resolved placeholder values

From 1004 runs (190 from this round).

- `[[E1-T-SLOPE]]` = -0.226
- `[[E1-T-CI]]` = [-0.28, -0.17]
- `[[E1-T-LAMBDA-25]]` = 0.0012
- `[[E1-T-LAMBDA-100]]` = 0.000877
- `[[E1-T-LAMBDA-200]]` = 0.000737
- `[[E1-PRODUCT-DRIFT]]` = 1.62x in eta*lambda (T=25 to T=200; prediction ours=8x, equilibrium=1x)
- `[[E1-LOWLR-SLOPE]]` = PENDING
- `[[E1-ETAT-COLLAPSE]]` = PENDING
- `[[E1-SCHED-RATIO]]` = PENDING
- `[[E2B-ISO-DROP]]` = 10.3
- `[[E2B-ISO-RANGE]]` = a factor of 5 in eta
- `[[E3-SLOPE]]` = 0.67
- `[[E3-INTERCEPT]]` = 0.08 (implies L = 0.2)
- `[[E3-LMAX]]` = PENDING
- `[[E3-MOM-RATIO]]` = 0.23
- `[[E4-TABLE]]` = see _data/e4_transfer_table.md
- `[[E4-OURS-MEAN]]` = 0.80
- `[[E4-OURS-WORST]]` = 4.38
- `[[E4-DEFAULT-MEAN]]` = 0.64
- `[[E4-KOSSON-MEAN]]` = 1.06
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

20 of 30 resolved.

## Notes

- E1 OUTCOME: interpolated slope -0.226 [-0.28, -0.17] over T in [25, 50, 100, 200]; grid argmax identical at [0.001]. Ours predicts -1, equilibrium predicts 0.
- E1: slope is incompatible with lambda ∝ 1/T; treat as a negative result against our discriminator and closer to the rotational-equilibrium account.
- E1: 4 training lengths (25, 50, 100, 200), slope -0.226 [-0.28, -0.17]. Interior optima: 4/4.
- E2b: 13 runs over 2 product levels, largest end-to-peak drop 10.29 points
- E3: NaN-divergence brackets for 14 (beta, lambda) pairs; median tightness 2.88x; median eta*lambda at boundary 0.431
- E4: all 6 settings complete
- E5b: 8 mis-specified runs analysed
- E6b: no runs yet