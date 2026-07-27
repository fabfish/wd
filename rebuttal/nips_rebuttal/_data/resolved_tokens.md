# Resolved placeholder values

From 974 runs (160 from this round).

- `[[E1-T-SLOPE]]` = PENDING
- `[[E1-T-CI]]` = PENDING
- `[[E1-T-LAMBDA-25]]` = PENDING
- `[[E1-T-LAMBDA-100]]` = PENDING
- `[[E1-T-LAMBDA-200]]` = PENDING
- `[[E1-PRODUCT-DRIFT]]` = PENDING
- `[[E1-LOWLR-SLOPE]]` = PENDING
- `[[E1-ETAT-COLLAPSE]]` = PENDING
- `[[E1-SCHED-RATIO]]` = PENDING
- `[[E2B-ISO-DROP]]` = 10.3
- `[[E2B-ISO-RANGE]]` = a factor of 5 in eta
- `[[E3-SLOPE]]` = 143.84
- `[[E3-INTERCEPT]]` = -0.40 (implies L = -0.8)
- `[[E3-LMAX]]` = PENDING
- `[[E3-MOM-RATIO]]` = 0.14
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
- `[[E6B-GAP-SGD]]` = PENDING
- `[[E6B-GAP-SGDM]]` = PENDING
- `[[E7-DIVERGENCE-RATIO]]` = PENDING
- `[[E7-PLATEAU]]` = PENDING
- `[[E7-BN]]` = PENDING

5 of 30 resolved.

## Notes

- E1: need >=2 training lengths with >= 6 lambda values each; have [(25, 3), (50, 8), (100, 3), (200, 3)]
- E2b: 13 runs over 2 product levels, largest end-to-peak drop 10.29 points
- E3: thresholds bracketed for 8 (beta, lambda) pairs; median bracket tightness 1.09x
- E4: 0/6 settings have all 5 strategies; means below cover only those
- E4: no setting has every strategy yet, nothing quotable
- E5b: no runs yet
- E6b: no runs yet