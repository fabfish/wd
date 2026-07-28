# Resolved placeholder values

From 1156 runs (355 from this round).

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
- `[[E3-LMAX]]` = 417.4
- `[[E3-MOM-RATIO]]` = 0.23
- `[[E4-TABLE]]` = see _data/e4_transfer_table.md
- `[[E4-OURS-MEAN]]` = 0.87
- `[[E4-OURS-WORST]]` = 1.58
- `[[E4-DEFAULT-MEAN]]` = 0.70
- `[[E4-KOSSON-MEAN]]` = 1.63
- `[[E4-WANG-MEAN]]` = 1.60
- `[[E4-ZERO-MEAN]]` = 9.00
- `[[E5B-3X]]` = 15.11
- `[[E5B-10X]]` = 69.74
- `[[E6B-LAMBDA-SLOPE]]` = 0.42 [-0.52, 1.75]
- `[[E6B-GAP-SGD]]` = 22.6 points
- `[[E6B-GAP-SGDM]]` = 23.7 points
- `[[E7-DIVERGENCE-RATIO]]` = 1.10 (final ||theta-theta'|| at lambda=0 over lambda=1e-3)
- `[[E7-PLATEAU]]` = yes under lambda=1e-3 (late drift -1.9%); lambda=0 keeps growing
- `[[E7-BN]]` = coupling survives without BN — bn=0: lambda* in {0.002..0.002} across eta, peak acc 59.3%; bn=1: lambda* in {0.0005..0.0005} across eta, peak acc 58.5%

29 of 30 resolved.

## Notes

- E1 soft-peak slope -0.280 [-0.39, -0.14]; values {25: 0.001304, 50: 0.000993, 100: 0.000806, 200: 0.000732}
- E1 headline eta=0.1: interp slope -0.226 [-0.28, -0.17]; grid argmax identical at [0.001].
- E1 RESCUE: eta=0.02 slope -0.523 [-1.03, 0.10] over T in [25, 50, 100, 200]; argmaxes [0.005, 0.005, 0.003, 0.002]. Two-constraint reading: timescale binds at small eta.
- E1-SCHED-RATIO withheld: const optima on ladder edge (raw ratio 0.01); wait for e1_rescue.
- E1: 4 training lengths (25, 50, 100, 200), slope -0.226 [-0.28, -0.17]. Interior optima: 4/4.
- E2b: 13 runs over 2 product levels, largest end-to-peak drop 10.29 points
- E3: NaN-divergence brackets for 14 (beta, lambda) pairs; median tightness 2.88x; median eta*lambda at boundary 0.431
- E4: all 6 settings complete
- E5b: 16 mis-specified runs analysed
- E6b: 33 runs
- E7a: final distances lambda=0/0.001/0.01 = 7.78/7.06/9.05
- E7b: 6 equilibrium runs
- E7c: 40 BN-ablation runs
- E3-LMAX from e3_hessian.csv (n=12 probes)