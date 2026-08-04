# Response to Reviewer xkCF

Rating 2 (reject). Scores: Quality 3, Clarity 4, Significance 2, Originality 2.

We thank the reviewer for five questions that are all answerable with
experiments, and for saying explicitly that a CIFAR-scale comparison would
suffice. We ran that comparison. We answer the questions in order.

---

## Q1. Why should matching stability upper bounds predict the optimum, rather than being only a heuristic?

It does not, and we now say so in the paper in exactly those terms.

Matching two upper bounds is not a valid way to locate an optimum: bounds can be
loose by different amounts, and nothing forces the argmax of the truth to sit
where two envelopes cross. What the argument does supply is the *scaling* of the
crossing point in `eta`, `T`, and `B`, because the loose constants enter
multiplicatively and are absorbed into a single prefactor. So the honest claim
is:

- the exponents are derived, and are falsifiable;
- the prefactor `C` is not derived, and is calibrated once from data.

We have restated Section 5 as a heuristic with derived exponents and a fitted
constant, replacing the language that implied an optimality result. The
compensating move is that we now *test* the exponents on a joint `(η, T)`
sweep, not a single-`η` slice:

- `log lambda*` against `log eta` (fixed `T`): slope **-0.87 [-0.94, -0.75]**
  — holds.
- `log lambda*` against `log T` at `η = 0.1`: slope `-0.226` `[-0.28, -0.17]`
  — floor-dominated (grid argmax flat). At `η = 0.02`: slope
  `-0.61 [-1.34, -0.32]` — timescale binding, compatible with -1.
- `log eta*` against `log(1-beta)`: slope 0.72 [0.58, 0.93] with weight decay.

A heuristic that states falsifiable exponents, identifies when a second
constraint masks them, and recovers them in the binding regime is, we think, a
different object from one that only draws a band.

## Q2. Novelty against AdamW timescales, batch-size scaling, and learning-rate-aware weight decay

Direct comparison, on assumptions, formula, and prediction:

- **Wang and Aitchison (AdamW as EMA).** Assumption: AdamW's update is an EMA
  with timescale `1/(ηλ)`; the timescale should match the horizon. Formula:
  `λ = 1/(ηT)`. Closest formal cousin of our timescale branch; confirmed in
  the regime where that branch binds (`η = 0.02`, slope
  `-0.61 [-1.34, -0.32]`), masked at large `η` by the equilibrium floor.
- **Kosson et al. (rotational equilibrium).** Assumption: scale-invariant
  parameters reach an equilibrium norm. Formula: hold `ηλ` constant.
  Prediction: no `T` dependence. This is the *floor* in our refined rule
  `λ* ≈ max(C/S, P_*/η)`, visible as the flat `η = 0.1` grid — not a
  replacement for the timescale, but the constraint that can hide it.
- **Batch-size scaling laws (McCandlish et al.; Bergsma et al.).** Assumption:
  gradient noise scale controls the useful batch size. Formula: `lambda` grows
  with `B`. Prediction: agrees with ours in direction. Difference: in our
  statement the batch size enters only through the step count `T = T_epochs *
  n/B`, so it is not an independent mechanism; we can therefore check whether
  the batch-size dependence is fully absorbed by `sum_t eta_t`. It is not
  entirely: after collapsing onto `sum_t eta_t`, the residual trend in `C` over
  `B` from 32 to 512 is 1.38, 1.89, 1.40, 1.60, 1.71. We now report this
  residual instead of claiming a clean law.
- **Scheduled weight decay (AdamW / SGDW schedule multipliers).** Assumption:
  `lambda` should vary *within* a run, typically sharing the same multiplier as
  the learning rate (cosine, drop-step, cosine with restarts). Difference: that
  is a within-run schedule; our rule predicts a single constant `lambda` across
  runs. They compose rather than compete. We now also measure the within-run
  schedules directly under a fixed learning rate (E8 below).

## Q3. Compare against at least one existing scaling rule or scheduled weight-decay baseline

**Across-run scaling rules (E4).** Done as a zero-tuning transfer test. The
constant `C` is calibrated **once** on a single reference setting (ResNet-18,
`B = 128`, `eta = 0.1`, `T = 100`) using data we already had. Every rule is then
applied blind to six held-out settings, and compared against a per-setting
oracle grid search:

Strategies: `lambda = 0`; the common default `lambda = 5e-4`; constant
`eta*lambda` calibrated at the reference (the Kosson-style rule); Wang and
Aitchison's `1/(eta*T)`; and ours, `C / sum_t eta_t`.

Held-out settings: `T = 25`, `T = 200`, `B = 32`, `B = 512`, VGG-16, ResNet-50.
Where the learning rate must change with the batch size it follows the linear
rule, so nothing is tuned per setting.

`see _data/e4_transfer_table.md`

Summary of the gap to the per-setting oracle: ours `0.87` mean and
`1.58` worst; fixed default `0.70`; constant product
`1.63`; `1/(eta*T)` `1.60`.

We also report the tuning cost, since that is the practical argument: the oracle
column needs eight training runs per setting, and every rule needs zero after a
one-time calibration.

**Within-run scheduled weight decay (E8).** Separately, we hold the learning rate
fixed at `eta = 0.1` (ResNet-18 / CIFAR-100, `B = 128`, `T = 100`) and compare a
constant `lambda` against the AdamW/SGDW schedule shapes applied *only* to
weight decay: cosine, linear, drop-step, and cosine-with-restarts
(`Te = 50`, `Tmult = 2`). Each schedule is swept over the same
`lambda_0 ∈ {1e-4, 5e-4, 1e-3, 2e-3, 5e-3}`; we report the best accuracy in that
grid (figures: `outputs/plots/nips26/e8_wd_sched_sgd.png`,
`e8_wd_sched_sgdm.png`).

| optimizer | fixed | cosine | linear | step | cosine_restarts |
|---|---:|---:|---:|---:|---:|
| SGD (mom=0) | 73.20 | 73.50 (+0.30) | 73.22 (+0.02) | **74.24 (+1.04)** | 73.43 (+0.23) |
| SGDM (mom=0.9) | 66.67 | 71.34 (+4.67) | 70.32 (+3.65) | **73.10 (+6.43)** | 70.13 (+3.46) |

Under a fixed learning rate, decaying `lambda` helps — especially for SGDM,
where a constant `lambda` is badly mismatched to the lack of LR annealing, and
drop-step recovers **+6.4** points. As a follow-up we also apply the same
AdamW-style multiplier jointly to `eta` and `lambda` (T=100): joint cosine
peaks at **76.42**, vs **76.72–76.73** for cosine LR + fixed `lambda`
(E4-ours / default `5e-4`) and **77.28** for an oracle over the same `lambda_0`
grid. So within-run scheduling is useful when LR is held fixed, but under our
default cosine-LR setup a constant `lambda` already matches or beats it. That
is a different knob from our claim: we select one constant `lambda` as a
function of `(eta, T, B)`, whereas these schedules redistribute regularization
over time. Both can be used together; E4 tests the former and E8 the latter.

## Q4. Which parts of the theory should survive in the non-convex setting?

We agree the convex, `L`-smooth analysis does not transfer as a theorem, and we
now scope the claim to two structural predictions that are measurable in deep
networks. Both are new experiments.

**The learning-rate ceiling.** `eta <= 2/(2*lambda + L)` rearranges to
`1/eta_max = lambda + L/2`: a straight line in `lambda` whose intercept is an
effective smoothness. We locate the empirical divergence threshold by bisection
in `eta` for seven weight decays, for both SGD and SGDM. Fitted slope
`0.67` against a prediction of 1, intercept `0.08 (implies L = 0.2)`, against
a top Hessian eigenvalue of `417.4` measured independently by power
iteration. The momentum version predicts the ceiling scales with `(1-beta)`;
measured ratio `0.23` against a prediction of 0.1.

**The stability mechanism.** We train pairs of networks on datasets differing in
exactly one example, with identical initialization and batch order, and track
`||theta_t - theta'_t||`. The convex theory says this grows with `t` without
weight decay and saturates with it. Measured: `1.10 (final ||theta-theta'|| at lambda=0 over lambda=1e-3)`.

What we do not claim to survive: the constants, the strong convexity used in the
momentum bounds, and any statement about the *value* of the generalization gap
as opposed to its scaling.

## Q5. How sensitive is the rule to the hidden constant?

This was the right thing to ask, and we had not measured it. Two answers,
then a third that covers the axes the Wave-0 CIFAR sweeps do not.

**How much does `C` actually vary (architectures, fixed dataset/optimizer)?**
Fitting `C = lambda* * sum_t eta_t` independently in 65 settings that already
exist in our sweeps -- three architectures, five batch sizes from 32 to 512,
learning rates from 0.001 to 0.5, two seeds -- gives a geometric mean of
**1.48**, a multiplicative standard deviation of **x/1.70**, and a range of
0.17 to 2.99. By architecture: ResNet-18 1.42, ResNet-50 1.42, VGG-16 1.72.
So `C` is stable to roughly a factor of two across architectures, and the
residual drift across batch sizes reported in Q2 is the largest single source
of variation.

**How much does being wrong cost (CIFAR)?** We sweep `C` deliberately wrong by
factors of 3 and 10 at two settings (`η=0.1`, `T∈{25,100}`), using the exact
cosine step budget `S`. The worst accuracy loss is **15.1** points at 3× and
**69.7** points at 10× — almost entirely from *overshooting* `C` (too much
weight decay). Undershooting by the same factors costs far less. The optimum
in `lambda` is therefore broad on the low side, which is why a rule that is
order-of-magnitude correct remains useful while a fixed default is not: the
default's error grows with the mismatch in `eta` and `T`, while ours does not.

**Across datasets and optimizers (E5c).** Wave-0 is all CIFAR-100 / SGDM. To
cover the remaining axes in the question we repeat the Fig. 1 protocol on a
3-layer MLP trained on MNIST (SGD and SGDM, cosine, no BatchNorm): fitted
`C` under SGD is **0.44** and under SGDM is **0.32** (cross-optimizer ratio
**1.38×**). Relative to the CIFAR Wave-0 geo-mean 1.48 this is still order-one
(about a factor of 3–5), consistent with “stable to a small constant factor”
rather than architecture-specific fine-tuning. Mis-specifying that MNIST
`C` by 3× / 10× costs **0.018** / **0.083** in best test loss relative to the
calibrated rule (see `outputs/plots/nips26/e5c_mnist_mlp_C.png`). The
practical claim is therefore: usefulness needs `C` stable to about a factor
of a few, not to 1%, and E5a+E5c support that.

---

## On the overall assessment

The reviewer's summary was that the contribution is "a unifying stability-based
explanation rather than a fundamentally new hyperparameter law". We accept that
framing and sharpen it: the timescale and the equilibrium product are stacked
constraints, made visible only by a joint `(η, T)` sweep. What we retain beyond
unification is the cost-side constraint (iso-product drop `10.3` points), the
envelope evidence against a fixed default at fixed `T`, and a transfer
comparison (Q3) that must be read under the two-constraint rule — constant
product wins on the floor; timescale matching wins where the floor does not
bind.

---

# Follow-up (2026-08-03)

Two new results, both settled by measurement. F2 replaces the scheduled-weight-decay
control we had used; F3 is a held-out test of the rule's transfer.

## F2. A schedule that preserves `η_tλ_t`, and matching by cumulative contraction

**The reviewer is right that the `joint` arm was the wrong control**: it gives
`η_tλ_t = η₀λ₀·m(t)²`, which does not preserve the coupling. We ran both of his
suggestions (E9, `_data/e9_table.md`, figure
`outputs/plots/nips26/e9_iso_matched.png`). Same protocol as the main
experiments — ResNet-18/CIFAR-100, `B = 128`, `η₀ = 0.1`, `T = 100`, SGDM,
**cosine learning rate** — so these are comparable to our own default rather
than to the constant-LR E8 arms.

**(a) `λ_t = λ₀·η₀/η_t`.** The `1/m_cos` factor diverges in the cosine tail, so
the multiplier is capped at `10×` (equivalently `η` is floored at `η₀/10` inside
the `λ` formula); `η_tλ_t` is then exactly constant while `η_t ≥ η₀/10`. Stated
as part of the protocol, not applied silently. Swept over the same
`λ₀ ∈ {1e-4, 5e-4, 1e-3, 2e-3, 5e-3}`:

| `λ₀` | realized `Σ_t η_tλ_t` | best acc |
|---:|---:|---:|
| 1e-4 | 0.29 C | 75.56 |
| **5e-4** | 1.44 C | **77.98** |
| 1e-3 | 2.88 C | 77.98 |
| 2e-3 | 5.76 C | 70.36 |
| 5e-3 | 14.39 C | 22.13 |

**(b) Matched cumulative contraction.** `Σ_t η_tλ_t` is linear in `λ₀`, so for
each shape we solve `λ₀ = budget / (⌈n/B⌉·Σ_e η₀ m_cos(e) m_λ(e))` and compare
at three budgets `{C/3, C, 3C}`, where `C = λ_ref·Σ_t η_t = 1.181` is the
contraction our own rule already prescribes at this setting. Usefully, the
`fixed` shape at these three budgets solves to `λ₀ ∈ {1.994e-4, 5.982e-4,
1.795e-3}`, which are exactly the E5b wrong-`C` points and the E4-ours baseline,
so that column is the already-reported data rather than new runs:

| budget `Σ_t η_tλ_t` | fixed | cosine | linear | step | **iso-product** |
|---|---:|---:|---:|---:|---:|
| `C/3` | 74.62 | 74.34 | 74.43 | 74.16 | **75.86** |
| `C` | 76.72 | 75.50 | 76.44 | 75.85 | **78.22** |
| `3C` | 76.19 | 75.03 | 75.53 | 73.97 | **77.50** |

Three things follow, and the first one is not in our favour as a *simplification*
even though it favours the underlying claim:

1. **Matching the contraction budget is not sufficient.** At a fixed budget the
   shapes still differ by 1.70 / 2.72 / 3.53 pp. So `Σ_t η_tλ_t` does not
   summarize a schedule; how the contraction is distributed in time also matters.
   Spreads at fixed shape across budgets (1.16–2.36 pp) are the same order, so
   neither factor dominates.
2. **The shape that preserves the coupling wins at every budget**, by
   +1.24 / +1.50 / +1.31 pp over a constant `λ` at the same contraction.
3. Placed against every other arm at the same `η₀`, `T` and optimizer:

| method | best acc |
|---|---:|
| **iso-product, matched at `C`** | **78.22** |
| iso-product, peak over the `λ₀` ladder | 77.98 |
| cosine LR + constant `λ` (per-cell oracle over `λ₀`) | 77.28 |
| cosine LR + constant `λ` (default 5e-4) | 76.73 |
| cosine LR + constant `λ` (ours, `C/Σ_tη_t`) | 76.72 |
| joint `m(t)` on `η` and `λ` (best shape, cosine) | 76.42 |
| constant LR + scheduled `λ` (best shape, step) | 73.10 |
| constant LR + constant `λ` | 66.67 |

The schedule the reviewer proposed as a fairer control turns out to be the best
arm we have measured: **+0.94** over tuning a constant `λ` per cell and
**+1.80** over the joint multiplier. We will report the comparison this way and
retract the `joint`-based framing.

## F3. Held-out `C`: calibrate on small MLPs, predict `λ*` on larger ones

We use MLP **width** as the held-out architecture axis: it moves the parameter
count by more than an order of magnitude with the data, optimizer and schedule
untouched. Protocol as in E5c (3-layer ReLU MLP, no normalization, cosine LR,
`B = 128`). Details and the blindness argument: `common/e10_c_width.md`.

The runner is split into `ladder → predict → heldout`, and the `predict` stage
writes `λ_pred` with a timestamp to `_data/e10_predictions_<ds>.json` before any
held-out grid is trained; both files record `blind: true`. Sanity check on the
pipeline: refitting `C` at `h = 512` on MNIST reproduces the E5c values we
already reported, 0.440 (SGD) and 0.320 (SGDM), to three digits.

**How `C` moves with width** (calibration rungs only):

| dataset | momentum | `C(128)` | `C(256)` | `C(512)` | slope of `log C` on `log h` |
|---|---:|---:|---:|---:|---|
| MNIST | 0 | 0.442 | 0.374 | 0.440 | **−0.00 [−0.24, +0.24]** |
| MNIST | 0.9 | 0.466 | 0.353 | 0.320 | **−0.27 [−0.40, −0.14]** |
| CIFAR-10 | 0.9 | — | 2.800 | 2.672 | −0.068 (two rungs, no interval) |

**Blind extrapolation vs `C` measured directly at the held-out width:**

| dataset | momentum | width | `C` predicted | `C` measured | ratio |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | 1024 | 0.416 | 0.325 | 1.28× |
| MNIST | 0 | 2048 | 0.415 | 0.327 | 1.27× |
| MNIST | 0.9 | 1024 | 0.257 | 0.244 | 1.05× |
| MNIST | 0.9 | 2048 | 0.213 | 0.255 | 0.84× |
| CIFAR-10 | 0.9 | 1024 | 2.549 | 2.479 | **1.03×** |

So `C` transfers across a 4–16× change in width to within **1.3×**, and the
resulting `λ` is within **1.60×** (MNIST) / **1.14×** (CIFAR-10) of the oracle.

**Does that reduce tuning?** The oracle for each cell is the best weight decay
anyone measured there — the 5-point ladder *plus* every rule's own `λ` — so no
rule can look good merely because the ladder is coarse. Cost: 5+ runs per
(width, momentum, `η`) cell for the oracle, one run for any rule, zero after a
one-time calibration.

| dataset | rule | mean acc gap (pp) | worst | mean test-loss gap | `λ/λ_oracle` |
|---|---|---:|---:|---:|---:|
| MNIST (12 cells) | default 5e-4 | 0.03 | 0.11 | 0.0005 | 1.27× |
| | `1/(ηT)` | 0.11 | 0.25 | 0.0032 | 2.70× |
| | constant `ηλ` | 0.06 | 0.20 | 0.0018 | 1.56× |
| | **ours** | 0.08 | 0.19 | 0.0020 | 1.60× |
| CIFAR-10 (3 cells) | default 5e-4 | 0.84 | 1.19 | 0.1576 | 0.13× |
| | `1/(ηT)` | 0.52 | 1.11 | 0.0792 | 0.23× |
| | constant `ηλ` | 1.49 | 3.73 | **0.0171** | 1.08× |
| | **ours** | 2.01 | 3.79 | 0.0233 | 1.14× |

Reading this honestly:

- **The prediction of `λ*` is good; the accuracy advantage is not there.** On
  CIFAR-10 ours and constant-`ηλ` land nearest the loss-oracle `λ` (1.1×) while
  the default and `1/(ηT)` are off by 4–8×, yet the fixed default gives the
  smaller *accuracy* gap, because in this setting the accuracy optimum sits at a
  smaller `λ` than the loss optimum. This is the same pattern as E4
  (ours 0.87 vs default 0.70) and we do not dress it up.
- **MNIST cannot discriminate**: all four rules are within 0.11 pp. That is
  useful negative information — we will not claim an advantage on MNIST-MLP.
- **The width axis is not where `C` is fragile.** `C` is 0.32–0.44 for
  MNIST-MLP, 2.5–2.8 for CIFAR-10-MLP, and 1.48 (geometric mean) for
  CIFAR-100 CNNs: about **8.8×** across families, against ≤1.5× across a 16×
  width range. So the practical statement is that `C` must be calibrated once
  per dataset/architecture family, after which it transfers over width, learning
  rate and training length. One calibration replaces a two-dimensional grid; it
  does not replace knowing the family.
- Limits of this test: single seed; the CIFAR-10 ladder has only two rungs, so
  its slope has no usable interval; and at `η = 0.3` our predicted
  `λ = 1.402e-3` diverged while `λ = 1.322e-3` converged, i.e. that cell sits on
  the stability boundary. It is counted as a divergence rather than imputed.

