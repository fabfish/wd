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
compensating move is that we now *test* the exponents, including the one that
fails:

- `log lambda*` against `log eta` (fixed `T`): slope **-0.87 [-0.94, -0.75]**
  on a dense 8 x 9 grid, two seeds — holds.
- `log lambda*` against `log T` (fixed `eta`): slope `-0.226`
  `[-0.28, -0.17]` on a dense ladder over `T ∈ {25, 50, 100, 200}` — **fails**
  relative to the predicted -1; grid argmax is identical at every `T`. We
  report this as a negative result.
- `log eta*` against `log(1-beta)`: slope 0.72 [0.58, 0.93] with weight decay.

A heuristic that states falsifiable exponents and then reports which of them
survive is, we think, a different object from one that only draws a band, and
we have reorganised the paper around that distinction.

## Q2. Novelty against AdamW timescales, batch-size scaling, and learning-rate-aware weight decay

Direct comparison, on assumptions, formula, and prediction:

- **Wang and Aitchison (AdamW as EMA).** Assumption: AdamW's update is an
  exponential moving average with timescale `1/(eta*lambda)`; the timescale
  should match the training horizon. Formula: `lambda = 1/(eta*T)`. Prediction:
  identical exponents to ours in `eta` and `T`. Our training-length sweep
  rejects that `1/T` dependence in this regime (slope `-0.226`); we
  therefore no longer present Wang--Aitchison as independently confirmed by
  our vision experiments, only as the closest formal cousin of the submitted
  claim.
- **Kosson et al. (rotational equilibrium).** Assumption: scale-invariant
  parameters reach an equilibrium norm. Formula: hold `eta*lambda` constant.
  Prediction: **no dependence on `T`**. This is the account our
  training-length sweep supports: measured slope `-0.226`
  `[-0.28, -0.17]` against their 0 and our -1. We say so directly.
- **Batch-size scaling laws (McCandlish et al.; Bergsma et al.).** Assumption:
  gradient noise scale controls the useful batch size. Formula: `lambda` grows
  with `B`. Prediction: agrees with ours in direction. Difference: in our
  statement the batch size enters only through the step count `T = T_epochs *
  n/B`, so it is not an independent mechanism; we can therefore check whether
  the batch-size dependence is fully absorbed by `sum_t eta_t`. It is not
  entirely: after collapsing onto `sum_t eta_t`, the residual trend in `C` over
  `B` from 32 to 512 is 1.38, 1.89, 1.40, 1.60, 1.71. We now report this
  residual instead of claiming a clean law.
- **Scheduled weight decay (Xie et al.).** Assumption: `lambda` should vary
  during training. Difference: their schedule is within a run, ours is a
  prediction across runs. These compose rather than compete, and we say so.

## Q3. Compare against at least one existing scaling rule or scheduled weight-decay baseline

Done, as a zero-tuning transfer test. The constant `C` is calibrated **once** on
a single reference setting (ResNet-18, `B = 128`, `eta = 0.1`, `T = 100`) using
data we already had. Every rule is then applied blind to six held-out settings,
and compared against a per-setting oracle grid search:

Strategies: `lambda = 0`; the common default `lambda = 5e-4`; constant
`eta*lambda` calibrated at the reference (the Kosson-style rule); Wang and
Aitchison's `1/(eta*T)`; and ours, `C / sum_t eta_t`.

Held-out settings: `T = 25`, `T = 200`, `B = 32`, `B = 512`, VGG-16, ResNet-50.
Where the learning rate must change with the batch size it follows the linear
rule, so nothing is tuned per setting.

`[[E4-TABLE]]`

Summary of the gap to the per-setting oracle: ours `[[E4-OURS-MEAN]]` mean and
`[[E4-OURS-WORST]]` worst; fixed default `[[E4-DEFAULT-MEAN]]`; constant product
`[[E4-KOSSON-MEAN]]`; `1/(eta*T)` `[[E4-WANG-MEAN]]`.

We also report the tuning cost, since that is the practical argument: the oracle
column needs eight training runs per setting, and every rule needs zero after a
one-time calibration.

## Q4. Which parts of the theory should survive in the non-convex setting?

We agree the convex, `L`-smooth analysis does not transfer as a theorem, and we
now scope the claim to two structural predictions that are measurable in deep
networks. Both are new experiments.

**The learning-rate ceiling.** `eta <= 2/(2*lambda + L)` rearranges to
`1/eta_max = lambda + L/2`: a straight line in `lambda` whose intercept is an
effective smoothness. We locate the empirical divergence threshold by bisection
in `eta` for seven weight decays, for both SGD and SGDM. Fitted slope
`[[E3-SLOPE]]` against a prediction of 1, intercept `[[E3-INTERCEPT]]`, against
a top Hessian eigenvalue of `[[E3-LMAX]]` measured independently by power
iteration. The momentum version predicts the ceiling scales with `(1-beta)`;
measured ratio `[[E3-MOM-RATIO]]` against a prediction of 0.1.

**The stability mechanism.** We train pairs of networks on datasets differing in
exactly one example, with identical initialization and batch order, and track
`||theta_t - theta'_t||`. The convex theory says this grows with `t` without
weight decay and saturates with it. Measured: `[[E7-DIVERGENCE-RATIO]]`.

What we do not claim to survive: the constants, the strong convexity used in the
momentum bounds, and any statement about the *value* of the generalization gap
as opposed to its scaling.

## Q5. How sensitive is the rule to the hidden constant?

This was the right thing to ask, and we had not measured it. Two answers.

**How much does `C` actually vary?** Fitting `C = lambda* * sum_t eta_t`
independently in 65 settings that already exist in our sweeps -- three
architectures, five batch sizes from 32 to 512, learning rates from 0.001 to
0.5, two seeds -- gives a geometric mean of **1.48**, a multiplicative standard
deviation of **x/1.70**, and a range of 0.17 to 2.99. By architecture:
ResNet-18 1.42, ResNet-50 1.42, VGG-16 1.72. So `C` is stable to roughly a
factor of two across architectures, and the residual drift across batch sizes
reported in Q2 is the largest single source of variation.

**How much does being wrong cost?** We sweep `C` deliberately wrong by factors
of 3 and 10 at two settings: `[[E5B-3X]]` and `[[E5B-10X]]` accuracy points
respectively. The optimum in `lambda` is broad, which is why a rule that is
order-of-magnitude correct is useful and a fixed default is not: the default's
error grows with the mismatch in `eta` and `T`, while ours does not.

---

## On the overall assessment

The reviewer's summary was that the contribution is "a unifying stability-based
explanation rather than a fundamentally new hyperparameter law". After this
round we would not dispute that framing, and we go further: the one prediction
that would have separated us from the closest prior account (`λ ∝ 1/T`) fails
the training-length test, and we report that as a negative result. What we
retain is the cost-side constraint (accuracy is not flat along an iso-product
line: `10.3` points), the envelope evidence that a fixed default is
dominated at fixed `T`, and a head-to-head transfer comparison (Q3) in which
the constant-product rule is the strongest baseline once `T` varies — consistent
with E1.
