# Response to Reviewer eC8H

Rating 4 (borderline accept). Scores: Quality 3, Clarity 3, Significance 4,
Originality 4.

We thank the reviewer for recognising that weight decay is an under-studied
hyperparameter and for a limitation statement that is, in our view, the sharpest
challenge in the whole review set. We address it first, with new experiments.

---

## 1. The main limitation: "it is enough to tune the step size"

The reviewer's argument is that if `lambda * eta` should scale as `1/T`, then
one can fix `lambda`, tune `eta`, and never think about weight decay again.

We take this in two parts. First: even at fixed training length, fixing
`lambda` and tuning only `eta` is *not* equivalent to tuning both — that is
1a and 1b below, and those measurements stand. Second: the submitted paper
suggested the coupling also moves with `T`. We tested that directly (1c); the
`1/T` dependence is **not** supported, and we report that as a negative
result rather than lean on it.

### 1a. A fixed weight decay cannot track the best achievable accuracy

Using the dense 8 x 9 learning-rate by weight-decay grid we already had on
ResNet-18 / CIFAR-100 (two seeds, 100 epochs, 144 runs), we computed the
*envelope*: the best accuracy reachable at each learning rate when weight decay
is allowed to move with it.

- The envelope is nearly flat: it varies by only **2.8 points** across two
  decades of learning rate. Coupling `lambda` to `eta` buys robustness to the
  learning rate.
- The common default `lambda = 5e-4` gives up **1.44 points on average and 3.77
  points at its worst learning rate** relative to that envelope.
- Crucially, *no* fixed weight decay does better. The best possible fixed choice
  still gives up 3.77 points somewhere in the range. The full table over nine
  candidate values is in the response figure.

So the strategy "fix `lambda`, tune `eta`" is not equivalent to tuning both. It
is dominated, and the size of the loss is several points, not a rounding error.

Fitting the optimum directly, `log lambda*` against `log eta` has slope
**-0.87, 95% bootstrap interval [-0.94, -0.75]**. A redundant `lambda` would
give slope 0.

### 1b. Accuracy is not flat along a line of constant `eta * lambda`

If only the product mattered, one could slide along `eta * lambda = const` at no
cost. Experiment E2b walks that line over two decades of `eta`, holding the
product at its optimal value. Accuracy falls by `10.3` points at the
ends, staying within one point of its peak only over `a factor of 5 in eta`.

The reason is that the product is not the only constraint. Our analysis gives a
*second*, one-sided constraint that prior work does not: weight decay tightens
the admissible learning rate to `eta <= 2/(2*lambda + L)`. Small `eta` fails to
make progress within the budget; large `eta` crosses that boundary. The good
region is the intersection of a band and a ceiling, which is two-dimensional
information, so two knobs are genuinely needed.

### 1c. Training length: two constraints, not redundancy

The submitted paper implied the optimum should move as `1/T`. We ran the
missing sweep at both `η = 0.1` and `η = 0.02`.

At `η = 0.1` the grid argmax sits at `10^{-3}` for every
`T ∈ {25, 50, 100, 200}` (interp slope `-0.226`, CI `[-0.28, -0.17]`). Read
alone, that looks like a constant-product rule. It is not the whole picture:

- Soft (accuracy-weighted) `λ*` inside 1 point of the peak still drifts down
  with `T` (interpolated peak `0.0012 → 0.000877 → 0.000737`).
- At `η = 0.02` the timescale reappears: slope `-0.61 [-1.34, -0.32]`, with
  `λ*` moving `5.1 → 3.3 → 1.3 × 10^{-3}` from `T = 25` to `200`. At `T = 25`
  the cross-`η` ratio is ≈ 4.3 against a predicted factor of 5 from `λ ∝ 1/η`.

Theory: the SGD-WD stability bound is itself `T`-independent; `λ ∝ 1/(η T)`
comes from matching it to SWA. Rotational equilibrium supplies a floor
`η λ ≳ P_*`. The workable rule is `λ* ≈ max(C/S, P_*/η)`. Large `η` pins the
optimum to the floor (headline grid); small `η` lets the timescale bind.
So a fixed `λ` is *not* justified by the training-length axis — it only looks
that way if one samples on the floor. Limitation section rewritten around
1a–1b plus this two-constraint reading.

---

## 2. Why uniform stability, and how it translates into generalization

Uniform stability bounds the expected generalization gap directly: if swapping
one training example changes the learned predictor's loss by at most `epsilon`
uniformly, then `E[test - train] <= epsilon` (Bousquet and Elisseeff; Hardt et
al.). So it is not a proxy for generalization, it is a bound on it. What it does
*not* do is capture the optimization side, which is why our story needs the
learning-rate ceiling as well as the stability bound.

We take the reviewer's question as asking for evidence that this mechanism is
actually operating in a deep network rather than only in the convex proof.
Experiment E7a supplies it: we train pairs of networks on datasets differing in
exactly one example, with identical initialization and batch ordering, and track
the parameter divergence `||theta_t - theta'_t||`. The theory predicts growth
with `t` without weight decay and saturation with it. Measured:
`[[E7-DIVERGENCE-RATIO]]`, with the weight-decay run plateauing at
`[[E7-PLATEAU]]`.

---

## 3. Is SGDM better at generalizing? What does the momentum parameter do?

We can answer this from sweeps we already have, plus one new one.

Re-analysing the existing momentum sweeps on ResNet-18 / CIFAR-100 (100 epochs,
`beta` from 0 to 0.99, learning rate re-tuned at every `beta`):

- **Without weight decay**, peak accuracy varies by **0.25 points** across
  `beta` in [0, 0.8]. Momentum does not generalize better.
- **With `lambda = 2e-3`**, peak accuracy varies by **1.38 points** across
  `beta` in [0, 0.95]. At `beta = 0.99` it collapses to 73.68%, which is the
  effective step `eta/(1-beta)` running past the stability boundary rather than
  a generalization effect.
- What momentum *does* change is where the optimum sits. Fitting `log eta*`
  against `log(1-beta)` gives slope **1.24, interval [0.80, 2.32]** without
  weight decay and **0.72, interval [0.58, 0.93]** with it, against a prediction
  of 1 from the effective-step argument `eta_eff = eta/(1-beta)`.

This matches the structure of our bounds, where momentum enters through a factor
that rescales the admissible learning rate but leaves the `1/(lambda * n)`
stability term untouched. In other words: momentum moves the optimum, weight
decay moves the achievable accuracy. Over the same sweeps, weight decay is worth
about five points at every value of `beta`.

Experiment E6b adds the missing arm: `beta` in {0, 0.5, 0.9, 0.99} crossed with
a coupled and a zero weight decay, logging *training* accuracy, which our older
CSVs did not record, so the train-test gap itself is measurable. Predictions
under test: `lambda*` scales as `(1-beta)` (measured slope
`[[E6B-LAMBDA-SLOPE]]`), and the gap ordering SGDM+WD < SGD+WD < SGD
(`[[E6B-GAP-SGDM]]` against `[[E6B-GAP-SGD]]`).

---

## 4. Summary of changes

- Training-length × learning-rate sweep (1c): timescale recovered at small `η`;
  headline large-`η` grid sits on an equilibrium floor — two constraints, not a
  concession.
- Envelope analysis: cost of a fixed `λ` at fixed `T` (1a).
- Iso-product walk: product alone is not sufficient (1b), drop of `10.3` points.
- Stability probe (2); momentum arm with train-test gaps (3).
- Limitation rewritten: `λ` is not absorbed into `η`, and across `T` it tracks
  `max(C/S, P_*/η)` rather than a single fixed default.
