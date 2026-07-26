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

This is correct at a *fixed* training length and *fixed* everything else, and we
should have said so. It stops being correct as soon as either changes, for two
distinct reasons. We now have measurements for both.

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
product at its optimal value. Accuracy falls by `[[E2B-ISO-DROP]]` points at the
ends, staying within one point of its peak only over `[[E2B-ISO-RANGE]]`.

The reason is that the product is not the only constraint. Our analysis gives a
*second*, one-sided constraint that prior work does not: weight decay tightens
the admissible learning rate to `eta <= 2/(2*lambda + L)`. Small `eta` fails to
make progress within the budget; large `eta` crosses that boundary. The good
region is the intersection of a band and a ceiling, which is two-dimensional
information, so two knobs are genuinely needed.

### 1c. The training length is the case where a fixed `lambda` is simply wrong

See the shared point S1: at fixed `eta = 0.1`, varying `T` over {25, 100, 200}
moves the optimal weight decay from `[[E1-T-LAMBDA-25]]` to
`[[E1-T-LAMBDA-200]]`, with fitted slope `[[E1-T-SLOPE]]` against `log T`
(95% interval `[[E1-T-CI]]`). A practitioner who changes the epoch budget and
leaves `lambda` at its default is off by that factor, and compensating through
`eta` alone is exactly the move that 1b shows is not free.

We have rewritten the limitation section around this: the claim is not that
`lambda` must be tuned by grid search, it is that `lambda` must *move*, and our
rule says where to move it without any search.

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

- New training-length experiment establishing the `1/T` dependence (S1).
- New envelope analysis quantifying what a fixed weight decay costs (1a).
- New iso-product experiment showing the product is not sufficient (1b).
- New empirical stability probe in a deep network (2).
- Momentum analysis and a new momentum arm with train-test gaps (3).
- The limitation section rewritten so that it states the scope of the redundancy
  argument accurately instead of overclaiming.
