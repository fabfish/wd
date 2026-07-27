# Response to Area Chair vXFZ (meta-review)

We thank the AC for a meta-review that isolates four concrete weaknesses. We
take them in order. Each is answered with either a change of claim or a new
experiment; three of the four are new experiments.

One correction of record before the substance: the paper is a study of weight
decay in SGD and SGD with momentum, with ResNet-18, ResNet-50 and VGG-16 on
CIFAR-100 as the main evidence and a small language-model experiment as a
secondary observation. The meta-review describes it as a study of weight decay
for Transformers, which we suspect is a slip, and we mention it only because the
choice of evidence is relevant to weakness 3.

---

## 1. "It essentially recovers existing results through a different approach, and there may be more related results not acknowledged"

We accept both halves and have acted on both.

**Missing related work.** Reviewer SijV was right that we omitted Kosson et al.
on rotational equilibrium (arXiv:2305.17212, arXiv:2510.19093), which is the
closest prior account of why the optimum keeps `eta*lambda` roughly constant.
This is now cited and discussed, along with Xie et al. on scheduled weight
decay.

**Recovery of existing results.** Our honest accounting is that the stability
machinery is Hardt et al.'s, that a strongly convex regularizer converting an
`O(eta*T/n)` bound into a `T`-independent one is standard, and that the
resulting `lambda ~ 1/(eta*T)` coincides with Wang and Aitchison's AdamW
timescale. We have rewritten the contribution statement to say this rather than
leaving the overlap for the reader to find.

What we retain is narrower, and one part of the submitted claim is now a
**negative result**. Prior accounts explain why the product should be held
constant; the stability analysis also supplies a one-sided constraint that the
product picture alone does not (`eta <= 2/(2*lambda + L)`), with the testable
consequence that accuracy falls along an iso-product line (measured drop
`10.3` points). The two accounts disagree on training length:
equilibrium predicts `eta*lambda = const`, our argument predicted
`eta*lambda ∝ 1/T`.

We ran that discriminator: ResNet-18 / CIFAR-100, dense `lambda` ladder,
`T ∈ {25, 50, 100, 200}`. Fitted slope `-0.226`, interval
`[-0.28, -0.17]`; grid argmax identical at every `T`. The data favour Kosson et
al. on this axis. We report the finding as a negative result against the
submitted `1/T` claim and reframe the related work accordingly.

## 2. "The theoretical analysis may not support the experiments; it relies on assumptions such as L-smoothness"

We agree that the convex, `L`-smooth analysis does not transfer to deep networks
as a theorem, and we have scoped the claim accordingly. Rather than assert that
it transfers, we now measure the two structural predictions that can be checked
directly.

**The learning-rate ceiling.** We locate empirical thresholds by bisection in
`eta`, counting only NaN/explosion (not under-fitting: at large `lambda` the
loss sticks at `log #classes` without diverging, which is not the L-smooth
ceiling). Explosion brackets are clean at `lambda = 0` and tighten with
momentum (`[[E3-MOM-RATIO]]` against a prediction of 0.1). At large positive
`lambda` the slope-1 test `1/eta_max = lambda + L/2` is not cleanly identified;
we report what we have (`[[E3-SLOPE]]`, `[[E3-INTERCEPT]]`, Hessian
`[[E3-LMAX]]`) and state this limitation rather than claim a verification.

**The stability mechanism itself.** We train pairs of networks on datasets
differing in exactly one example, with identical initialization and batch
ordering, and track `||theta_t - theta'_t||`. Without weight decay the convex
theory predicts growth with `t`; with weight decay, saturation. Measured
`[[E7-DIVERGENCE-RATIO]]`.

These are the parts we claim survive. We explicitly do not claim the constants,
the strong convexity used in the momentum bounds, or any statement about the
value rather than the scaling of the generalization gap.

## 3. "No convincing argument why a practitioner should use this rather than simpler existing approaches"

This was the fairest criticism in the set, and the submission genuinely did not
contain the experiment that would settle it. We have run it as a zero-tuning
transfer test.

The constant is calibrated **once**, on a single reference setting (ResNet-18,
`B = 128`, `eta = 0.1`, `T = 100`), from data we already had. Five strategies
are then applied blind to six held-out settings (`T = 25`, `T = 200`, `B = 32`,
`B = 512`, VGG-16, ResNet-50) and scored against a per-setting oracle grid
search: no weight decay; the common default `5e-4`; a constant `eta*lambda`
calibrated at the reference; Wang and Aitchison's `1/(eta*T)`; and ours.

`[[E4-TABLE]]`

Gap to the oracle: ours `[[E4-OURS-MEAN]]` mean, `[[E4-OURS-WORST]]` worst;
fixed default `[[E4-DEFAULT-MEAN]]`; constant product `[[E4-KOSSON-MEAN]]`;
`1/(eta*T)` `[[E4-WANG-MEAN]]`. The oracle column costs eight training runs per
setting; every rule costs zero after the one-time calibration.

Two supporting numbers, both from data we already had:

- On the dense 8 x 9 grid, coupling `lambda` to `eta` keeps accuracy within
  **2.8 points** across two decades of learning rate, whereas the common default
  `5e-4` gives up **1.44 points on average and 3.77 at its worst**. No fixed
  choice does better than 3.77 worst-case.
- Fitting the prefactor in 65 independent settings (three architectures, five
  batch sizes, two seeds) gives a geometric mean of **1.48** with a spread of
  **x/1.70**, so a single calibration transfers across architectures to within
  about a factor of two, and being wrong by 3x costs `[[E5B-3X]]` points.

## 4. "The discussion about momentum is not complete, and the opportunity to contrast with concurrent work is missed"

Both are addressed.

**Momentum.** Re-analysing sweeps we already had (ResNet-18 / CIFAR-100, `beta`
from 0 to 0.99, learning rate retuned at each `beta`): at its own optimal
learning rate, momentum changes peak accuracy by **0.25 points** without weight
decay (`beta` in [0, 0.8]) and **1.38 points** with `lambda = 2e-3` (`beta` in
[0, 0.95]); the collapse at `beta = 0.99` is the effective step `eta/(1-beta)`
crossing the stability boundary. What momentum changes is where the optimum
sits: `log eta*` against `log(1-beta)` has slope 1.24 [0.80, 2.32] without
weight decay and 0.72 [0.58, 0.93] with it, against a prediction of 1. This is
the structure our bounds have -- momentum rescales the admissible learning rate
and leaves the `1/(lambda*n)` stability term alone -- and the paper now states
it as a result rather than a remark. A new arm (`beta` in {0, 0.5, 0.9, 0.99}
crossed with coupled and zero weight decay, logging training accuracy so the
train-test gap is measurable) tests `lambda* ∝ (1-beta)`: measured
`[[E6B-LAMBDA-SLOPE]]`.

**Concurrent work.** The contrast with rotational equilibrium is now explicit
and, as described under weakness 1, is the discriminating experiment rather than
a discussion point. We also note that the equilibrium mechanism requires scale
invariance; our ablation on networks without normalization (`[[E7-BN]]`) probes
whether the coupling persists where that mechanism does not apply.

---

## What changed, in one place

- New measurements: training-length sweep (**negative** for `λ ∝ 1/T`),
  iso-product walk (accuracy not flat: `10.3` points), zero-tuning
  transfer against four rules, explosion-boundary brackets, momentum arm,
  prefactor sensitivity.
- Reframed: contribution type Negative Results is now load-bearing — we
  withdraw the `1/T` empirical claim and align the training-length axis with
  Kosson et al.; we keep the cost-side / non-redundancy claims at fixed `T`.
  The `sum_t eta_t` restatement resolves the `1/(eta*T)` vs `2/(eta*T)`
  bookkeeping inconsistency (schedule difference, not a typo). Section 5 is
  labelled a heuristic; related work states the overlap directly.
- Demoted: Qwen LoRA is no longer offered as validation (SijV is right that WD
  is not load-bearing at standard LoRA settings).
