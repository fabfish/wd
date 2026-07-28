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
This is now cited and discussed, along with AdamW/SGDW within-run weight-decay
schedules (cosine / drop-step / restarts).

**Recovery of existing results.** Our honest accounting is that the stability
machinery is Hardt et al.'s, that a strongly convex regularizer converting an
`O(eta*T/n)` bound into a `T`-independent one is standard, and that the
resulting `lambda ~ 1/(eta*T)` coincides with Wang and Aitchison's AdamW
timescale. We have rewritten the contribution statement to say this rather than
leaving the overlap for the reader to find.

What we retain is a refined coupling, not a concession. Prior accounts explain
why the product should be held constant; the stability analysis also supplies a
one-sided constraint that the product picture alone does not
(`eta <= 2/(2*lambda + L)`), with the testable consequence that accuracy falls
along an iso-product line (drop `10.3` points). On training length the two
accounts looked like rivals — equilibrium predicts `ηλ = const`, we predicted
`ηλ ∝ 1/T` — but a joint `(η, T)` sweep shows they are stacked: at `η = 0.1`
the grid sits on the equilibrium floor (slope `-0.226`); at `η = 0.02` the
timescale returns (slope `-0.61 [-1.34, -0.32]`). Operational rule:
`λ* ≈ max(C/S, P_*/η)`. Related work is reframed accordingly.

## 2. "The theoretical analysis may not support the experiments; it relies on assumptions such as L-smoothness"

We agree that the convex, `L`-smooth analysis does not transfer to deep networks
as a theorem, and we have scoped the claim accordingly. Rather than assert that
it transfers, we now measure the two structural predictions that can be checked
directly.

**The learning-rate ceiling.** We locate empirical thresholds by bisection in
`eta`, counting only NaN/explosion (not under-fitting: at large `lambda` the
loss sticks at `log #classes` without diverging, which is not the L-smooth
ceiling). Explosion brackets are clean at `lambda = 0` and tighten with
momentum (`0.23` against a prediction of 0.1). At large positive
`lambda` the slope-1 test `1/eta_max = lambda + L/2` is not cleanly identified;
we report what we have (`0.67`, `0.08 (implies L = 0.2)`, Hessian
`417.4`) and state this limitation rather than claim a verification.

**The stability mechanism itself.** We train pairs of networks on datasets
differing in exactly one example, with identical initialization and batch
ordering, and track `||theta_t - theta'_t||`. Without weight decay the convex
theory predicts growth with `t`; with weight decay, saturation. Measured
`1.10 (final ||theta-theta'|| at lambda=0 over lambda=1e-3)`.

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

`see _data/e4_transfer_table.md`

Gap to the oracle: ours `0.87` mean, `1.58` worst;
fixed default `0.70`; constant product `1.63`;
`1/(eta*T)` `1.60`. The oracle column costs eight training runs per
setting; every rule costs zero after the one-time calibration.

We also compare against *within-run* scheduled weight decay (E8): fixed learning
rate `eta = 0.1`, and AdamW/SGDW schedule shapes applied only to `lambda`
(cosine, linear, drop-step, cosine-with-restarts). Under SGDM, drop-step lifts
peak accuracy from **66.67%** (fixed `lambda`) to **73.10%** (+6.4 points); under
SGD the gain is smaller (+1.0 point for drop-step). That confirms scheduling
`lambda` is useful engineering when the learning rate is held fixed, but it is
orthogonal to selecting a constant `lambda` from `(eta, T, B)` — E4 tests the
latter, E8 the former.

Two supporting numbers, both from data we already had:

- On the dense 8 x 9 grid, coupling `lambda` to `eta` keeps accuracy within
  **2.8 points** across two decades of learning rate, whereas the common default
  `5e-4` gives up **1.44 points on average and 3.77 at its worst**. No fixed
  choice does better than 3.77 worst-case.
- Fitting the prefactor in 65 independent settings (three architectures, five
  batch sizes, two seeds) gives a geometric mean of **1.48** with a spread of
  **x/1.70**, so a single calibration transfers across architectures to within
  about a factor of two, and being wrong by 3x costs `15.11` points.

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
`0.42 [-0.52, 1.75]`.

**Concurrent work.** The contrast with rotational equilibrium is now explicit
and, as described under weakness 1, is the discriminating experiment rather than
a discussion point. We also note that the equilibrium mechanism requires scale
invariance; our ablation on networks without normalization (`coupling survives without BN — bn=0: lambda* in {0.002..0.002} across eta, peak acc 59.3%; bn=1: lambda* in {0.0005..0.0005} across eta, peak acc 58.5%`) probes
whether the coupling persists where that mechanism does not apply.

---

## What changed, in one place

- New measurements: joint `(η, T)` sweep recovering the timescale at small `η`,
  iso-product walk (drop `10.3` points), zero-tuning transfer, explosion
  brackets, momentum arm, prefactor sensitivity; `e1_rescue` densifies the
  peak / short-`T` / const-LR ladder.
- Reframed: `λ* ≈ max(C/S, P_*/η)` — timescale plus equilibrium floor — rather
  than abandoning `1/T`. The `sum_t eta_t` restatement resolves the
  `1/(ηT)` vs `2/(ηT)` bookkeeping inconsistency. Section 5 is a heuristic;
  related work states the overlap and the composition directly.
- Demoted: Qwen LoRA is no longer offered as validation (SijV is right that WD
  is not load-bearing at standard LoRA settings).
