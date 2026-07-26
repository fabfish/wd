# Response to Reviewer SijV

Rating 3 (borderline reject). Scores: Quality 3, Clarity 3, Significance 2,
Originality 2.

We thank the reviewer for a precise and fair review. Three of the four points
are ones we accept outright, and one of them (the missing rotational-equilibrium
literature) turned out to point at a genuinely missing experiment, which we have
now run. We answer the two direct questions first, since they were asked
plainly and deserve plain answers.

---

## Q1. How much of this is new, honestly?

**Not new.** The stability machinery is Hardt et al.'s, and the fact that adding
a strongly convex regularizer turns an `O(eta*T/n)` uniform-stability bound into
a `T`-independent `O(1/(lambda*n))` one is a standard consequence rather than a
discovery. The resulting `lambda ~ 1/(eta*T)` coincides numerically with Wang
and Aitchison's AdamW-as-EMA timescale, which we cite but did not sufficiently
foreground as an equivalent prior result. That good `(eta, lambda)` pairs form a
band is already visible in D'Angelo et al. and, as the reviewer notes, in Kosson
et al.

**New.** We claim three things, and we have rewritten the contribution statement
to say only these.

1. **The cost side of weight decay.** Existing accounts explain why the product
   should be held roughly constant. None of them supplies the accompanying
   constraint that weight decay *tightens* the admissible learning rate, here
   `eta <= 2/(2*lambda + L)` for SGD-WD and `eta <= 2/(2*lambda + L*Gamma_beta)`
   with momentum. This is what turns a one-dimensional band into a two-sided
   region, and it makes a prediction the band picture does not: accuracy must
   fall off along a line of constant `eta*lambda`. We measure this in a new
   experiment; the drop is `[[E2B-ISO-DROP]]` points.
2. **One derivation for four couplings.** Learning rate, weight decay, batch
   size and training length fall out of the same stability argument rather than
   being four separately motivated empirical rules. We think this is worth
   something even where individual endpoints are known, but we now say
   explicitly that the value is unification, not rediscovery.
3. **A falsifiable discriminator against the equilibrium account** (see Q2 and
   the next section).

We would rather state this honestly than claim more. If the reviewer's view is
that unification plus one new constraint is insufficient for the bar, that is a
judgement we understand; we have at least removed the ambiguity about what is
being claimed.

## Q2. There are many versions of this relationship; some proportionality, some a precise law. Which is it?

Proportionality is what the analysis supports; the constant is not predicted and
we now say so in those words.

Concretely, the exponents are stable and measurable, and the constant is neither:

- `log lambda*` against `log eta`: slope **-0.87, 95% interval [-0.94, -0.75]**
  (dense 8 x 9 grid, ResNet-18 / CIFAR-100, two seeds).
- `log lambda*` against `log T`: slope `[[E1-T-SLOPE]]`, interval
  `[[E1-T-CI]]` (new experiment).
- The constant: fitting `C = lambda* * sum_t eta_t` in 65 independent settings
  spanning three architectures, five batch sizes, two seeds and learning rates
  from 0.001 to 0.5 gives a geometric mean of **1.48 with a multiplicative
  spread of x/1.70** (range 0.17 to 2.99). By architecture: ResNet-18 1.42,
  ResNet-50 1.42, VGG-16 1.72.

So the honest statement of our result is a scaling relation with a
setting-dependent prefactor of order one, calibrated once. Papers that state a
precise law are, we believe, reporting the prefactor of their own setting. We
now report the spread instead of a single number, and we measure what a wrong
prefactor costs: being off by 3x costs `[[E5B-3X]]` points and by 10x costs
`[[E5B-10X]]`.

---

## 1. Framing: Kosson et al. and the optimization-dynamics view

We accept this criticism. Both papers (arXiv:2305.17212, arXiv:2510.19093) are
now cited and discussed in the related work, and we agree that in normalized
networks the dominant role of weight decay is control of the effective learning
rate rather than capacity control.

The reviewer's summary -- weight decay prevents norm growth, weights rotate at a
controlled speed in equilibrium, and the optimum keeps `eta*lambda` roughly
constant -- is, we think, complementary to ours rather than in conflict.

It is worth being precise about where the two differ, because Kosson et al. are
careful about this themselves: theirs is a **steady-state** statement, and they
explicitly note that `(eta, lambda)` pairs with equal products behave
differently during the initial phase, with high-`lambda` pairs acting as an
implicit warmup (arXiv:2510.19093, Sec. 5). So the pure product picture is
already known to be incomplete in the transient. Our departure from it is a
different one, and it concerns the end of training:

- Rotational equilibrium describes a stationary state reached after a
  transient, and carries no dependence on the training horizon. The optimal
  `eta*lambda` it predicts therefore does not move with `T`.
- Our stability argument bounds how far a trajectory can be driven from a
  neighbouring one over `T` steps, so it predicts `eta*lambda ∝ 1/T`.

Every experiment in the submission was at 100 epochs, which is precisely why the
paper could not tell these apart -- a fair reading of the reviewer's novelty
concern. We have now run the training-length sweep: at fixed `eta = 0.1` and
`B = 128`, with `T` in {25, 100, 200} on ResNet-18 / CIFAR-100, the optimal
product moves by a factor of `[[E1-PRODUCT-DRIFT]]`, with fitted slope
`[[E1-T-SLOPE]]` against `log T` (interval `[[E1-T-CI]]`).

We also note that the equilibrium mechanism relies on scale invariance, i.e. on
normalization layers. Our ablation on networks without normalization
(`[[E7-BN]]`) tests whether the coupling persists where that mechanism does not
apply.

## 2. The Qwen LoRA experiment

We accept this. The reviewer is right that weight decay is not a load-bearing
hyperparameter at standard LoRA settings, where the default is frequently zero,
and that a sweep at those magnitudes does not establish that the coupling is
doing meaningful work.

We have therefore demoted this result: it is no longer presented as validation
of the coupling, it is described as a suggestive observation in a setting where
weight decay is known to be weakly load-bearing, and the limitation is stated in
the text. We agree that the convincing version requires full-parameter language
model training, which we could not run within the rebuttal window and now list
as the primary piece of missing evidence rather than implying it is covered.

## 3. Minor inconsistencies

The reviewer is right that the conclusion says `lambda ≈ 2/(eta*T)` (line 283)
while Eq. 16 says `1/(eta*T)`. This was not a typo but an undocumented
difference between schedules, and we have fixed the statement rather than the
number.

The quantity the coupling is about is the total step budget `S = sum_{t<T}
eta_t`, so the law is

    lambda* = C / S.

For a constant schedule `S = eta*T`. For cosine annealing to zero stepped once
per epoch, `sum_{k<T} 0.5*(1 + cos(pi*k/T)) = (T+1)/2` exactly, so `S = eta*T/2`
and the same law reads `2C/(eta*T)`. All of our experiments use cosine, which is
where the factor of two entered. Writing the law in terms of `S` removes the
ambiguity and makes a testable prediction: at matched `eta` and `T`, a
constant-LR run should prefer half the weight decay of a cosine run. Measured
ratio: `[[E1-SCHED-RATIO]]`.

"bitch size" and the other typos are fixed.
