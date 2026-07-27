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

**New, after the training-length test.** We claim less than the submission did,
and we have rewritten the contribution statement accordingly.

1. **The cost side of weight decay, at fixed training length.** Existing
   accounts explain why the product should be held roughly constant. The
   stability analysis also supplies a one-sided constraint the product picture
   alone does not: weight decay tightens the admissible learning rate. The
   testable consequence is that accuracy falls along a line of constant
   `eta*lambda`. We measure this; the drop is `10.3` points.
2. **A negative result on the timescale claim.** Matching stability upper
   bounds suggested `lambda ∝ 1/T`. We ran the discriminator against the
   equilibrium account (next section); the measurement lands closer to Kosson
   et al. than to us. The paper's contribution type is Negative Results; we
   report the finding in those terms.

We would rather state this honestly than claim more. If the reviewer's view is
that a cost-side constraint plus a negative result on the timescale is
insufficient for the bar, that is a judgement we understand; we have at least
removed the ambiguity about what is being claimed, and we have stopped
defending a prediction the data reject.

## Q2. There are many versions of this relationship; some proportionality, some a precise law. Which is it?

Proportionality is what the analysis supports; the constant is not predicted and
we now say so in those words.

Concretely, after the new measurements:

- `log lambda*` against `log eta` (fixed `T = 100`): slope **-0.87, 95%
  interval [-0.94, -0.75]** — this exponent is real at fixed training length.
- `log lambda*` against `log T` (fixed `eta = 0.1`, dense ladder): slope
  `-0.226`, interval `[-0.28, -0.17]`. Grid argmax is `10^{-3}` at every
  `T ∈ {25, 50, 100, 200}`. This is **incompatible with slope -1** and close
  to the equilibrium prediction of 0. We withdraw `lambda ∝ 1/T` as an
  empirical claim in this regime.
- The prefactor `C = lambda* * sum_t eta_t`, fitted at fixed `T = 100` across 65
  settings, has geometric mean **1.48 with spread x/1.70**. Because `lambda*`
  does not fall with `T`, the same `C` is *not* stable across training
  lengths — that is the content of the negative result above.

So the honest statement is: at fixed training length the `eta`--`lambda`
coupling is a real scaling with a setting-dependent prefactor; across training
lengths a constant-product rule is the better guide. Being wrong by 3× in
`lambda` at fixed `T` costs `[[E5B-3X]]` points; by 10×, `[[E5B-10X]]`.

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
implicit warmup (arXiv:2510.19093, Sec. 5). Our stability argument made a
different departure, predicting `eta*lambda ∝ 1/T` from the horizon dependence
of uniform stability.

Every experiment in the submission was at 100 epochs, which is precisely why the
paper could not tell these apart -- a fair reading of the novelty concern. We
have now run the training-length sweep. The result favours their account on
this axis: grid `lambda* = 10^{-3}` at every `T`, interpolated slope
`-0.226` `[-0.28, -0.17]`, product drift only `1.62x in eta*lambda (T=25 to T=200; prediction ours=8x, equilibrium=1x)`. We
say so directly and reframe the related work accordingly.

What we still add, and where a pure product rule is incomplete even on their
terms: walking along a line of constant `eta*lambda` over two decades of `eta`
costs up to `10.3` points (E2b). The product pins a band; it does
not pin the pair. Our ablation without normalization (`[[E7-BN]]`) remains the
test of whether any of this coupling survives where scale invariance fails.

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
