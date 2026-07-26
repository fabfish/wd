# Shared argument blocks

Reusable text. Each reviewer only sees their own response, so repeating these is
expected; keeping one copy here is so the versions do not drift apart.

---

## S1. The one experiment the paper was missing

Every experiment in the submission was run at a single training length of 100
epochs. That is the reason several of the criticisms land, and we have now
fixed it.

At fixed `T`, the following three rules are numerically indistinguishable:

- ours, `lambda* = C / (eta * T)`
- a constant optimal product, `eta * lambda = const`
- a well-chosen fixed `lambda` combined with tuning `eta`

They stop being indistinguishable the moment `T` varies, and they make
*opposite* predictions:

- ours: `log lambda*` falls with slope -1 against `log T`
- constant product: slope 0
- fixed `lambda`: slope 0, and `eta*` must absorb the change instead

We therefore ran a training-length sweep on ResNet-18 / CIFAR-100 at fixed
`eta = 0.1`, `B = 128`, with `T` in {25, 100, 200} and a weight-decay ladder
that lines up exactly with the existing 100-epoch grid. The measured slope is
`[[E1-T-SLOPE]]` with 95% bootstrap interval `[[E1-T-CI]]`, and the optimal
weight decay moves from `[[E1-T-LAMBDA-25]]` at `T = 25` to
`[[E1-T-LAMBDA-200]]` at `T = 200`.

---

## S2. Restating the law in terms of the summed step size

We adopt a change of statement that removes an inconsistency Reviewer SijV
correctly spotted (Eq. 16 says `1/(eta*T)`, the conclusion says `2/(eta*T)`).

The quantity the coupling is really about is the total step budget

    S = sum_{t<T} eta_t

so that the law reads

    lambda* = C / S.

For a constant schedule `S = eta*T`. For cosine annealing to zero, stepped once
per epoch, `sum_{k<T} 0.5*(1 + cos(pi*k/T)) = (T+1)/2`, so `S = eta*T/2` and the
same law reads `lambda* = 2C/(eta*T)`. The stray factor of two was the
difference between a constant and a decayed schedule, not a typo, and writing
the law in terms of `S` removes the ambiguity. It also yields a prediction we
test directly: at matched `eta` and `T`, a constant-LR run should prefer half
the weight decay of a cosine run. Measured ratio: `[[E1-SCHED-RATIO]]`.

---

## S3. Relation to Kosson et al. (rotational equilibrium)

We were wrong to omit this line of work and now discuss it explicitly
(arXiv:2305.17212, arXiv:2510.19093).

Kosson et al. analyse scale-invariant parameters, where the weight norm reaches
an equilibrium at which weight decay shrinkage balances gradient growth. The
equilibrium norm goes as `sqrt(eta/lambda)` and the magnitude of the relative
update in the steady state depends only on the product `eta*lambda`, which is
why the recommendation is to hold `eta*lambda` constant.

The two accounts are complementary rather than competing. It is worth being
precise about where they differ, because Kosson et al. are themselves careful
here: their claim is a *steady-state* one, and they explicitly observe that
`(eta, lambda)` pairs with the same product behave differently during the
initial phase, with high-`lambda` pairs acting like an implicit warmup
(arXiv:2510.19093, Sec. 5). So the product picture is already known not to be
the whole story in the transient.

Our claim is a different departure from the product picture, and it concerns the
end of training rather than the beginning:

- The equilibrium argument describes a stationary state reached after a
  transient, and carries no dependence on the training horizon. It therefore
  predicts an optimal `eta*lambda` that does not move with `T`.
- Our stability argument bounds how far the iterate can be driven from a
  neighbouring trajectory over `T` steps, so it predicts
  `eta*lambda ∝ 1/T`.

Experiment E1 measures this directly: the optimal product moves by a factor of
`[[E1-PRODUCT-DRIFT]]` between `T = 25` and `T = 200`, where the equilibrium
account predicts no movement at all.

We also note that the equilibrium mechanism requires scale invariance, i.e.
normalization layers. Our ablation without normalization (`[[E7-BN]]`) tests
whether the coupling survives where the equilibrium argument does not apply.

---

## S4. Honest accounting of what is new

Asked directly, our view is:

**Not new.** The stability machinery is Hardt et al.'s. That a strongly convex
regularizer converts an `O(eta*T/n)` stability bound into a `T`-independent one
is a standard consequence. The resulting `lambda ~ 1/(eta*T)` coincides
numerically with Wang and Aitchison's AdamW-as-EMA timescale. The observation
that good `(eta, lambda)` pairs lie on a band is in D'Angelo et al. and in
Kosson et al.

**New.** Three things.

1. The *cost* side of weight decay. Prior work explains why `eta*lambda` should
   be held roughly constant; none of it gives the accompanying constraint that
   weight decay tightens the admissible learning rate to
   `eta <= 2/(2*lambda + L)`. That constraint is what makes the coupling a
   two-sided one rather than a one-dimensional band, and it is what E2b
   measures.
2. A single derivation that produces the learning-rate, weight-decay, batch-size
   and training-length couplings together, from one stability argument, rather
   than as four separate empirical rules.
3. The `1/T` dependence as a *falsifiable* discriminator against the
   equilibrium account, which we now test (S1, S3).

We have rewritten the contribution statement to say this plainly rather than
leaving the reader to work out the overlap.

---

## S5. What the theory claims outside the convex setting

We agree the convex, `L`-smooth analysis does not transfer as a theorem. What we
claim is that two of its structural predictions are measurable in deep networks,
and we now measure them.

1. **The learning-rate ceiling.** The bound `eta <= 2/(2*lambda + L)` rearranges
   to `1/eta_max = lambda + L/2`, a straight line in `lambda` whose intercept is
   an effective smoothness. Experiment E3 locates the empirical divergence
   threshold by bisection for seven weight decays and finds slope
   `[[E3-SLOPE]]` and intercept `[[E3-INTERCEPT]]`, against a top Hessian
   eigenvalue of `[[E3-LMAX]]` measured by power iteration.
2. **The stability mechanism itself.** Experiment E7a trains pairs of networks on
   datasets differing in exactly one example and tracks
   `||theta_t - theta'_t||`. Without weight decay this grows with `t`; with
   weight decay the theory says it saturates. Measured ratio at the end of
   training: `[[E7-DIVERGENCE-RATIO]]`.

What we do not claim: that the constant `C` is predicted by the theory. It is
fitted once (see S6), and the paper now says so in those words.

---

## S6. The constant, and how much it matters

The exponents come from the analysis; the constant does not. We therefore
measured it rather than asserting it.

Fitting `C = lambda* * S` independently in 65 settings that already exist in our
sweeps -- three architectures (ResNet-18, ResNet-50, VGG-16), five batch sizes
(32 to 512), learning rates from 0.001 to 0.5, two seeds -- gives a geometric
mean of `C = 1.48` with a multiplicative spread of x/1.70 and a full range of
0.17 to 2.99. By architecture: ResNet-18 1.42, ResNet-50 1.42, VGG-16 1.72.

The practical question is not how tight that spread is but how much a wrong `C`
costs. Experiment E5b sweeps `C` deliberately wrong by factors of 3 and 10:
being off by 3x costs `[[E5B-3X]]` accuracy points, and by 10x costs
`[[E5B-10X]]`. This is the sense in which the rule is usable: the optimum in
`lambda` is broad, so an order-of-magnitude-correct prediction captures most of
the available accuracy, while a fixed default does not, because it is wrong by a
factor that *grows* with the mismatch in `eta` and `T`.
