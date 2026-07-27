# Shared argument blocks

Reusable text. Each reviewer only sees their own response, so repeating these is
expected; keeping one copy here is so the versions do not drift apart.

**Status note (2026-07-27).** Experiment E1 is in. The training-length
discriminator that several of these blocks previously framed as decisive came
out against our `λ ∝ 1/T` prediction and closer to the rotational-equilibrium
account. The drafts below have been rewritten around that fact. Do not paste an
older version that still claims a slope of -1.

---

## S1. The training-length experiment (negative for λ ∝ 1/T)

Every experiment in the submission was run at a single training length of 100
epochs. At fixed `T`, our rule `λ* = C/(η T)`, a constant-product rule
`η λ = const`, and a well-chosen fixed `λ` are numerically hard to tell apart.
They make opposite predictions once `T` varies:

- ours: `log λ*` against `log T` has slope -1
- constant product (Kosson et al.): slope 0

We therefore ran the missing sweep: ResNet-18 / CIFAR-100, SGDM, `η = 0.1`,
`B = 128`, cosine schedule, dense weight-decay ladder
`{1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2}`, and `T ∈ {25, 50, 100, 200}`,
all from one code path.

**Result.** The grid argmax is `λ* = 10^{-3}` at every training length. The
log-parabola interpolation gives slope `-0.226` with 95% bootstrap
interval `[-0.28, -0.17]`, and interpolated optima
`0.0012` / `0.000877` / `0.000737` at
`T = 25/100/200`. The optimal product `η λ*` moves by only
`1.62x in eta*lambda (T=25 to T=200; prediction ours=8x, equilibrium=1x)`, against an 8× movement predicted by `λ ∝ 1/T`.

We treat this as a **negative result** against the `1/T` discriminator that the
submission implied and that our rebuttal set out to test. On this axis the data
are much closer to the rotational-equilibrium account than to ours. The paper's
stated contribution type is Negative Results; we will report the finding in
those terms rather than retrofit the theory.

What the same sweep still shows: the accuracy peak in `λ` is broad but real
(moving from `5e-4` to `5e-3` costs several points at every `T`), so weight
decay continues to matter — it just does not move with `T` the way a
stability-timescale argument predicts.

---

## S2. Restating the law in terms of the summed step size

We still adopt a change of statement that removes an inconsistency Reviewer
SijV correctly spotted (Eq. 16 says `1/(η T)`, the conclusion says `2/(η T)`).

The natural quantity is the total step budget

    S = sum_{t<T} η_t

so that a timescale rule would read `λ* = C / S`. For constant step size
`S = η T`; for cosine annealing to zero, stepped once per epoch,
`sum_{k<T} ½(1 + cos(π k/T)) = (T+1)/2`, so `S = η T/2` and the same rule
reads `2C/(η T)`. The stray factor of two was a schedule difference, not a
typo.

E1 changes what we claim about this form: because `λ*` did not scale as `1/S`
in the sweep above, we no longer present `λ* = C/S` as an empirically supported
coupling across training lengths. We keep the `S`-based statement only as the
correct reading of the *submitted* equations, and as the form that makes the
schedule ambiguity disappear. The constant-LR vs cosine prediction
(`[[E1-SCHED-RATIO]]`) remains a test of that bookkeeping and is reported
separately.

---

## S3. Relation to Kosson et al. (rotational equilibrium)

We were wrong to omit this line of work and now discuss it explicitly
(arXiv:2305.17212, arXiv:2510.19093).

Kosson et al. analyse scale-invariant parameters, where the weight norm reaches
an equilibrium at which weight decay shrinkage balances gradient growth. The
equilibrium norm goes as `√(η/λ)` and the relative update in the steady state
depends on the product `η λ`, which is why the recommendation is to hold
`η λ` constant. That account is a *steady-state* statement and carries no
dependence on the training horizon.

Our rebuttal was designed to separate the two accounts by varying `T`. The
measurement (S1) lands on their side of that particular question: the optimal
product barely moves across an 8× range in `T`. We say so directly, and we
reframe the related-work discussion accordingly: on the training-length axis,
rotational equilibrium is the better guide to the optimum we measure.

Two points of contact remain where the accounts are not the same claim, and
where our measurements still add something:

1. **Along a line of constant `η λ`, accuracy is not flat** (E2b). Holding the
   product fixed and walking `η` over two decades costs up to
   `10.3` points. A pure product rule is therefore necessary but
   not sufficient for choosing the pair; the learning-rate ceiling and the
   under-fitting floor both bite. Measured usable range within 1 point of the
   peak: `a factor of 5 in eta`.
2. **A fixed default `λ` is dominated by the envelope** (E2a, from data already
   in the submission). Coupling `λ` to `η` at fixed `T` keeps accuracy within
   2.8 points across two decades of learning rate; `λ = 5e-4` gives up 1.44
   points on average and 3.77 at worst, and no fixed choice does better than
   3.77 worst-case.

So the correction E1 forces is specific: we withdraw the claim that the
optimum moves as `1/T`. We do not withdraw the claim that weight decay is a
load-bearing, non-redundant hyperparameter, nor the claim that the product
alone does not pin down the pair.

---

## S4. Honest accounting of what is new

Asked directly, our view after E1 is:

**Not new, and in one place wrong as a prediction.** The stability machinery is
Hardt et al.'s. That a strongly convex regularizer converts an `O(η T/n)`
stability bound into a `T`-independent one is standard. The resulting
`λ ~ 1/(η T)` coincides with Wang and Aitchison's AdamW-as-EMA timescale, and
our direct test of that `1/T` dependence on ResNet-18 / CIFAR-100 fails
(S1). The observation that good `(η, λ)` pairs lie on a band is in D'Angelo et
al. and in Kosson et al.; the band's weak dependence on `T` is what E1 finds,
in line with Kosson.

**What we still claim.**

1. **The cost side of weight decay, at fixed training length.** Prior work
   explains why `η λ` should be held roughly constant. The stability analysis
   also supplies a one-sided constraint that the product picture alone does
   not: weight decay tightens the admissible learning rate. E2b measures the
   consequence — accuracy falls along an iso-product line.
2. **Weight decay is not absorbed into tuning `η`.** The envelope analysis
   (E2a) quantifies what a fixed default costs against a `λ` that is allowed
   to move with `η`.
3. **A negative result on the timescale claim.** Matching stability upper
   bounds suggested `λ ∝ 1/T`; the measurement says otherwise in this regime.
   We report that rather than defend it.

We have rewritten the contribution statement to say this plainly.

---

## S5. What the theory claims outside the convex setting

We agree the convex, `L`-smooth analysis does not transfer as a theorem. After
E1 and E3 we are more specific about what survives measurement.

1. **The learning-rate ceiling.** True NaN-divergence thresholds are clean at
   `λ = 0` and tighten once momentum is added (`[[E3-MOM-RATIO]]` against a
   prediction of `1-β = 0.1`). At large positive `λ`, however, training fails
   by under-fitting (loss stuck at `log #classes`) long before it explodes, so
   the linear test `1/η_max = λ + L/2` is not cleanly identified in that
   regime. We report the explosion brackets we have (`[[E3-SLOPE]]`,
   `[[E3-INTERCEPT]]`) and state this limitation explicitly rather than claim
   a slope-1 verification.
2. **The stability mechanism itself.** Experiment E7a (leave-one-out trajectory
   pairs) remains the direct probe of whether weight decay saturates
   `‖θ_t - θ'_t‖`. Status: `[[E7-DIVERGENCE-RATIO]]`.

What we do not claim: that the constant `C` is predicted by the theory, or that
`λ* = C/S` holds across training lengths (see S1).

---

## S6. The constant, and how much it matters

The Wave-0 estimate of `C = λ* · S` was taken entirely at fixed `T = 100`.
Fitting it in 65 settings (three architectures, five batch sizes, two seeds)
gave a geometric mean of `C = 1.48` with multiplicative spread x/1.70. That
number is still the right summary of *cross-architecture, cross-batch*
variation at fixed training length.

E1 changes the interpretation across `T`: because `λ*` stays near `10^{-3}`
while `S` grows with `T`, the implied `C` grows roughly in proportion to `T`
rather than staying constant. So `C` is not a universal constant of the
training horizon; treating it as one is exactly the mistake E1 exposes.

The practical question that remains well-posed is how much a wrong `λ` costs
*at fixed `T`*. Experiment E5b sweeps the multiplier around a calibrated value:
being off by 3× costs `[[E5B-3X]]` points, and by 10× costs `[[E5B-10X]]`.
Combined with E2a, this is the sense in which a product-style rule (or even a
decent fixed default) can be useful within a fixed training length, while a
`1/T` extrapolation across training lengths is not.
