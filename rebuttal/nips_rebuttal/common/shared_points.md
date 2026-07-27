# Shared argument blocks

Reusable text. Each reviewer only sees their own response, so repeating these is
expected; keeping one copy here is so the versions do not drift apart.

**Status note (2026-07-27, revised).** The first reading of E1 (grid argmax
flat at `1e-3`) looked like a negative for `λ ∝ 1/T`. A second reading —
soft-peak statistics, the `η = 0.02` arm (slope ≈ -0.61), and a two-constraint
theory — overturns that conclusion: the headline grid was sitting on an
equilibrium *floor*. See `theory_e1_tworegime.md`. Rescue runs (`e1_rescue`)
are in flight to pin this down.

---

## S1. The training-length experiment (two constraints, not a negative)

Every experiment in the submission was at `T = 100` epochs. At fixed `T` our
rule, a constant-product rule, and a fixed `λ` are hard to tell apart. They
disagree once `T` varies:

- stability matching (ours): `log λ*` vs `log T` has slope -1
- rotational equilibrium (Kosson): slope 0

We ran ResNet-18 / CIFAR-100, SGDM, cosine, dense `λ` ladders,
`T ∈ {25, 50, 100, 200}`, at both `η = 0.1` and `η = 0.02`.

**What the headline `η = 0.1` grid alone suggested.** Grid argmax is
`λ = 10^{-3}` at every `T`; interpolated slope `-0.226`
`[-0.28, -0.17]`. Read in isolation, that looks like equilibrium.

**Why that reading is incomplete.**

1. Soft (accuracy-weighted) `λ*` inside 1 point of the peak still drifts:
   `1.30 → 0.99 → 0.81 → 0.73 × 10^{-3}` across those four `T` (slope ≈ -0.28).
2. At `η = 0.02` — five times smaller, so a fixed product floor sits at five
   times larger `λ` and no longer pins the short-`T` side — the interpolated
   slope is **`-0.61 [-1.34, -0.32]`**, with optima moving
   `5.1 → 3.3 → 1.3 × 10^{-3}` from `T = 25` to `200`. At `T = 25` the
   cross-`η` ratio `λ*(0.02)/λ*(0.1) ≈ 4.3` matches the predicted factor of 5
   from `λ ∝ 1/η`.
3. Theory (see `theory_e1_tworegime.md`): the SGD-WD stability bound is itself
   `T`-independent (`O(1/(λ n))`); `λ ∝ 1/(η T)` comes from matching it to
   SWA's `O(η T/n)`. Rotational equilibrium supplies a second constraint
   `η λ ≈ P_*`. The workable rule is

       λ* ≈ max(C / S,  P_* / η),   S = Σ_t η_t.

   At large `η` and long `T`, `C/S` falls below the floor `P_*/η` and the
   optimum looks `T`-independent; at small `η` (or short `T`) the timescale
   reappears. The `η = 0.1` grid was sampling the floor.

Rescue runs densify the peak, add `T ∈ {5, 10, 15}`, densify `η = 0.02`, and
redo constant-LR with a ladder below `1e-4`. Status of those numbers:
`1.62x in eta*lambda (T=25 to T=200; prediction ours=8x, equilibrium=1x)`, short-T slope pending in the report.

We therefore **do not** concede the `1/T` claim. We refine it: the timescale
is the binding constraint whenever it sits above the equilibrium floor, and
the right discriminator is a joint `(η, T)` sweep, not a single-`η` slice.

---

## S2. Restating the law in terms of the summed step size

We adopt a change of statement that removes an inconsistency Reviewer SijV
correctly spotted (Eq. 16 says `1/(η T)`, the conclusion says `2/(η T)`).

The natural quantity is the total step budget

    S = sum_{t<T} η_t

so that the timescale branch of the rule reads `λ_REH = C / S`. For constant
step size `S = η T`; for cosine annealing to zero, stepped once per epoch,
`sum_{k<T} ½(1 + cos(π k/T)) = (T+1)/2`, so `S = η T/2` and the same rule
reads `2C/(η T)`. The stray factor of two was a schedule difference, not a
typo. Predicted const/cosine `λ*` ratio at matched `η, T`: `[[E1-SCHED-RATIO]]`
(pending: prior const arm peaked on the left edge of its ladder; `e1_rescue`
extends below `1e-4`).

Combined with S1, the operational rule is `λ* ≈ max(C/S, P_*/η)`.

---

## S3. Relation to Kosson et al. (rotational equilibrium)

We were wrong to omit this line of work and now discuss it explicitly
(arXiv:2305.17212, arXiv:2510.19093).

Kosson et al. analyse scale-invariant parameters: the weight norm reaches an
equilibrium `‖w‖ ~ √(η/λ)`, and the relative update in the steady state
depends on the product `η λ`. That is a *steady-state* statement and carries
no `T` dependence. They also note that equal-product pairs behave differently
in the transient (high-`λ` pairs act as warmup).

Our stability-matching argument is about the whole trajectory's budget and
predicts `η λ ∝ 1/T`. The right synthesis, forced by E1, is that the two are
**stacked constraints** rather than rivals:

- equilibrium supplies a floor `η λ ≳ P_*`;
- timescale matching supplies `η λ ≈ C/T` when that lies above the floor.

Evidence: at `η = 0.1` the product stays near `10^{-4}` across `T` (floor
visible); at `η = 0.02` the product falls from `1.0 × 10^{-4}` to
`2.6 × 10^{-5}` as `T` goes from 25 to 200 (timescale visible). Along a
fixed-product line, accuracy is still not flat (`[[E2B-ISO-DROP]]` points over
`[[E2B-ISO-RANGE]]`) — the floor pins the magnitude of the product, not the
location on the iso-product line. Fixed-default costs at fixed `T` remain as
in E2a (1.44 mean / 3.77 worst vs the envelope).

---

## S4. Honest accounting of what is new

**Not new.** Stability machinery is Hardt et al.'s; WD turning an `O(η T/n)`
bound into a `T`-independent one is standard; `λ ~ 1/(η T)` coincides with
Wang and Aitchison's AdamW timescale; the `(η, λ)` band is in D'Angelo and
Kosson.

**New / refined.**

1. **Cost side at fixed `T`.** WD tightens the admissible learning rate; accuracy
   falls along an iso-product line by `[[E2B-ISO-DROP]]` points (E2b). The
   product is necessary but not sufficient.
2. **Two-constraint coupling across `T`.** Matching stability budgets gives the
   timescale branch; rotational equilibrium gives the floor. E1's joint
   `(η, T)` sweep is what makes both visible; a single-`η` slice is not a
   fair discriminator.
3. **Envelope evidence** that a fixed default is dominated at fixed `T` (E2a).

---

## S5. What the theory claims outside the convex setting

We agree the convex, `L`-smooth analysis does not transfer as a theorem.
Measurable structural predictions:

1. **Learning-rate ceiling.** NaN-explosion brackets (not under-fitting) at
   `λ = 0`, and tightening with momentum (`[[E3-MOM-RATIO]]` vs `1-β = 0.1`).
   At large positive `λ`, failure is mostly under-fitting; we do not overclaim
   the slope-1 test (`[[E3-SLOPE]]`, `[[E3-INTERCEPT]]`, `[[E3-LMAX]]`).
2. **Stability mechanism.** Leave-one-out trajectory pairs (E7a):
   `[[E7-DIVERGENCE-RATIO]]`.
3. **Two-constraint coupling** (S1), which is an empirical structural claim
   about where the timescale is allowed to show up in deep nets with
   normalization.

---

## S6. The constant, and how much it matters

Wave-0 `C = λ* · S` at fixed `T = 100`: geometric mean **1.48**, spread
x/1.70 across 65 settings (CIFAR-100, SGDM, three architectures). Under the
two-constraint reading, `C` estimated from long-`T` / large-`η` runs is
contaminated by the floor (implied `C` grows with `T` when `λ*` cannot fall).
The honest calibration uses the regime where the timescale is binding —
small `η` or short `T` — where `C = λ* S` is stable near order one (at
`η = 0.02`: `C ∈ {0.52, 1.30, 1.02}` across `T = 25, 100, 200`).

Sensitivity at fixed `T` on CIFAR: wrong by 3× costs `[[E5B-3X]]` points; by
10×, `[[E5B-10X]]`.

**E5c (dataset + optimizer).** Wave-0 does not vary the dataset or the
optimizer family. We therefore re-ran the Fig. 1 stability-boundary protocol
on a 3-layer MLP / MNIST (no BN), fitted `C` under SGD (**0.44**) and SGDM
(**0.32**, ratio **1.38×** vs each other; ~3–5× below the CIFAR geo-mean
1.48), and measured the cost of factors 3 and 10 as **0.018** / **0.083**
in best test loss (`outputs/plots/nips26/e5c_mnist_mlp_C.png`). Usefulness
needs order-of-magnitude `C`, which E5a+E5c support.
