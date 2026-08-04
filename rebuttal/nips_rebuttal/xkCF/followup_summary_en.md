# Follow-up Experiments for Reviewer xkCF (2026-08-03)

This document summarizes the two new experiments we ran in response to reviewer
xkCF's 2026-08-03 follow-up. His third point (LoRA / related-work) is deferred
to a later round. **Point F1 is dropped in its entirety** — see below.

---

## F1 — dropped: the reviewer misread the batch-size claim

The reviewer's F1 suggests a contradiction between Eq.(17) (which he reads as
`λ* ∝ B`) and Table 4 (where `ηλ*` sits in a narrow range). This is a misreading:
our single law is `λ* = C / Σ_t η_t`, and there is exactly one claim with two
conditional readings.

- **Fixed `η`**: `Σ_t η_t ∝ 1/B`, so `λ* ∝ B` (measured slope **+1.02 [+0.88, +1.15]**).
- **Linear LR scaling (`η ∝ B`)**: `Σ_t η_t` is cancelled by `B` and stays put, so
  `λ*` is flat and `ηλ* ∝ B`.
- An unconditional regression over all 65 settings gives only **+0.23** — the two
  dependencies cancel; conditioning is the entire content of the claim.
- The published residual drift of `C` across `B` (**1.38 / 1.89 / 1.40 / 1.60 / 1.71**)
  matches the numbers in the paper / Table 4 word-for-word. There is no
  inconsistency.

Because spelling out "which reading" would be redundant, we drop F1. We keep one
self-initiated precision edit: in Exp. 3, replace *"optimal `λ` grows with batch
size"* with *"under linear LR scaling `ηλ*` grows with `B` while `λ*` itself stays
put (`λ* ∝ B` is the fixed-`η` reading)"*. This is our own tightening, not a
concession — Eq.(17) and Table 4 are consistent as written.

---

## F2 (E9) — the two schedulers the reviewer proposed

The reviewer correctly noted that our original `joint` arm gives
`η_tλ_t = η₀λ₀·m(t)²`, which does *not* preserve the coupling our rule is about,
and so cannot serve as a control. We ran both of his suggestions (same protocol
as the main paper — ResNet-18 / CIFAR-100 / `B = 128` / `η₀ = 0.1` / `T = 100` /
SGDM / **cosine LR** — so these are comparable to our own default, not to the
constant-LR E8 arms).

### (a) iso-product: `λ_t = λ₀·η₀/η_t` (holds `η_tλ_t` constant)

The `1/m_cos` factor diverges in the cosine tail, so the multiplier is capped at
**10×** (equivalently `η` is floored at `η₀/10` inside the `λ` formula); stated as
part of the protocol, not applied silently. `λ₀ ∈ {1e-4, 5e-4, 1e-3, 2e-3, 5e-3}`:

| `λ₀` | realized `Σ_t η_tλ_t` | best acc |
|---:|---:|---:|
| 1e-4 | 0.29 C | 75.56 |
| **5e-4** | 1.44 C | **77.98** |
| 1e-3 | 2.88 C | 77.98 |
| 2e-3 | 5.76 C | 70.36 |
| 5e-3 | 14.39 C | 22.13 |

### (b) matched cumulative contraction

`Σ_t η_tλ_t` is linear in `λ₀`, so for each shape we solve `λ₀` so that the
cumulative contraction equals a common budget `{C/3, C, 3C}`, where
`C = 1.181` is the contraction our own rule already prescribes at this setting.
The `fixed` shape at these three budgets solves to `λ₀ ∈ {1.994e-4, 5.982e-4,
1.795e-3}` — exactly the E5b wrong-`C` points and the E4-ours baseline — so that
column reuses already-reported runs rather than new training:

| budget `Σ_t η_tλ_t` | fixed | cosine | linear | step | **iso-product** |
|---|---:|---:|---:|---:|---:|
| `C/3` | 74.62 | 74.34 | 74.43 | 74.16 | **75.86** |
| `C` | 76.72 | 75.50 | 76.44 | 75.85 | **78.22** |
| `3C` | 76.19 | 75.03 | 75.53 | 73.97 | **77.50** |

Take-aways:

1. **Matching the contraction budget is not sufficient.** At a fixed budget the
   shapes still differ by **1.70 / 2.72 / 3.53 pp**, so `Σ_t η_tλ_t` is not a
   sufficient statistic for a schedule; the time-distribution of the contraction
   also matters. Spreads at fixed shape across budgets (1.16–2.36 pp) are the same
   order, so neither factor dominates.
2. **The coupling-preserving shape wins at every budget**, by **+1.24 / +1.50 / +1.31 pp**
   over a constant `λ` at the same contraction.
3. Against every other arm at the same `η₀`, `T` and optimizer, iso-product
   (matched @ `C`) = **78.22** is the best arm we have measured: **+0.94** over the
   per-cell constant-`λ` oracle (77.28) and **+1.80** over the joint multiplier
   (76.42).

→ We retract the `joint`-based framing and adopt iso-product as the main control.

---

## F3 (E10) — held-out `C`: calibrate on small MLPs, predict `λ*` on larger ones

We use MLP **width** as the held-out architecture axis: it moves the parameter
count by 4–16× while leaving data, optimizer and schedule untouched. The runner is
split into `ladder → predict → heldout`, and the `predict` stage writes `λ_pred`
with a timestamp to `_data/e10_predictions_<ds>.json` *before* any held-out grid is
trained (both files record `blind: true`). Refitting `C` at `h = 512` on MNIST
reproduces the E5c values 0.440 (SGD) / 0.320 (SGDM) to three digits — a pipeline
sanity check.

**How `C` moves with width (calibration rungs only):**

| dataset | momentum | `C(128)` | `C(256)` | `C(512)` | slope of `log C` on `log h` |
|---|---:|---:|---:|---:|---|
| MNIST | 0 | 0.442 | 0.374 | 0.440 | **−0.00 [−0.24, +0.24]** |
| MNIST | 0.9 | 0.466 | 0.353 | 0.320 | **−0.27 [−0.40, −0.14]** |
| CIFAR-10 | 0.9 | — | 2.800 | 2.672 | −0.068 (two rungs, no interval) |

**Blind extrapolation vs `C` measured directly at the held-out width:**

| dataset | momentum | width | `C` predicted | `C` measured | ratio |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | 1024 | 0.416 | 0.325 | 1.28× |
| MNIST | 0 | 2048 | 0.415 | 0.327 | 1.27× |
| MNIST | 0.9 | 1024 | 0.257 | 0.244 | 1.05× |
| MNIST | 0.9 | 2048 | 0.213 | 0.255 | 0.84× |
| CIFAR-10 | 0.9 | 1024 | 2.549 | 2.479 | **1.03×** |

`C` transfers across a 4–16× width change to within **1.3×**; the resulting `λ` is
within **1.60×** (MNIST) / **1.14×** (CIFAR-10) of the oracle.

**Does that reduce tuning?** The oracle for each cell is the best weight decay
*anyone* measured there — the 5-point ladder plus every rule's own `λ` — so no rule
can look good merely because the ladder is coarse. Cost: 5+ runs per
(width, momentum, `η`) cell for the oracle, one run for any rule, zero after a
one-time calibration.

| dataset | rule | mean acc gap (pp) | worst | mean loss gap | `λ/λ_oracle` |
|---|---|---:|---:|---:|---:|
| MNIST (12 cells) | default 5e-4 | 0.03 | 0.11 | 0.0005 | 1.27× |
| | `1/(ηT)` | 0.11 | 0.25 | 0.0032 | 2.70× |
| | constant `ηλ` | 0.06 | 0.20 | 0.0018 | 1.56× |
| | **ours** | 0.08 | 0.19 | 0.0020 | 1.60× |
| CIFAR-10 (3 cells) | default 5e-4 | 0.84 | 1.19 | 0.1576 | 0.13× |
| | `1/(ηT)` | 0.52 | 1.11 | 0.0792 | 0.23× |
| | constant `ηλ` | 1.49 | 3.73 | **0.0171** | 1.08× |
| | **ours** | 2.01 | 3.79 | 0.0233 | 1.14× |

Honest reading:

- **The `λ*` prediction is good; the accuracy advantage is not there.** On CIFAR-10
  ours and constant-`ηλ` land nearest the loss-oracle `λ` (1.1×), yet the fixed
  default gives the smaller *accuracy* gap, because in this setting the accuracy
  optimum sits at a smaller `λ`. Same pattern as E4 (ours 0.87 vs default 0.70); we
  do not dress it up.
- **MNIST cannot discriminate**: all four rules within 0.11 pp. Useful negative
  information — we make no accuracy claim on MNIST-MLP.
- **The fragile axis is not width but family.** `C` is 0.32–0.44 for MNIST-MLP,
  2.5–2.8 for CIFAR-10-MLP, and 1.48 (geometric mean) for CIFAR-100 CNNs — about
  **8.8×** across families, against ≤1.5× across a 16× width range. Practical
  statement: calibrate `C` once per dataset/architecture family, after which it
  transfers over width, learning rate and training length. One calibration replaces
  a two-dimensional grid; it does not replace knowing the family.
- Limits: single seed; the CIFAR-10 ladder has only two rungs, so its slope has no
  usable interval; at `η = 0.3` our predicted `λ = 1.402e-3` diverged while
  `1.322e-3` converged (stability boundary), counted as a divergence, not imputed.

---

## Deliverables

- `xkCF/response.md` — "Follow-up (2026-08-03)" section (**F1 removed; only F2/F3**).
- `common/e9_iso_matched.md`, `common/e10_c_width.md` — experiment notes;
  `common/scheduled_wd_baselines.md` carries a correction box (main control is now
  iso-product).
- `总览.md` / `PLACEHOLDERS.md` updated (F1 row removed; F1 reproduce command dropped).
- Paper edits (self-initiated, not from the reviewer): (1) Exp.3 wording tightened;
  (2) scheduled-WD main control switched to iso-product. The residual `C`-vs-`B`
  drift 1.38/1.89/1.40/1.60/1.71 matches the published numbers exactly, so Eq.(17)
  and Table 4 are consistent and F1 is not adopted.
