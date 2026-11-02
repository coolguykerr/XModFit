# XPR: what the parameter-dependent mesh actually costs

**To:** Elias Boshara
**From:** Daniel Kerr (XModFit / XModJointFit side)
**Date:** 2026-09-10
**Subject:** measuring the 2026-09-09 fixed-slab-count design on real data

Verdict up front: **the trade you made was the right one, and I now have numbers for
it.** The kink you removed was fatal to us; the parameter-dependent discretization
error you introduced instead is not — it shifts fitted parameters by less than 4% of
their own error bars, orientation included. There is one place it has real
consequences (ranking near-degenerate candidates in our global search) and one
performance observation at the end that I think is worth your time.

This document is longer than a results table because the *design* of each test is the
part that makes the numbers interpretable, and because several obvious ways to measure
this give badly misleading answers. I've included the ones I got wrong first.

Everything here was run against your unmodified `XPR/` at the 2026-09-10 revision.

---

## 0. The question your design raises

Before 2026-09-09 the profile lived on a rigid 0.5 Å lattice and the *number* of cells
moved with the parameters. That gave a genuine kink: at `d = [11.0, 9.0]`, where
`sum(d)` lands exactly on a node, the one-sided `dlogL/dd1` slopes were 4.3 apart and
**stayed** 4.3 apart as h → 0, and AD disagreed with central differences by ~1e-2. The
mechanism was `sort(unique([gridz; 0.0; -cumsum(box_lengths)]))` merging a box
interface with the node it coincided with.

After 2026-09-09 the count is fixed and the *spacing* moves instead:

```
outside the boxes    spacez                        (rigid)
inside box b         d_b * spacez / max_box_length
```

That kills the kink completely — same test point now agrees to 1.5e-9, and the
one-sided slopes converge instead of staying split. But it replaces a
**non-smoothness** with a **parameter-dependent bias**: the discretization error is
now a function of `d`, so `dlogL/dd` contains a spurious term `d(error)/dd` in
addition to the physics.

For a least-squares fit that swap is obviously good — a kink breaks the gradient
outright, a smooth bias just perturbs it. But we use the model in two other ways where
it was not obvious, and that is what this study is about.

---

## 1. What we do with the model, and the three different sensitivities

Recapping briefly since this document should stand alone. `XLayersPDB_XPR` is an
XModFit model class driven three ways:

1. **lmfit least-squares**, ~8–10 free parameters, using `XPR_grad` and a
   reflectivity Jacobian as the analytic `Dfun`.
2. **emcee MCMC** for posteriors, which evaluates the model far from any optimum.
3. **A multistart global search** — Sobol or physics-seeded starts, Ipopt L-BFGS and
   Newton on your exact `XPR_grad`/`XPR_hess`, one Hessian per endpoint to classify
   MIN / SADDLE / DEGENERATE, then **dedup by a Δ(−logL) tolerance of 11–20**.

Those give three *distinct* sensitivities, and the study is organised around them:

| we need | why a moving mesh might break it |
|---|---|
| unbiased **parameter estimates** | the spurious `d(error)/dd` term shifts where `dlogL/dp = 0` |
| **absolute −logL comparable across parameter space** | step 3 ranks and merges candidate minima on a Δ(−logL) tolerance; if the error differs between two candidates, it tilts that comparison |
| an unbiased **pose** (θ, φ) | the protein is binned on the *rigid* `spacez` grid, so rotating it sweeps atoms across fixed cell boundaries — a priori the most exposed quantity, and the one our campaigns are actually about |

The third is why the pose gets its own test rather than being lumped in with the rest.

---

## 2. Setup, and how "converged" is defined

**Data.** mTIM3 `d20_tim3_1`, a real 7:3 SOPC:SOPS monolayer with mTIM3 bound,
94 points over q = 0.028–0.560 Å⁻¹, median σ/R = **0.94%**. Template `t3hmmm`
(`tim3_hmmm_ext_all_bestframe.pdb`, 1815 atoms), 2 boxes. Real data matters here
because every "is this error big?" question is answered relative to the measurement
noise, and synthetic errors would beg the question.

Starting pose is our archived XModFit fit for this dataset (θ = 51.376°, φ = 125.370°,
top-referenced insertion 8.662 Å → `d_protein` = −14.065 with maxh = 22.728), film
d = [12.731, 8.747], ρ = [0.2413, 0.4655], σ = 3.4. Default mesh (`spacez` 0.5,
`max_box_length` 30) gives an in-box spacing of **0.212 Å** and 693 smearing nodes.

**The reference.** "Converged" is a Richardson extrapolation over `spacez`
(0.5 / 0.25 / 0.125 / 0.0625). This is a clean uniform h-refinement, and worth
spelling out why: because *both* the rigid outer spacing **and** the in-box spacing
`d_b·spacez/max_box_length` are proportional to `spacez`, halving it halves the mesh
everywhere at once. Refining `max_box_length` instead would refine only the boxes and
leave the outer regions untouched — a non-uniform refinement that does not converge to
the same thing (see §3C, which is exactly about that).

**This test was not possible before 2026-09-10.** With `spacez` hard-coded I had to
edit your source in a scratch copy to do it. Exposing it is what made a proper
convergence study a one-liner, which is the main reason I asked for it.

---

## 3. The tests

### A. Is the error a well-behaved quadrature error, or something else?

**Why this first.** Everything downstream assumes the error is an ordinary
discretization error: smooth, O(h²), refinable to zero. If it were not — if the
observed order were ~1, or erratic, or non-monotone — that would signal a residual
non-smoothness or an inconsistency in the scheme, and no amount of "the bias is small"
would be reassuring, because it wouldn't be a *bias*, it would be noise.

**How.** Evaluate logL against the real data at four meshes and take successive
differences. Two successive ratios give two independent estimates of the convergence
order; consistency between them is the real test.

**Result.**

| `spacez` | nodes | in-box (Å) | logL | eval (ms) |
|---|---|---|---|---|
| 0.5 | 693 | 0.2122 | −193.27851 | 9.3 |
| 0.25 | 1383 | 0.1061 | −194.56165 | 26.4 |
| 0.125 | 2764 | 0.0530 | −194.91378 | 86.5 |
| 0.0625 | 5525 | 0.0265 | −195.00395 | 313.8 |

Successive ΔlogL: −1.2831, −0.3521, −0.0902 → **observed order 1.87 then 1.97**.
Richardson limit **logL = −195.035**. The default mesh is off by **+1.76 in logL**
(−3.51 in χ²).

**What it means.** Textbook second order, monotone, with the two order estimates
agreeing and tightening toward 2. The scheme is consistent and the error is a genuine
quadrature error. That licenses everything below, and it also means the error is
*predictable*: halving `spacez` divides it by four.

**What failure would have looked like.** Order ≈ 1 (a first-order scheme, or a
surviving kink being smoothed over), erratic ratios, or non-monotone logL.

---

### B. Where in q does the error live, and does R/R_F behave differently?

**Why.** Two separate concerns. First, our protein signal is at high q — the film
fringes dominate below ~0.25 Å⁻¹ and the protein's contribution is what shapes the
tail — so an error concentrated at high q is worse for us than the same error spread
uniformly. Second, our data files are `.rrf` (R/R_F) and we fit Fresnel-normalised, so
I needed to know whether normalising changes the picture. It was not obvious a priori:
if R_F were itself computed on the mesh, the ratio could partially cancel the error,
or amplify it.

**How.** Band-average the relative error ΔR/R between the default and converged
meshes, fit a power law in q, and repeat the whole thing with `fres=false` to get raw
R. Also express the error in units of the measured σ, because that is the only scale
on which "big" means anything.

**Result.**

| q band | n | mean ΔR/R | max \|ΔR/R\| | mean \|ΔR\|/σ |
|---|---|---|---|---|
| 0.028–0.100 | 36 | −5.95e-05 | 1.13e-04 | 0.008 |
| 0.100–0.200 | 20 | −2.98e-05 | 1.13e-04 | 0.008 |
| 0.200–0.300 | 11 | +3.72e-05 | 3.25e-03 | 0.054 |
| 0.300–0.400 | 10 | −2.88e-03 | 3.36e-03 | 0.147 |
| 0.400–0.500 | 10 | −1.65e-03 | 2.30e-03 | 0.077 |
| 0.500–0.560 | 7 | +4.64e-03 | 1.11e-02 | 0.038 |

Power law **|ΔR/R| ~ q^1.91**. Σ(ΔR/σ)² = 0.355 over 94 points. Worst single point
**0.168 σ** at q = 0.330.

**R and R/R_F carry the same relative error to 1.7e-16.**

**What it means.** The q^1.91 is exactly the Born-limit prediction and worth stating
because it identifies the mechanism rather than just the magnitude. In the kinematic
limit R/R_F ≈ |∫ρ′(z)e^{iqz}dz|², and a staircase profile turns ρ′ into a delta comb,
making the integral a midpoint quadrature with relative error ~(qh)²/24 — i.e. growing
as q². Measured 1.91. So the error is the transform quadrature, nothing more exotic.
(Aliasing is not in play: q_Nyquist = π/h is 6.3 Å⁻¹ outside the boxes and 15 Å⁻¹
inside, against q_max = 0.55 — margins of 11× and 27×.)

The R/R_F result says the Fresnel divisor is analytic in q and carries no mesh
dependence, so normalising is a no-op for this question and I can quote one set of
numbers for both.

And in the units that matter: the worst point is displaced by **0.17 σ**, in a band
where the protein signal lives. Visible if you look for it; irrelevant to a fit.

**What failure would have looked like.** q^0 (a scale error rather than a quadrature
error), oscillation in q (interference from the mesh structure — i.e. leakage), or R
and R/R_F disagreeing (R_F contaminated by the mesh).

---

### C. How much of the error is actually parameter-dependent?

**Why this is the crux.** Only the *in-box* spacing moves with the parameters;
everything outside the boxes is rigid at `spacez`. A rigid error is a near-constant
offset — it cancels from any Δχ² and cannot distort a likelihood surface. Only the
parameter-dependent part can bias a fit or reorder candidates. So the total error is
the wrong thing to worry about; I needed to separate the two.

**How.** `max_box_length` refines **only** the in-box mesh and leaves the outer regions
alone, so it isolates the parameter-dependent half exactly. Comparing
`max_box_length` 30 → 480 at fixed `spacez` gives the in-box contribution; comparing
against the fully converged mesh gives the total.

The metric here is Σ(ΔR/σ)², **not** ΔlogL — see §4, this matters.

**Result, at the fitted d₁ = 12.73:**

| | mean \|ΔR/R\| | Σ(ΔR/σ)² | Δχ² |
|---|---|---|---|
| in-box mesh only (parameter-dependent) | 2.71e-04 | 0.0240 | **+1.24** |
| total mesh error | 9.92e-04 | 0.3547 | **−3.45** |

The in-box mesh is **6.8%** of the total by the model-difference norm — the rigid
`spacez` region dominates. And note the two Δχ² have **opposite signs**: the in-box
and outer errors partially *cancel*.

In-box error alone, sweeping d₁ (this is the parameter-dependent part in isolation):

| d₁ (Å) | h_box (Å) | mean \|ΔR/R\| | Σ(ΔR/σ)² | max \|ΔR\|/σ |
|---|---|---|---|---|
| 4.0 | 0.067 | 1.21e-04 | 0.002 | 0.015 |
| 8.0 | 0.133 | 1.09e-04 | 0.025 | 0.069 |
| 12.73 | 0.212 | 2.71e-04 | 0.024 | 0.050 |
| 18.0 | 0.300 | 1.48e-03 | 0.116 | 0.115 |
| 24.0 | 0.400 | 1.24e-03 | 1.769 | 0.573 |
| 30.0 | 0.500 | 3.46e-03 | 7.980 | 1.379 |
| 40.0 | 0.667 | 7.81e-03 | 5.694 | 0.754 |
| 55.0 | 0.917 | 9.91e-03 | 66.584 | 4.495 |

**mean |ΔR/R| ~ h_box^1.96** — again clean h².

**What it means.** Three things, in increasing order of practical importance.

1. The parameter-dependent part is the *minor* part, which is why the bias in §D
   comes out so small.
2. **Refining `max_box_length` alone makes absolute χ² worse, not better.** Because
   the in-box and outer errors have opposite signs, removing one destroys a partial
   cancellation. I first saw this on your `protein.pdb` test case, where pushing
   `max_box_length` 30 → 400 moved logL *away* from the converged limit
   (−464.54 → −464.36 against a limit of −465.68). This is a genuine trap for anyone
   who reads `max_box_length` as "the accuracy knob", and it is the strongest argument
   for `spacez` being the exposed one.
3. The error grows as h_box², so it is governed by `d_b / max_box_length`. At the
   default `max_box_length = 30`, our films (d ≈ 9–27 Å) sit at h_box = 0.15–0.45 and
   are fine. A 55 Å box would reach 4.5 σ at one q point. That's a usage rule, not a
   defect — but it does mean `max_box_length` should be set to at least the thickest
   box one intends to fit, which is not obvious from the name.

---

### D. Does it bias the fitted parameters?

**Why.** This is the question that decides whether any of the above matters for a
published number. A 3.5 χ² offset in the absolute likelihood is irrelevant if the
*location* of the minimum doesn't move.

**How.** Fit the real data properly — 8 free parameters (`qoff`, `protins`, `cov`,
`sig`, d₁, d₂, ρ₁, ρ₂), same starting point — once on the default mesh and once at
`spacez = 0.125`, and compare parameter by parameter. Two design choices worth
flagging:

- **Shifts are expressed in units of each parameter's own standard error.** An
  absolute shift in Å is uninterpretable: 0.006 Å is negligible for a thickness with
  σ = 0.2 Å and would be alarming for one with σ = 0.001 Å. The dimensionless ratio is
  the only thing that answers "does this change my answer?".
- **A 2×2 cross-evaluation** on top: score both parameter sets on both meshes. This
  separates two things that a single comparison conflates — "the mesh moved the
  optimum" (a real bias) from "the mesh moved χ² everywhere" (a common-mode offset
  that cancels from any Δχ² and costs nothing).

**Result.** χ² at the two optima: 129.763 (default) vs 128.957 (fine), over 94 points.

| parameter | fit @ 0.5 | fit @ 0.125 | shift | stderr | **shift/σ** |
|---|---|---|---|---|---|
| qoff | 6.056e-05 | 6.009e-05 | +4.7e-07 | 1.60e-05 | 0.030 |
| protins | −14.1972 | −14.1888 | −0.0083 | 0.291 | −0.029 |
| cov | 15.123 | 15.120 | +0.0029 | 0.168 | 0.017 |
| sig | 3.2116 | 3.2107 | +0.0009 | 0.115 | 0.007 |
| d₁ | 12.3347 | 12.3289 | +0.0058 | 0.205 | 0.029 |
| d₂ | 9.87689 | 9.88524 | −0.0083 | 0.638 | −0.013 |
| ρ₁ | 0.233962 | 0.233870 | +9.3e-05 | 0.0041 | 0.023 |
| ρ₂ | 0.448384 | 0.448224 | +1.6e-04 | 0.0085 | 0.019 |

**Worst shift = 0.030 σ.**

Cross-evaluation:

| χ² | at P(0.5) | at P(0.125) |
|---|---|---|
| mesh 0.5 | 129.7626 | 130.0501 |
| mesh 0.125 | 129.2451 | 128.9568 |

**What it means.** Read the second row: scored on the fine mesh, the default-mesh
parameters cost **+0.29 χ²** relative to the fine-mesh parameters. That is the entire
cost of the mesh bias, on a fit with χ² = 129. The **+0.52 χ²** offset between the two
rows is the common-mode mesh error, which is what makes absolute χ² non-transferable
but cancels from every Δχ² taken within one mesh.

So: for parameter estimation the default mesh is simply not a consideration. A 0.03 σ
shift is two orders of magnitude below anything we would report.

**What failure would have looked like.** shift/σ approaching 1 for any parameter, or a
large bias cost in the cross-table.

---

### E. Does it move the pose?

**Why separately.** §D held θ and φ fixed. That leaves the most exposed quantity
untested, for a specific reason: the protein's density is binned onto the **rigid**
`spacez` grid, so rotating the protein sweeps atoms across cell boundaries that do not
move. If any parameter were going to pick up a mesh artifact, orientation is the
candidate — and orientation is the entire point of our campaigns, so "small bias in d"
would not have been a sufficient answer.

**How.** Same as §D but with θ and φ released (10 free parameters), same start, both
meshes.

**Result.** χ² 128.810 (default) vs 128.202 (fine).

| parameter | fit @ 0.5 | fit @ 0.125 | shift | stderr | shift/σ |
|---|---|---|---|---|---|
| **theta** | 51.6111 | 51.5742 | **+0.0368°** | 0.984 | 0.037 |
| **phi** | 126.656 | 126.543 | **+0.113°** | 7.673 | 0.015 |
| (the other 8) | — | — | — | — | ≤ 0.020 |

**What it means.** The mesh moves the fitted orientation by **0.04° in θ and 0.11° in
φ**. Our pose-resolution tolerance is of order 5°, and replicate disagreement between
nominally identical datasets is 11–20 in −logL. So this is three orders of magnitude
inside anything we can resolve. The most exposed quantity turned out to be the least
affected — which makes sense in hindsight, since the protein contribution is a sum
over 1815 atoms and their individual binning errors average down.

---

### F. Does it reorder near-degenerate candidates?

**Why.** This is the one place a *varying* error can do real damage. Our multistart
returns a list of minima and merges them on a Δ(−logL) tolerance of 11–20. If the
mesh error differs between two candidates, it tilts that comparison directly — and
unlike §D, no amount of "the optimum barely moves" helps, because here we are
comparing *different* optima against each other.

**How, and this design point took two attempts.** The relevant ensemble is "other
candidate minima at comparable χ²", so I perturb around the optimum and keep points
within +100 χ². Crucially the statistic is the **spread** of the mesh error across
candidates, not its mean: a common-mode offset cancels from every pairwise comparison
and cannot reorder anything. Only the variation can.

**Result.** 55 candidates within +100 χ²:

- mesh error (χ²_default − χ²_fine): min **−5.23**, max **+7.94**, spread **13.16**
- **sd = 2.53 χ² = 1.26 logL**

**What it means.** The mesh contributes a **~1.3 logL noise floor on candidate
ranking**. Against our 11 logL merge tolerance that is ~11% — the tolerance itself is
safe. But it means two candidates within ~1.3 logL of each other **cannot be reliably
ordered on the default mesh**. In a study whose whole output is a ranked list of
minima, that is a real if narrow constraint, and it is the one operational consequence
of the moving mesh that we have had to change our procedure for (we now re-score the
shortlist at `spacez/2` before declaring a winner).

---

## 4. Three metrics that do NOT work, and why

Worth recording, because they explain why the tables above look the way they do — and
because two of them were my first attempt.

**ΔlogL across a parameter sweep is useless.** My first §C swept d₁ and reported
`logL(coarse) − logL(fine)` at each point. The numbers were nonsense: ±1740 at
d₁ = 4 Å, ±14000 at d₁ = 55 Å. The reason is that far from the optimum logL itself is
~10⁶, so a 10⁻⁴ *relative* change in the model shows up as a huge absolute ΔlogL that
says nothing about the mesh. Σ(ΔR/σ)² measures how much the *prediction* moves and is
independent of how bad the fit happens to be at that point. Any mesh comparison across
a parameter sweep needs a metric of that kind.

**Relative error ΔR/R is ill-conditioned near fringe minima.** At one point in a d₁
sweep on your test protein, R passed within 1.7e-06 of zero, and ΔR/R there was 7%
with 89% of its power at short periods — which looked exactly like a leakage artifact
and was purely division by a near-zero. Recomputing in ΔR/σ dropped the short-period
fraction to 0.3%. Anything expressed as ΔR/R needs a check on where R is small.

**A ratio of oppositely-signed Δχ² is meaningless.** I initially reported the in-box
mesh as "36% of the total error in χ²" from |+1.24| / |−3.45|. Those have opposite
signs — they partially cancel — so the ratio describes nothing. The model-difference
norm (6.8%) plus an explicit statement about the cancellation is the honest version.

**And independent 1σ perturbations do not sample near-degenerate candidates.** My
first §F perturbed each of 8 parameters by its own marginal σ independently, and
*every* draw landed beyond +100 χ². These parameters are strongly correlated —
`qoff`, `protins`, `cov`, d and ρ all trade against each other in XRR — so the joint
1σ ellipsoid is far narrower than the box of marginal σ's. Drawing from the fit
covariance gives the intended ensemble with E[Δχ²] = ndim.

---

## 5. Verdict on the design change

The 2026-09-09 trade was correct, and the numbers say so cleanly:

- **What you removed** was a kink with one-sided slopes 4.3 apart that did not vanish
  as h → 0, in a model whose entire purpose is exact derivatives. It broke `XPR_hess`
  wherever it occurred, which silently corrupts our MIN/SADDLE classification rather
  than failing loudly.
- **What you introduced** is a smooth O(h²) bias that moves fitted parameters by
  ≤ 0.04 σ, orientation included, and contributes ~1.3 logL of noise to candidate
  ranking.

That is a very good exchange. As a bonus the fixed-count design also removed the
entire segfault class — `d1 = 300` with `d_protein = −500` now returns a finite
answer, where before it wrote past the end of a preallocated array and took the whole
Qt process down with it. We have retired the Python-side `check_headroom` guard that
existed only for that, and replaced it with a mirror of your new window check.

---

## 6. What we changed on our side

For completeness, since some of it is a direct consequence of the above:

- **The sizing arguments are now split by whether they change the answer.** Measured:
  `spacez` and `max_box_length` do; `d_protein_bounds` (4e-11), `sig_max` (6e-11) and
  `min_box_length_sum` (4e-13) do not. So we derive the *fatal window* from the fitted
  parameter bounds and the *answer-bearing mesh* from fixed model attributes —
  deliberately **not** from the bounds, so that widening a bound cannot silently
  change χ².
- **We deleted a ratcheting sizing heuristic.** Our old code grew `max_box_length`
  from live parameter values, quantized upward, because it only ever sized buffers.
  Under the new code that would have stepped the objective mid-descent every time the
  ratchet clicked.
- **`spacez` is exposed to the user** as an accuracy knob, and we publish the slab
  count and in-box spacing as fit outputs so the cost of one's bounds is visible.
- **Documented rule:** parameters and Δχ² within one mesh are trustworthy; absolute χ²
  is not transferable across `spacez`/`max_box_length` or across code versions.

---

## 7. Two things for you

### 7.1 A documentation point: `max_box_length` is not the accuracy knob

Worth a comment in the source, because the name invites the opposite reading. Since
the in-box and outer errors partially cancel at the defaults, **raising
`max_box_length` alone degrades the absolute likelihood** — measured on your
`protein.pdb` case, −464.54 → −464.36 against a converged −465.68. `spacez` is the
knob that converges; `max_box_length` sets *where* the in-box mesh equals `spacez` and
should be chosen to be at least the thickest box the caller intends to fit.

### 7.2 A performance observation: `eval_smeared!` is doing a lot of provably zero work

The double loop is O(B²) in erf calls:

```julia
@inbounds for i = 1:(B-1)
    @simd for j = 2:(B-1)
        ρ1 += (ρ_hard_z[j] - ρ_hard_z[j-1]) * erf((zi - smear_lattice[j]) / sqrt2sig)
```

and B is now fixed and fairly large. Measured on `protein_2box` at `spacez = 0.5`:

| case | M | B | ms/eval |
|---|---|---|---|
| protein, `sig_max` 5 | 358 | 557 | 12.14 |
| protein, `sig_max` 12 | 358 | 837 | 17.33 |
| **no protein**, `sig_max` 5 | 122 | 321 | **3.63** |
| **no protein**, `sig_max` 12 | 122 | 601 | **7.99** |

The no-protein rows are the diagnostic: with `include_protein = false` there is no
area/volume work at all, so that 3.63 ms is essentially the erf sum alone. It is the
dominant cost in both cases.

**But most of the `j` terms are structurally zero.** `ρ_hard_z` is filled with
`ρ_bottom` across the whole bottom tail and `ρ_top` across the whole top tail, so
`ρ_hard_z[j] - ρ_hard_z[j-1]` is exactly zero there — **200 of 556 terms (36%)** in
the protein case at `sig_max = 5`. And with no protein, `ρ_hard_z` is constant within
each box as well, so only **`nbox + 1` = 3 of 320 terms (0.9%)** are nonzero. A
bare-film evaluation is spending ~99% of its time multiplying erf by zero.

That matters to us specifically because our workflow fits a lot of protein-free
lipid-only scans — the homometry seeding that supplies membrane candidates to the
multistart runs on them.

Two options, in increasing order of effort:

1. **Precompute the indices where `ρ_hard_z` actually changes** and loop over those.
   One caveat worth flagging: do *not* implement this as a runtime `iszero(Δρ)` test,
   because under ForwardDiff a Dual can have a zero value with nonzero partials and
   you would silently drop derivative information. Build the index list
   *structurally* — from `top_cells` / `box_cells` / `bottom_cells` and whether a
   protein is present — which is known at load time and is exactly where the zeros
   come from. (In the tails the difference is genuinely zero including partials, since
   `ρ_top`/`ρ_bottom` are `Float64` constants; within a box without a protein
   consecutive cells hold the *identical* Dual, so that difference is exactly zero
   too. But the structural route avoids having to rely on that.)
2. **Exploit the Toeplitz structure.** In the rigid regions both grids are uniform
   with spacing `spacez`, so `erf((z_i − lattice_j)/√2σ)` depends only on `i − j`.
   That makes the sum a discrete convolution — tabulate erf once per lag, or FFT it
   for O(B log B). More invasive, but it is the general fix and would decouple cost
   from `sig_max` and `d_protein_bounds` entirely.

Option 1 looks like a contained change with a large payoff on the no-protein path, and
no numerical consequences at all if the index list is built structurally.

---

Happy to send the probe scripts — the study is a single self-contained file
(`julia/resolution_study.py`) that prints all of the above plus its own conclusions,
and it now runs unmodified against your tree because `spacez` is a keyword argument.
