# XPR review notes — the 2026-09-08 and 2026-09-09 updates

**To:** Elias Boshara
**From:** Daniel Kerr (XModFit / XModJointFit side)
**Date:** 2026-09-09

Short version: **the fixed-slab-count change works.** It fixes the last remaining
non-smoothness, and as a bonus it removes an entire crash mode we had been guarding
against in Python. Everything below is either a measurement you might want for your
own records, or a small concrete request. Four requests, in priority order, are in
§2; everything after that is evidence.

---

## 0. What we do with XPR, so the priorities make sense

XPR is wrapped as an XModFit model class, `XLayersPDB_XPR`, and driven three ways:

1. **lmfit least-squares** on ~10 free parameters (`qoff, theta, phi, d_protein, C,
   sig, d1..dn, rho1..rhon`), using `XPR_grad` and a reflectivity Jacobian for the
   analytic `Dfun`.
2. **emcee MCMC** for the posterior, 16–100 walkers, which means the model gets
   evaluated at parameter values well away from any optimum.
3. **A multistart global search** — Sobol or physics-seeded starts, Ipopt L-BFGS and
   Newton with your exact `XPR_grad`/`XPR_hess`, then one Hessian per endpoint to
   classify each as MIN / SADDLE / DEGENERATE, then dedup by a Δ(−logL) tolerance.

So we lean on four properties, and they're why the notes below emphasise what they do:

- **Exact derivatives, hence genuine C¹/C² smoothness.** A kink makes `XPR_grad`
  return a one-sided derivative and makes `XPR_hess` meaningless, which silently
  corrupts the MIN/SADDLE classification rather than failing.
- **Absolute −logL comparable across parameter space**, because the multistart merges
  basins on a Δ(−logL) tolerance (currently 11–20).
- **No crashes.** This runs inside a PyQt GUI process; a segfault takes the whole
  application down and loses the user's session.
- **Cost at wandering parameter values**, not just near the optimum, because of (2).

Test system throughout: your `test_data/protein_2box.rrf` + `protein.pdb` (1697
atoms), 2 boxes, `vol_calc="approx"`, 78 q-points over 0.026–0.550 Å⁻¹. Everything
was run against an unmodified copy of your `XPR/` at a separate project path; where I
needed to vary `spacez` I edited a scratch copy, noted as such.

---

## 1. Verdict on the 2026-09-09 update

**The smoothness problem is fixed, comprehensively.** Every discrete operation is
gone from the hot path — no `ceil`/`floor`/`round`, no `sort`/`unique`, no `collect`
over a parameter-dependent range. The node count is a load-time constant and node
positions are affine in the box lengths.

The specific defect that survived your 2026-09-08 version was a kink whenever a
cumulative box length landed exactly on the 0.5 Å lattice, because
`sort(unique([gridz; 0.0; -cumsum(box_lengths)]))` merged the box interface with the
grid node it coincided with. That's gone: box interfaces are now nodes *by
construction* (`τ = k/K` hits `box_bottom` exactly at `k = K`), so there is nothing
to merge.

| test, at `d = [11.0, 9.0]` (sum exactly 20.0, i.e. on the old lattice) | 2026-09-08 | 2026-09-09 |
|---|---|---|
| AD vs central FD, `d1` | ~1e-2 | **1.5e-9** |
| one-sided `dlogL/dd1`, left vs right, as h→0 | stuck at 4.3 | **→ 0 linearly** (9.6 → 0.96 → 0.093 → 0.019) |

And the step-size ladder is clean for every parameter — AD vs central FD falls as
exactly h², the signature of pure truncation with no kink floor:

```
          h=1e-2   h=3e-3   h=1e-3   h=3e-4   h=1e-4
theta    3.8e-06  8.2e-07  1.6e-07  1.8e-08  3.0e-08
phi      5.8e-07  1.7e-07  3.8e-08  3.1e-08  2.7e-07
protins  7.9e-06  5.0e-07  2.4e-07  5.0e-08  9.2e-09
sig      6.9e-03  6.2e-04  6.9e-05  6.2e-06  6.8e-07
d1       3.2e-03  2.9e-04  3.2e-05  2.9e-06  3.3e-07
rho2     2.3e+00  2.1e-01  2.3e-02  2.1e-03  2.3e-04
```

The centroid-referenced `d_protein` from 2026-09-08 also does what it was meant to.
Along φ = 112.59° the topmost atom changes at 10 values of θ in [1°, 179°], and
`d(maxh)/dθ` genuinely jumps at each (4e-4 to 1.2e-1). `dlogL/dθ` no longer does:
worst left/right disagreement over those 10 angles is **3.3e-5** relative, with the
AD gradient sitting between the two one-sided differences.

**The fixed count is genuinely fixed.** `B = 557` and `M = 358` were invariant across
`d1 ∈ [0.5, 80]`, `d_protein ∈ [−500, 400]`, `sig ∈ [1, 200]` — as they must be,
since only load-time constants enter them.

**And the segfault class is gone.** `d1 = 300` together with `d_protein = −500`
returns a finite result. We had built a Python-side `check_headroom` guard purely to
stop an out-of-range proposal from overrunning `max_ed_slabs`/`max_refl_slabs`; we
can retire it. This is a real improvement for us and I don't think it was the headline
you had in mind.

---

## 2. Requests

### 2.1 Please expose `spacez` (highest value of the four)

`spacez = 0.5` is hard-coded at line 210, and it is now **the dominant error term**,
while `max_box_length` — which *is* exposed — controls the part that matters least.

Mesh refinement study at the fit point, refining `spacez` and the box mesh together
(scratch copies with `spacez` edited, `max_box_length = 30` throughout):

| `spacez` | box cells K | h_box | logL |
|---|---|---|---|
| 0.500 | 60 | 0.194 | −464.541686 |
| 0.250 | 120 | 0.097 | −465.299332 |
| 0.125 | 240 | 0.049 | −465.550971 |

Observed convergence order p = 1.59, extrapolating to a continuum limit of
**−465.676**. Against that:

| | logL | error |
|---|---|---|
| 2026-09-09 at defaults | −464.5417 | **+1.13** |
| 2026-09-09, box mesh refined (`max_box_length = 400`) | −464.3556 | +1.32 |
| 2026-09-08, uniform 0.5 Å | −466.4557 | −0.78 |

Two things fall out of that table. First, **refining `max_box_length` makes the
absolute likelihood slightly worse, not better** — the box-mesh and outer-mesh errors
have opposite signs and partially cancel at the default, and refining only the boxes
destroys the cancellation. Second, the new scheme is not more accurate than the old
one in absolute −logL; it's comparable magnitude, opposite sign. (Its *smoothness* is
much better, which is what we actually needed.)

The reason `spacez` dominates is geometric. At the fit point the protein spans
z = −61.7 to −7.9 Å while the film is 0 to −21.3 Å. So only 13.4 Å of the protein's
53.7 Å extent lies inside the boxes at h = 0.194 Å; the other **40.4 Å — 75% of it —
lies below the stack at the hard-wired 0.5 Å**. `max_box_length` refines the 25%.

Either a `spacez` keyword or a "target spacing for the non-box regions" keyword would
let us trade cost against accuracy deliberately. Right now we can only make the
absolute likelihood worse.

### 2.2 `top_cells` is over-provisioned by `sum(d)`

```julia
top_cells = ceil(Int, max(0.0, dmax + prot_extent) / spacez)
```

but the protein's top sits at `d_protein + prot_extent − sum(box_lengths)`. The bound
is therefore taken at `sum(d) = 0`. It is correct and safe — I checked the coverage
inequality holds for all `d_protein ∈ [dmin, dmax]` — just larger than needed by
`sum(d)/spacez` cells. At our defaults that's 118 cells (59 Å) above z = 0 where
37.7 Å would do, and since `eval_smeared!` is O(B²) those cells are not free.

If `load_data` took a lower bound on `sum(box_lengths)` (or a `min_box_length`), you
could subtract it. Minor, purely a speed matter.

### 2.3 Out-of-range `d_protein` clips the protein silently

With the defaults (`d_protein_bounds = (-30, 30)`, `prot_extent = 28.97`,
`sum(d) = 21.31`) the grid reaches 59.0 Å above z = 0, so the protein's top hits the
ceiling at `d_protein = +53.2`. Past that it is quietly truncated:

| `d_protein` | relative error vs a grid wide enough to contain it |
|---|---|
| 50 | 6.7e-15 (still inside) |
| 60 | 2.7e-2 |
| 80 | 3.5e-1 |
| 150 | 4.3e-1 |
| 400 | 4.5e-1 |

At 150 and 400 the result is **bit-identical** (−94966.9092): the protein is entirely
outside the grid and the model has silently become the bare film.

The old behaviour here was a segfault, which is worse in a GUI but better as a signal.
A wrong-but-finite-and-*smooth* likelihood is the hardest kind for us, because an
MCMC sampler will accept it and a gradient-based solver will walk into it happily —
nothing anywhere reports a problem. An `error()` when
`d_protein + prot_extent − sum(box_lengths) > top_cells*spacez` (and the mirror
condition below) would be ideal; a returned flag would also work. We're adding a
Python-side bounds check regardless, but the library knows the true condition and we
only know an approximation of it.

### 2.4 One-line bug in `XPRsim._import_pdb` (predates this update)

```python
element_key = 'Ca' if atom.name == 'CAL' else atom.element
electrons[ind] = atominfo[element_key][0]
radii[ind]     = atominfo[element_key][1] / 100.0
masses[ind]    = atominfo[atom.element][2]      # <-- not element_key
```

The `CAL` override is applied to the electron count and the vdW radius but not the
mass. In `test_data/protein.pdb` the element columns (77–78) are blank for that atom,
so Bio.PDB guesses `'C'` and it gets 12.011 instead of calcium's 40.078.

That was cosmetic before 2026-09-08, because the masses were unused. Now `prealloc`
mass-centres the coordinates, so it moves the centroid — and therefore **the zero of
`d_protein`** — by **0.0345 Å** along the rotated z axis for this structure. It's
worth 60 in logL at your demo point. Concretely: we could not reproduce the demo
notebook's protein cell (−897.7146) at the stated `d_protein = -15.058030329245691`;
we get −958.3173. Correcting for the centroid offset, i.e. evaluating at
−15.023486, gives **−897.7154**, which matches to 8e-4. That offset is the whole of
the discrepancy and nothing else.

Related, same line: Bio.PDB upper-cases element symbols while `mendeleev` keys the
table by proper-case symbol, so `atominfo[atom.element]` raises a bare `KeyError` on
any **two-letter element** — `Nd`, `Zn`, `Se`, `Cl`, `Mg`. Single-letter elements (all
of a plain protein) are unaffected, which is why it hasn't surfaced. We hit it on an
unrelated structure. A `.capitalize()` fallback covers it.

---

## 3. The question we were most worried about, and the answer

Your note said the runtime is a little slower but you believed the issue was patched.
The thing we went looking for beyond that was whether **making the resolution a
function of the fit parameters** introduces artifacts downstream in Parratt. The
worry was by analogy with Fourier methods: in the Born limit
R/R_F ≈ |∫ρ′(z)e^{iqz}dz|², a staircase profile makes ρ′ a delta comb, and changing
the sampling grid as a parameter varies is the kind of thing that produces
leakage/scalloping. Since the multistart merges basins on a Δ(−logL) tolerance, a
rippling numerical error would manufacture spurious local minima and we would spend
compute cataloguing them.

**It does not happen, and structurally it cannot.** The artifact needs a *fixed*
lattice for the profile to beat against:

- **Your old scheme** pinned nodes to a fixed 0.5 Å lattice and let the *count* vary
  with d. So `d mod 0.5` was a real variable — that is a commensurability, and it is
  exactly the kink measured in §1.
- **The new scheme** puts nodes at `k/K` of each box with the count fixed. There is no
  lattice, so `d mod anything` is not a variable. The only d-dependence remaining is
  the smooth amplitude `h = d/K` of the quadrature error.

Measured on a `d1` sweep from 6 to 26 Å in 0.05 Å steps, comparing the default mesh
against `spacez = 0.125` / K = 240, with the error expressed in units of the data σ
(`dR/R` is ill-conditioned near fringe minima and gives misleading numbers):

| quantity | result |
|---|---|
| power at 0.5 Å period (the old commensurability) | **1.2e-6 – 1.8e-4** of total |
| short-period power (< 1 Å) | 6e-3 – 3.5e-2 of total |
| sign changes in d(error)/d(d1) over 401 points | **1** |
| local maxima of logL(d1), coarse / fine | **1 / 1** |
| local maxima of logL(d_protein) across z = 0, coarse / fine | **1 / 1** |
| error in the total-reflection plateau (q < 0.05) | 1.9e-5 |

The remaining power is smooth long-period trend. Three readings:

- **The coarse mesh did not invent a local optimum** — identical extremum counts on
  both meshes, in `d1` and in `d_protein` including the z = 0 crossing. That's the
  direct answer.
- **The mesh-density steps at z = 0 and at each box interface do not add reflection.**
  A spurious interface would give a q-independent offset in the plateau; instead the
  error vanishes as q → 0. Which is as it should be — Parratt is exact for whatever
  contiguous stack it's given, and the stack is a legitimate sampling of the true
  profile however the spacing varies along it.
- **No aliasing.** q_Nyquist = 6.3 Å⁻¹ (outer mesh) and 16.2 Å⁻¹ (box mesh) against
  q_max = 0.55 Å⁻¹ — margins of 11× and 29×.

The error's q-dependence is exactly the textbook quadrature signature. Mean relative
error by band, default vs refined mesh at the fit point:

| q band | 0.026–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.56 |
|---|---|---|---|---|---|---|
| mean dR/R | −3.9e-5 | +6.0e-5 | +1.1e-3 | −2.4e-3 | −2.2e-3 | +1.1e-3 |

A power-law fit gives **|dR/R| ~ q^2.23**, against the midpoint-rule prediction of q².
Only 3 sign changes across 78 points, and those sit at fringe minima where R is small.
So it is a smooth, q²-growing multiplicative distortion — concentrated, unavoidably,
exactly where the protein signal is.

### 3.1 The one consequence that is real

Because `h = d/K`, the error amplitude grows with box thickness:

| | at the fit point (h = 0.19) | worst in the sweep, d1 = 24.9 (h = 0.41) |
|---|---|---|
| max mesh error, in data σ | 0.13 σ | **1.48 σ** (at q = 0.28) |
| Σ(ΔR/σ)² of the model difference | 0.58 | **9.87** |

For **fitting** this is harmless. Solving `dlogL/dp = 0` in one dimension on each
mesh moves the optimum by:

| | d1 | d2 | C | sig | d_protein | rho1 |
|---|---|---|---|---|---|---|
| `spacez` 0.5 → 0.125 | 0.0019 Å | 0.0010 Å | 0.04% | 0.0008 Å | 0.0009 Å | 0.007% |

— orders of magnitude inside any realistic uncertainty, because near an optimum the
error's *gradient* is small even where its magnitude isn't.

For the **global search** it matters more. Comparing −logL across endpoints with very
different `d` means comparing points whose mesh error differs by ~10 in χ², and our
basin-merge tolerance is 22–40 in χ². It's a systematic tilt, not noise. That's a
consequence of the design rather than a bug, and it is the strongest argument for
§2.1: with `spacez` exposed we could run the search at a mesh where the tilt is
below our tolerance and only pay for it during the search.

---

## 4. Cost, and the reference values

Cost, warm, at the fit point — **taken under CPU contention from another job, so treat
the absolute numbers as ±2× and the pattern as the finding**:

| | 2026-09-08 | 2026-09-09 |
|---|---|---|
| `XPR_sim_ref` | ~3.5 ms | ~4.8 ms |
| `XPR_grad` (12 params) | ~12 ms | ~29 ms |
| `XPR_hess` | ~288 ms | ~832 ms |

Slower at the optimum, as you said. But it is now **flat across parameter space**,
where the old code ran 1.8 ms at the fit point and 15–20 ms for a far-flung walker
(the profile z-span used to set the cost). At a walker sitting at `d = [210, 320]` the
new code takes ~7.5 ms against the old ~15 ms. For our MCMC that is probably a net
win, and it removes a pathology where a chain crawled until burn-in pulled the walkers
back. The Hessian cost is the one that hurts us — we take one per multistart endpoint.

Two knobs we found: `d_protein_bounds` and `sig_max` change B, and therefore cost,
but **do not change the answer** (logL identical to 4 decimals across
(−30,30) → (−60,60) and sig_max 5 → 20; they only extend constant-density far field).
`max_box_length` **does** change the answer, since it sets K. Worth a comment in the
source — we had been ratcheting `max_box_length` from live parameter values on the
assumption it only sized buffers, which under the new code would step the objective
mid-descent. Fixed on our side.

Finally, the reference log-likelihoods all moved, purely from the new discretization.
These are with no PDB, so they're directly comparable to your `demo.ipynb` outputs:

| | 2026-09-08 (your demo) | 2026-09-09 (measured) |
|---|---|---|
| 2-box | −1214.1600 | **−1212.2127** |
| 3-box | −600.6398 | **−600.1768** |
| 4-box | −62408.5320 | **−62404.0635** |

You may want to refresh the notebook. One request if you do: the protein cell's
`d_protein = -15.058030329245691` is not the same physical pose as the previous
version's `13.358030329245691` (the conversion between the two conventions is
`d_protein_new = d_protein_old − maxh(θ,φ)`, which would be −13.717), so we couldn't
use that cell to check that the centroid redefinition was a pure reparameterisation.
We verified it another way — our stored reference −466.4557 reproduces **exactly** at
the converted insertion, and `protein_height` comes back as 13.358030329246 — but
keeping the notebook's protein point at a physically corresponding value across a
convention change would make that check a one-liner for us next time.

---

## 5. Summary of asks

| # | ask | why |
|---|---|---|
| 1 | Expose `spacez` (or a target spacing for the non-box regions) | it is the dominant error term, and the exposed knob refines the wrong 25% and makes absolute −logL worse |
| 2 | Subtract a lower bound on `sum(box_lengths)` from `top_cells` | pure speed; `eval_smeared!` is O(B²) |
| 3 | Error (or flag) when `d_protein` pushes the protein past the grid | currently silent clipping to a smooth, wrong likelihood; MCMC accepts it |
| 4 | `masses[ind] = atominfo[element_key][2]`, and a case-fallback on the element lookup | 0.0345 Å centroid shift on `protein.pdb`; `KeyError` on any two-letter element |

Nothing here blocks us. We'll be integrating the 2026-09-09 version shortly and can
work around all four. Happy to send the probe scripts if you'd like to reproduce any
of it.
