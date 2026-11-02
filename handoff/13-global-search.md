# 13 — Fast global search (`fastscan/`)

Status: **engine and tool implemented; Increment-0 gates and the full Qt wiring
verified by running them.** 178 headless checks plus 33 against a real `XModFit`
widget, all passing. Unlike the rest of `handoff/`, the numbers in §4 were measured
in this tree, not read off the source. What has NOT been done is a human driving the
real GUI on a real protein-free scan — see §8.

## 1. Why this exists

`XLayersPDB_XPR`'s pose block (`theta, phi, protins, cov`) is badly multimodal, and
XModFit's single lmfit descent lands in whichever basin the user started near. The
external seeding study established a two-stage seeded multistart that fixes this and
specified an XModFit feature for it. The design plan and its evidence base live
**outside this repository**, in `../xpr-seeding-study` — deliberately, and
`.githooks/pre-push` blocks those filenames, so do not copy them in here. The two
documents that matter:

- `PLAN_XMODFIT_FASTSCAN.md` — the original feature spec.
- `reports/INTERPRETATION-2026-09-11.md` §§3.1–3.4 — **post-dates and partly
  supersedes it.** Read it before changing any of the reporting behaviour.

The correction that shapes the whole design: the method is a **good optimiser and a
poor enumerator**, both measured. It finds the deepest basin reliably (51/53
datasets out of sample). But a 64-start screen finds a median of **7** distinct
certified minima where a 384-start reference finds **49**, which collapses the
apparent pose span from 90.4° to 23.7° and calls **11.9%** of cases pose-*determined*
that the reference calls pose-*ambiguous*. And the ambiguity threshold the original
spec hard-coded (Δ ≈ 11–20) is **retired**: that scale varies ten-fold between strata
of one corpus (2.55 → 25.90).

So this ships as a **basin-finder with a calibrated humility statement**, and that is
enforced structurally, not just in wording — see §3.3.

## 2. Files

| path | what |
|---|---|
| `fastscan/modes.py` | `Mode` / `ModeSet` / `Provenance` / `Power`; position-only dedup, great-circle metric, bound-aware certification, Laplace weights. **Qt-free and Julia-free**, so every rule is unit-testable. Read its module docstring first. |
| `fastscan/core.py` | `scan_box`, `load_for_scan`, `active_arrays`, `sobol_starts`, `scan`, `to_modes`. numpy + scipy + `xpr_backend` only. |
| `fastscan/seeds.py` | membrane-seed screening (drops, never clips), pooling across scans, `.seeds`/study-JSON IO. Owns `RHO_PHYS_MAX = 0.62`. |
| `fastscan/io.py` | the `.xmodes` sidecar, and `export_mode_as_xpar` for promoting one mode. |
| `fastscan/worker.py` | `spawn`-child entry point. Must not import Qt, `xmodfit` or `Fit_Routines`. |
| `fastscan/solvers.jl` | Ipopt bridge, ported from the study harness, names prefixed `xpr_`. |
| `fastscan/homometry.py` | stage-0 enumerator adapter: array input, per-file caching, and the two refusals. |
| `fastscan/_xrr_vendor.py` | the enumeration pipeline itself, vendored near-verbatim (78 kB) from the `xrr-homometric-pipeline` skill's `xrr.py`. Self-contained numpy/scipy; matplotlib made lazy. Do not add features here — keep it re-syncable. |
| `Tools/GlobalSearch/FastGlobalSearch.py` | the whole UI: four steps in one dialog. |
| `validate_fastscan.py` | parts A (pure Python), B (Julia), C (Qt/plugin contract), D (spawn child), E (stage-0 enumerator). 178 checks. |
| `validate_fastscan_gui.py` | the same feature against a REAL `XModFit` widget offscreen: the live parameter tables, `update_plot`, `undoFit`, save/load. 33 checks. Separate process by necessity — see §4a. |

Changed shared file: **`Functions/xpr_backend.py` only**, +146 lines, purely
additive (`include_solvers`, `solvers_available`, `solve`, `IPOPT_STATUS`).
Verified absent from `origin/develop`, so zero merge cost. `xmodfit.py`,
`Fit_Routines.py` and `PlotWidget.py` are **untouched** — the Tools plugin
mechanism (`create_menus:646-658`, `launch_tApp:670-682`) gives the menu entry free.

## 3. How it works

### 3.1 The algorithm

Stage 0 seeds (three routes, §3.4) → stage 1 draws `N1=64` scrambled Sobol points
over the pose block with the membrane block overwritten by a cycled seed, solved
sloppily (`max_iter=30, tol=1e-3`) → stage 2 converges the top `K=6` → each endpoint
certified by the exact Hessian → dedup on position → rank by depth.

Two things are easy to get wrong and are pinned by tests:

- **Dedup is position only** (great-circle ≤ 10°, |Δprotins| ≤ 1 Å, |Δcov| ≤ 1).
  Putting −logL in the identity rule splits one basin into several (~3× over-count).
  And the metric must be the great-circle angle, never `|Δθ|+|Δφ|`, which is wrong at
  the poles and at the φ wrap.
- **K is the safety parameter, not the stage-1 tolerance.** The cheap ranking
  correlates with the converged ordering at only ρ ≈ 0.20. If basins are missed,
  raise K.

### 3.2 The search box is not the fit bounds

`scan_box` extends `XLayersPDB_XPR._param_box` to all 8+2n columns: θ/φ get the
fundamental domain [0,180]×[0,360] (lossless for optimisation, and **a scan artefact
that is never written back** — CLAUDE.md rule 5 keeps them unbounded in lmfit); `cov`
is capped at 1.2·`cov_scale()` rather than guessed; box `rho` at the *physicality*
limit 0.62, not the class `RHO_MAX=5.0`; and a coordinate the user has pinned keeps
its value with degenerate bounds, so the Ipopt dimension stays 8+2n.

**Consequence that is invisible in the reference implementation** (where the free set
is hard-coded): pinned columns carry a `(hi−lo)≈1e-9` factor through the chain rule,
so their Hessian rows are numerically zero. Restricting the Hessian to the free block
with `np.ix_(free, free)` before taking eigenvalues is **mandatory** — without it
every endpoint classifies DEGENERATE.

### 3.3 Honest reporting, enforced structurally

`ModeSet` has **no `is_ambiguous` attribute and no threshold constant anywhere in the
type**. The API is `n_distinct(delta)` and `pose_span(delta)` — functions of delta. A
caller wanting a boolean must pick a threshold in its own visible code. The retired
number is not available to be hard-coded because there is nowhere to put it. A
`ModeSet` also cannot be constructed without a `Provenance`, and the provenance
carries the method's measured power, so a set reopened later still reports what it
could not have seen. `validate_fastscan.py` A3 and A10 pin all of this, including
that the code never says "the minima".

### 3.4 Step 1: seeds

(a) enumerate one or more **protein-free** scans; (b) load a seed table (our
`.seeds`, or the study's `CONFIG` JSON `membrane_seeds` key); (c) the current
membrane as a single seed. Route (c) is the degraded path and costs **7–16×** in hit
rate, which the results banner always states. Screening **drops and reports**, never
clips: a clipped `rho = −0.02 → 0.0` is a vacuum layer that silently wastes every
start assigned to it.

Three layered checks guard route (a), and **the layering is the point** — each one
catches what the previous cannot:

1. **Data convention, first.** A raw-R curve must not be enumerated at all. Measured
   discriminator on this tree's fixtures: median reflectivity over the upper half of
   the q range is ~0.15 for R/R_F and **1.5e-07** for raw R — six orders of magnitude,
   so the 1e-3 threshold needs no tuning. This is checked *before* the Patterson and
   is **not overridable**: raw R does not merely fit badly, its q^-4 decay dominates
   the transform and manufactures peaks at hundreds of Angstroms (measured: a 508 Å
   span on an 86-point curve). It is checked first because it has the *same symptom*
   as a protein-loaded scan, so diagnosing it second reports the wrong cause — which
   is exactly what an earlier version of this code did.
2. **Patterson span**, overridable. A protein-free film should sit under ~30 Å.
   **This is a screen, not a guarantee, and the limit is resolution:** the span it can
   see is capped by π/q_max, so measured here, a protein-loaded TIM3 scan out to
   q = 0.55 resolves a single peak at 23.8 Å and sails through.
3. **Post-fit χ²/N**, reported. This is what caught that case, at **380**. Neither
   2 nor 3 alone is sufficient; both are always reported.

### 3.5 Process model

The scan runs in a **`spawn` child**, polled by a `QTimer`. Not for speed: the
backend is process-wide single-slot, and on the Qt thread `processEvents()` could
dispatch a plot refresh whose `xb.load` drops `data_R`, silently changing the
objective mid-run with no error, because the fingerprint check is on the load side
and not the eval side. A child cannot be raced by the GUI, gives real cancellation,
and keeps the Hessian-time RSS out of the GUI process.

## 3.6 Relationship to the TIM corpus runs

**It is the same scheme, with the same defaults, and three deliberate deviations.**

The corpus was run with both schemes, and they had different jobs. Counting the
archives in `../xpr-seeding-study/julia/bakeoff/` (2587 `.npz` in the tracked set):

| scheme | archives | role |
|---|---|---|
| `*_twostage_*` | **1151** | the cheap scheme, i.e. this feature |
| `*_B_homoseed_*` | **968** | arm B, fully converged starts -- the *reference* the screen was scored against |
| `*_armD*` | 415 | the rejected cost-map arm |
| A / C / C2 / B5 / Bnarrow | 43 | the null seed-quality experiments |

So the two-stage scheme was run at scale on TIM1, TIM3 and TIM4, and its published
scores (TIM1 10/11, TIM4 20/30 and 26/30, TIM3 51/53) come from those archives. The
knobs recorded inside a `*_twostage_*` archive are `N1=64, K=6, iter1=30, tol1=1e-3`
and `FREE=[0,1,2,3,4,5,8,9,10,11]` -- identical to this implementation's defaults,
and the same `ipopt_bfgs` from the same `solvers.jl`.

**Three deviations, of which the first two change the numbers**, so a run from this
tool is NOT bit-comparable to an archive:

1. **Band-stratified starts, default ON here, never run in the study.**
   `run_twostage.py:62` is a plain `sob.copy()` with the membrane block overwritten;
   there is no theta stratification. It is recommended by
   `reports/INTERPRETATION-2026-09-11.md` §3.4 on the grounds that the admissible set
   is dominated by the opposite band -- but recommended is not measured. **Turn it off
   to reproduce an archive.**
2. **The regression guard.** `run_twostage.py:84` accepts `r.minimum`/`r.obj`
   unconditionally, so a truncated Ipopt solve that ended worse than its start is
   taken at face value and enters the ranking. Measured here: −logL 4335 → 115881.
   The corpus archives may therefore contain such endpoints; `xb.solve(keep_best=True)`
   removes them, which can reorder a shortlist.
3. **`FREE` is derived from `vary` here, hard-coded to all ten there.** A user who has
   pinned `sig` gets a 9-dimensional search, and the `N1=64` calibration was measured
   at ten. Surfaced in the banner. This is also why the bound-aware `-BOUND` label
   (§6) has no counterpart upstream: `classify()` at `run_twostage.py:72` does restrict
   to `FREE`, but to a *hard-coded* `FREE`, so the pinned-column problem never arises
   there and neither does the railed-coordinate one.

## 4. Measured in this tree (2026-09-14)

Fixtures: `Data/i2_tim3_1000_test.txt` (76 active points in q ∈ [0.027, 0.550]) and
`Data/tim3_hmmm_ext_all_bestframe.pdb`, 2 boxes, `vol_calc='approx'`, slab counts
(644, 1043).

**Gate 1 — objective identity. PASS, exactly.**
`Σ Fit.residual(p,'Linear')² = 2·(−logL_xpr)` to **1.9e-16** relative. So for the
default fit scale there is nothing to reconcile: the number the scan ranks by is the
number the parameter dock shows. `Fit.residual`'s non-dict branch is `(data−fit)/err`
with no normalisation (`Fit_Routines.py:304`), and XPR computes `−0.5·Σ((R−R̂)/σ)²`.
Under `Log`/`Log w/o error` the objectives genuinely differ and the ranking can
differ; the dialog warns and the mode table gains a `chi2_fit/N` column.

**Gate 2 — mesh reproduction. PASS, exactly.** `max|R_scan − R_model| = 3.8e-15` and
identical slab counts, because `load_for_scan` reproduces `_prepare`'s
`alloc_bounds_for(*model._param_box(...), spacez=..., box_ref_length=...)` call
verbatim. Those two arguments **change the answer**, not just the cost.

**Gate 3 — spawn child. PASS.** Time to a warm solve in a fresh child: import 0.6 s,
`get_jl` 3.4 s, `load_data` 4.5 s, first `sim` 4.2 s, `include_solvers` 0.2 s
≈ **13 s**. Warm `sim` 8.7 ms.

**Gate 4 — `terminate()` mid-`IpoptSolve`. PASS, cleanly.** 0.04 s, exit code −15,
and a follow-up child reproduces the same objective to the digit (7255.3996). No
depot or juliapkg poisoning, so cancel-by-kill is safe.

**The budget does not transfer, and this is the most important operational finding.**

| | study (TIM1, 1697 atoms) | here (TIM3 template) |
|---|---|---|
| warm `sim` | 4.3 ms | **8.7–13.7 ms** |
| warm `grad` | 29.5 ms | **92 ms** |
| warm `hess` | 288 ms | **1302 ms** |
| first `hess` (JIT, per process) | ~32 s | **36–47 s** |
| stage-1 solve (`max_iter=30`) | 0.61–0.64 s | **4.2–7.0 s** |
| **N1=64, K=6 total** | **~90 s** | **~735 s (12.2 min)** |

Eight-fold. So `N1`/`K`/iteration limits are all user-exposed, and the Run step has a
*Measure cost on this model* probe that projects the real wall clock before
committing. Do not quote 90 s to a user.

**Nothing converged on this fixture.** Every stage-1 and stage-2 solve ended
`Maximum_Iterations_Exceeded` or `Maximum_CpuTime_Exceeded` at `max_iter=300,
max_cpu=60`. The convergence warning is therefore load-bearing rather than
decorative, and the classifications on such a run are provisional.

## 4a. Driving the real widget, and two traps in doing so

`validate_fastscan_gui.py` builds an actual `XModFit`, loads two datasets, selects
`XLayersPDB_XPR`, opens the tool through the Tools loader, installs a mode into the
live tables and undoes it. **All 33 checks pass**, which is what closes the Qt-wiring
gap: `theta` reaches `fit.fit_params`, the model attribute follows (because
`Fit.params` *is* `func.__dict__`), the visible cell shows it, `cellChanged` is
reconnected, `theta.min` stays `-inf`, and `undoFit` restores.

Two traps cost real time to find and are encoded in that harness so they cost nobody
else any:

1. **`addData` blocks headlessly on a single file.** `xmodfit.py:4214`:
   ```python
   if len(fnames) > 1: data_dlg.accept()
   else:               data_dlg.exec_()     # modal -- hangs with no display
   ```
   So a headless test must pass **two or more** datasets. This is why
   `validate_perdataset_fit.py` works and a naive one-file version does not. Worth
   promoting into `handoff/09-validation-and-env.md`, since it applies to every GUI
   harness, not just this one.
2. **The console dock hijacks stdout.** `create_consoleDock` starts a live IPython
   kernel, and `print` after that can block on flush ("IOStream.flush timed out").
   The harness reports through `os.dup(1)` taken *before* `import xmodfit`. This is
   also the reason the GUI checks cannot share a process with parts A-E, hence the
   separate file -- the same split the repo already uses for
   `validate_jointfit_gui.py`.

## 5. Three defects found by these gates, and fixed

1. **`cpu_limited` tested `status == 6`.** In Ipopt.jl `6` is `Feasible_Point_Found`;
   the CPU-limit code is `-4`. So the one signal that detects a corrupted ranking
   never fired. Now driven by a named `IPOPT_STATUS` table.
2. **Ipopt can return a point far worse than its start.** It is an interior-point
   method with no monotone-descent guarantee, and when truncated it returns whichever
   iterate it stopped on. Measured: a stage-2 refine went from −logL 4335 to
   **115881**. Since the scheme *ranks* these endpoints, accepting that silently
   corrupts the ranking. `xb.solve(keep_best=True)` now returns the better of
   {start, endpoint} and flags `regressed`; `core.scan` records it, refuses to
   certify such a point (it is not stationary), and flags the mode.
3. **`Mode.__eq__` on numpy arrays.** The generated dataclass `__eq__` made
   `mode in list` raise "truth value of an array is ambiguous". `Mode` is now
   `eq=False`, which is also the correct semantics.

## 6. A correction to the published method

`classify()` takes a `-BOUND` suffix and excludes coordinates railed at a unit-cube
face. The reference implementation tests the full free block regardless, so a
*constrained* minimum on a box face — which happens, since θ ∈ [0,180] and `cov` is
bounded — is reported SADDLE when it is the best point available inside the search
region. `MIN-BOUND` counts as a minimum for the sampler hand-off but is flagged,
because the box is a scan artefact and an unbounded refine may leave that point.
**This fired on real data in the very first end-to-end run.** How much of the
study's quoted saddle fraction it accounts for is not yet quantified — see §7.

## 7. Claims to verify

| # | claim | status |
|---|---|---|
| 1 | objective identity under `Linear` | **DONE**, 1.9e-16 |
| 2 | mesh reproduction | **DONE**, 3.8e-15 |
| 3 | spawn child reaches a warm solve | **DONE**, ~13 s |
| 4 | `terminate()` is safe | **DONE**, exit −15, follow-up clean |
| 5 | how much of the published saddle fraction is really `MIN-BOUND`? | **OPEN** — replay the study's `endpoints_*.npz` through `modes.classify` and count. Cheap, and it bears on a published number. |
| 6 | does `N1=64` hold when a solve costs 8× more? | **OPEN.** Never varied on any corpus, and the cost model here is different enough that it deserves its own curve. |
| 7 | Sobol bit-reproducibility across scipy versions | **OPEN.** Version recorded in `.xmodes`; reproducing corpus results needs the same one (here: scipy 1.7.3). |
| 8 | Laplace weights against a converged sampler | **OPEN**, inherited from the study. |
| 9 | corpus acceptance (TIM1 10/11; TIM3 p3i2 −logL 80.286 cold; the `p1d2` saddle) | **OPEN** — needs the study's fixtures, which are out of tree. |

## 8. Known gaps

- **The enumerator has never been run on a genuinely protein-free scan**, because
  this tree has no such fixture: `i2`/`d6`/`n1` are all protein-loaded. So Part E
  pins the code path, the seed layout and both refusals, but *not* the science. Point
  it at a real monolayer-only scan before trusting a candidate table.
- `homometry.candidates(..., include_family=True)` emits homometric partners as well,
  but only those whose non-zero steps happen to match the box count exactly; the rest
  would need a fit to convert. Off by default.
- Multi-template runs, the Mollweide orientation thumbnail, the MCMC hand-off
  (`ModeSet.seed_positions` → `lmfit/minimizer.py`'s `_seed_pos`) and the docked
  panel are all later increments. The tool sets `Fit.modes` as a plain attribute for
  now, so nothing in `Fit_Routines.py` had to change; increment 2 makes it a declared
  field with staleness invalidation.
- `physical_members` in the enumerator uses `rho_hi=0.60` where `seeds.py` uses 0.62.
  Reconcile when route (a) lands.
- `.xmodes` deliberately does not store Hessians: they are only meaningful against the
  exact same data and mesh and are re-derivable at the stored point.
