# Protein-orientation MCMC seeding and angle handling — XModJointFit change record

**Category:** (theta, phi) walker seeding and angle wrapping for the XLayersPDB model family's emcee runs
**Status:** Working — individual-model path validated live by the user; every joint path validated headless only
**Owner chat(s):** memory `pdb-mcmc-seeding.md`, `multimodal-seeding-plan.md`, `parallel-joint-sampling-plan.md` (sessions 6f65f848, 22bf29f6, c0878594)
**Last touched:** 2026-08-25 (all of it is in the uncommitted working tree; see section 2)

## 1. Why this exists

In an XLayersPDB reflectivity fit a rigid protein template is posed against a layered
interface by two Euler angles (`theta`, `phi`) plus an insertion depth and a coverage.
The reflectivity is only weakly sensitive to orientation, so the (theta, phi) posterior
is broad and genuinely multimodal — the known mTIM3 case has two minima separated by
~110 deg in phi whose chi-squared differ by only ~6 percent. Stock XModFit seeded emcee
walkers by drawing (theta, phi) uniformly on the sphere and jittering the other
parameters, which puts almost every walker in a bad region; the ensemble then either
collapses into whichever mode it stumbles into first, or wastes tens of thousands of
steps climbing out of nowhere. This work replaces that with a cheap deterministic survey
of the orientation landscape, uses the survey to place walkers proportionally to the
tempered posterior *and* to force walkers into every detected basin, and then refines
each walker's nuisance parameters locally. A second, independent problem is handled here
too: theta/phi have a fundamental domain ([0,180] x [0,360)) with a fold identity
(theta -> 360-theta requires phi -> phi+180), so the sampler must wrap coordinates every
step or phi random-walks to plus/minus thousands of degrees and the chain's
autocorrelation becomes meaningless. Everything here applies to single-dataset fits and
to joint (multi-dataset) fits under every theta/phi sharing configuration.

Note: there is a *separate*, non-MCMC global-seeding programme for the Julia-backed
`XLayersPDB_XPR` model (Sobol multistart, Ipopt, homometry seeding) — see
`05-xpr-julia.md`, `STAGE2_SEEDING_PLAN.md` and `SEEDING_STUDY_TIM1.md`. It shares no
code with this document.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `lmfit/minimizer.py` | 3400 (589 changed) | modified (uncommitted) | seeding helpers, constants, the `Minimizer.emcee` seeding block, parallel-scan dispatch |
| `lmfit/customsampler.py` | 336 (45 changed) | modified (uncommitted) | per-step angle wrapping; `anglepairs`; `step_callback` |
| `pdb_seed_worker.py` | 98 | new-untracked | single-model parallel grid scan (spawn Pool) |
| `pdb_joint_seed_worker.py` | 126 | new-untracked | joint both-shared parallel grid scan |
| `Joint_Fit_Routines.py` | 1114 (515 changed) | new in commit `c5f697d`, then heavily modified (uncommitted) | `orientation_layout`, `_build_orientation_seed`, `_resample_regular`, `_seed_shared_axis`, `set_emcee_minimizer` |
| `xmodfit.py` | 7740 (2534 changed) | modified (uncommitted) | `seedTempSpinBox`, `seedModeCombo`, `update_emcee_parameters`, `runFit`, `run_joint_emcee` |
| `Fit_Routines.py` | (13 changed, committed) | modified | passes `seed_temperature` through `perform_fit` -> `set_emcee_minimizer` |

**Bucket summary — this is important for the merge.** `origin/develop` already contains a
`CustomSampler` with the theta/phi `np.piecewise` wrap, and a `Minimizer.emcee` that
special-cases `funcname == 'XLayersPDB'` with a *uniform-on-sphere random* walker seed
(upstream `minimizer.py:1443-1462`). The three-phase grid-scan scheme first appears in
the local commits `d3e353e`/`d955174` (which also added `Minimizer.__walkerresidual`,
`prepare_walker_fit`, `_calculate_walker_uncertainties_correlations` — all absent
upstream), in its *old* form: serial scan, a 50/50 likelihood/uniform mix, and a
rejection loop. **Everything described in section 3 below — the module-level helpers, the
temperature knob, basin detection, inverse-CDF draws, parallel scans, all joint seeding,
`anglepairs`, `step_callback` — exists only in the uncommitted working tree.** It has
never been committed anywhere.

## 3. How it works

### 3.1 The three phases

Phase 1 (survey). Build a near-uniform-on-sphere grid over (theta, phi): theta rings
every `thetares = 10` deg, and per ring a phi step chosen so each cell subtends roughly
the same solid angle (`lmfit/minimizer.py:222-243`, `build_orientation_grid`; the same
construction is inlined at `1897-1911` for the in-`emcee` path). At `thetares = 10` this
is 544 cells. For each cell, theta and phi are *fixed* and every other varying parameter
is profiled out by an unconstrained `scipy.optimize.least_squares`; the cell's score is
`ret.cost`, which is exactly chi-squared/2 of the real objective. The result is an
orientation cost map with the nuisance parameters marginalised (profiled, strictly) out.

Phase 2 (cell probabilities). The ragged grid is flattened into per-cell arrays and turned
into a categorical distribution over cells:

- `_cell_solid_angles` (`minimizer.py:122-131`) gives each cell its true solid angle
  `(cos theta_lo - cos theta_hi) * dphi`, clipped to [0,180]. This replaces the old
  `uniform_inverse += len(phimeshrange)` weight, which added a constant (the number of
  theta rings) instead of a cell count or an area — that bug is gone from the working tree.
- `_seed_cell_probs` (`minimizer.py:134-154`) forms
  `p ∝ area * exp(-0.5 * (chi2 - min chi2) / (T * chimin))`, where `chimin` is the
  reduced-chi-squared scale `min(cost)/(ndata - ndof)`, then blends in a solid-angle-uniform
  floor with weight `SEED_UNIFORM_FRAC`. `T -> inf` gives the area-uniform distribution;
  `T -> 0` concentrates on the argmin. This *replaces* the old hard-wired 50 percent
  likelihood / 50 percent uniform mix with an explicit knob.
- `_detect_orientation_basins` (`minimizer.py:165-219`) finds local-minimum cells: a cell
  qualifies if its cost is `<=` both phi-neighbours in its own ring (wrapped) and `<=` the
  nearest-phi cell in the theta +/- 1 rings, and its relative excess over the global min is
  `<= SEED_BASIN_REL_TOL`. Near-duplicates (within 15 deg theta / 25 deg phi) are merged,
  lowest cost kept. This is the mechanism that guarantees a *narrow* secondary well gets
  walkers even when random draws would miss it.

Phase 3 (draw and refine). `_inverse_cdf_draw` (`minimizer.py:157-162`) draws cell indices
by `searchsorted` on the CDF — exact and O(1) per walker, replacing the old `while True`
rejection loop. Detected basins then overwrite the head of the draw with
`WALKERS_PER_BASIN` walkers each, capped at `nwalkers // 2` so forced walkers can never
take over the ensemble (`minimizer.py:2006-2012`). Each walker's angles are jittered
uniformly inside its cell, then that walker's *other* parameters are set by a local
`least_squares` with theta/phi frozen, perturbed by the local relative standard errors
(`minimizer.py:2014-2071`). So every walker starts at a locally consistent point, not just
a random one.

`seed_thetaphi_from_costmap` (`minimizer.py:246-284`) is the composed, model-free version
of phases 2 and 3 — same helpers in the same order — so the joint seeder can reuse the
exact logic without going through `Minimizer.emcee`. The helpers take arrays only (no
model, no `self`), which is what made them unit-testable on synthetic two-well maps.

Constants live at `minimizer.py:108-112`: `SEED_TEMPERATURE = 1.0`,
`SEED_UNIFORM_FRAC = 0.1`, `WALKERS_PER_BASIN = 2`, `SEED_BASIN_REL_TOL = 0.5`.
`_pdb_sampler_moves()` (`115-119`) is the single source of the emcee move mix
(StretchMove 0.3 / KDEMove 0.3 / DEMove 0.25 / DESnookerMove 0.15), lifted out of the
two places that used to duplicate it. `_PDB_FUNCNAMES` (`94-95`) replaced scattered
or-chains of function names.

### 3.2 Which fits get seeded

`minimizer.py:1825`: `orient_seed = (funcname in _PDB_FUNCNAMES) or self._orient_seed`.
The first disjunct is the single-model case (`funcname` comes from
`Fit_Routines.set_emcee_minimizer`, which passes `type(self.func).__name__`); the second
is set by `JointFit.set_emcee_minimizer` for a joint fit. `orient_seed` gates four things
at once: the `angleindex`/`p0` setup block (`1827-1849`), the seeding branch (`1868`),
the choice of `CustomSampler` over `emcee.EnsembleSampler` (`2208-2221`), and the use of
`custom_run_mcmc` instead of `run_mcmc` (`2246-2247`).

Inside the seeding branch the dispatch is:

1. `self._seed_pos is not None` -> use the caller's precomputed `p0` and skip the internal
   scan entirely (`1870-1874`). This is the joint per-dataset / mixed / both-shared-individual
   path.
2. else `thetaflag and phiflag` -> run the internal single-pair grid scan (`1875-2071`).
   Used by single-model fits and by the joint both-shared "joint landscape" mode, where
   `self.userfcn` is `JointFit.residual` so the per-cell `least_squares` profiles the whole
   concatenated joint objective.
3. else -> the old jitter-only fallback (`2073-2078`).

### 3.3 Parallel seed scans

Phase 1 is deterministic and the cells are independent, so it parallelises bit-identically.
`minimizer.py:1923` reads `seed_workers = getattr(self, '_seed_workers', workers)` — this
deliberate decoupling lets a joint fit scan across all cores while its *sampling* stays
serial (the joint objective was, at the time, unpicklable). Two workers exist:

- `pdb_seed_worker.py` — single model. Ships only the model instance (~25 KB; the Qt signal
  lives on the `Fit` object, not the model) via a spawn-Pool initializer, sets
  `func.__fit__ = True` and `func.x` to the sliced range, and reproduces `Fit.residual`'s
  non-dict single-dataset residual per scale (`_scale_residual`, lines 35-44). Dispatched
  at `minimizer.py:1929-1942`, guarded on `funcname in _PDB_FUNCNAMES` and on `fitobj.x`
  not being a dict.
- `pdb_joint_seed_worker.py` — joint both-shared. `JointSeedScan` ships all models and data
  (reusing `pdb_joint_logprob.init_worker` and `_scale_residual_norm`), reconstructs
  `JointFit.residual`'s no-covariance branch including the per-curve max normalisation and
  the Gaussian prior terms, and profiles all remaining master params (shared *and* local,
  including a shared `protins`) per cell. `__getstate__` drops the models so the scan object
  itself pickles small. Dispatched at `minimizer.py:1943-1959`, gated on
  `not fitobj._use_cov`.

Both fall back to the serial loop (`1971-1977`) on any exception, with the traceback printed.
There is a deliberate comment at `1964-1970` explaining why no analytic Jacobian is used for
these sub-fits even when the model has one: with ~2 profiled parameters finite differences
are 3 evaluations, cheaper than one AD pass, and in the parallel path every worker would
additionally pay ForwardDiff's ~3 s specialisation (measured 30 s -> 34 s serial, 51 s -> 74 s
at 16 workers).

### 3.4 Joint seeding across sharing configurations

`JointFit.orientation_layout` (`Joint_Fit_Routines.py:622-664`) inverts `param_map` to find,
per dataset, which master key drives its `theta` and its `phi`, and that key's column in the
varying-parameter order. It returns `col`, `datasets`, `loc2master`, and the two
classifications `both_shared` (one theta and one phi, each mapped to all K datasets) and
`neither_shared` (every dataset owns its own pair). Anything else is "mixed".

`JointFit._build_orientation_seed` (`666-781`) handles everything except both-shared-joint
(it returns `None` there, deferring to the in-`emcee` scan). It first calls
`self.push_params(emcee_params)` to **pin the shared structure at the joint best fit**, then
scans each dataset's own model with `pdb_seed_worker.map_cell_costs`, profiling only that
dataset's *local* varying nuisances (`719-724`: a param is local iff its master key maps to
exactly one target). The rationale recorded in the notes is that only orientation basins
consistent with the shared structure should survive. Then:

- both-shared with `both_shared_mode='individual'`: seed the single shared pair from the
  *sum* of the per-dataset maps (`730-738`).
- per-dataset pairs (theta column and phi column each used by exactly one dataset):
  straight 2-D `seed_thetaphi_from_costmap` on that dataset's map (`749-759`).
- mixed — shared theta with per-dataset phi, or the reverse — via `_seed_shared_axis`
  (`799-853`): each member's ragged map is resampled onto a regular `nt x 72` grid by nearest
  cell (`_resample_regular`, `783-797`); the shared axis gets the marginal
  `sum_i min_other M_i` with solid-angle ring weights; each member's other angle is then
  drawn *conditional* on that walker's shared bin.

It returns `(p0, angleindex, anglepairs)`, which `set_emcee_minimizer` (`890-1002`) attaches
to the fitter as `_seed_pos` / `_angleindex` / `_anglepairs` plus `_orient_seed = True`. The
`angleindex` must be supplied by the caller here because per-dataset angles are named
`d{i}_theta`, which `Minimizer.emcee`'s bare-name test at `1833-1844` misses;
`minimizer.py:1848-1849` honours the override.

`_attach_joint_parallel` (`855-888`) attaches the picklable `JointLogProb` and its spawn-Pool
initializer as `fitter._ext_objective` / `_ext_obj_init`, gated on `emcee_cores > 1 and not
self._use_cov`. It is called from *all* branches. `Minimizer.emcee`'s generic external-objective
block (`2152-2191`) builds the pool; the one-line dispatch at `2218` routes
`step_callback = pdb_step_cb if pdb_step_cb is not None else ext_step_cb`, which is what makes
an orientation-seeded joint fit sample over the pool *and* wrap angles in the main process.
(The picklable-objective machinery itself belongs to `02-mcmc-stack.md`.)

### 3.5 Angle wrapping

The fold identity is applied to `state.coords` in the main process after every proposal, inside
`CustomSampler.custom_sample`:

- Legacy path (`customsampler.py:263-282`, upstream code): from `angleindex`, columns marked 1
  are theta and 2 are phi; theta is folded `theta -> 360 - (theta % 360)` when `theta % 360 > 180`
  and phi is simultaneously reflected `phi -> (phi + 180) % 360`. Pairing is by sorted column
  order, which is only correct for a single theta/phi pair (or one shared pair).
- New `anglepairs` path (`customsampler.py:236-262`): an explicit list of
  `(theta_col_or_None, phi_col_or_None)` per dataset. Each *distinct* theta column is folded
  exactly once and its reflection mask recorded; then each phi is reflected using *its* theta's
  mask. A phi driven by more than one theta is degenerate and only gets `% 360`. This handles
  unequal counts, one shared theta driving many phis, and per-dataset pairs. Set only when
  `self._anglepairs` is present (`minimizer.py:2204-2205`), so the single-model and both-shared
  cases keep the legacy behaviour bit-for-bit.

`angleindex` moved from a `custom_sample`/`custom_run_mcmc` argument to a constructor attribute
(committed change), which is what enabled the resume fix below.

**The resumed-sampler wrapping bug.** `angleindex` used to be applied only in the fresh-seed
branch. On a load-and-continue, `if reuse_sampler:` (`minimizer.py:1852`) caught the run first
and the `elif pdb and reuse_sampler` branch below it was dead code — so a resumed XLayersPDB run
built its `CustomSampler` with **no** `angleindex`, wrapping silently stopped, and phi random-walked
to plus/minus thousands of degrees from the load seam onward (visible in the user's phi trace).
The dead `elif` also carried a `backend.reset` that would have wiped the resumed chain. Fix:
the dead branch was deleted and the `moves`/`angleindex`/`anglepairs` `setdefault`s were moved to
`minimizer.py:2200-2206`, immediately before the sampler is constructed, so they apply on both
fresh and resumed runs; `setdefault` leaves fresh-run values untouched, and there is no
`backend.reset` there. Two related fixes in the same region: the reuse branch's
`self.sampler.pool = auto_pool` is now guarded on `self.sampler` existing (`1858`), because
XModFit builds a fresh `Minimizer` per run; and the dead scalar-posterior `sin(theta)` prior at
`1514-1526` no longer calls `theta(angleind)` (a `NameError`-in-waiting) but indexes theta out
of the parameter vector. That prior branch is still dead for the vector XRR residual, so **no
sin(theta) prior is applied during sampling.**

### 3.6 Control flow, GUI to sampler

Individual: MCMC window "Start sampling" -> `start_emcee_sampling` (`xmodfit.py:3325`; calls
`update_emcee_parameters`, warms the numba JIT under the "Initializing..." label) ->
`doFit(fit_method='emcee', ...)` -> `runFit` -> `Fit.perform_fit(..., seed_temperature=...)`
(`xmodfit.py:3420-3425`) -> `Fit.set_emcee_minimizer` (`Fit_Routines.py:487-516`, puts
`seed_temperature` and `funcname` into the `Minimizer` `**kws`) -> `Minimizer.minimize` merges
`self.kws` into the call (`minimizer.py:3135`) -> `Minimizer.emcee(seed_temperature=...)`.

Joint: the joint MCMC window sets `_mcmc_is_joint`, and `doFit` routes emcee straight to
`run_joint_emcee` before any individual-fit guard (`xmodfit.py:1731-1737`).
`run_joint_emcee` (`xmodfit.py:7149-7198`) passes `seed_temperature` and `seed_mode` into
`JointFit.perform_fit` (`Joint_Fit_Routines.py:1004-1038`) -> `set_emcee_minimizer`. The
`_workers` gate at `1031-1033` only raises workers above 1 when `_ext_objective` was attached.

## 4. How to use it

**GUI.** Open the MCMC window ("Calculate Param Error", or the joint panel's
"Joint MCMC / Param Error..." button). Two controls are added programmatically into free rows
of `gridLayout`, directly under MCMC Walkers/Steps (`xmodfit.py:2802-2856`):

- **"Seed temperature (T)"** (row 4), `QDoubleSpinBox`, range 0.1-20, step 0.5, default 1.0.
  Enabled only when `_mcmc_is_pdb()`. T=1 seeds posterior-weighted; **T=2 to 5 is what you set
  to spread walkers across competing orientation modes** if a run collapses into one mode; T<1
  concentrates on the best mode. It affects only where walkers *start*, never the posterior.
  Detected basins get their guaranteed walkers at any T.
- **"Seed from"** (row 5), `QComboBox` with `['Joint landscape', 'Individual landscapes']`.
  Enabled only for a joint fit whose theta *and* phi are both shared across all datasets
  (checked via `self.fit.orientation_layout()`), because per-dataset orientations are always
  seeded individually. "Joint landscape" (default, index 0 -> `seed_mode='joint'`) scans the
  shared pair against all datasets at once, re-profiling shared parameters such as `protins`
  jointly per cell. "Individual landscapes" (index 1 -> `'individual'`) scans each dataset's own
  model with shared params pinned at the joint best fit and sums the maps.

Both are read in `update_emcee_parameters` (`xmodfit.py:3304-3323`) into `self.seed_temperature`
and `self.seed_mode`, each wrapped in try/except so a missing widget falls back to the
`__init__` defaults (`xmodfit.py:550`, `552`).

**Parallelism.** Set `ParallelCoresLineEdit` (`self.emcee_cores`) > 1. That value drives the seed
scan (`pdb_seed_worker` / `pdb_joint_seed_worker`) and, separately, the sampling pool. For a joint
covariance/GLS fit both stay serial and a message is printed.

**Not exposed anywhere:** `seed_uniform_frac` and `walkers_per_basin` are `Minimizer.emcee`
parameters (`minimizer.py:1538-1539`) but no caller ever sets them — the module constants are
always in effect. Change them at `minimizer.py:109-110` if needed.

**Known multimodal test case** (from the notes): data `i2_tim3_1000.rrf`, 3-column q/R/err, fit
range 0.026-0.550, 78 points, function `XLayersPDB`, 9 varying, PDB
`tim3_hmmm_ext_all_bestframe.pdb`. Primary minimum theta ~40.9, phi ~74.1, protins ~6.1,
cov ~0.158, chi-squared 148.9; secondary theta ~31.4, phi ~149, protins ~14.0, cov ~0.147,
chi-squared 157.2. A self-contained copy sits in this repo: `Data/i2_tim3_1000_test.txt` plus
`Data/i2_tim3_1000_test_fitparam.par`, and the `.par` is parked at the **secondary** mode
(theta 31.93, phi 149.64, protins 13.996, cov 0.1469) — so a good seeding run should discover
the primary mode from it.

**Measured numbers, with their caveats** (all asserted in the notes, none re-measured here):

- Phase 1 dominates: 544 cells at `thetares=10`, ~72 s serial, ~0.12 s/cell. Phase 3 for
  20 walkers ~2.6 s. First `y()` JIT ~1.1 s.
- Per `y()` eval ~1.6 ms with `ED_profrot` cached (fixed angle) / ~3.1 ms fresh angle.
- Parallel scan == serial to 0.0, both known modes are the two lowest cells, ~6.2x on 32 cores.
- Sampling (separate from seeding): 70.7 -> 13.8 ms/step at 8 workers, ~5x. Caveat: pool spawn +
  numba warmup is a one-time 15-25 s, so short runs look slower — measure by differencing two
  step counts.
- `XLayersPDB_dev_smooth` did *not* improve mixing over `XLayersPDB` (tau ~384 vs ~238,
  acceptance 0.075 vs 0.139, ~3x slower per step). Caveats: tau in the "chain too short" regime
  (N/tau ~11-20, well under 50), single stochastic run, and `_dev_smooth` forced to dz=0.5
  rather than its recommended 0.25. The conclusion drawn was that its smoothness rework helps
  derivative-based fitting, not MCMC mixing.
- Both posteriors broad and multimodal: theta std ~20-25 deg, phi std ~25-53 deg.

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `lmfit/minimizer.py` | module top (`94-284`) | new `_PDB_FUNCNAMES`, 4 seeding constants, 7 new module-level functions | additive |
| `lmfit/minimizer.py` | `Minimizer.emcee` signature | 3 new kwargs (`seed_temperature`, `seed_uniform_frac`, `walkers_per_basin`) with defaults | additive |
| `lmfit/minimizer.py` | `emcee` PDB gate (`1825`) | `funcname == 'XLayersPDB'` -> `funcname in _PDB_FUNCNAMES or self._orient_seed`, at 5 sites | rewrite of the gate |
| `lmfit/minimizer.py` | `emcee` seeding branch (`1868-2084`) | upstream drew (theta,phi) uniform-on-sphere per walker; now a grid scan + tempered categorical + basins + per-walker refine | **full rewrite** |
| `lmfit/minimizer.py` | `emcee` reuse branch (`1852-1866`) + `2200-2206` | guarded `self.sampler.pool`; moved `moves`/`angleindex` setdefaults so resume wraps; deleted a dead `elif` containing `backend.reset` | rewrite (bug fix) |
| `lmfit/minimizer.py` | `_lnprob` scalar-posterior prior (`1514-1526`) | `theta(angleind)` NameError replaced by an index into the parameter vector | fix, branch still dead |
| `lmfit/minimizer.py` | `emcee` objective/pool blocks (`2102-2191`) | picklable objectives + spawn pools + main-process step callbacks | additive (detailed in `02-mcmc-stack.md`) |
| `lmfit/customsampler.py` | `__init__` | `angleindex` promoted to ctor arg (committed); `anglepairs`, `step_callback` added (uncommitted) | additive |
| `lmfit/customsampler.py` | `custom_sample` wrap block | new `anglepairs` branch *before* the upstream `piecewise` branches; `step_callback` invoked per stored step | additive (legacy branch untouched) |
| `lmfit/customsampler.py` | `custom_sample`/`custom_run_mcmc` signatures | `angleindex` argument removed (now an attribute) | breaking for any external caller |
| `Fit_Routines.py` | `perform_fit`, `set_emcee_minimizer` | `seed_temperature` parameter threaded through | additive |
| `xmodfit.py` | `__init__`, `confInterval_emcee`, `update_emcee_parameters`, `runFit`, `doFit`, `run_joint_emcee` | two new widgets built programmatically (no `.ui` change), two new state fields, plumbing | additive |

`Joint_Fit_Routines.py`, `pdb_seed_worker.py` and `pdb_joint_seed_worker.py` are wholly new — no
upstream contact surface.

## 6. Known limitations, gotchas and TODOs

1. **Covariance/GLS joint fits get no parallelism at all here** — neither the seed scan nor
   sampling. `pdb_joint_seed_worker` reconstructs only the no-covariance residual (the whitened
   branch needs the shipped Cholesky), and `_attach_joint_parallel` is gated `not self._use_cov`.
   A message is printed. Recorded as the remaining follow-up, blocked on having a covariance
   dataset to validate the whitened residual bit-for-bit.
2. **Two spawn Pools per run** in the individual-PDB parallel case (one for the seed scan, one for
   sampling), each paying process spawn plus numba JIT warmup. The notes list unifying them as a
   deferred optimisation.
3. **`__main__` guard is mandatory** in any headless script that reaches a spawn Pool — otherwise
   the script re-imports itself and fork-bombs. The GUI is already guarded.
4. **Three different lists of "PDB function names" exist and they disagree:**
   `minimizer._PDB_FUNCNAMES` has 6 entries (`XLayersPDB`, `_box`, `_G`, `_H`, `_dev_smooth`,
   `_XPR`); `xmodfit._PDB_MCMC_FUNCS` (`xmodfit.py:2634`) has 3 (`XLayersPDB`, `_box`,
   `_dev_smooth`); `Fit_Routines.set_emcee_minimizer` (`Fit_Routines.py:491`) tests a different 3
   (`XLayersPDB`, `_dev_smooth`, `_XPR`). Consequences: for `XLayersPDB_G`/`_H`/`_XPR` the seed
   temperature spinbox is disabled and the GUI does not use the angle-aware backend, yet
   `Minimizer.emcee` *does* seed and wrap them; and for `_box`/`_G`/`_H` `fiterr` stays the scalar
   `0`, so the jitter-only fallback at `minimizer.py:2077` (`fiterr[ind]`) would raise `TypeError`
   if only one of theta/phi varies.
5. **`_seed_shared_axis` skips basin detection.** The mixed/shared-axis path applies only the
   tempered categorical (`Joint_Fit_Routines.py:829`); it never calls
   `_detect_orientation_basins`, so the "guaranteed walkers per basin" property that motivated
   the whole scheme does not hold on that path.
6. **`SEED_UNIFORM_FRAC` and `WALKERS_PER_BASIN` are unreachable from the GUI**, and
   `_build_orientation_seed` never forwards `walkers_per_basin` either — defaults only.
7. **Sampler move mix is untuned.** `_pdb_sampler_moves()` is deliberately the pre-existing
   default, kept "pending live tuning"; item 4 of the original four-item plan (find a mix that
   lifts the low acceptance / high tau on this multimodal target) was never done.
8. **Deferred conceptual ideas that remain unimplemented**, per the notes: none of the phase-2/3
   items are outstanding any more (all four were implemented), but the move-mix study and the
   pool unification are. An explicitly **rejected** idea: optimising the `lru_cache` key for
   `ED_profrot`/`protvolcalc`, which hashes full atom-coordinate tuples — profiling showed it is a
   cache *hit* inside a cell (theta/phi fixed) so the O(N_atoms) hash is under 1 percent of
   runtime.
9. **Validation harnesses live in per-session temp scratchpads, not the repo.** The seeding tests
   (`verify_seed_helpers.py`, `verify_seedmode.py`, `verify_seed_gui.py`, `verify_joint_orient.py`,
   `verify_joint_neither.py`, `verify_joint_mixed.py`, `verify_joint_seed_worker.py`,
   `verify_reuse_wrap.py`, `verify_joint_4ds_parallel.py`, `verify_joint_parallel_sample.py`) are
   under `%LOCALAPPDATA%\Temp\claude\C--Users-kerr-Documents-GitHub-XModJointFit\{22bf29f6...,
   c0878594...}\scratchpad\`. Also noted: `22bf29f6.../verify_joint_parallel_scan.py` is
   **obsolete** — it asserts `sampler.pool is None`, the behaviour that was later flipped;
   `verify_joint_4ds_parallel.py` is the corrected copy.
10. **Lesson recorded in the notes:** always validate against `CustomHDFBackend` (the GUI backend),
    not emcee's stock `HDFBackend` — an earlier joint validation used the stock backend and masked
    a real autocorrelation-exception bug.
11. `CustomSampler.get_autocorr_time` (`customsampler.py:335-337`) still calls
    `super().get_autocorr_time(**kwargs)` and throws the result away before delegating to the
    backend. The notes describe removing exactly this kind of redundant double-autocorrelation in
    `custombackend.py`; the sampler-level one is still there.

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| 1 | `pdb_seed_worker`'s cost map is bit-identical to the serial scan | `pdb-mcmc-seeding` | Run the i2_tim3 case at cores=1 and cores=8, dump `chi2mesh`, assert max abs diff 0.0. **Then repeat with a per-column fit limit active** — see claim 2. |
| 2 | The parallel scan reproduces `Fit.residual` | `pdb_seed_worker.py` docstring | Suspected **wrong**: `Fit.residual` applies `self.imask_for()` (`Fit_Routines.py:291-296`) but `pdb_seed_worker._init/_cell_residual` slices only `imin:imax` and never applies a mask. Same for `pdb_joint_seed_worker.JointSeedScan.residual` vs `JointFit.residual`'s `obj.get("imask")` (`Joint_Fit_Routines.py:552-558`) and for `_build_orientation_seed`'s data slices (`Joint_Fit_Routines.py:716-718`). Check: enable a per-column limit, compare parallel vs serial cell costs. |
| 3 | `pdb_joint_seed_worker` cost == `JointFit.residual`-based cost bit-for-bit, with `protins` shared | `multimodal-seeding-plan` | Re-run `verify_joint_seed_worker.py` on a no-cov 2-dataset shared-`protins` case; assert 0.0 diff and serial == parallel. |
| 4 | ~6.2x scan speedup on 32 cores; 544 cells ~72 s serial; ~0.12 s/cell | `pdb-mcmc-seeding` | Time the scan at cores=1 vs cores=32 on i2_tim3; count `len(cells)`. |
| 5 | Both known i2_tim3 modes are the two lowest cells of the map | `pdb-mcmc-seeding` | Sort the 544 cell costs, check the top two are near (40,70) and (30,150). |
| 6 | `least_squares` cost == chi-squared/2 exactly, so the scan reproduces the real objective | `pdb-mcmc-seeding` | Compare `ret.cost` to `0.5*sum(Fit.residual**2)` at the same parameters. |
| 7 | The new seeding "resolves more modes" than the old likelihood-profile scheme | user, live, via `multimodal-seeding-plan` | Compare seeded `p0` mode coverage old vs new on i2_tim3 (the old code is recoverable from `git show HEAD:lmfit/minimizer.py`, lines ~1640-1710). |
| 8 | `_detect_orientation_basins` finds both a wide primary and a narrow secondary well and dedups | `multimodal-seeding-plan` | Re-run `verify_seed_helpers.py` (synthetic two-well map); add the real 544-cell map. |
| 9 | `_inverse_cdf_draw` matches the target distribution to TV = 0.005 | `multimodal-seeding-plan` | Draw 1e6 samples, compare the empirical histogram to `probs`. |
| 10 | `_cell_solid_angles` sums to ~4*pi over a full grid | docstring at `minimizer.py:125` | `np.sum(_cell_solid_angles(...))` over `build_orientation_grid()` output. |
| 11 | Resumed samplers now wrap angles (angleindex `[1,2,0]`, phi stays in [0,360)) | `multimodal-seeding-plan` | Re-run `verify_reuse_wrap.py`; also confirm the `.h5` chain shows no phi discontinuity at the load seam. |
| 12 | `anglepairs` wrapping is correct for neither-shared and mixed layouts | `multimodal-seeding-plan` | Re-run `verify_joint_neither.py` (expects `angleindex [1,2,0,1,2,0]`) and `verify_joint_mixed.py` (expects `angleindex [1,2,0,2,0]`, `anglepairs [(0,1),(0,3)]`, theta~60 shared, phi_0~70, phi_1~220). |
| 13 | Parallel joint sampling preserves seeding and wrapping; abort and resume work through the pool | `parallel-joint-sampling-plan` | Re-run `verify_joint_parallel_sample.py`, `verify_joint_parallel_edge.py`, `verify_joint_4ds_parallel.py`. |
| 14 | `seed_mode='individual'` precomputes `_seed_pos` while `'joint'` runs the in-`emcee` scan | `multimodal-seeding-plan` | Re-run `verify_seedmode.py`. |
| 15 | The temperature spinbox is enabled for all PDB fits and the seed-mode combo only for a both-shared joint fit | `multimodal-seeding-plan` | Re-run `verify_seed_gui.py`; note it asserts row 4 / row 5 placement in `gridLayout`. |
| 16 | `_dev_smooth` mixes *worse* than `XLayersPDB` (tau 384 vs 238, acceptance 0.075 vs 0.139) | `pdb-mcmc-seeding`, self-flagged as one run in the too-short-chain regime | Re-run both to N/tau > 50 with matched dz; repeat 3 seeds. Low priority — it only informs model choice. |
| 17 | No `sin(theta)` prior is applied during sampling (the prior branch is dead for a vector residual) | `pdb-mcmc-seeding` and the comment at `minimizer.py:1516-1519` | Assert `lnprob.size > 1` for an XLayersPDB residual, so `float_behavior` never reaches the scalar branch; decide whether the uniform-on-sphere prior *should* be applied. |
| 18 | Model pickles in ~25 KB and carries no Qt signal | `pdb_seed_worker.py` docstring | `len(pickle.dumps(fit.func))`. |
| 19 | The GUI seeding path has been exercised live at cores > 1 | flagged "GUI path not yet live-tested at cores>1" in `pdb-mcmc-seeding`; later notes claim headless-only for joint | Click through: individual PDB at cores=8, then a 2-dataset joint at cores=4, both seed modes. |
| 20 | Joint seeding correctly pins the shared structure (only basins consistent with it survive) | `multimodal-seeding-plan` | Verify `push_params` before the scans changes the per-dataset maps vs not pinning, on a 2-dataset shared-`protins` case. |

## 8. Consolidation candidates

| Item | Recommendation | Risk |
|---|---|---|
| Three disagreeing PDB-function-name lists (`minimizer._PDB_FUNCNAMES`, `xmodfit._PDB_MCMC_FUNCS`, the tuple in `Fit_Routines.set_emcee_minimizer`) | **Merge** into one exported list, imported by the other two | Low mechanically, but it changes behaviour for `_G`/`_H`/`_XPR` (they would start getting the angle-aware GUI backend and an enabled temperature knob). Needs a deliberate decision per model, not a blind merge. |
| `build_orientation_grid` (`minimizer.py:222-243`) vs the inline grid construction in `emcee` (`1897-1911`) | **Merge** — replace the inline copy with a call | Low; they are the same arithmetic, but the inline version also fills `chi2mesh` and `phimeshrange`, so the refactor must keep those. Confirm identical `cells` first. |
| `seed_thetaphi_from_costmap` (`246-284`) vs the inline phases 2-3 in `emcee` (`1984-2020`) | **Merge** — the inline block is a duplicate of the composed helper, minus the per-walker `least_squares` refine | Medium: the inline version also writes `p0` and drives the refine loop; a botched refactor would silently change the seeding distribution. Add a distribution regression test first. |
| `_seed_shared_axis`'s hand-rolled tempered draw (`Joint_Fit_Routines.py:846-849`) | **Merge** into `_seed_cell_probs` + basin detection so the mixed path gains the guaranteed-basin property | Medium — changes seeding behaviour on the mixed path. |
| `CustomSampler.get_autocorr_time`'s discarded `super()` call (`customsampler.py:336`) | **Drop** the `super()` line | Low, but verify the base call has no side effects (the notes assert the base `Backend.get_autocorr_time` is a pure `get_chain` -> `integrated_time`). |
| Duplicated spawn-Pool construction: `pdb_seed_worker.map_cell_costs`, `pdb_joint_seed_worker.map_cell_costs`, and two blocks in `Minimizer.emcee` | **Merge** into one pool helper and, per the notes, reuse a single pool for seeding and sampling | Medium-high: touches the parallel paths that were validated bit-for-bit. Do it after the audit, not before. |
| Seeding validation scripts stranded in session temp scratchpads | **Move** the ~10 `verify_*seed*/orient/wrap` scripts into the repo next to the existing `validate_*.py` harnesses, and delete the obsolete `verify_joint_parallel_scan.py` | Low, and it is a prerequisite for the audit pass — these files are the only executable record of what was tested. |
| `run_mcmc_kwargs` / `progress=` reaching `custom_run_mcmc` | **Keep** | None; noted only because `custom_run_mcmc`'s signature changed (dropped `angleindex`), which breaks any out-of-tree caller. |
