<!-- Template for handoff/NN-<topic>.md. Keep the section order and headings. -->

# XPR Julia backend + global seeding - XModJointFit change record

**Category:** the differentiable Julia reflectivity model (`XLayersPDB_XPR`) and the homometry-seeded Ipopt multistart research programme built on it.
**Status:** Research harness on top of a Working-but-unvalidated-live model class. `validate_xpr.py` is the only gate and it needs an out-of-tree `XPR-main/test_data`. No global-search fit method exists in the GUI.
**Owner chat(s):** memory `xpr-julia-backend.md` and `homometry-seeding-for-xpr.md`; in-repo `STAGE2_SEEDING_PLAN.md` and `SEEDING_STUDY_TIM1.md`.
**Last touched:** 2026-09-10 - integrated the 2026-09-10 upstream (all four of our asks accepted; two of our divergences retired), and measured what the parameter-dependent mesh costs on real data. See 3.9 and 3.10. Later the same day, migrated the 32 downstream harnesses onto the new sizing API and corrected the source fingerprint in circulation to `2402cb195df0ce6f`. See 3.10.1.

> **MOVED, 2026-09-04 — read this first.** The seeding *study* half of this document has
> been extracted to its own repository, `../xpr-seeding-study`: all of `julia/bakeoff/`
> (the harness and the 107 `endpoints_*.npz` archives), `STAGE2_SEEDING_PLAN.md`,
> `SEEDING_STUDY_TIM1.md`, and the newer `TIM1_SEEDING_REPORT.md` /
> `PLAN_XMODFIT_FASTSCAN.md` / `PLAN_CLUSTER_TOOL.md`. It is research, not an XModFit
> feature, and must not reach `chemmatcars/XModFit` — see fix-list A0.6 for the one
> remaining step (commit `7942458` still holds the tracked half in local history) and for
> the `.githooks/pre-push` guard. Everything this document says about those files remains
> accurate, but their path is now that repository. **The model side stays here and is
> XModFit's**: `Functions/xpr_backend.py`, `Functions/XRR/XLayersPDB_XPR.py`,
> `Functions/pdb_geometry.py`, `julia/XPR/`, `validate_xpr.py`, `requirements_xpr.txt`,
> and the `lmfit`/`Fit_Routines` wiring. One feature is owed from the study: the fast-scan
> in `PLAN_XMODFIT_FASTSCAN.md`.

> **BUCKET WARNING, read first.** Essentially all of this category is **working-tree-only**. `git log --oneline origin/develop..HEAD` contains nothing from it: `Functions/xpr_backend.py`, `Functions/XRR/XLayersPDB_XPR.py`, `Functions/pdb_geometry.py`, `validate_xpr.py`, `requirements_xpr.txt`, `julia/**`, `STAGE2_SEEDING_PLAN.md`, `SEEDING_STUDY_TIM1.md` are **untracked**, and the two upstream files it modifies carry their XPR changes as **uncommitted `git diff`**. Verified: `git diff origin/develop...HEAD -- Fit_Routines.py | grep -c can_jac` = 0, while `git diff -- Fit_Routines.py` holds the whole analytic-Jacobian block. One `git checkout .` destroys the upstream-side wiring; the untracked tree is not even in the reflog. **Commit before doing anything else.**

## 1. Why this exists

`XLayersPDB` fits specular reflectivity of an n-box film with a rigid PDB protein at a fitted orientation and depth. It is not differentiable (hard slab edges, grid volume calculation), so every fit and MCMC step runs on finite differences, and the (theta, phi, protins, cov) pose problem is badly multimodal: the same data admits several poses at comparable likelihood. XPR (Elias Boshara) reimplements that physics in Julia with every step continuously differentiable, so ForwardDiff returns exact gradients, Hessians and reflectivity Jacobians of the log-likelihood.

The plan had three stages: stage 1, the model class and forward evaluation (done); stage 2, replace a single lmfit descent with a global search returning a *list* of minima with weights; stage 3, feed those minima to MCMC, possibly HMC using the gradient. Stage 2 exists only as an offline harness (`julia/bakeoff/`) plus two design documents, but it delivered the scientific answer stage 2 needed: how to place multistart points, how many are needed, and what likelihood tolerance counts as the same answer. Target systems are the user's mTIM1 and mTIM3 protein-monolayer XRR campaigns (`.rrf`, 78 points, q = 0.026-0.550, 2-box lipid monolayer plus protein, 10 free parameters).

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `julia/XPR/src/XPR.jl` | 1415 | new-untracked | vendored + patched Julia model (upstream `XPR-main/XPR`, **2026-09-08 revision**) |
| `julia/merge_upstream.py` | 426 | new-untracked | re-applies the four divergences onto an upstream drop; anchored + asserted + idempotent (3.9) |
| `julia/XPR/{Project,Manifest}.toml` | 16 + lock | new-untracked | ForwardDiff 1.4.5, SpecialFunctions 2.8.3, **Ipopt 1.15.0** (added here) |
| `Functions/xpr_backend.py` | 621 | new-untracked | one juliacall session per process; load fingerprint; slab guard; `sim`/`sim_ref`/`grad`/`hess`/`jac`/`edp`/`sim_edp`/`maxh` |
| `Functions/XRR/XLayersPDB_XPR.py` | 772 | new-untracked | the XModFit model class |
| `Functions/pdb_geometry.py` | 245 | new-untracked | numba union volume + projected area for coverage bookkeeping; `com_extent`/`com_extent_grad` for the insertion conversion |
| `validate_xpr.py` | 582 | new-untracked | regression gate; needs `../XPR-main/test_data` |
| `requirements_xpr.txt` | 25 | new-untracked | pinned juliacall 0.9.28 + transitive deps |
| `julia/README.md` | 171 | new-untracked | setup, juliapkg/SSL notes, divergences, Jacobian notes |
| `STAGE2_SEEDING_PLAN.md` | 423 | new-untracked | stage-2 design + bake-off; **its own section 2 is flagged superseded** |
| `SEEDING_STUDY_TIM1.md` | 315 | new-untracked | how-many-starts / what-tolerance study; supersedes the plan's phase-1 design |
| `julia/bakeoff/README.md` | 47 | new-untracked | harness index - **stale**, covers 6 of 15 scripts |
| `julia/bakeoff/solvers.jl` | 61 | new-untracked | `ipopt_bfgs` / `ipopt_newton`, unit-cube bounds, f/g/h counters |
| `julia/bakeoff/solvers_profile.jl` | 42 | new-untracked | `ipopt_bfgs_bounded`, per-coordinate lo/hi so a coord can be pinned `lo==hi` |
| `julia/bakeoff/run.py` | 156 | new-untracked | the 5-solver bake-off, 16 starts, p3i2; **saves no endpoints** |
| `julia/bakeoff/run_seed.py` | 133 | new-untracked | paired arm A vs B on p3i2; dataset and candidates hardwired |
| `julia/bakeoff/run_multistart.py` | 197 | new-untracked | general driver, `CONFIG.json`; arms A/B/B5/P_paired/Bnarrow |
| `julia/bakeoff/run_armC.py`, `run_armC2.py` | 142, 157 | new-untracked | pose-conditioned membrane (one per pose / one per candidate) |
| `julia/bakeoff/run_armD.py` | 236 | new-untracked | seed from XModFit's own (theta,phi) cost map. Newest; **no verdict recorded anywhere** |
| `julia/bakeoff/phys_seed.py` | 145 | new-untracked | Patterson-derived `cov` and `protins` starting values |
| `julia/bakeoff/seed_utils.py` | 61 | new-untracked | physicality screen for homometry membrane seeds |
| `julia/bakeoff/{analyse,distinct,crossseed,catalogue}.py` | 104, 69, 121, 80 | new-untracked | p3i2 analysis: split-norm, merge-rule sweep, cross-seed recovery, depth-banded catalogue |
| `julia/bakeoff/compare_arms.py` | 186 | new-untracked | master cross-arm/dataset/template comparison + exact Sobol-prefix N-curve |
| `julia/bakeoff/tolerance_scale.py` | 140 | new-untracked | empirical tolerance from replicate pose disagreement |
| `julia/bakeoff/profile_sig.py`, `hess_report.py` | 126, 78 | new-untracked | `sig` profile likelihood; Hessian errors/correlations/eigenspectrum |
| `julia/bakeoff/endpoints*.npz` | 92 files, 2.4 MB | new-untracked | every saved multistart run (see 4.5) |
| `julia/bakeoff/results_p3i2.txt` | 13 | new-untracked | the bake-off table as printed |
| `lmfit/minimizer.py` | - | modified (uncommitted) | `XLayersPDB_XPR` added to `_PDB_FUNCNAMES` (`:95`) |
| `Fit_Routines.py` | - | modified (uncommitted) | `can_jac` / `residual_jac` / `_canonicalize_result` / `_report`; `set_emcee_minimizer` tuple |
| `Joint_Fit_Routines.py` | - | modified (uncommitted) | `_report`, appending each objective's derived block with a namespace keymap |
| `xmodfit.py` | - | modified, **no XPR change** | grep for `xpr` is empty; modified for other categories |

Out of scope, cross-reference only: the numba XLayersPDB family is `handoff/04-pdb-xrr-models.md`; MCMC/orientation walker seeding inside `lmfit` is `handoff/03-orientation-seeding.md`; the joint-fit engine is `handoff/01-*`.

## 3. How it works

### 3.1 Julia project resolution

`Functions/xpr_backend.py:61 julia_project_dir()` takes the first of `$XPR_JULIA_PROJECT`, `<repo>/julia/XPR`, `<repo>/../XPR-main/XPR` that has `src/XPR.jl`. On this machine the third resolves to `GitHub/XPR-main/XPR`, so `$XPR_JULIA_PROJECT` is how you develop against your own checkout without editing code. The vendored copy is version-pinned by its `Manifest.toml`.

### 3.2 Divergences of the vendored `XPR.jl`

The header enumerates four; the file carries five distinct patches, all marked inline. **Do not hand-merge an upstream drop** - run `python julia/merge_upstream.py` (3.9), which re-applies all of them from anchors and fails loudly if upstream has moved one.

1. **Optional keywords on `load_data`/`prealloc`** (`XPR.jl:210`, `:246`), each defaulting to the upstream literal: `xray_energy` (was 10.0), `fres` (was `true`), and the three slab-sizing ones `max_box_length` (was a literal 30 A/box), `max_sig` (was 5 A), `slab_headroom` (new, 0). The sizing ones exist because overrunning `max_ed_slabs`/`max_refl_slabs` **segfaults, it does not raise** - and a segfault in the Qt process takes XModFit down.
2. **`XPR_edp` / `XPR_sim_edp` / `XPR_maxh`**, returning the smeared EDP (the second alongside R and the protein height from one `XPR!` call - `sim` then `edp` runs the forward model twice; `sim_edp` is claimed ~1.9x faster), and `XPR_maxh(theta, phi)` -> `(maxh, mindepth, atom)`, the centroid-to-top offset with no model evaluation at all. `maxh` is what converts between the two definitions of the insertion parameter (3.9).
3. **`XPR_jac(params, active)`** (`:1335`), `dR_i/dp_j` by forward-mode AD over a subset. `const JAC_CHUNK = 4` (`:1328`) **pins** the ForwardDiff chunk: the default derives from `length(x)`, and letting it float makes Julia recompile the model (~3 s) every time the varying set changes.
4. **Unit-cube reparameterisation** (`to_physical`, `:87`, applied at `:1178` inside `XPR!`). `load_data(...; param_lower, param_upper)`, each 8+2n long, makes every entry point take a `[0,1]` vector; because the map is *inside* the AD tape, `XPR_grad`/`XPR_hess`/`XPR_jac` come back already chain-ruled in normalised coordinates. Motivation: raw `cond(J^T J)` ~ 1e13. Omit the bounds and behaviour is unchanged in physical units.
5. **`int_pts` allocation fix** (`:336-352`), not in the header. `TempVars.int_pts` is `zeros(T, 2*max_intersect, 2)` with `max_intersect = N(N-1)/2`, 6.6e6 entries for 1815 atoms; `TempVars` is parameterised by `T`, so nested Hessian duals (~169 components) made that one array ~9 GB. It is read only by `greens!`, i.e. `vol_calc="exact"`, so it (with `cell_max`/`max_neighbor`) is now sized to 1 under `"approx"`.

Sixth change, in `Project.toml`: **Ipopt 1.15.0**, purely for the bake-off. Pre-Ipopt `Project.toml`/`Manifest.toml` were backed up to `/tmp/XPR_*.bak` (volatile; regenerate with `Pkg.rm("Ipopt")`). `E` is inert throughout: XPR allocates its absorption array and never writes it, so `beta == 0` and lambda only ever appears multiplied by beta. Parratt is written in q, so `q_c` carries no wavelength - E moves the critical *angle*, not the critical *q*.

### 3.3 The Python bridge, `Functions/xpr_backend.py`

* **Module-level session, deliberately.** `_jl` is a module global (`:51`); `get_jl()` (`:132`) is the double-checked lazy starter. Keeping the handle off the model instance is what keeps `XLayersPDB_XPR` **picklable**, required by the parallel emcee / seed-scan paths. The file is in `Functions/` root, not `Functions/XRR/`, because XModFit treats every `.py` in a category folder as a fittable model class (same rule as `pdb_cache.py`).
* **Env workarounds, all applied before `import juliacall`** (`:153-162`). `PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=no`: juliacall auto-loads an IPython extension whenever `IPython` is merely in `sys.modules` with a live shell (it is, via pyqtgraph), and that extension rewires Julia's stdio and **kills the Qt process silently with exit status 0** the instant juliacall is imported. `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` so Ctrl-C raises into Python. `PYTHON_JULIAPKG_PROJECT` defaulted to `default_juliapkg_project()` (`:77`) = `<depot>/environments/pyjuliapkg`, because juliapkg's own default keys on `$CONDA_PREFIX`, so every *activated* conda env would get a private ~1 GB Julia install while the same interpreter invoked by absolute path resolves elsewhere.
* **SSL retry** (`:167-181`): juliapkg pings `julialang-s3.julialang.org` on every startup over stdlib urllib. If the import fails and a runtime already exists, `get_jl` sets `PYTHON_JULIAPKG_OFFLINE=yes`, drops juliacall/juliapkg from `sys.modules` (`_forget_juliapkg_modules`, `:99` - juliapkg snapshots its config into a module-level `STATE` at import time) and retries once. Safe because the failure is inside `juliapkg.executable()`, before libjulia loads.
* **StructUtils workaround** (`:207-209`): after `Base.set_active_project` onto XPR, juliapkg's project is **appended** to `LOAD_PATH` (appended, so the XPR manifest still wins). Without it, ForwardDiff pulls in StaticArraysCore, firing the StructUtils package-extension callback (StructUtils arrives via PythonCall -> JSON), and Julia dumps a 26-frame "required but does not seem to be installed" error - once per process, i.e. once per emcee/seed worker.
* **Fast activation** (`:195`): `Base.set_active_project` + `using XPR`, falling back to `Pkg.activate`/`instantiate` only on failure. `Pkg` costs ~4.4 s of the ~16 s a worker needs to become useful, and `instantiate` takes depot locks, so a 16-way pool serialises on them.
* **Load fingerprinting** (`load`, `:269`; `fp` at `:309`) over q/R/err arrays, box count, PDB path + mtime, `vol_calc`, `rho_top`, `rho_bottom`, `E`, `rrf`, `electron_scale` and the normalisation bounds. A matching fingerprint whose `_Alloc` *covers* the requested sizing is a no-op; headroom already paid for is retained (`:323`) so a fit that grows the bounds once does not oscillate. numpy buffers become real Julia `Array`s (`_jarr`, `:332`) because XPR keeps the data for the life of the load and must not alias memory Python may free.
* **Slab guard** `check_headroom` (`:412`) raises `ValueError` for a box longer than `max_box_length`, a `sig` above `max_sig`, or a `protins` stretching the profile past `protlen + max_box_length*nbox + slab_headroom`. Both a negative `protins` (protein below the stack) and one larger than the protein (lifted clear above) overrun it. Without this the call is a segfault.
* **Two sizing strategies.** `alloc_bounds_for(LO, HI, nbox)` (`:399`) sizes from the *bounds*, so a bounded global search never reallocates mid-run; the model class instead quantizes from *live values* (`_quantize_up`, `XLayersPDB_XPR.py:280`, used at `:506-513`) so the budget is a monotone staircase that settles after one or two steps.
* Unit-cube helpers `is_normalized` (`:380`), `to_physical` (`:385`), `to_normalized` (`:392`). Entry points `sim` (`:453`), `grad` (`:461`), `hess` (`:474`), `edp` (`:481`, z negated into XModFit's convention: 0 at the top of the stack, increasing into the film), `jac` (`:493`, 0-based indices converted to 1-based), `sim_edp` (`:509`), `reset` (`:524`).

### 3.4 The model class, `Functions/XRR/XLayersPDB_XPR.py`

* **mpar layout:** one `Model` phase with `Layers`/`d`/`rho`. Row 0 is the top medium, the last row the bulk; only their `rho` is used, `d` ignored. Rows between are the boxes, top to bottom (`_full_params`, `:408`). At least 3 layers required (`:494`). `mpar` is deep-copied in `__init__` (`:152`) - it is a mutable default argument and XModFit builds every instance as `funcClass()`.
* **Roughness is ONE scalar `sig`** (`:193`), not an mpar column: XPR smears every interface with the same value. A real physical difference from `XLayersPDB`.
* **Bounding media are load-time constants.** XPR bakes `rho[0]` and `rho[-1]` into `SimData` as `Float64`, so varying either forces a full reallocation every iteration - and it is why `parameter_columns` returns `None` for them (`:478`).
* **`cov` semantics, the load-bearing decision.** The fitted `cov` reaches Julia **raw and untouched** (`:420`), which is what keeps `xb.grad`/`hess` exact: nothing the gradient needs sits outside the tape. Raw `cov` is `n_s * A`, and `A` differs by mode - `sum(pi r^2)` for `'approx'` vs the true footprint for `'exact'`, ~70x apart for a 1700-atom protein - so it is not a covered fraction and is not even on the same scale in the two modes. The fraction is **derived downstream**: `cov_scale()` (`:317`) returns `(V_union/N) * (A_xpr / A_proj(theta,phi))` for `'approx'`, 1.0 for `'exact'`, NaN under `cov_norm='xpr'`. `y()` publishes `cov_scale`, `coverage`, `footprint_Angs2`, `prot_volume_Angs3` as scaler outputs (`:571-577`); `coverage_fraction()` (`:385`) converts a whole MCMC chain after the fact.
* **`cov_norm='protvol'` (default) vs `'xpr'`.** XPR's fast path builds `alpha[j] = sum_i f_ij/(A*dz)` from dimensionless segment *fractions* (`XPR.jl:926 area_slices_sphere_seg!`), so it has units of 1/volume where `area_slices_gl!`'s `volume/(A*dz)` is dimensionless. The model's electron term and its excluded-volume term therefore imply surface densities differing by `V_union/N` (6.1x on the test protein) - exactly what `XLayersPDB`'s `protvol` normalisation removes. `'protvol'` closes it by scaling the loaded electron array by `N/V_union` (`electron_scale`, `:294`): a per-structure, rotation-invariant CONSTANT folded into the data at load time, hence invisible to the parameters and to AD. `'xpr'` reproduces upstream bug-for-bug and is what `validate_xpr.py` uses for its reference log-likelihoods (it defaults `cov_norm='xpr'`).
* **`canonical_params()`** (`:247`). theta/phi are deliberately unbounded (`:199-200`) so emcee can cross the poles and the phi wrap without a reflecting wall; the fold in `y()` (`_fold_angles`, `:228`) is local, because lmfit re-sets parameters from its own variable vector each residual call and writes that vector into `result.params` at the end. So the *reported* angle is the raw optimiser coordinate - observed: reported theta=379.3, phi=-8208.2 for a fit evaluating 19.3 / 71.8 (identical R). `canonical_params` is a pure relabelling called from `Fit._canonicalize_result()` after minimize, before `fit_report`; chi2 and residuals untouched. Caveat: a reflected theta flips the *sign* of its correlations (magnitudes stand).
* **Default finite bounds** `D_MAX=100`, `RHO_MAX=5`, `SIG_MAX=10`, `PROTINS_MAX=100` (`:430-433`) - a *performance* decision. Evaluation cost is proportional to the z-span of the profile, walked on a fixed 0.5 A lattice; an unbounded `d` lets a walker sit at `d=[210,320]`, 1292 slabs / 15 ms instead of 269 / 1.8 ms. Walkers are seeded by scattering each parameter by its *relative* stderr, so a poorly determined thickness lands there routinely and the chain crawls until burn-in. With finite bounds `_lnprob`'s prior rejects the proposal before the model runs, at zero cost. **A saved `.par` restores its own bounds**, so pre-existing setups keep their old infinite ones and must be fixed in the GUI.
* `vol_calc='exact'` costs ~300x (`7*(slabs + 2*atoms)` union-of-circles solves): 3.5 -> 1160 ms per `y()`, 12 -> 1634 ms per gradient, on 1697 atoms / 259 slabs.

### 3.5 `Functions/pdb_geometry.py`

`union_volume` (`:122`, `lru_cache`, rotation-invariant so computed once per structure) and `projected_area` (`:144`, `lru_cache(512)` on the exact float angles - deliberately *not* rounded, since quantising would put a staircase into the objective), over numba `njit` grids (`_mark_volume` `:36`, `_mark_area` `:66`). `sphere_sums` (`:107`) gives `(N, sum pi r^2, sum 4/3 pi r^3)`. `xpr_rotation_matrix` (`:88`) reproduces XPR's `rotate_protein!` rows. **Why not imported from `XLayersPDB`** (which carries its own `Vol_gridcalc`/`Area_gridcalc`): `Functions/XRR/XLayersPDB.py` imports `Functions/pdb_cache.py`, so a shared helper importing back into a category folder would be circular *and* would break the rule that a category folder holds only model classes (`pdb_geometry.py:14-21`).

### 3.6 Analytic Jacobian into lmfit

`parameter_columns(names)` (`XLayersPDB_XPR.py:441`) maps lmfit names to `(column, sign)` in the 8+2n vector, returning `None` if *any* name is unmappable so the caller falls back to finite differences for the whole Jacobian rather than mixing exact and missing columns. Column `-1` means provably zero (top/bulk `d`); theta carries sign `-1` when the fold reflected it; `rho` of the first/last layer is unmappable (baked into `SimData` as `Float64`). `dydp(names)` (`:526`) calls `xb.jac` seeded only on live columns.

`Fit.can_jac` (`Fit_Routines.py:315`) gates it: opt out with `fit.use_jacobian=False` (`:54`); refused for joint fits (`type(self.x)==dict`), for any `expr` constraint (needs an unimplemented chain rule), and **when fewer than 4 parameters vary** (`:330`). `Fit.residual_jac` (`:338`) mirrors `residual()`'s scaling term by term for all four fit scales. Handed to lmfit as `Dfun` for **leastsq only** (`:442`): `least_squares` takes its own `jac` with a different signature and no `Minimizer.__jacobian` wrapper, so the bounded-to-internal gradient rescaling would be missing. The `nvarys>=4` gate is measured - a clear win from 4 up, a loss at 2 - and in the emcee orientation seed scan (sub-fits with ~2 nuisance params) it made things *worse*: 30 -> 34 s serial, 51 -> 74 s at 16 workers, because every worker re-pays the ~3 s AD specialisation. End to end on 5 varying parameters: 261 model calls -> 41, same optimum to 1.3e-5.

### 3.9 The 2026-09-08 upstream revision: insertion redefined, lattice rebuilt

Elias Boshara's updated `XPR-main/XPR` changed two things in the physics layer, both
for **smoothness**, and one of them changes the meaning of a fitted parameter.

**(a) `d_protein` (XModFit's `protins`) is now measured from the protein's MASS
CENTROID.** It used to be measured from the TOP of the protein: `rotate_protein!`
translated by `z_top = d_protein - sum(box_lengths) - maxh` where
`maxh = max_i (r3(theta,phi).c_i + rad_i)`. That `maximum` over atoms is only
piecewise smooth in (theta, phi) - the topmost atom changes as the protein turns -
so the *translation itself* carried a kink, which is the one thing a differentiable
backend cannot afford. Upstream now subtracts the mass-weighted centroid from the
coordinates in `prealloc` (hence the new `masses_py` argument to `load_data`) and
translates by `z_offset = d_protein - sum(box_lengths)`, which is exactly linear in
`d_protein` and exactly smooth in the angles. `rotate_protein!` returns the old
quantity as `protein_height = d_protein + maxh`, and `XPR_sim_ref` passes it back as
a third value.

**(b) `smear_xy_interface!` was rebuilt onto one rigid lattice.** A single `spacez`
grid spans the whole padded profile, the box interfaces are injected into it with
`sort(unique(...))`, everything is ascending in z and `reflectivity!` reverses the
views. Previously the smearing lattice and the reflectivity lattice were assembled
separately out of parameter-dependent `collect(a:step:b)` ranges, so their *length*
moved with the parameters. `eval_smeared!` lost its `ns` argument accordingly, and
`smear_lattice`/`rho_hard_z`/`mu` are now sized `max_refl_slabs`.

**Neither is a change of physics.** `validate_xpr.py` reproduces the stored protein
reference `-466.4557` **exactly** at `protins = 13.358030329245691 - maxh`, and
`protein_height` comes back as `13.358030329246`. The box references moved in the
fourth decimal from the new lattice (`-1214.2981 -> -1214.1600`,
`-600.6506 -> -600.6398`) and those are upstream's own new `demo.ipynb` numbers,
which we now match to all printed digits, 4-box included.

**Measured smoothness gain** (1697-atom test protein, `protein_2box.rrf`):

* Along `phi = 112.59` the topmost atom changes at **10** values of `theta` in
  `[1, 179]`, and `d(maxh)/d(theta)` genuinely jumps at each (4e-4 to 1.2e-1). The
  old convention would have put that jump straight into `dlogL/dtheta`. It now does
  not: worst left/right disagreement over those 10 angles **3.5e-5 relative**, with
  the AD gradient sitting between the two one-sided differences. `validate_xpr.py`
  asserts this and names the mechanism if it ever fails.
* AD vs central differences for `(theta, phi, protins)` over 24 random orientations:
  worst **7.4e-6** relative, at the round-off floor (it stops falling with `h`).
* The `10*sig` grid-alignment kink is **gone**: at `sig = 3.4` (i.e. `10*sig = 34.0`,
  exactly on a node) AD vs central differences is **1.4e-8**, where the old code
  disagreed by ~1e-2. The smearing extent now goes through `floor`/`ceil`, which
  carries no derivative.
* The `sum(d)` kink **remains**, because `sort(unique(...))` merges a box interface
  with the grid point it lands on. At `d = [11.0, 9.0]` the one-sided `dlogL/dd1`
  slopes are `+39549.0` and `+39553.3` and stay ~4.3 apart as `h -> 0`; AD returns
  the right-hand one. ~9e-3 relative in `dR/dd` at the worst q point. Measure zero,
  but round-number defaults sit exactly on it.
* Crossing a lattice node in `protins` (which moves the padded grid ends by 0.5 A
  and changes the slab count) leaves the derivative stable to six digits.
* Hessian symmetric to 2.3e-10; its `theta`/`phi`/`protins` rows match finite
  differences of the AD gradient to 1.3e-8 / 1.2e-6 / 5.3e-7.

**What XModFit does with the redefinition.** `protins` is passed to Julia raw, as
always, so the fit is in the new coordinate and the AD chain stays exact. The old
quantity is *derived and reported*, three ways:

* `y()` publishes `protins_top` and `maxh` as scaler outputs, taking `protins_top`
  from Julia's own `protein_height` so it cannot drift from what was evaluated.
* `XLayersPDB_XPR.protins_top(protins, theta, phi)` / `.protins_from_top(...)`
  broadcast, so an emcee chain converts in one pass with no model evaluations -
  which is the honest way to get its credible interval.
* `derived_params(result)` / `derived_report(result)` give it with a **propagated
  error and its covariance against every other varying parameter**, linearising
  `protins_top = protins + maxh(theta, phi)` through `result.covar` with
  `d/d(protins) = 1` and `d(maxh)/d(theta, phi)` taken analytically from the atom
  that currently attains the maximum (`pdb_geometry.com_extent_grad`).
  `Fit._canonicalize_result` calls it **before** folding the angles, because
  `result.covar` is in the optimiser's raw coordinates and, while `maxh` is exactly
  invariant under the fold (`r3(360-theta, phi+180) == r3(theta, phi)`, so the
  *value* does not care), the SIGN of `d(maxh)/d(theta)` does.
  `Joint_Fit_Routines._report` does the same per objective through a namespace
  keymap built from `_master_key`, collapsing duplicates when the pose is shared.
  Both append a `[[Derived]]` block to the fit report.

The linearisation is verified exact (Monte Carlo from the same covariance shrunk to
1% agrees to 1.0002), but it is a linearisation of a `max` over atoms: on the
validator's test fit, where `sigma(phi) = 8.2 deg`, the full-covariance Monte Carlo
gives 0.374 against the linearised 0.488, a 31% difference. That is nonlinearity,
not error. `derived_report` prints the active atom index and a tie count so a fit
sitting on a kink is visible.

**Sizing.** `protins = 0` now puts the centroid on the bottom interface, so roughly
half the protein always hangs below the stack: there is *always* an overhang where
the old convention needed none. `_prepare` therefore derives `slab_headroom` from
`com_extent(theta, phi)` rather than from `protins` alone, and
`xb.check_headroom` / `alloc_bounds_for` take `prot_up`/`prot_down`. The `min=0`
bound on `protins` was wrong under the new convention and is now `-PROTINS_MAX`.

### 3.10 The 2026-09-10 upstream, and what the moving mesh actually costs

Upstream took all four asks from `XPR_FEEDBACK_FOR_ELIAS_2026-09-09.md`: `spacez` and
`min_box_length_sum` are keyword arguments, an out-of-window protein is a hard
`error("protein out of bounds")`, and `XPRsim._import_pdb` gives the CAL ion calcium's
mass (plus `.capitalize()` on the element lookup). **Two of our divergences became
upstream features and were retired** -- the `max_box_length`/`max_sig` sizing keywords
and `slab_headroom`. `julia/merge_upstream.py` is rebaselined; it re-applies four
divergences now, not five, and is idempotent.

All four of upstream's `demo.ipynb` references reproduce to every printed digit:
2-box **-1212.2127**, 3-box **-600.1768**, 4-box **-62404.0635**, protein+2box
**-464.6397** at `protins = -13.717` (the demo now quotes the centroid-referenced
pose directly, which was the fifth thing we asked for).

**The sizing contract split in two, and conflating the halves is silent.** Measured:

| argument | changes the answer? | why |
|---|---|---|
| `spacez` | **YES** | mesh outside the boxes, and `cells_per_box = ceil(box_ref/spacez)` |
| `box_ref_length` (-> `max_box_length`) | **YES** | in-box mesh `d_b*spacez/box_ref` |
| `d_protein_bounds` | no (4e-11) | extends constant-density far field; but Julia ERRORS outside it |
| `sig_max` | no (6e-11) | Gaussian tail length |
| `min_box_length_sum` | no (4e-13) | trims top cells |

So `_param_box`/`alloc_bounds_for` derive the *fatal* window from the lmfit bounds
and the *answer-bearing* mesh from `spacez`/`box_ref_length`, which are model
attributes and deliberately NOT bounds-derived: two people fitting the same data with
different `d` bounds must get the same chi2. The old ratcheting `_quantize_up` sizing
is gone -- under the new code it would have stepped the objective mid-descent.
`check_headroom` now mirrors Julia's window test (using the rotation-invariant
`prot_extent`, as upstream does) and RAISES; box-length and `sig` overruns only warn,
because they degrade gracefully.

**The resolution study (`julia/resolution_study.py`, report in
`julia/resolution_study_report.txt`), on real data:** mTIM3 `d20_tim3_1` + `t3hmmm`,
94 points, q 0.028-0.560, median sigma/R 0.94%.

1. **Convergence is clean O(h^2)** -- observed order 1.87 then 1.97 over
   `spacez` 0.5/0.25/0.125/0.0625, Richardson limit logL = -195.035. The default mesh
   is off by **+1.76 in logL (-3.51 in chi2)**.
2. **The error is a smooth q^1.91 distortion of R**, from -6e-5 (q<0.1) to +4.6e-3
   (q>0.5), worst point 0.168 sigma, `sum((dR/sigma)^2) = 0.355`. **R/R_F and raw R
   carry the SAME relative error to 1.7e-16** -- the Fresnel divisor is analytic in q
   and mesh-independent, so normalising changes nothing.
3. **The parameter-dependent half is the MINOR half.** Only the in-box spacing
   moves with the parameters; by the model-difference norm `sum((dR/sigma)^2)` it is
   **6.8%** of the total at the fitted point, and the rigid `spacez` region dominates.
   The two also have OPPOSITE SIGNS in chi2 (+1.24 in-box vs -3.45 total), i.e. they
   partially cancel -- which is why raising `box_ref_length` alone makes absolute chi2
   *worse*, and why exposing `spacez` was the ask that mattered. The in-box part alone
   scales as **h_box^1.96** and is 2.7e-4 in R at the fitted `d1`, 35x below the noise.
4. **It does not bias the answer, pose included.** Two real 8-parameter fits,
   default vs `spacez=0.125`: every fitted value moves by **<= 0.030 of its own
   standard error**. Releasing `theta`/`phi` as well (10 parameters) the worst shift is
   **0.037 sigma**, with the orientation moving **0.037 deg in theta and 0.113 deg in
   phi** -- against a campaign pose tolerance of ~5 deg. Cross-evaluating, the
   default-mesh parameters cost only **+0.29 chi2** on the fine mesh, while the
   **+0.52 chi2** common-mode offset cancels from any Delta(chi2).
5. **For the global search it is small but not nothing.** Over 55 candidates drawn
   from the fit covariance within +100 chi2, the mesh error has sd **2.53 chi2 =
   1.26 logL** -- about **11%** of the tightest basin-merge tolerance (11 logL).
   Common-mode to first order; only its variation between candidates can reorder them.

**Recommendations** (full list printed by the study; summary here): keep
`spacez=0.5`/`box_ref_length=30` as production defaults, the bias is 25x smaller than
the parameter errors; fix both once per campaign and RECORD them, because absolute
chi2 is not transferable across them or across code versions; set `box_ref_length` >=
the thickest box you will fit (above it the in-box mesh coarsens past `spacez` and the
model warns); confirm convergence once per campaign by refitting the final answer at
`spacez/2`; for the multistart treat **~1.3 logL as a mesh noise floor on candidate
RANKING** and re-score the shortlist at `spacez/2` before declaring a winner; and
refine `spacez`, never `box_ref_length` alone.

### 3.10.1 The downstream migration, and a fingerprint correction (2026-09-10)

Retiring the two divergences in 3.10 **broke every offline fitting harness**, in both this
repo and `../xpr-seeding-study`. The model itself was fine; the callers were not:

* `xb.alloc_bounds_for()` returns a **dict** (`max_box_length`, `sig_max`, `spacez`,
  `min_box_length_sum`, `d_protein_bounds`). Every harness still did
  `mbl, ms, sh = xb.alloc_bounds_for(...)` -> `ValueError: too many values to unpack
  (expected 3)`.
* `xb.load()` no longer accepts `max_sig=` or `slab_headroom=`.

**Migrated the same day: 32 files, +81/-89 lines.** 31 harnesses (10 in
`julia/bakeoff/`, 10 in the study's mirror of it, 11 in the study's
`stage0_tim4/`) now do `sizing = xb.alloc_bounds_for(...)` and splat `**sizing` into
`xb.load()`, which is the pattern `Functions/XRR/XLayersPDB_XPR.py:707` already used. Five
scripts also echoed the old 3-tuple (`alloc(box=.. sig=.. head=..)`) and now print the
whole dict, so nothing is silently dropped from a run log.

`slab_headroom` was **dropped, not translated.** `julia/merge_upstream.py:120` records it
as obsolete because the slab count no longer depends on the parameters, so there is no
proposal it could protect against; no replacement value was invented for it.
`min_box_length_sum` is its nearest relative but is not the same quantity.

`tests/recovered/check_aproj.py` is the one caller that talks to Julia directly rather than
through the bridge. Besides `max_sig` -> `sig_max` it needed **`masses_py=`** added, which
the new `load_data` requires for the mass-centroid recentring; the script already unpacked
the masses from `pdbread` and was discarding them.

**Verification.** `stage0_tim4/tolerance_scale_tim3.py t5lowis` under
`XPR_PIN=pose XPR_CFGDIR=configs_newconv/tim3_cond`: exit 0, 12 datasets x (own optimum +
pinned-pose re-optimise) = 24 real fits, every `-logL` finite (20.267-52.128), no NaN and
no warnings. Separately, `alloc_bounds_for`'s dict keys were asserted to be a subset of
`load()`'s parameters, and all 32 files byte-compile. **No published number was recomputed
and no completed analysis was re-run.**

> **Correction for anyone checking provenance.** The source fingerprint of this tree is
> **`2402cb195df0ce6f`**, not the `432da5d0da1b10fe` that was in circulation on the day --
> that value appears to have been taken mid-integration, between `XPR.jl` (11:51) and
> `XLayersPDB_XPR.py` (12:19). The migration itself **cannot** move the fingerprint:
> `source_fingerprint()` hashes only the six MODEL sources, and none of them is a harness
> script. The conclusion that mattered is unaffected -- the corpus is still either
> unstamped (pre-2026-09-08) or `b815f595af37b2cc`, so **archives produced under this
> revision must not be pooled with it**, and `stage0_tim4/archive_guard.py` enforces that
> (re-verified after the migration: 2587 archives, all `PRE-2026-09-08`, single
> convention).

**`run_multistart.py` had diverged into a dead duplicate.** The study's copy carried
neither `XPR_OUT_DIR` nor the `src_sha256` stamp, and could not run at all: it derived
`REPO` as `HERE/../..`, but that repo carries the harness and **not** the model (no
`Functions/`, no `julia/XPR/`), so it died on `ModuleNotFoundError: No module named
'Functions'`. Porting the two blocks verbatim would have been worse than leaving it --
`source_fingerprint()`'s `root` is derived the same way, so all five model files would have
hashed to `MISSING` and every archive would have carried an authoritative-looking stamp of
nothing. It now resolves `REPO` through `XMODFIT_ROOT` (the convention the other 16 study
scripts already used unanimously), reads the five model files from there and `solvers.jl`
from its own directory (the copy `jl.include` actually loads; byte-identical to ours
today), and keeps the label strings unchanged so **both copies emit a bit-identical stamp**
-- verified, both give `2402cb195df0ce6f`. Apart from that path resolution the two files
are now the same, and the rest of the two `julia/bakeoff/` trees was already byte-identical.

Two things an auditor should know about that copy: `stage0_tim4/run_wave.py` defaults
`XMODFIT_BAKEOFF` to **this** repo's `julia/bakeoff`, so production waves have always
executed our copy and the study copy is still not on any live path; and
`archive_guard.assert_single()` refuses via `SystemExit`, not a catchable `Exception`, so a
caller wrapping it in `except Exception` will not see it fire.

### 3.7 Registration in upstream files

`lmfit/minimizer.py:95` adds `'XLayersPDB_XPR'` to `_PDB_FUNCNAMES`, enabling orientation-grid seeding, CustomSampler angle wrapping and the picklable parallel objective (consumed at `:1515`, `:1825`, `:1929`, `:2102`). `Fit_Routines.py:491` adds it to the `set_emcee_minimizer` relative-stderr tuple. `xmodfit.py` needs **no** change - the class is auto-discovered from `Functions/XRR/`. The GUI wiring described in `STAGE2_SEEDING_PLAN.md` ("new entry in `xmodfit.py:574 self.fitMethods` and the allowlist at `xmodfit.py:1765`") is for the not-yet-written global-search fit method.

### 3.8 The harness (`julia/bakeoff/`)

Every driver has the same shape: read data; build `LO`/`HI` for the 12-slot vector with `FREE = [0,1,2,3,4,5,8,9,10,11]` (yscale/bkg pinned by a null range); `xb.load` with `param_lower/param_upper` + `alloc_bounds_for`; `jl.include('solvers.jl')`; loop Ipopt L-BFGS over starts; classify each endpoint from the exact Hessian; save an `.npz`. Classification is uniform: `w = eigvalsh(-hess)`, `tol = 1e-8*max(|w|,1)`; `w[0] > tol` -> MIN, `< -tol` -> SADDLE, else DEGENERATE (`run_seed.py:72`, `run_multistart.py:88`, `run_armD.py:89`).

**Arm naming convention**, used in filenames and both design docs: **A** unseeded Sobol over all 10 coords; **B** Sobol over the 5 pose coords with the membrane block overwritten by a homometry candidate, round-robin (**the production scheme**); **B5** as B but the pose is a *native* 5-D Sobol set, not the first 5 columns of a 10-D one; **C** pose-conditioned membrane sub-fit; **C2** one conditional sub-fit per candidate, deduped and each promoted; **D** seeded from the orientation cost map.

| script | what it does |
|---|---|
| `solvers.jl` | `ipopt_bfgs` (limited-memory, exact `XPR_grad`) and `ipopt_newton` (exact `XPR_hess` as lower-triangular sparse), bounds `0..1`, `max_cpu` 60 s, `max_iter` 300, `tol` 1e-8, `mu_strategy=adaptive`, plus f/g/h counters |
| `solvers_profile.jl` | `ipopt_bfgs_bounded(x0, lo, hi)` for profile likelihoods. Separate file so `solvers.jl` stays untouched while long runs are in flight |
| `run.py` | the 5-solver bake-off; prints only, output preserved in `results_p3i2.txt` |
| `run_seed.py` | paired A vs B, `python run_seed.py [N] [seed]`; candidates hardcoded at `:36`; writes `endpoints[_seed<N>].npz` + `timing*.json` |
| `run_multistart.py` | `python run_multistart.py CONFIG.json [N] [seed] [arms]`. Arms `A_sobol10`, `B_homoseed`, `B5_native`, `P_paired` (every pose x every candidate), `Bnarrow` (cov recentred/narrowed on a Patterson estimate for 80% of starts, the rest at full range as insurance). Runs `seed_utils`' screen; warns if the best endpoint rails at a bound (`:181`) |
| `run_armC.py`, `run_armC2.py` | pin the pose and solve the membrane inside a physical window around the lipid-only reference, then release everything; C2 does it once per candidate |
| `run_armD.py` | imports `build_orientation_grid`, `_detect_orientation_basins`, `seed_thetaphi_from_costmap` from `lmfit/minimizer.py`. Mode `fast` = detect basins, polish each with one full 10-D solve (candidate design for a ~2 min GUI first pass); mode `full` = draw N poses from the tempered area-weighted posterior exactly as `Minimizer.emcee` does. `costmode` in `eval`/`posins`/`full`/`phys`/`physins`/`physins2` |
| `phys_seed.py` | `data_patterson_peaks` (outermost two local maxima of \|G(z)\| in 15-60 A; local maxima, never argmax over a window), `cov_from_patterson`, `protein_depth`/`protins_from_depth` (the validated rule), `protins_for_pose`/`target_extent_from_data` (the failed rule, kept on record) |
| `seed_utils.py` | `screen_membrane_seeds`: drop candidates with `rho < rho_top`, `rho > 0.62`, non-positive `d`, or outside the optimiser bounds. Guards a quiet failure - `run_multistart.py` used to *clip* `rho=-0.02` to exactly 0.0, a vacuum layer, wasting the start |
| `analyse.py` | single-seed per-arm summary, split-norm (membrane \|\|du\|\| vs great-circle pose angle on the misses), per-block mean \|du\|, paired A-vs-B, top 8 |
| `distinct.py` | sweeps the merge rule to show which clause binds |
| `crossseed.py`, `catalogue.py` | cross-seed hit rates + Jaccard; catalogue under 4 identity rules with recovery by -logL depth band (this is what establishes the phase-4 depth cut) |
| `compare_arms.py` | reads every `endpoints_*.npz`: per (dataset, template, arm, seed) stats; **exact Sobol-prefix hit counts at N = 2..128**; pooled hit rate vs N; per-series pose consistency; depth-banded catalogue. `HIT_TOL` env var |
| `tolerance_scale.py` | pins each dataset's pose to a sibling's and re-optimises the rest. `cov` is re-optimised, not transferred |
| `profile_sig.py`, `hess_report.py` | `sig` profile by continuation outward from the optimum; marginal 1-sigma, correlations, eigenspectrum and softest eigenvector at the best endpoint |

## 4. How to use it

### 4.1 Install and environment

`python -m pip install -r requirements_xpr.txt` is the only pip step. First `import juliacall` downloads Julia 1.11.9 (262 MB zip, ~1 GB installed); first `using XPR` resolves and precompiles ForwardDiff + SpecialFunctions. Budget 10-20 min once; afterwards a session starts in ~6-7 s. Verified under Python 3.9 in both `cmc-py39` and `cmcars` (see the `cmc-py39-env` memory). Keep `juliacall` at the same version in every environment sharing the Julia project - juliapkg pins PythonCall to an exact compat bound. Env vars: `$XPR_JULIA_PROJECT` (which checkout), `$PYTHON_JULIAPKG_PROJECT` (where the 1 GB lands), `$JULIA_DEPOT_PATH` (honoured by the default), `$XRR_SKILL_SCRIPTS` (`run_armD.py:38`, where `xrr.py` from the `xrr-homometric-pipeline` skill lives).

### 4.2 GUI use of the model

Pick `XLayersPDB_XPR` under XRR. Fixed-parameter cells: `fname` (PDB), `rrf`, `use_protein`, `vol_calc` (`approx`/`exact`), `cov_norm` (`protvol`/`xpr`), `fslice`, `E`. Fitting parameters: `qoff`, `yscale`, `bkg`, `sig`, `theta`, `phi`, `protins`, `cov` plus `__Model_d_iii` / `__Model_rho_iii`. Read the `coverage` and `cov_scale` scaler outputs before bounding `cov`.

The first MCMC step is slow for three reasons, none of them the sampler: Julia session ~7 s, first-`y()` JIT ~8 s (both once per process), and the (theta,phi) orientation grid seed scan ~29 s (560 cells, ~14700 model evals - XModFit's own seeding for every `XLayersPDB*` model, not an XPR cost). To skip the scan set theta/phi `vary=0` **before** the least-squares fit: `set_emcee_minimizer` copies from `self.result.params`, so a flag changed afterwards is ignored. Use comfortably more than `2*ndim` walkers; the stretch move only evaluates `nwalkers/2` at a time, so 16 cores cannot help 22 walkers at ndim=10.

### 4.3 Regression gate

`python validate_xpr.py [path/to/XPR-main/test_data]` (defaults to `<repo>/../XPR-main/test_data`). **Run 2026-09-08 against the new upstream: ALL CHECKS PASSED.**

Checks the published `demo.ipynb` log-likelihoods to 1e-3 - 2-box `-1214.1600`, 3-box `-600.6398`, 4-box `-62408.5320` (new), protein+2box `-466.4557` at `protins = 13.358030329245691 - maxh` - plus, added with the 2026-09-08 integration:

* `maxh`/`mindepth` from `XPR_maxh` against `pdb_geometry.com_extent`, **bit-identical**, same argmax atom;
* Julia's `protein_height == protins + maxh` to 1e-9, and `protins_top`/`protins_from_top` round-tripping;
* `com_extent_grad` against central differences (worst 1.2e-9);
* **the smoothness assertion**: `dlogL/dtheta` one-sided differences at all 10 atom-switch angles, failing with the message "the insertion offset has leaked back into the forward model" if it regresses;
* upstream demo parity for the protein cell after correcting the documented CAL-mass divergence (below), `-897.7146` to 2e-3;
* `derived_params`' propagated error against Monte Carlo at 1% of sigma (must match to 2%, this is the Jacobian test) and at full sigma (loose bound only - see 3.9).

Then as before: the grid footprint against XPR's own union area, coverage consistency between `vol_calc` modes, EDP output, grad/hess finiteness, leastsq and emcee through `Fit_Routines`, angle canonicalisation, the normalised-load round trip, and the analytic Jacobian against central differences (worst 3.7e-6; `use_jacobian` 271 -> 33 residual calls, same chi2 to 3e-9).

**One knowing divergence from upstream `XPRsim.py`, found here.** `test_data/protein.pdb` line 2 is a calcium ion named `CAL` whose element columns are blank, so Bio.PDB guesses element `'C'`. `XPRsim._import_pdb` overrides the name to `'Ca'` for the electron count and the radius but takes the **mass** from `atom.element`, i.e. 12.011; `Functions/pdb_cache.pdbread` uses calcium's 40.078 for all three, which is right and is what every numba `XLayersPDB*` already does. One atom in 1697, but it moves the mass centroid - and hence the zero of `protins` - by **0.0345 A** on this structure. `validate_xpr.py` corrects for it explicitly and then matches `-897.7146` to 8e-4, which is how we know that is the whole of the difference. Worth telling Elias: his line is inconsistent with itself, not just with us.

### 4.4 The recommended production recipe (settled)

Per dataset per template, **~10 minutes**:

1. **Homometry pass, once per film series, ~75 s** - pure numpy, no Julia and no PDB: `python xrr.py LIPID_ONLY.rrf --rt <rho_top> --rb 0.334 --N 2 --quick --no-qoff` from the `xrr-homometric-pipeline` skill, on the **protein-free (lipid-only) scan**. Returns `sig, d1, d2, rho1, rho2` per basin candidate.
2. **Arm B multistart, N = 64 x ~8 s**: Sobol over the 5 pose coords, membrane overwritten by the candidates round-robin, all 10 coords released for the solve. `python run_multistart.py CONFIG.json 64 2024 B_homoseed`.
3. One exact Hessian per endpoint (288 ms) to classify MIN/SADDLE/DEGENERATE.
4. Dedup, depth-cut and rank; tolerances in 4.6.

**Number of starts (settled; TIM1, arm B, MD template, 13 runs over 11 datasets, exact Sobol prefixes):** P(find the optimum) = 0.31 / 0.31 / 0.46 / 0.54 / 0.85 / 0.85 / 0.92 / **1.00** / 1.00 / 1.00 at N = 1 / 2 / 4 / 8 / 16 / 24 / 32 / **48** / 64 / 128. **90% at N=32, 100% at N=48; recommend N=64** (35% margin over the worst observed first-hit, start 37 on p1d3). N=128 is wasted; N=16 is a coin flip on the hard datasets. **Only holds when the template is right** - on crystal, arm B reaches only 0.85 at N=128 and 2 of 13 runs never find the optimum.

Why Sobol: perfect stratification at powers of two (64 Sobol points put exactly one in each of 64 cells; 64 uniform points leave 24 empty), and nested prefixes, so one N=128 run answers N=16/32/64 *exactly*. The price is that points are not independent, so binomial statistics do not apply and prefixes give one realisation - vary the scramble seed for error bars.

### 4.5 The saved `endpoints_*.npz`

92 files, 2.4 MB total, so every table in both design docs can be recomputed without re-running anything. Two schemes: p3i2/TIM3 from `run_seed.py` - `endpoints.npz` (seed 2024) and `endpoints_seed{7,101,2718}.npz`, i.e. 2 arms x 4 seeds x 64 starts; TIM1 and later from `run_multistart.py`/`run_armC*.py`/`run_armD.py` - `endpoints_<tag>_<pdb>[_<armset>]_s<seed>.npz` with `<pdb>` in {`md`, `orig`}. Contents: `<arm>_starts`, `<arm>_us`, `<arm>_fs`, `<arm>_cls`, `<arm>_oks` (+ `<arm>_cand`), plus `FREE`, `NAMES`, `LO`, `HI`, `SEEDS`, `POSE`, `MEMB`, `N`. Free-coordinate order is always `qoff, theta, phi, protins, cov, sig, d1, d2, rho1, rho2`; indices 0-4 are `POSE`, 5-9 `MEMB`. Arm D also saves `cell_theta`, `cell_phi`, `cell_cost`, `basins`, `t_map`, `t_polish`, `armD_prov`.

**Filename clobber trap (fixed 2026-09-02).** `run_multistart.py` used to write `endpoints_<tag>_<pdb>_s<seed>.npz` regardless of which arms ran, so launching a new arm at an existing (tag, pdb, seed) silently overwrote the earlier file - this destroyed f6/orig's A+B results. The name now carries the arm set unless it is exactly {A_sobol10, B_homoseed} (`run_multistart.py:189`) and warns on overwrite.

### 4.6 Tolerances (settled)

* **`Delta(-logL) < 1` for "same basin"** bookkeeping - safe, and the answer is insensitive: the N-curve is *identical* for every tolerance from 1.0 to 19.8, because the gap from the global basin to the next distinct minimum is median **34** in -logL (min 3.5, max 317, n=13 datasets).
* **`Delta(-logL) ~ 11-20` when deciding whether two DATASETS disagree.** Measured by pinning each dataset's theta/phi/protins to a sibling's and re-optimising the rest: median 11.1 for the strictest replicates (p1o2/3/4, identical conditions), 19.8 pooled over all 32 pairs, max 79.9. Reproduced on the crystal template (17.5 / 11.9 / 17.5), so the scale is a property of the **data**, not the template.
* **Quote parameter errors from replicate scatter, not the Hessian.** Hessian says theta +/- 0.54 deg; replicates of the same film disagree by 1.3-6.8 deg. Systematics beat statistics by ~6x.
* `Delta(chi2) = 2*Delta(-logL)`, so the conventional 1.0 is `Delta(chi2) = 2` - right for about one parameter, far too tight for ten (68% joint for p=10 is `Delta(-logL) = 5.77`).
* Diagnostic: **if the N-vs-tolerance curve is sensitive to the tolerance, the model is probably mis-specified.** On crystal the same table runs 0.38 -> 0.92 at N=16 as the tolerance goes 0.5 -> 3.0.

### 4.7 Headline results, with sample sizes and caveats

**5-solver bake-off** (`results_p3i2.txt`; p3i2, 16 Sobol starts, 10 free params, all solvers on the same starts and the same exact AD derivatives, unit cube). time / best -logL / MIN / SADDLE: scipy L-BFGS-B 40 s / 94.15 / **0** / 14; Ipopt L-BFGS 117 s / 84.66 / 8 / 8; scipy trust-constr 529 s / 83.86 / 11 / 4; Ipopt Newton 478 s / 84.65 / 8 / 6; Ipopt BFGS->Newton escalation 544 s / 84.66 / **14** / 2. Standing conclusions: **coverage is the bottleneck, not the solver** (every method except scipy L-BFGS-B reached the then-best basin exactly 1 in 16, and the four good methods' best values span `Delta(chi2) < 2`); scipy L-BFGS-B is **disqualified** (0/16 MIN and it reports "converged" at saddles - it stops on a projected-gradient tolerance saddles satisfy); Ipopt earns its dependency on the L-BFGS row alone (8/16 vs 0/16 on the *identical* gradient - it tests KKT error); Ipopt exact-Hessian Newton is **dominated** by Ipopt L-BFGS (same 8 MIN, 4x cost); escalation buys **certification, not depth**, so use it as a finisher on good endpoints only.

**SUPERSEDED: that best value was not the bottom.** A 64-start rerun found **-logL 80.286**, 3.57 deeper; the whole 83.6-85.3 cluster is a distinct, shallower pose mode 25-28 deg great-circle away. `u*` physical: qoff 1.006e-4, theta 42.67, phi 149.8, protins 12.06, cov 9.092, sig 3.437, d=[12.48, 8.756], rho=[0.2404, 0.4642]. The pose saved in `p3i2_tim3.xpar` (theta 34.6 / phi 107.7) **is the second mode**. This is why the banner at the top of `STAGE2_SEEDING_PLAN.md` marks its section 2 superseded by section 2b: **read section 2b, not section 2, for basin depths and hit rates**; section 2's *solver* conclusions above are unaffected.

**Seeding on p3i2/TIM3, properly powered** (4 Sobol seeds x 2 arms x 64 = 512 starts): hits within 1.0 of 80.286 **A 9/256 (3.5%) vs B 30/256 (11.7%), 3.3x, Fisher p=6.7e-4**; MIN certification **A 125/256 vs B 182/256, p=3.9e-7** (the most stable effect measured); median -logL per seed A 2943/144/2321/2349 (erratic) vs B 108/142/104/141 (stable). Equal wall clock, 500 vs 518 s per 64 starts. All 8 runs found 80.286. **The earlier "1 vs 7, 7x" reading was A's unlucky seed - 3.3x is the number; the "2943 vs 108, 27x" median figure was A's worst seed.**

**Seeding on TIM1** (128 starts/arm): f6/md A 1 hit vs B **21**; p1o4/md 3 vs **21**; p1o4/orig 1 vs **18**, all Fisher p<0.001. So **6-21x on TIM1**, larger than p3i2's 3.3x, and overwhelmingly *coverage*: on p1o4 the MIN counts were identical (108/108, 102/102). An earlier claim that "on TIM1 the benefit is quality not coverage" came from the crystal template on f6 alone and is **wrong**.

**Arm comparison** (Ipopt L-BFGS, 128 solves/run): B vs A **B wins 6-21x, p<0.001**; B vs B5 49/384 vs 40/384, p=0.367, **no difference**; B vs C 49/384 vs 32/384, p=0.060, **no benefit, maybe worse**; B vs C2 49/384 vs 41/360, p=0.576, **no difference**. **Use arm B.** C and C2 cost ~63% more per solve (12.7 vs 7.8 s) and buy nothing; B5's better stratification (24.9 vs 31.2 empty cells at N=128) does not convert into hits. **Caveat constraining every comparison:** arm B's own seed-to-seed spread on one dataset is 12-21 hits (+/-35%), so **no single-seed comparison of two arms is trustworthy** - an earlier C-vs-B p=0.031 evaporated to p=0.060 with three seeds.

**Why arm C fails (measured):** at 128 starts the conditional seeds spanned sig 2.50-5.00, d2 1.50-13.50, rho2 0.35-0.75 - *both* bounds of all three windows - against sig 2.83-4.56 at only 16 starts. More poses means more *wrong* poses, and a membrane fitted to a wrong pose is worse than a membrane fitted to no pose at all.

**Seed accuracy does not matter - overturned once, then restored. Current reading:** section 2b.2 concluded "dimension reduction, not seed accuracy" from ONE Sobol seed (2/22, 2/21, 3/21). Pooling 4 seeds appeared to overturn it (19.3/6.0/9.5% for candidates at unit-cube distance 0.150/0.196/0.520, chi2 p=0.018) - **that was a confound**: candidates were assigned by `i % 3` and Sobol points are structured, so the groups were unbalanced in pose (mean start theta 0.441/0.510/0.552, Kruskal-Wallis p=0.036, against theta_u*=0.237). The clean paired test (all 64 poses x all 3 candidates, 192 starts, arm `P_paired`) gives 8/64, 8/64, 5/64, **chi2 p=0.618**, Friedman p=0.086. **So seed accuracy does NOT predict success and the original conclusion stands** - do not make the homometry seed precise, spend the freed budget on pose coverage. Two corollaries: the factorial design buys nothing per start (21/192 = 0.109 hits/start, identical to round-robin B); but a hit is **not** a property of the pose either - of 16 poses with at least one hit, 11 hit with exactly one candidate, 5 with two, **0 with all three**, so the (pose, membrane) *pair* determines the basin. Consistent with the split-norm result that misses are wrong in both blocks (arm A misses, n=63: membrane ||du|| median 0.411, great-circle 75.4 deg, per-coordinate mean |du| pose 0.214 vs membrane 0.140; 87% of misses have a membrane block farther from u* than the nearest candidate).

**Pose landscape** (52 pooled competitive MIN, p3i2, seed 2024): two theta families of 26 each - theta 34.3-45.7 deg (best 80.29) and theta 108.3-153.6 deg (best 94.20) - separated by 66-122 deg great-circle, phi spanning 105-258 deg in the upper band. Orientation-only clustering decays: 8 clusters at gc<=5-10 deg, 7 at 20, 5 at 30, 3 at 45-60 - a band, not isolated points. This independently **reproduces** the up/down pole-band + flat-phi structure the `xrr-homometric-pipeline` skill records for TIM3 from three other forward models: different code, different Euler convention, same structure.

**Reproducibility is a function of depth** (identity `gc<=10 deg & dprotins<=1 A & dcov<=1`, cross-checked against `||du||<=0.10` and a coarser physical rule - all agree): -logL < 90 -> 3 minima, mean recovery 6.3/8; 90-110 -> 7 minima, 4.7/8; 110-200 -> 10 minima, **3.0/8, none found by all 8 runs**. So **apply a depth cut** (recovery >= 5/8, or `Delta(-logL) < 15`) before publishing a ranked list, or its *length* is a property of the Sobol seed, not of the data. The discarded tail is harmless: at `Delta > 30`, `exp(l)` alone suppresses it by ~1e-13 in the Laplace weights.

**Template result, decided by the seeding work:** **MD beats crystal by 26-31 in -logL** on every properly searched dataset (f6 31.0, p1o4 26.0, p1d2 25.9). The crystal template's f6 best has `d2=1.695 A, rho2=1.126 e/A^3` - a slab 3.4x denser than bulk water, *not* railed against its bound, i.e. a genuine likelihood preference of a wrong template contorting the box model. Corollary: **template comparisons need an adequate multistart per template** - MD's basin is found 1-3 times in 128 unseeded starts. The one contrary reading (crystal beating MD on f2 by 15.5, from `profile_sig.py`'s 24-start warm-up) should be **discounted**.

**Other measured facts.** Gauss-Newton `J^T J` is within 0.2% of the exact Hessian and PSD by construction where exact `-H` had a negative eigenvalue, so use `J^T J` for Laplace weights. `cond(J^T J)` ~ 1e13 raw, ~1e11 normalised. `sig` **is** inferable (profile likelihood 3.50-4.07 A over six dataset/template combinations, statistical error 0.02-0.11 A, sharply parabolic), so pinning it at 3.40 is a misspecification. Sobol beat uniform at 64 starts each (best -logL 83.99 vs 94.17) for comparable evaluations. `p1d2_tim1`'s published XLayersPDB pose (theta 129.5, protins collapsing to 4.18 A) is a **seeding artifact** - under a proper multistart both templates pull it into the up band and the flipped solution is not even a local minimum (a saddle, 179 worse). The p1d series is anomalous: distinct `qoff` (~2.2e-4 vs the f-series' ~3e-5), highest coverage (0.37-0.40 vs 0.25-0.27), and uniquely the protein fails to break the membrane homometry there (chi2 ratio 1.00-1.19x vs 1.4-3.0x); its experimental conditions are still unknown.

**Costs, p3i2/TIM3 (78 q points, 1815 atoms, 10 free params):** `y()`/`sim` 4.3 ms; `grad` 29.5 ms; `jac` (10 cols) 37.6 ms; `hess` **288 ms** warm (was 605 ms before the `int_pts` fix; peak RSS 10.7 -> 1.19 GB, compile 50 -> 32 s) plus ~32 s on the first call per process; Julia session ~7 s, ~8.5 s to first useful evaluation. Warm `sim` 1.7 ms without protein / 3.5 ms with. Coverage bookkeeping overhead ~0.2 ms/call.

### 4.8 Parameter transfer to/from XLayersPDB (in scope, and important)

* **theta/phi DO transfer.** Recorded once as "do not transfer"; that was **wrong**, corrected 2026-09-02. XPR's hand-written matrix in `rotate_protein!` (`julia/XPR/src/XPR.jl:406-409`) - rows `[cosP,-sinP,0]`, `[cosT sinP, cosT cosP, -sinT]`, `[sinT sinP, sinT cosP, cosT]` - is bit-identical to scipy `R.from_euler('zx',[phi,theta],degrees=True)`, which is exactly what `XLayersPDB.ED_profrot` uses; `R_xpr @ v == rot.apply(v)` to 4e-16. `XPR.jl:1185-1186` converts deg->rad internally, so the *parameter* is in degrees on both sides. Verified at 4 angle pairs and empirically: XPR's p1d2 pose (34.7, 115.2) lands inside the XLayersPDB cluster for the same film series (34.3-35.7, 104-115). **Safe to seed XLayersPDB from XPR angles and to feed them to the NGL viewer.** **Both source docstrings that asserted the old wrong claim are now fixed (2026-09-06):** `Functions/pdb_geometry.py:xpr_rotation_matrix` and the Euler-convention bullet in `Functions/XRR/XLayersPDB_XPR.py` both state the equality and cite this section.
* **`cov` transfers, but only after rescaling** - it is not a raw copy: XPR's `cov` is the raw `n_s*A` (~26) where XLayersPDB carries a covered fraction (~0.37). The conversion is **asymmetric between volume modes** (`validate_xpr.py:133`): `vol_calc='approx'` (what `run_multistart.py` uses) wants `cov = frac*cov_scale(theta,phi)`, `vol_calc='exact'` wants `cov = frac` directly. `cov_scale` is 63-90 for `Tim4-orig-cen.pdb`. With that rescaling applied, XPR's free-`cov` optimum at an archived XLayersPDB pose lands **within 1%** of the archived value on all five TIM4 datasets tested. Passing the *scaled* value to `'exact'` inflates chi2 ~2x and looks convincingly like a model disagreement; reading XPR's `cov` as a fraction gives chi2 15,000-78,000 against archived 321-729. Both mistakes were made and caught.
* **`protins` NO LONGER transfers - `protins_top` does.** This bullet used to say
  `protins` transfers directly, verified 2026-09-04 (`xpr-seeding-study/TIM4_STAGE0_INVENTORY.md`
  §6b): profiling `protins` in XPR with theta/phi/boxes/sig pinned to an archived
  XLayersPDB fit and `cov` matched by `frac*cov_scale` put the XPR optimum at
  **-0.01 A** (`Tim4_F2`, archived 7.010) and **-0.17 A** (`p4m2_tim4`, archived
  6.174). **That measurement stands and its conclusion still holds - about the OLD
  parameter.** The mechanism it identified was that XPR's z reference
  `z_top = protins - sum(d) - maxh` agreed with XLayersPDB's
  `ztop_grid = -sum(d[:-1]) + protins`: both anchor the TOP of the protein.

  The 2026-09-08 upstream revision moved XPR's anchor to the mass centroid (3.9), so
  the transferable quantity is now `protins_top = protins + maxh(theta, phi)`, and
  raw `protins` differs from XLayersPDB's by `maxh` - **27.08 A on the 1697-atom test
  protein at (32.92, 112.59)**, i.e. larger than the film. Use
  `XLayersPDB_XPR.protins_top()` and `.protins_from_top()`; both broadcast, so an
  archived chain or a whole seeding-study endpoint table converts in one pass. The
  old measurement is worth re-running once in the new coordinate, as claim 11 now
  says, but the geometry says it must come out the same.

  **MIGRATION, and it is not optional.** Every stored XPR `protins` predating
  2026-09-08 is in the old convention: saved `.par`/`.xpar` files, the
  `xpr-seeding-study` `endpoints_*.npz` tables, the TIM1/TIM3/TIM4 campaign records,
  and anything seeded from an XLayersPDB fit. Reading one back into the new model
  silently misplaces the protein by `maxh`, which is tens of Angstroms - it will not
  look like a subtle error, but it will look like a bad starting guess rather than a
  unit mismatch. Also note `protins`' default lower bound changed from 0 to
  `-PROTINS_MAX`, and **a saved `.par` restores its own bounds**, so an old file
  reloaded into the new model keeps `min=0` and will rail against it.

  Combined with the theta/phi result above: **theta, phi and cov transfer; the
  insertion transfers only through `protins_top`**; absolute `logL` does not.
* **`sig` does not map structurally**: one scalar here vs a per-layer mpar column there.

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `lmfit/minimizer.py` | `_PDB_FUNCNAMES` (`:94-95`) + comment | append `'XLayersPDB_XPR'` | Yes, one tuple entry |
| `Fit_Routines.py` | `Fit.__init__` (`:53-54`) | `self.use_jacobian=True` opt-out flag | Yes |
| `Fit_Routines.py` | new `can_jac` (`:315`), `residual_jac` (`:338`), `_canonicalize_result` (`:466`) | new methods | Yes |
| `Fit_Routines.py` | `perform_fit` leastsq branch (`:442`) | pass `Dfun=self.residual_jac` when `can_jac` | Additive, but inside an existing line; falls back to `{}` |
| `Fit_Routines.py` | `perform_fit` (`:458`, `:463`) | call `_canonicalize_result()` after minimize | Additive; no-op without `canonical_params` |
| `Fit_Routines.py` | `set_emcee_minimizer` (`:491`) | add `'XLayersPDB_XPR'` to the relative-stderr tuple | Yes |
| `Fit_Routines.py` | `evaluate` / `_install_func_x` (`~:216`) | split out so `residual_jac` reproduces the same x | **Refactor of existing code - inspect on merge** |
| `xmodfit.py` | - | none for XPR | n/a |
| `julia/bakeoff/run_armD.py` | *consumes* `build_orientation_grid`, `_detect_orientation_basins`, `seed_thetaphi_from_costmap` | import only | Yes, but couples the harness to category 03's private API |

Pre-existing upstream bug fixed in the same working-tree diff (belongs to category 03/04 but overlaps): `lmfit/minimizer.py` called `self.backend.reset(...)` unconditionally, so **any** emcee run with `backend=None` (user cancels the "Start MCMC file as" dialog) died with `AttributeError`.

Merge risk is low on the additive side and concentrated in two places: the `evaluate`/`_install_func_x` split, and the fact that none of it is committed.

## 6. Known limitations, gotchas and TODOs

**Not done / deferred**

1. **No global-search fit method in the GUI.** All of stage 2 is offline. The design (phases 0, 0.5, 1, 2, 3, 4 plus GUI wiring) is `STAGE2_SEEDING_PLAN.md` section 3, which asks for it to be **capability-gated, not name-gated** (extend the `can_jac` pattern with a Hessian probe) so `XLayersPDB`/`_dev_smooth` also benefit.
2. **The phase-0 Julia fix is STILL NOT DONE.** `XPR.jl:262` is `any(p_hi .<= p_lo) && error(...)`, so a pinned parameter cannot be `lo == hi`. Every harness therefore carries the `hi = lo + 1e-9` hack (`run_seed.py:27-28`: `HI[6]=1.0+1e-9`, `HI[7]=1e-9` for yscale/bkg). Relax to `any(p_hi .< p_lo)` and drop the hack.
3. **The acceptance test against the statistics-department reference is unwritten.** Section 4 of the plan specifies `validate_xpr_seed.py` against `C:\Users\kerr\Documents\Statistics Dept MCMC Collaboration\xpr_test\xpr_test\` (`1pt.py`, `1pt.jl`, `cluster_run.jl`, `XPRf/`, `Data/`). His `output_logs/` was empty, so the reference numbers must be regenerated. **Read `XPRf/src/XPRf.jl` first** - it hardcodes `sig`, `vol_calc`, `rho_top`, `rho_bottom` and presumably `yscale=1`/`bkg=0` (his `load_data` takes only 6 arguments). Do not guess them. Note `XPR-main` itself contains **no** optimiser; the student's Ipopt setup is the real reference, not the "last year code_hengjia" notebooks.
4. **Parallelism is unimplemented.** The plan wants Julia `Distributed` mirroring `cluster_run.jl` (`addprocs`, `@everywhere`, `remotecall` the data, **precompile on each worker**, then `pmap`), because a Python-side pool pays ~20 s x N of juliacall import + activation + `load_data`. `Distributed` inside a PyQt process is **untested**: verify `addprocs` does not fight Qt signal handling and worker stdout does not deadlock the GUI.
5. **No physicality filter in phase 4.** `rho2 = 1.126 e/A^3` should be flagged automatically. `seed_utils.screen_membrane_seeds` and the homometry skill's `physical_members` have usable criteria.
6. **Only 2 protein systems.** The planned next study was 3 directories from 3 systems, one at a time, which is what would make N=64 a general recommendation. Larger N (256/512) was never tested because everything saturates by 48.

**Traps that cost real time**

7. **juliacall + IPython kills the Qt process** silently, exit status 0 (see 3.3).
8. **Overrunning the slab budget segfaults**, it does not raise. `check_headroom` must run before every Julia call; `y()` does it via `_prepare` (`XLayersPDB_XPR.py:523`).
9. **Grid-alignment kinks.** The `spacez = 0.5` lattice gives R a small slope discontinuity when `sum(d)` or `10*sig` is a multiple of 0.5. AD gives the correct one-sided derivative; central differences straddle and disagree by ~1e-2. The round defaults (`d=[11.0,9.0]`, `sig=3.4`) sit **exactly** on them - use generic values for any FD cross-check, and expect this to matter for gradient-based MCMC in stage 3.
10. **`xrr.py` has no `--qmin` flag** and `precondition` extrapolates below `q_min`. Truncate above the critical edge (q >= 0.026, ideally 0.033 on raw R) or every candidate rails `qoff` at its bound.
11. **Prefer a lipid-only scan for the homometry pass, and read N=2.** On protein-loaded data a free 2-box model absorbs the protein by stretching the span from ~22 to 33-37 A, chi2/N 260-1770, `qoff` railed (lipid-only: chi2/N 5.7-18.3). An earlier TIM3 finding that a protein-loaded N=2 read gives essentially the same monolayer (the protein excess being too weak to move the two strong interfaces) is not contradicted, but the TIM1 statement is the current guidance. N=3 on a protein-loaded curve does **not** reveal the protein - its best candidate is a ~15 A near-vacuum box, the translation degeneracy soaking up the extra box.
12. **Counting-rule discipline.** Two counting mistakes were made and corrected, both the same bug: **a composite AND merge rule where the clause being swept is not the binding one**, so the count looks reassuringly insensitive to its tolerance. (a) "38/42 distinct endpoints" was a ~3x over-count (it included SADDLEs and -logL~3000 junk, and the -logL clause bound); the defensible figure is MIN-classified and -logL<200, merged at 0.05 -> **A 13, B 26**. (b) "Pose clusters insensitive to gc 5-30 deg" was wrong - the protins/cov clause bound, and nearest-neighbour gc between retained representatives was median 2.4-3.9 deg, *below* the smallest tolerance swept. Rules: never put -logL in an identity rule (it is a consequence of position); always sweep the tolerance and report which clause binds; compare orientations by **great-circle angle** ((theta,phi) -> unit vector -> `arccos(dot)`), never `|dtheta|+|dphi|` (wrong at the poles and the phi wrap); filter to MIN-classified and competitive before counting anything.
13. **`max_cpu=60` confounds arm A**: its junk starts may be timing out rather than converging, inflating its f-eval count. Arguably that *is* the cost seeding avoids, but name it when quoting arm A timings.
14. **`choices` values must not be Python names.** `'approx'`/`'exact'`/`'protvol'`/`'xpr'` survive XModFit's `eval()` of Fixed-Parameter cells; the code still defends with `getattr(self.vol_calc, '__name__', None) or str(...)` (`XLayersPDB_XPR.py:502`). See the `choices-eval-gotcha` memory.
15. **Broken environments seen in practice.** `cmcars` has broken stdlib SSL (OpenSSL 3.5.7 rejects a malformed cert in the Windows store; `SSL_CERT_FILE` does not help because `load_default_certs` enumerates the store first) - handled by the offline retry, but the consequence is that juliapkg **cannot install or upgrade** anything there, so do Julia-side updates from `cmc-py39`. `cmcars` also had a broken `tomli` (dist-info present, package dir gone); fixed with `pip install --force-reinstall --no-deps tomli`.

**DEAD ENDS - do not retry**

16. **Escape along the most-negative eigenvector, then RE-RUN L-BFGS**: 2/16 MIN, no better than plain L-BFGS - the nudge lands in another saddle because the solver is still blind to negative curvature.
17. **The `int_pts` allocation fix as a runtime lever**: real in isolation (Hessian 605 -> 288 ms, RSS 10.7 -> 1.19 GB) but moved **neither** Newton method end to end (<2%).
18. **Arm C and arm C2** - no benefit, ~63% more cost per solve (mechanism in 4.7).
19. **Arm B5 (native 5-D Sobol)** - the stratification defect is real and does not convert into hits. Sobol construction is not the lever.
20. **A factorial pose x candidate design** buys no coverage efficiency: 3x candidates is just 3x starts.
21. **`costmode='physins'`** in arm D: equalising the model's 1%-threshold z-extent across poses drives `protins` to its bounds - the threshold extent is smearing-tail dominated, so anchoring it to a Patterson peak *difference* is dimensionally inconsistent. On f6/md: 1 basin instead of 5, polish reached -logL 481 instead of 194.9. Kept in the source only for the record (`run_armD.py:137-148`).
22. **Matching the MODEL's outermost Patterson peak to the data's** - discontinuous: it vanishes when the protein does not protrude far enough (absent at theta=90 deg, collapses onto the film peak at theta=30 once protins>=13), giving apparent slopes of -0.31 to -2.28 A/A (`phys_seed.py:22-26`).
23. **A true per-pose homometry enumeration** (not implemented, probably not worth it): the protein does not subtract cleanly, it *multiplies* the membrane by `[1 - cov*(1-layarea(z))]`, reaching ~0.27 in the protein core, so "subtract the protein and run Patterson" needs the enumeration generalised to modulated boxes.
24. **Using the analytic Jacobian below 4 varying parameters, or in the emcee seed scan** - measured losses (3.6).

**Stale documentation inside the source (fix on merge)**

25. ~~`XLayersPDB_XPR.py:54-56` and `pdb_geometry.py:90-92` still say theta/phi do not transfer.~~ **FIXED 2026-09-06** -- both now state the equality (they do transfer, 4.8).
26. `XPR.jl`'s header says "Divergence 1 of 2" / "2 of 2" then "Divergence 3"; `julia/README.md:93` says "Two changes" then lists three; `STAGE2_SEEDING_PLAN.md:26` says "the 4 divergences". The actual count is 5 source patches plus the Ipopt dependency.
27. `julia/bakeoff/README.md` covers 6 of 15 scripts and neither TIM1-era naming scheme. `run_multistart.py:10` references `make_tim1_config.py`, which **does not exist anywhere in the repo**.
28. `STAGE2_SEEDING_PLAN.md:38` still describes the harness as living in `%TEMP%/bake`; it was repointed to the in-repo `julia/bakeoff/solvers.jl`, which is authoritative.

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| 1 | `validate_xpr.py` passes | memory; plan `:20` | run it in `cmc-py39`; confirm `../XPR-main/test_data` still exists and note the out-of-repo coupling |
| 2 | The reference log-likelihoods are upstream `demo.ipynb` outputs, reproduced to 1e-4 | memory | **DONE 2026-09-08** against the new upstream. 2-box `-1214.1600`, 3-box `-600.6398`, 4-box `-62408.5320` all reproduce to every printed digit. The protein cell reproduces to 8e-4 *after* correcting the CAL-mass centroid offset of 0.0345 A (4.3) - without that correction it is 60 in logL out, which is the whole discrepancy and nothing else. Gate tolerance is 1e-3, not 1e-4 |
| 3 | `XPR_grad` is EXACT for `XLayersPDB_XPR` because nothing derived is folded into `cov` | `xpr_backend.py:461-467` | complex-step or high-order FD on `logL` over all 10 free params at a **generic (non-grid-aligned)** point; expect ~1e-9 |
| 4 | `projected_area` matches XPR's Green's-theorem union area to 0.07-0.31% at `fslice=0.5` | memory | re-run `validate_xpr.py:106-129` and record the per-orientation numbers |
| 5 | `cov = frac*cov_scale` makes `'approx'` reproduce `'exact'` to 0.03-0.25% rms (2.8% raw) | memory; `validate_xpr.py:135-158` | **CONFIRMED (approximately) 2026-09-04 on mTIM4/Tim4-orig-cen**, `xpr-seeding-study/TIM4_STAGE0_INVENTORY.md` §6b. rms dR/R = 0.173% (`p4m2_tim4`) and 0.358% (`Tim4_F2`) - same order, but F2 sits *above* the quoted 0.03-0.25% band, so treat that band as test-protein-specific rather than a bound. **Important corollary the claim does not state:** on these data (median error 0.60-0.71%, chi2_red ~10) that sub-percent R difference is a **19.6-41.5% chi2 difference**. So `'approx'` and `'exact'` are interchangeable for *parameters* and NOT for absolute `logL`. Also note the mapping is asymmetric and easy to misuse: `'approx'` takes `cov = frac*cov_scale`, `'exact'` takes `cov = frac` directly (`validate_xpr.py:133`); passing the scaled value to `'exact'` inflates chi2 ~2x and looks like a model disagreement |
| 6 | `alpha_approx` integrates to exactly `N/sum(pi r^2)` while the truth is `V_union/A_proj` | memory | integrate `alpha` from `XPR_edp` with all `rho` zeroed, at two orientations |
| 7 | `electron_scale = N/V_union` is rotation invariant to 0.13% ("grid noise") | `XLayersPDB_XPR.py:301-305` | it is exactly invariant analytically, so 0.13% is grid error - confirm it scales with `fslice` |
| 8 | Gauss-Newton `J^T J` within 0.2% of exact `-H`, and PSD where `-H` is not | plan `:330-336` | `hess_report.py` + `xb.jac` at a saved best endpoint; report `norm(H - J^T J)/norm(H)` and both eigenspectra |
| 9 | `cond(J^T J)` ~ 1e13 raw / ~1e11 normalised; p3i2 has one near-flat direction | memory; plan phase 2 | `hess_report.py CONFIG.json md`; also name the softest eigenvector's composition |
| 10 | theta/phi bit-identical to scipy `'zx'`, `R_xpr @ v == rot.apply(v)` to 4e-16 | memory, corrected 2026-09-02 | **DONE 2026-09-06.** Re-verified over 475 pairs (theta 0-180 x phi 0-359): `4.4e-16` on the matrix vs `from_euler('zx',[phi,theta],degrees=True)`, `2.1e-14` on `rot.apply(v)` vs `M @ v` at 20 A, `3.3e-16` vs `R_x(theta) @ R_z(phi)`. Both stale docstrings fixed (was 6.25) |
| 11 | `protins` transfers between XPR and XLayersPDB | memory; resolved 2026-09-04 | **SUPERSEDED 2026-09-08 by the upstream redefinition.** The 2026-09-04 measurement was correct and still is, but it was about the OLD top-anchored `protins`; upstream now anchors on the mass centroid, so the transferable quantity is `protins_top = protins + maxh` and raw `protins` differs by `maxh` (27.08 A on the test protein). See 4.8 for the migration. **Re-run the TIM4 profile in the new coordinate** to confirm the -0.01/-0.17 A agreement survives the reparameterisation; the geometry says it must, and `validate_xpr.py` already shows the *test* protein reproduces `-466.4557` exactly under the conversion, but the archived-fit comparison is the one that was quoted and it should be redone rather than assumed |
| 12 | N=48 -> 100%, recommend 64, generalises | study sec 5 | it explicitly does **not** yet: 2 systems only. Run the planned third/fourth before quoting N=64 as general |
| 13 | Arm B "wins 6-21x" pooled over seeds | study sec 3 | Sobol points within a run are dependent, so pooling 3 seeds x 128 as 384 independent trials is questionable. Recompute with the **run** as the unit of analysis |
| 14 | "MD beats crystal by 26-31 in -logL" | study sec 6 | confirm both templates have effectively the same atom count (1693 vs 1697) and identical `electron_scale` treatment before comparing likelihoods |
| 15 | The 11-20 replicate tolerance is a property of the data | study sec 4 | reproduce `tolerance_scale.py` on the p3/TIM3 series; a different scale there makes the claim TIM1-specific |
| 16 | **Arm D exists but has no recorded verdict** - 11 `armD-fast-*` npz files dated 2026-09-03 plus 2 `armD-full`, mentioned in no memory file or design doc | `run_armD.py` + the npz timestamps | run `compare_arms.py 'endpoints_*armD*'` and write the verdict down. **The single biggest documentation gap in this category** |
| 17 | `phys_seed`'s `DEPTH_OFFSET = 19.32 A, sd 0.36` over 11 TIM1 datasets, and the depth rule prunes 38 of 61 cells for f6 | `phys_seed.py:98-118` | recompute from the saved endpoints; confirm the sd and that pruning reproduces the up/down bands. The constant is explicitly system- and threshold-specific |
| 18 | `cov_from_patterson` calibration | `phys_seed.py:3-11` says r=0.969, residual +/- 1.06; `run_multistart.py:128` says LOO sd 1.23, max 2.09 | **two different numbers for the same thing** - reconcile, and re-fit from the 11 TIM1 endpoints |
| 19 | `sig` profile is sharply parabolic (3.50-4.07 A, error 0.02-0.11 A) | study sec 6 | those numbers came from `profile_sig.py`'s 24-start warm-up, whose template verdict was itself discounted. Re-run the profile from a properly searched optimum |
| 20 | `fit_sig`/`fit_yscl`/`fit_bkg` are "unconditionally True" and harmless | `xpr_backend.py:350-352` | confirm `XPR.jl` substitutes its own defaults only when False and that nothing relies on the False path |
| 21 | `E` is inert (Parratt in q, `beta==0`) | `XLayersPDB_XPR.py:41-51` | grep `XPR.jl` for any write to the absorption array; evaluate `y()` at E=8 and E=12 and confirm bit-identical R |
| 22 | `sim_edp` is ~1.9x faster than `sim`+`edp` | `xpr_backend.py:509-516` | time all three warm on p3i2 |
| 23 | "261 model calls -> 41, same optimum to 1.3e-5" | `julia/README.md:126-130` | `validate_xpr.py:352` prints a with/without comparison; re-run and record |
| 24 | The 92 `.npz` files are internally consistent (same `LO`/`HI`/`FREE` per dataset) | this document | assert `LO`/`HI` match across arms for a given tag before any pooled statistic; also check `compare_arms.py`'s `FNAME` regex (`:32`) against the `BACKUP`/`PILOT16`/`B5only` names |
| 25 | `p1d2_tim1`'s published pose is a seeding artifact and the flipped solution is a saddle 179 worse | study sec 6 | classify that pose directly with `xb.hess`. A strong claim about published work - verify before repeating it |
| 26 | The `sum(d)` grid-alignment kink is a property of `sort(unique(...))` merging a box interface with the grid point it lands on, and AD returns the correct right-hand derivative | 3.9, measured 2026-09-08 | the *existence* of the kink is measured (one-sided slopes 4.3 apart, stable as `h -> 0`, at `d = [11.0, 9.0]`); the *mechanism* is inferred from reading `smear_xy_interface!`. Confirm by instrumenting `length(rall)` either side of the coincidence, and check whether `unique` on `ForwardDiff.Dual` compares by value (which would drop the interface point and its partials) or by object identity (which would not). If it is the former, upstream has a measure-zero AD bug worth reporting |
| 27 | `protins_top`'s linearised error is exact, and its 31% disagreement with the full-covariance Monte Carlo on the validator's test fit is nonlinearity of `maxh` rather than a bug | 3.9, `validate_xpr.py` | **partly DONE 2026-09-08**: the small-step Monte Carlo agrees to 1.0002, which is the correctness half. The adequacy half is only shown on one fit at `sigma(phi) = 8.2 deg`; on a real TIM3/TIM4 fit with a better-determined orientation the two should converge, and if they do not, `protins_top()` over the emcee chain is the number to quote. Record the ratio the first time this is run on real data |
| 28 | Elias's `XPRsim._import_pdb` is internally inconsistent for the `CAL` ion (name-override for electrons and radius, `atom.element` for mass) | 4.3, measured 2026-09-08 | measured and quantified here (0.0345 A of centroid shift on `protein.pdb`, worth 60 in logL at the demo point). **Send it upstream** - it is a one-line fix in `XPRsim.py`, and until it lands, any protein `protins` compared between XPRsim and XModFit carries that offset |

## 8. Consolidation candidates

| Item | Recommendation | Risk |
|---|---|---|
| **Everything untracked** | **Commit immediately**, before anything else. Consider making `julia/XPR/` a submodule or adding an explicit vendor note instead of a copy | Low; not committing is the risk |
| ~80 lines of identical setup duplicated across `run.py`, `run_seed.py`, `run_multistart.py`, `run_armC.py`, `run_armC2.py`, `run_armD.py` (LO/HI assembly, `emb`, `classify`, load, npz write) | **Freeze** `run.py`/`run_seed.py` as historical (their outputs are cited verbatim) and **merge** the common core out of the rest into `seed_utils.py`, keeping arms as strategy functions | Medium - a refactor mistake makes the published numbers unreproducible. Freeze first |
| Hardcoded `REPO = r'C:\Users\kerr\...'` in `run.py:7`, `run_seed.py:14` | **Fix** to the `HERE`/`REPO` derivation the newer drivers use (`run_multistart.py:27-28`) | Low |
| External data root `C:\Users\kerr\Documents\XLayersProtTest\...` in `run.py:12`, `run_seed.py:20` and every `CONFIG.json` `data` field | **Document** it as a required env var. Nothing in `julia/bakeoff/` runs on a fresh clone | Low to fix, high impact if left |
| **No `CONFIG.json` and no `make_tim1_config.py` are committed** | **Add** one worked example config plus the generator, or every driver except `run.py`/`run_seed.py` is unrunnable | Low |
| `endpoints_f6_tim1_1250_md_s2024.BACKUP.npz`, `endpoints_p1d2_tim1_{md,orig}_s2024.PILOT16.npz`, `endpoints_f6_tim1_1250_{md,orig}_B5only_s2024.npz` | **Keep but rename** into the standard scheme, or move to `archive/`. `BACKUP` exists because of the clobber bug (4.5) and its provenance is only in a memory file | Low - but do not silently delete, and check whether the odd names parse against `compare_arms.py`'s regex either way |
| `analyse.py`, `distinct.py`, `crossseed.py`, `catalogue.py` all hardwire `ARMS = ['A_sobol10','B_homoseed']` and read only the p3i2 files | **Merge** into `compare_arms.py`, which already generalises, keeping thin wrappers if their exact printed output is cited | Medium - the plan's section 2b tables came from these, so retire them only once `compare_arms.py` reproduces those numbers |
| `julia/bakeoff/README.md` | **Rewrite** to cover all 15 scripts and both npz schemes, or point at this document | Low |
| `STAGE2_SEEDING_PLAN.md` section 2 vs 2b | **Keep both** - the superseded numbers are the record of how the conclusion changed - but add a short "current answers" table at the very top; the banner is doing a lot of work. **Do not delete section 2**: its solver conclusions are the only bake-off on record | Low |
| `SEEDING_STUDY_TIM1.md` vs plan section 2b | Overlapping, not redundant: 2b is TIM3/p3i2, the study is TIM1 and supersedes the plan's *phase-1 design*. **Keep both**, add a one-line cross-reference to each | Low |
| `Functions/pdb_geometry.py` vs `XLayersPDB`'s `Vol_gridcalc`/`Area_gridcalc` | **Keep duplicated** - the reason is structural and documented (`pdb_geometry.py:14-21`). If ever unified, the shared code must live in `Functions/` root | Medium if merged carelessly: it recreates a circular import via `pdb_cache` |
| `Ipopt` in `julia/XPR/Project.toml` | **Move** to a separate research environment if the shipped model class never needs a solver - only `julia/bakeoff/*.jl` uses it | Low, but it enlarges every user's Julia install |
| `solvers.jl` vs `solvers_profile.jl` | `ipopt_bfgs` is `ipopt_bfgs_bounded` with `lo=0, hi=1`. **Merge** once no long run is in flight; the split is documented as exactly that temporary measure (`solvers_profile.jl:4-6`) | Low |
