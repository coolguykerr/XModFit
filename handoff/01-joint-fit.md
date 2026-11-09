<!-- handoff/01-joint-fit.md -->

# Joint / global multi-dataset fitting — XModJointFit change record

**Category:** the joint (global) fitting engine, its GUI panel, and the `.jpar` file format
**Status:** Working-but-unvalidated-live (engine + GUI exercised by headless harnesses; the
correlated-error/footprint branch is only exercised on synthetic or offline data)
**Owner chat(s):** `merge-with-upstream-develop.md`, `column-fit-limits.md`,
`multimodal-seeding-plan.md`; skills `xmodjointfit` (engine) and `bilayer-jointfit-analysis`
(the campaign that drove the covariance work)
**Last touched:** 2026-08-27 / 2026-08-28 (working tree); last commit touching it `babe1aa`

## 1. Why this exists

Stock XModFit fits one dataset at a time. Multi-dataset selection just loops the same model
over each file independently, so there is no way to state "these three measurements share
one physical structure". Two distinct scientific needs drove this fork:

* **Shared structure across measurements (Axis 1).** For `Functions/XRR/XLayersPDB.py` a
  protein's orientation (`theta`, `phi`) and insertion depth (`protins`) are properties of
  the molecule, not of the measurement, so several reflectivity curves (different
  coverages, contrasts or buffers) must be fitted with ONE orientation while each keeps its
  own nuisance parameters (`yscale`, `bkg`, `qoff`). The same applies to `XLayers`
  substrate parameters shared across all runs on one wafer.
* **Correlated errors across measurements (Axis 2).** A corrected specular-XRR bundle is
  stitched from several attenuator settings and footprint-corrected with a single beam. The
  scale-factor correction factors `c_k` and the beam FWHM are *shared, uncertain*
  quantities, so points in *different* datasets have correlated errors. Fitting the
  datasets independently with diagonal errors throws that information away and reports
  wrong error bars. The engine instead builds the full covariance over the concatenation
  and does generalized least squares, which yields correct GLS parameter uncertainties from
  lmfit's Jacobian-based `.covar` — no MCMC needed.

The concrete target configuration is the DOPC supported-bilayer campaign: ~17 corrected XRR
datasets (`El_Cat_Si_1`, `El_Cat_Si_2`, `MO_Qz`, `TGP_Qz`) fitted with `XLayers`-family
models under one shared scale-factor covariance `Sigma_logc` plus one shared beam-FWHM
variance, with substrate parameters shared per wafer and bilayer parameters shared per
replicate. The second target is joint `XLayersPDB` fits with shared `theta`/`phi`/`protins`.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `Joint_Fit_Routines.py` | 1114 | new (`c5f697d`) + uncommitted | the `JointFit` engine: namespacing, sharing, residual, covariance whitening, fit driver |
| `joint_covariance.py` | 372 | new (`c5f697d`) | pure-numpy sidecar readers + full covariance assembly + Cholesky |
| `joint_fit_spec.py` | 294 | new (`c5f697d`) + uncommitted | editable, name-keyed spec that `compile()`s into a fresh `JointFit` |
| `joint_fit_io.py` | 369 | new (`c5f697d`) + uncommitted | `.jpar` writer/parser, `.par` parser, model dump/apply |
| `joint_fit_panel.py` | 689 | new (`c5f697d`) + uncommitted | the "Joint Fit" panel (overlay under the Data list) |
| `joint_fit_orchestration.py` | 112 | new (`c5f697d`) | one-shot "build a JointFit from a selection" glue; effectively dead (see 8) |
| `xmodfit.py` | 7740 | modified | all host wiring: `doFit` routing, `run_joint_fit`, `.jpar` save/load, joint view/plot, model comparison |
| `validate_jointfit.py` | 287 | new (`c5f697d`) | headless engine tests: whitening identity, N=1 == stock `Fit`, 3-dataset GLS on real data |
| `validate_jointfit_gui.py` | 362 | new (`c5f697d`) + uncommitted | offscreen GUI regression (~55 checks) for the whole joint workflow |
| `validate_perdataset_fit.py` | 199 | new (`c5f697d`) | per-dataset fit memory + PDB cache (feeds `joint_model_for`) |
| `validate_joint_mask.py` | 236 | new-untracked | joint mask/covariance/footprint composition tests (owned by doc 08, joint-specific) |
| `validate_xlayerspdb_joint.py` | 260 | new-untracked | real joint `XLayersPDB` fit with shared `qoff` + old-`.par` load |

Bucket detail for `xmodfit.py`: `run_joint_fit`, `saveJointFit`, `loadJointFit`,
`loadParIntoJoint`, `show_joint_datasets`, `_update_plot_joint`, `_sync_joint_fixed_params`,
`create_jointFitDock`, `computeModelComparison` and the `joint_active` branch of `doFit`
exist in `HEAD` (commit `c5f697d`, patched by `babe1aa`). `run_joint_emcee`,
`clear_joint_view`, `_joint_masks`, `_parse_fitrange` and the `_mcmc_is_joint` routing at
the top of `doFit` are **uncommitted working-tree only**. Nothing joint-related exists in
`origin/develop`.

## 3. How it works

### 3.1 Two orthogonal axes, one branch point

`Joint_Fit_Routines.py:11-34` states the invariant that makes the design tractable:
`residual()` branches ONLY on "is a covariance attached?" (`self._use_cov`), never on "are
parameters shared?". Sharing is entirely a property of the parameter map. So shared
parameters and GLS whitening compose without any combinatorial code.

### 3.2 Axis 1 — parameter namespacing and sharing

Each registered dataset keeps its own model instance and its own local lmfit `Parameters`.
`build_master_params` (`Joint_Fit_Routines.py:293-368`) flattens them into one global
`master_params` plus `param_map: master_key -> [(obj_idx, local_key), ...]`:

* endpoints claimed by a share-group get ONE master parameter, whose
  value/vary/min/max/brute_step are inherited from the group's FIRST endpoint
  (`:339-346`); an endpoint in two groups raises (`:349-351`);
* every unclaimed local key becomes a namespaced master key via `_master_key`
  (`:237-252`): scalars -> `d{i}_qoff`, multipar -> `__d{i}_<mkey>_<pkey>_<num>`.

The multipar naming is load-bearing: `pkey` and `num` never contain `_`, so the downstream
`rsplit('_', 2)` in `loadParameters`/`apply_block` still recovers the parts from a
namespaced mkey. Subset-share group masters use the same trick with a `g{gid}_` prefix
(`_group_master_key`, `:370-390`). A group flagged `bare=True` keeps the *un-namespaced*
name (`theta` stays `theta`) specifically so the emcee angle gate that matches on
`var_names == 'theta'` keeps firing — that is the one place where Axis 1 leaks into the
MCMC stack (documented in `handoff/02-mcmc-stack.md`).

`push_params` (`:423-440`) writes each master value BOTH to the model's attribute
(`func.__dict__[lkey]`, which is what `XLayers.y()` reads directly) and to the local lmfit
parameter (which `update_parameters()` reads for multipar), and clears any local `expr` so
the master value is authoritative. Three convenience group builders exist —
`share_all` / `share_within` / `share_endpoints` (`:261-291`) — of which only `share_all`
is used in-tree; `share_endpoints` is the one that supports a physically-shared parameter
living at *different* local keys in different datasets (e.g. the oxide thickness is
`__Model_d_005` in a bilayer run but `__Model_d_001` in a bare-buffer run).

### 3.3 The global objective and residual concatenation

`evaluate()` (`:442-462`) evaluates every objective's model on its own `[imin, imax]`
window (and only the active points if the model opts in with `__masked_x__`).
`residual()` (`:511-583`) then has two branches:

* **no covariance** (`:546-571`): per dataset, slice, apply the mask, normalize by that
  dataset's own `np.max(data)`, apply the chosen fit scale, and concatenate. This is
  byte-identical to stock `Fit.residual` per curve — which is what makes the N=1 no-cov
  regression against stock `Fit` (`validate_jointfit.py` T2) meaningful. The mask is applied
  *before* `maxd` deliberately (an excluded hot pixel must not rescale the whole vector, and
  `log10` must never see an excluded non-positive point).
* **covariance / GLS** (`:515-544`): concatenate the *raw* `y_corr - fit` in absolute units,
  apply `_cov_mask`, then return `L^{-1} r` by triangular forward substitution
  (`scipy.linalg.solve_triangular`, O(N^2), with a `np.linalg.solve` fallback). Under
  `leastsq`/`least_squares` the minimized sum of squares is then exactly
  `(d-f)^T C^{-1} (d-f)`.

Optional Gaussian priors (`set_priors`, `:397-418`) append `(value - mu)/sigma` terms
*after* whitening (`:576-579`) — a prior constrains a parameter, not a datum, so it must not
enter `C`. Each prior adds 1 to `ndata`/dof, which is the right accounting for a soft
constraint. Intended use per the docstring: informing a low-leverage buried roughness from
an individual-fit value ± stderr instead of hard-fixing it.

### 3.4 Axis 2 — cross-dataset covariance and Cholesky whitening

`joint_covariance.build_full_covariance` (`joint_covariance.py:312-353`) assembles

    C_ij = delta_ij * sigma_meas_i^2 + y_corr_i*y_corr_j*Sigma_logc[k_i,k_j] + var_fwhm*v_i*v_j

The middle term is the scale-factor model, with `k_index` naming which attenuator segment
each point came from — off-diagonal blocks between *different* datasets are exactly what
makes this a joint fit rather than N single fits. The third term is the shared-beam
footprint: one beam FWHM was used to correct every dataset, so its variance is a rank-1
update `var_fwhm * outer(v, v)` over the whole concatenation. The footprint *amplitude*
error is deliberately absorbed by a free per-dataset `yscale` and is NOT in `C`.

`JointFit.add_dataset` (`:112-177`) turns Axis 2 on when any dataset carries a `cov` dict,
and takes `y_corr`/`sigma_meas` from the sidecar in place of the GUI's `y`/`yerr`.
`_register_shared_sigma` (`:214-224`) and `_register_shared_var_fwhm` (`:226-232`) *refuse*
datasets whose `Sigma_logc` or `var_fwhm` disagree — cross-dataset correlation is only
meaningful under one shared calibration and one beam.

`_build_chol` (`:467-506`) assembles `C` over the concatenated windows and caches
`L = chol(C)`; `cholesky_with_jitter` (`joint_covariance.py:356-373`) escalates diagonal
jitter and finally floors eigenvalues if `C` is not numerically PD. The mask composition
order is the riskiest thing in this file and is commented as such at `:474-479`: slice
`[imin,imax]` -> apply this objective's `imask` -> concatenate -> apply `_cov_mask`, folded
in symmetrically in `_build_chol` and `residual`, with a length guard at `:532-536` because
misalignment would silently produce a plausible wrong answer.

Sidecar formats (`joint_covariance.py:47-211`): `*_c_factors.txt` (needs the raw 4-column
`x, y, err, sf` file so `segment_index` can map each row to its `k`), a self-contained
`*_corrected_cov.npz`, and the multi-dataset `xrr_corrected_bundle` npz
(`load_bundle_npz`, `:133-197`) which is the only reader that returns `v` / `var_fwhm`.

### 3.5 Per-dataset fit ranges and column masks

`_set_ranges` (`:591-617`) gives each objective its own `[imin,imax]`. Without a `ranges`
dict every objective uses the scalar `(xmin,xmax)` from the Xrange box; with one (keyed by
dataset name, index fallback) each objective uses its own bounds, and a missing entry or a
`None` side falls back to the scalar. Index resolution goes through
`Fit_Routines._window_indices` (`Fit_Routines.py:8`, imported at
`Joint_Fit_Routines.py:43`) so the joint and single paths cannot drift; the old `0.99*xmin`
factor was removed because it moved a negative bound the wrong way.

`_set_ranges` also freezes `obj['imask']` = the full-length eligibility mask restricted to
the window. The mask is built by the HOST (`xmodfit.py:_joint_masks`, `1466-1481`) because
only the host owns the DataFrames and column names; the engine never learns about columns.
`add_dataset` asserts `len(mask) == len(x)` at registration (`:153-162`) so a future
row-alignment violation fails loudly instead of misaligning the whitening. Full mask design
rationale is in `handoff/08-core-app.md`.

### 3.6 The spec layer (why there are two layers)

`JointFitSpec` (`joint_fit_spec.py:115-284`) is the persistent, editable description the GUI
binds to; `JointFit` is a throwaway compiled artifact. Everything in the spec is keyed by
dataset NAME, because the engine's per-dataset master keys are POSITIONAL (`d{i}_`), so a
spec that stored sharing by index would silently re-target after the user removed a middle
dataset (`joint_fit_spec.py:23-27`). `compile()` (`:231-284`) registers datasets in spec
order, loads any covariance sidecar, translates name-keyed `ShareGroup`s into engine
`(idx, local_key)` groups (dropping endpoints whose dataset is gone or which lack the key),
drops groups that fall below 2 members, and calls `build_master_params(groups=...)`.
`prune_missing` (`:175-187`) is called from the host whenever the Data list changes, so a
deleted dataset can never leave a stale name that `compile()` would `KeyError` on.

The spec holds *persistent model instances* (`DatasetEntry.model`), not classes, so edits
made in the stock parameter tables survive a re-`compile()`. `constrain_model` (`:128`) is a
GUI-affordance flag only: heterogeneous per-dataset models already work in `compile()`.

### 3.7 GUI panel and the host contract

`JointFitPanel` (`joint_fit_panel.py:178-690`) is a plain `QWidget` mounted *under* the Data
list inside a `QScrollArea` (`xmodfit.py:create_jointFitDock`, `6867-6876`) — not a separate
tab — with everything below the "Enable joint fit" checkbox in a collapsible `self.body`
(`joint_fit_panel.py:195-198`, `312-313`). Its coupling to the host is the documented
contract in the module docstring, implemented at `xmodfit.py:6879-6959`:
`selected_data_names`, `joint_model_name`, `joint_model_factory` (deep-copies the LIVE
`self.fit.func` rather than reconstructing from the function list, because a `.par` load
does not leave `curr_module` in a reconstructable state), `joint_model_for` (prefers that
dataset's remembered individual fit so each dataset enters the joint fit at its own
best-fit), `show_joint_dataset(s)`, `refresh_param_tables`, `run_joint_fit`,
`clear_joint_view`, `clear_individual_selection`, `_overlay_modifier_active`.

Panel sub-widgets: dataset list (`JointDatasetList`, `153-172`, whose `mousePressEvent`
makes a plain click on an already-selected item switch the *detail* without collapsing the
multi-selection), shared-parameter checkbox table (`226-234`), covariance sidecar table
(`237-247`, double-click to attach), per-dataset fit-range table (`249-260`,
`_on_range_item_changed` at `655-679` writes back to the spec and reverts non-numeric text),
master/detail dataset selector (`262-270`; items display basenames, `itemData` carries the
real `<num><>path` key), read-only array readout (`272-280`), save/load/run buttons, and the
joint MCMC button. `_resync_all` (`524-532`) rebuilds every sub-widget from the spec and is
the single refresh entry point; `_reexpand_shared_all` (`608-619`) re-expands bare
share-all groups over the *current* dataset set, because `set_shared` snapshots names at
click time and a dataset added later would otherwise be silently excluded.

Master/detail editing works because `JointFit` duck-types `Fit` for the stock parameter
tables: `.func`, `.fit_params` (the current objective's lmfit `Parameters`) and `.params`
(the current objective's `__dict__`, so `self.fit.params['choices']` keeps working) at
`Joint_Fit_Routines.py:1067-1101`.

Fixed parameters are NOT per-dataset by design: `_sync_joint_fixed_params`
(`xmodfit.py:5854-5877`) unifies all fixed params (including a PDB `fname` and the
`filepaths` dict) across every objective when a joint fit is compiled or loaded, and
`_fan_fixed_to_objectives` (`5836-5852`) fans a table edit out to all objectives. The stated
reason is that a joint fit must test ONE structural hypothesis.

### 3.8 Control flow: click -> result

Main **Fit** button: `doFit` (`xmodfit.py:1721`). First, if the method is `emcee` and the
dedicated joint MCMC window is open, route to `run_joint_emcee` *before* the individual-fit
guards (`1731-1737`) — those guards need a Data-list selection and a function-list
selection, which do not exist while that window is up. Then the individual guards run, and
only afterwards does the joint branch fire: `joint_active` requires
`jointFitEnabled` AND at least one dataset selected in the *joint panel's* list AND
>= 2 datasets in the spec (`1774-1777`), i.e. selection in the joint list is what makes the
Fit button mean "joint". Selecting a dataset in the Data list with a plain click clears the
joint selection, so the Fit button reverts to an individual fit even with joint mode on.

`run_joint_fit` (`7085-7147`): `spec.compile(self.data, masks=self._joint_masks(spec))` ->
`self.fit = self.jointFit` -> `_sync_joint_fixed_params` -> scalar `(xmin,xmax)` parsed from
the Xrange box (deliberately NOT the per-file `self.xmin`/`self.xmax` dicts) plus
`ranges = {name: (fit_xmin, fit_xmax)}` from the panel -> `functionCalled` connected to
`fitCallback` -> `showFitInfoDlg` -> `JointFit.perform_fit(...)` -> `_is_fitted = True` ->
`set_current(detail)` -> `refresh_param_tables()` -> `update_plot()` -> `FitResultDialog`.

Plotting: `update_plot` early-returns to `_update_plot_joint` whenever `self.fit` has
`objectives` (`6155-6157`); `_update_plot_joint` (`6523-6647`) draws each overlaid dataset's
data curves plus its model curve named `<Model> [joint:<num>]`, and lets the DETAIL
objective drive the parameter tables and `output_params`/`genParamListWidget`.
`fitCallback` skips the live single-Fit curve redraw for a `JointFit` (`3481`) but still
shows iteration/chi-sqr. `clear_joint_view` (`1160-1172`) drops the overlay on a blank
click. Details of the overlay/legend machinery belong to `handoff/08-core-app.md`.

Confidence intervals: `_update_ci_button_state` (`2160-2171`) enables "Calculate Param
Error" for a fitted `JointFit` with no Data-list selection; the individual error button
refuses to run while joint mode is enabled and points the user at the panel's own
"Joint MCMC / Param Error…" button (`2213-2226`). Everything past that point is
`handoff/02-mcmc-stack.md`.

`perform_fit` (`Joint_Fit_Routines.py:1004-1062`) supports `leastsq`, `least_squares`,
`differential_evolution` (popsize 300, `updating='immediate'`), a generic fallback, and
`emcee`; `nan_policy='raise'` on all least-squares paths; `**fit_kws` are forwarded to
`Minimizer.minimize` so TRF `ftol`/`xtol`/`gtol` can be tuned.

## 4. How to use it

**GUI workflow**

1. Load the datasets (Data tab) and pick a function; optionally fit each dataset
   individually first (each dataset's fit is remembered, and `joint_model_for` seeds the
   joint fit from it).
2. Tick **Enable joint fit** in the Joint Fit panel (under the Data list) to expand the body.
3. Select datasets in the Data list, then **Add selected <- Data**.
4. Tick the parameters to share in **Shared parameters** — e.g. `theta`, `phi`, `protins`
   for a joint `XLayersPDB` fit; a substrate layer's `__Model_d_00N`/`__Model_rho_00N` for
   `XLayers`. A checkbox means "tied across ALL datasets in this joint fit".
5. Optional: double-click in **Correlated-error covariance** to attach a `.npz` or
   `*_c_factors.txt` sidecar (plus the raw 4-column file for the text flavour). Attaching one
   on any dataset switches the whole fit to GLS whitening.
6. Optional: per-dataset `xmin`/`xmax` in **Per-dataset fit range**; blank falls back to the
   main Xrange box.
7. Set method/scale/iterations in the usual boxes, then **Run joint fit** (or the main
   **Fit** button while a joint dataset is selected).
8. Click a dataset in the joint list to inspect/edit it in the stock parameter tables
   (multi-select overlays curves; a plain click inside a multi-selection only moves the
   detail; Ctrl+click keeps the Data-list selection so individual (dashed) and joint (solid)
   curves overlay).
9. Save/Load via the panel buttons or **File > Save Joint Fit (.jpar)** /
   **Load Joint Fit (.jpar)** / **Load .par into Joint Fit**.

**`.jpar` format** — grammar in `joint_fit_io.py:15-47`, written by `format_jpar`
(`:182-230`), parsed by `parse_jpar` (`:236-331`). It is a superset of the stock `.par`:

    #XModFit Joint Fit v1
    #Fit Range=0.026:0.550          <- GLOBAL scalar fit window (note the space)
    #Column Limits=...              <- global non-x column limits (optional)
    #Fit Method=leastsq
    #Fit Scale=Log
    #Fit Iterations=1000
    #Dataset: C:/data/El_Cat_Si_1_DOPC_wtrans.txt
    #Category: XRR
    #Function: XLayers
    #Xrange=np.linspace(0.0,0.7,400)   <- the MODEL's evaluation grid expression
    #FitRange=0.03:0.45                <- PER-DATASET fit window (no space; blank => global)
    #ColumnLimits=qz:0.1:0.4;...        <- per-dataset non-x column limits (optional)
    #Cov: .../El_Cat_Si_1_DOPC_wtrans_c_factors.txt
    #Raw: .../El_Cat_Si_1_DOPC_wtrans.txt
    #Fixed Parameters: / #Single fitting parameters: / #Multiple fitting parameters:
    ... one stock-.par-style block per dataset ...
    #Shared:
    #group   bare=1   name=-
    C:/data/a.txt<TAB>theta
    C:/data/b.txt<TAB>theta

Datasets are keyed by FILE PATH, not by the volatile in-session `"<num><>path"` key;
`_loaded_key_for_path` (`xmodfit.py:7279-7291`) re-associates on load (exact path first,
then a *unique* basename fallback). Unknown `#` lines are skipped by the parser, so old and
new files load in either direction. `loadJointFit` (`7350-7456`) offers to load missing
datasets from their stored paths, rebuilds each model with `_build_joint_model`
(`7239-7277`, which pulls a PDB `fname` out of the fixed block so a PDB model can be
constructed), restores sharing via `set_shared_endpoints`, restores the global fit settings,
and compiles a viewable `JointFit` flagged `_jpar_incomplete` if any dataset was skipped.

**Headless / scripted use** (the covariance and footprint route, as used by the bilayer
campaign's `run_fit.py`):

    metas, shared = jcov.load_bundle_npz('xrr_corrected_bundle.npz', datasets=[...])
    jf = JointFit()
    for nm, meta in metas.items():
        jf.add_dataset(nm, funcClass=XLayers, x=meta['x'], cov=meta)
    jf.build_master_params(groups=[jf.share_all('theta'),
                                   jf.share_endpoints([('a', '__Model_d_005'),
                                                       ('b', '__Model_d_001')])])
    report, msg = jf.perform_fit(0.026, 0.55, fit_scale='Linear', fit_method='leastsq')

**Harnesses** (do not run them as part of this handoff; an audit pass is deferred):
`python validate_jointfit.py` with `ELCAT_BASE` pointing at the El_Cat_Si_1 data;
`QT_QPA_PLATFORM=offscreen python validate_jointfit_gui.py`;
`validate_joint_mask.py`; `validate_xlayerspdb_joint.py`.

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `xmodfit.py` | module imports (`:58`) | `from joint_fit_panel import JointFitPanel, NameDelegate` | additive |
| `xmodfit.py` | `__init__` state (`:521-552`) | new `jointFit`, `_joint_overlay_names`, `_mcmc_is_joint`, `seed_mode` attrs | additive |
| `xmodfit.py` | `create_jointFitDock` call site (`:595`) + `create_jointFitDock` (`6867-6876`) | new panel mounted under the Data list | additive |
| `xmodfit.py` | File menu (`:614-622`) | 3 new actions (Save/Load `.jpar`, Load `.par` into joint) | additive |
| `xmodfit.py` | `doFit` (`1721-1789`) | two early-return routes (joint MCMC, joint fit) inserted before the per-file loop | additive branch; the per-file loop is untouched |
| `xmodfit.py` | `update_plot` (`6150-6157`) | early return to `_update_plot_joint` when `self.fit` has `objectives` | additive guard |
| `xmodfit.py` | `fitCallback` (`3478-3481`) | live curve redraw skipped for a `JointFit` | additive guard |
| `xmodfit.py` | `dataFileSelectionChanged` (`1114-1129`) | plain click clears the joint-list selection (mutual exclusion) | modifies existing selection behaviour |
| `xmodfit.py` | `fixedParamChanged` (`5828-5830`) | fans the edit to all joint objectives | additive call |
| `xmodfit.py` | `_update_ci_button_state` (`2160-2171`), individual error button (`2213-2226`) | joint-aware enabling + a refusal message when joint mode is on | modifies existing behaviour |
| `xmodfit.py` | `_joint_masks`, `run_joint_fit`, `run_joint_emcee`, `.jpar` I/O, `show_joint_datasets`, `_update_plot_joint`, `clear_joint_view`, `_sync_joint_fixed_params`, model comparison, contract methods | ~1000 new lines, all new methods | additive |
| `Fit_Routines.py` | `_window_indices` (`:8`) | new module-level helper, imported by `Joint_Fit_Routines` | additive (owned by doc 08) |

Merge risk: the only *rewrites* of stock behaviour are the selection-mutual-exclusion in
`dataFileSelectionChanged` and the CI-button gating. Everything else is a guard or a new
method. `origin/develop` has no joint code at all, so no joint file can conflict; the
friction is confined to the `doFit` / `update_plot` / `fitCallback` regions of `xmodfit.py`,
which upstream also touched for its new MCMC framework (`merge-with-upstream-develop.md`).

## 6. Known limitations, gotchas and TODOs

1. **Subgroup sharing is not reachable from the GUI.** The engine
   (`share_within`/`share_endpoints`), the spec (`set_shared(dataset_names=[...])`,
   `set_shared_endpoints`) and the `.jpar` grammar all support arbitrary-subset sharing, but
   the panel's only control is a per-parameter checkbox that calls `set_shared(lkey)` with
   no subset (`joint_fit_panel.py:409`), i.e. share-across-ALL. Subset sharing today requires
   either a hand-written `.jpar` or a script. This is the single biggest gap between the
   advertised feature and the UX.
2. **The footprint covariance term is unreachable from the GUI.** **FIXED 2026-11-06 (WS1 B2a, `1b83aae`): `load_cov_npz` and `load_dataset_cov` now carry `v`/`var_fwhm`, so the term reaches any sidecar-based fit (the GUI attaches sidecars this way). The text below is the pre-fix state.** `_use_footprint` needs
   `cov['v']` and `cov['var_fwhm']` (`Joint_Fit_Routines.py:146-148`), and only
   `load_bundle_npz` produces them. The GUI attaches sidecars through
   `jcov.load_dataset_cov`, which returns neither key for either flavour
   (`joint_covariance.py:277-309`). So a GUI joint fit can use the scale-factor covariance
   but never the shared-beam term; the bilayer campaign gets it via a script.
3. **`load_cov_npz` drops `v`/`var_fwhm` even if present in the `.npz`.** **FIXED 2026-11-06
   (`1b83aae`): both keys are carried through; a sidecar with only one of the two now
   raises `ValueError` at load rather than turning the footprint term half-on. The text
   below is the pre-fix state.** `load_cov_npz` (`joint_covariance.py:200-211`) returned
   neither key, which is why (2) was unreachable for the npz flavour.
4. **`doFit`'s individual-fit guards run before the joint branch.** `self.sfnames` and a
   function-list selection must be non-empty for a joint fit via the main Fit button, even
   though a joint view has no Data-list selection. This is survivable only because
   `clear_individual_selection` blocks signals and leaves `sfnames` stale. The joint-MCMC
   route had to be hoisted above the guards for exactly this reason (`1727-1737`); the
   least-squares route was not.
5. **The joint fit uses the scalar Xrange box, never `self.xmin`/`self.xmax` per file.**
   Per-dataset windows come only from the panel's range table / `.jpar` `#FitRange=`.
   Editing the Xrange box while an individual dataset is selected does not change a joint
   fit's window for that dataset unless you also set its row in the range table.
6. **`_eval_chisqr` (model comparison) ignores per-dataset ranges and column masks.**
   **FIXED 2026-11-06 (WS1 A1.4, `e1e8b9f`): both models are scored on
   `JointFit.active_index(i)`, the model evaluated on the objective's own `[imin, imax]`
   window (`chisqr_on(..., imin=, imax=)`, final fix wave M1). A `.jpar` or `.par` loaded
   into the joint fit and never re-fitted now has its fit ranges and column masks frozen
   at load (`XModFit._apply_joint_ranges`, which calls `JointFit._set_ranges` with the same
   arguments as `run_joint_fit`; final fix wave I2), so the comparison honours
   `#FitRange=` and the column limits there too. Under covariance the comparison scores
   both models on raw `y`/`yerr`, a plain chi-square, not `y_corr`/`sigma_meas`; that was
   already so and is intended. The text below is the pre-fix state.**
   `_eval_chisqr` (`xmodfit.py:7532-7551`) masked with the scalar `_joint_range()` only. A joint fit
   using per-dataset windows will be compared against individuals over a different point
   set; the comparison already warns when point counts differ, but the joint side itself is
   computed on the wrong window.
7. **`'Linear w/o error'` scale weights datasets by their own maximum.** **WARNED 2026-11-06 (`6b727d0`): the weighting is unchanged (decision X1), and `perform_fit` adds a `jf.warnings` entry for >=2 datasets without covariance. The text below is the pre-fix state, and still describes the weighting.** In the no-cov
   branch each dataset is divided by its own `np.max(data)`; the factor cancels in `Log`,
   `Log w/o error` and `Linear` (via `err`), but NOT in `Linear w/o error`, where the
   relative weight between datasets becomes `1/maxd`.
8. **A share-group inherits value/bounds/vary from its FIRST endpoint only.** **FIXED
   2026-11-06 (`e177e2f`, `9518ef9`, final fix wave): differing start values warn (one
   `jf.warnings` entry per group); differing `min`/`max`/`vary` raise `ShareMismatchError`
   unless `build_master_params(on_mismatch='first')`. `'first'` copies the first
   endpoint's `min`/`max`/`vary` onto every other endpoint's local Parameter
   (`_unify_share_group`), so `push_params` can no longer clip a later endpoint, and adds a
   warning saying the bounds were unified. `±inf` bounds compare by equality. Premise
   correction: `run_joint_fit` did NOT catch exceptions. The GUI now compiles through
   `XModFit._compile_joint_guarded` on the Fit path and on both load paths: a
   `ShareMismatchError` asks Yes/No ("Apply the first dataset's bounds/vary to all members
   of each share group and continue?"; Yes recompiles with `on_mismatch='first'`, No
   leaves everything as it was), and any other `ValueError` (e.g. a covariance sidecar
   problem) is shown in a warning box instead of reaching the Qt slot. A refused load shows
   no success box. The text below is the pre-fix state.** If two datasets disagreed on a
   shared parameter's starting value or bounds, the master took the first endpoint's, and
   the push wrote that value into each later endpoint's own local Parameter, whose
   narrower bounds clipped it silently (lmfit's value setter). That dataset was then
   evaluated at a different value from the one the master reported (`build_master_params`
   step 1 and `push_params`, `Joint_Fit_Routines.py:475-486` and `:576` at the final fix
   wave).
9. **`remove_dataset` on the engine renumbers every later dataset's master keys**, so the
   caller must re-run `build_master_params`; the GUI never does this, it edits the spec and
   recompiles instead (which is why engine-level `remove_dataset` is nearly dead code).
10. **`set_priors` has no GUI.** Priors are script-only, and `remove_dataset`/`clear` wipe
    them because the master keys they name are rebuilt.
11. **Covariance/GLS joint MCMC is deliberately serial** and orientation seeding is skipped
    for it (`Joint_Fit_Routines.py:868`, `956-964`) — flagged as follow-up work by the notes.
    Details in `handoff/02-mcmc-stack.md`.
12. **Deferred by the user (from `merge-with-upstream-develop.md`):** folding the joint-fit
    overlay controls INTO the "fit options" panel, sharing buttons with the enable toggle,
    instead of the current collapsible panel under the Data list.
13. `array_summary_rows(spec, jf=...)` accepts a compiled `JointFit` but `_param_value`
    never reads it (`joint_fit_panel.py:135-143`); it works only because `compile()` reuses
    the spec's model instances.
14. **WARNED 2026-11-06 (`6b727d0`): `compile()` adds a `jf.warnings` entry when a dataset has more than one curve.** `_primary_curve` (both in `joint_fit_spec.py:290-294` and
    `joint_fit_orchestration.py:39-46`) takes only the FIRST sub-curve of a dataset. A
    multi-curve dataset silently contributes one curve to a joint fit.
    **2026-11-09 (audit gap AG.4, `3ba60b4`):** the orchestration copy warns too, into
    `JointFit.warnings` from `build_jointfit_from_selection`, or through `warnings.warn`
    from a bare `_primary_curve` call (`validate_joint_ws1.py` T4c').

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| 1 | A 1-dataset, no-covariance `JointFit` reduces *exactly* to stock `Fit` | `Joint_Fit_Routines.py:6-9` docstring; (retired; was `validate_jointfit.py` T2) | **VERIFIED 2026-11-04 on synthetic data** by `validate_joint_gls.py` G2 (Linear and Log fit scales: chisqr, values and stderrs agree with stock `Fit`). Not re-run on real data |
| 2 | `sum((L^-1 r)^2) == r^T C^-1 r` to <1e-8 on the real El_Cat covariance | `validate_jointfit.py` T1 | **Verified on SYNTHETIC data with the footprint term ON 2026-11-04** (not the real El_Cat covariance) by `validate_joint_gls.py` G1 (synthetic, no real data). Original check: run T1 with `ELCAT_BASE` set; also assert the same identity with the footprint term switched on (T1 currently builds `C` without `v`) |
| 3 | Cross-dataset covariance measurably changes stderrs vs diagonal-only | (retired; was `validate_jointfit.py` T3) | **PARTLY COVERED 2026-11-04** by `validate_joint_gls.py` G3's cross-vs-block comparison on synthetic data: stderr(a) differs by ~60% between the cross-block and block-diagonal covariance, and each matches its own analytic value. Real-data stderr ratios not recorded |
| 4 | lmfit's `.covar` under the whitened residual gives correct GLS uncertainties without MCMC | `Joint_Fit_Routines.py:22-27` | **PARTLY CONFIRMED 2026-11-04** by `validate_joint_gls.py` G3 (redchi computed independently as chi2/(N-n_free), and checked against the engine's): engine stderr matches the redchi-scaled GLS covariance (lmfit `scale_covar=True`) for a linear model; whether reported GLS errors should be scaled or unscaled is open -- a WS2 reporting-contract decision; MCMC comparison untested. The match is to ~1e-8 in both the cross-block and block-diagonal variants. The unscaled textbook covariance `(X^T C^-1 X)^-1` is ~6% smaller here. Compare `result.covar` stderrs against a serial covariance-aware MCMC posterior on one small GLS fit |
| 5 | The `bare=True` master name is what keeps the emcee `theta`/`phi` angle gate firing | `Joint_Fit_Routines.py:265-266`, `370-378` | Grep the gate in `lmfit/minimizer.py` / `customsampler.py` and confirm it matches on the bare name; then check a subset-shared (`g0_theta`) joint MCMC does NOT wrap angles (expected failure mode) |
| 6 | `rsplit('_', 2)` in `loadParameters` (cited as xmodfit.py ~3246/3254) tolerates the `d{i}_`/`g{gid}_` prefixes | `Joint_Fit_Routines.py:243-244`, `377-378` | Those line numbers are stale (the file has grown); locate the current `rsplit('_', 2)` sites and confirm a `__d0_Model_d_005`-style key round-trips through `.par` load |
| 7 | Per-dataset parameter edits in the stock tables survive a re-`compile()` | `joint_fit_panel.py:31-33` | Edit a value for dataset 1, add a dataset (forces `_resync_all`), re-run, confirm the edit is in `jf.master_params['d1_...']` |
| 8 | Fixed-parameter sharing across objectives is always correct for PDB models (`filepaths`) | `xmodfit.py:5854-5877`; `validate_jointfit_gui.py:165-170` | Run the GUI harness checks; additionally confirm a PDB swap re-reads the structure per objective rather than leaving a stale cached geometry |
| 9 | `.jpar` round-trips old and new files in either direction (the `#FitRange=`/`#ColumnLimits=` additions are back-compatible) | `joint_fit_io.py:190-195`; `column-fit-limits.md` | Load a pre-August `.jpar` (no `#FitRange=`) and confirm `fit_xmin/fit_xmax` come out `None`; write a new one and load it in a build without the column-limit code |
| 10 | The `0.99*xmin` -> exact-bound change shifts `imin` by zero points on the real test data | `column-fit-limits.md` (measured claim) | Re-measure on `Data/i2_tim3_1000_test.txt` at its own `#Fit Range=0.026:0.550`; the note admits a one-point shift is possible for a datum in `[0.99*xmin, xmin)` |
| 11 | Masks compose identically in `_build_chol` and `residual` (whitening rows aligned) | `Joint_Fit_Routines.py:474-479`, `532-536` | Run `validate_joint_mask.py`; add a case where the two datasets have *different* masks and a non-trivial `Sigma_logc` off-diagonal, and check the length guard is not the only protection **Seed scans and parallel MCMC sampling, 2026-11-06 (WS1, `63e984e`, `bfdd234`):** the same frozen window mask now reaches the seed workers and the parallel log-probs; `validate_seed_mask.py` shows serial == parallel with a mask on, and differs from mask-off. Covariance-whitening side was covered 2026-11-04 (`validate_joint_gls.py` G4/G4b) |
| 12 | The panel's headless-testable-with-a-stub-parent claim | `joint_fit_panel.py:6-8` | Instantiate `JointFitPanel(parent_app=stub)` with a stub exposing only the documented contract and confirm no `AttributeError` (the panel calls `_refresh_data_labels`, `_overlay_modifier_active`, `joint_model_for`, `confInterval_joint_emcee`, none of which are in the documented list) |
| 13 | `differential_evolution` with `popsize=300` is usable on a joint objective | `Joint_Fit_Routines.py:1048-1053` | Time one small joint DE run; 300 * nvar population on a concatenated objective may be impractical |
| 14 | A joint fit run through the main **Fit** button behaves identically to **Run joint fit** | `xmodfit.py:1780-1789` | Both call `run_joint_fit()`, but verify `fit_method`/`fit_scale` are set consistently (doFit sets them before routing; `run_joint_fit` re-reads the combo boxes at `7100-7101`) |

## 8. Consolidation candidates

| Item | Recommendation | Risk |
|---|---|---|
| `joint_fit_orchestration.py` (112 lines) | **Drop or fold in.** `build_jointfit_from_selection` is imported at `xmodfit.py:1723` but its only call site is inside a triple-quoted dead block (`1798-1811`); `push_result_to_tables` has no caller at all; `_primary_curve` is duplicated verbatim in `joint_fit_spec.py:290-294`. The spec layer superseded it. | Low. Keep the module docstring's "GUI hook" example somewhere, and delete the dead import + dead string block together. |
| The dead `doFit` block at `xmodfit.py:1798-1811` | **Delete.** It documents the pre-spec orchestration path and references `self._cov_path`/`self._raw_path`/`self.jointSharedParams`, none of which exist. | None. |
| `xmodfit_old.py` (untracked) | **Drop.** It contains a third copy of the same dead orchestration code plus a live `push_result_to_tables` import; it will confuse a grep during the merge. | None (untracked, superseded). |
| `JointFit.share_within` / `share_endpoints` / `set_shared_expr` / `set_priors` | **Keep**, but only if item 1 of section 6 (GUI subset sharing) is actually built; otherwise they are an untested API surface. `share_endpoints` is the one with a real, named use case (same oxide at different layer indices). | Medium — dropping them would remove the only mechanism for heterogeneous-key sharing. |
| `JointFit.remove_dataset` | **Keep but mark internal.** The GUI never calls it (spec edit + recompile instead) and it leaves `param_map` stale until `build_master_params` re-runs. | Low. |
| `_primary_curve` duplication | **Merge** into one helper (it also embeds the "first sub-curve only" limitation, section 6 item 14, which should be fixed in one place). | Low. |
| `validate_jointfit.py` hard-coded `BASE = '/mnt/project'` and `El_Cat_Si_1_*` filenames | **Move** the data paths to env/args before upstreaming; the harness cannot run on another machine as written. | None. |
| Joint MCMC surface (`set_emcee_minimizer`, `_attach_joint_parallel`, `_build_orientation_seed`, `_seed_shared_axis`, `_resample_regular`) — ~460 of the 1114 lines in `Joint_Fit_Routines.py` | **Consider splitting** into a `joint_mcmc.py` mixin so the engine file stays about the objective. Documented in `handoff/02-mcmc-stack.md` / `handoff/03-orientation-seeding.md`. | Medium — it touches the emcee path that `babe1aa` just stabilised; do it after the merge, not during. |
