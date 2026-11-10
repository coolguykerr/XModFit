# Spec: general correlated-error (covariance) fitting, and the LSS_Reader contract

**Written 2026-11-09.** This is a design document, not a task plan. A task-by-task plan
(`superpowers:writing-plans`) is written from it once it is approved. It covers two
programs that have to agree on a file:

- **XModFit** (this repo): fits any dataset, with any fit function, using a covariance
  supplied for that dataset, in single and joint fits.
- **LSS_Reader** (`chemmatcars/LSS_Reader`, a separate repo): the reduction program that
  stitches reflectivity scans. It must write that covariance. §8 is its part.

Ordering: this workstream comes **before WS2** in `14-roadmap.md`. WS2 decomposes χ² by
dataset, and with correlated errors that decomposition is no longer a plain sum, so WS2
has to be designed against this.

Calls marked **[call]** were made by the author of this document, not by the user. They
are collected in §12 so they can be checked.

---

## 1. Intent

Reduced reflectivity data is assembled from several scans or detector regions. Each
region is matched to the one before it by a scale factor, and that factor is uncertain.
All points in a region therefore share one scaling error. The matching chains region to
region, so each region's error also accumulates the errors of the regions before it.
These are **correlated** errors. Per-point error bars cannot express them, and a fit
that ignores them over-weights the stitching steps.

The user wants:

- to supply a covariance for **any dataset**, and have **any fit function** respect it;
- this to work for a **single-dataset fit**, not only inside a joint fit;
- the covariance to come from the **reduction**, which knows the stitching. XModFit never
  reconstructs the stitching itself;
- the two programs to "talk to each other correctly", through a file format that both
  sides implement exactly.

The engine already contains one special case of this: the bilayer campaign's attenuator
scale-factor covariance plus a shared beam-footprint term, joint fits only. This
workstream generalises it and keeps that case working.

## 2. What exists today (as of `7d2549e`)

> **Status 2026-11-10: implemented (XModFit side) by doc 18, commits `b55bf41`..COV T8
> (`git log --oneline --grep "COV T"`); the list below is the pre-implementation state.
> Verified headless by `validate_covariance.py`, `validate_covariance_gui.py`,
> `validate_cov_stitching.py`, `validate_joint_gls.py` and the full `tests/run_all.py`. The
> LSS_Reader side (§8) is not started.**

From reading `Joint_Fit_Routines.py`, `joint_covariance.py` and `Fit_Routines.py`:

- **Joint only.** Covariance exists only in `JointFit`. `Fit` (single dataset) has none.
- **One fixed formula** (`joint_covariance.build_full_covariance`):
  `C = diag(σ_meas²) + outer(y,y)·Σ_logc[k_i,k_j] + var_fwhm·outer(v,v)`.
  It is built from the measured `y` (*data-based*), so it is biased low (§4.4).
- **Three file formats**, all from the scale-factor-correction workflow:
  - `*_corrected_cov.npz`;
  - `*_c_factors.txt`, plus a raw 4-column file;
  - the multi-dataset bundle (`load_bundle_npz`), which carries per-dataset `v`/`var_fwhm`.

  All of them *replace* the dataset's `y`/`yerr` with `y_corr`/`sigma_meas`.
- **GUI:** the joint panel already has a per-dataset covariance and raw-file picker
  (`joint_fit_panel.py:414-425`, `covTable`). It is for joint fits only.
  `xmodfit._cov_path`/`_raw_path` are read in dead code but never set. Single fits have no
  route at all.
- **Forced sharing.** Every dataset with a covariance must share one `Sigma_logc`, or
  the fit raises `ValueError`. The footprint term additionally requires one shared
  `var_fwhm`.
- **Mixing crashes.** One attached covariance turns on the whitened path for the whole
  fit. A dataset without one then fails in `_build_chol` (`obj["cov"]` is `None`). This
  was found by reading the code and has not been run.
- **Fit scale ignored.** With a covariance the residual is always linear. `Log` is ignored
  without notice.
- **MCMC.** Joint covariance MCMC is serial only.

## 3. Goals, non-goals, anti-goals

**Goals**

1. A covariance can be attached to any single-curve dataset and is honoured by every fit
   function, in single fits and in joint fits.
2. The covariance model is general. It is made of named error **terms** of three kinds
   (§4.2). The per-region stitching error and the existing attenuator and footprint terms
   are instances of it.
3. Datasets are independent by default. Correlation across datasets exists only through
   terms explicitly declared **shared**, which is how the bilayer case is expressed.
4. Datasets with and without a covariance can be fitted together.
5. Both fit scales work: `Linear` (model-based covariance, §4.4) and `Log` (§4.5).
6. The bias of the data-based form is removed by default. The data-based form remains
   available to reproduce earlier results.
7. File formats are read by separate loaders into one internal object, so that the
   planned combined data-plus-covariance file from LSS_Reader is one more loader (§7).
8. LSS_Reader gets a precise specification of what to compute and write (§8).

**Non-goals (this workstream)**

- Parallel MCMC with a covariance. It stays serial, with a warning (§6.4).
- Quantifying shared instrument parameters, such as beam width, for LSS data. The term
  model allows adding them later.
- The small correlation between neighbouring overlap points caused by LSS_Reader's spline
  interpolation (§8.6).
- The combined LSS_Reader file format. Only the loader interface for it is in scope.
- Any change to fits **without** a covariance. They must behave exactly as today.
- GUI redesign. Only a minimal entry point is in scope (§6.6).
- Implementing the LSS_Reader changes. §8 specifies them. Doing them is a separate task in
  that repo.

**Anti-goals: it fails even if it technically works when any of these is true**

- Fitted per-region scale factors are presented as results, or as a measure of how well
  the data is stitched (§4.6).
- A no-covariance fit changes in any way.
- Something was dropped or downgraded and the user is not told, both in the warnings and
  in the report.
- The stitching error is counted twice: once inside the per-point error bars and again in
  the covariance (§8.2).
- The bilayer GLS results can no longer be reproduced.

## 4. The covariance model

### 4.1 Notation

One dataset has N points, x (qz), data y, model f(θ). Points are indexed i, j, and
regions or components k, l.

### 4.2 Terms

A dataset's covariance is **measurement error plus a list of terms**:

    C = diag(σ_meas²)  +  Σ_terms C_term

Each term has:

| field | meaning |
|---|---|
| `name` | identifier; see `scope` |
| `scope` | `"dataset"` (default): private to this dataset. `"shared"`: the same random variables as every other dataset's shared term of the same name |
| `kind` | `"multiplicative"`, `"additive"` or `"dense"` |
| `labels` | K component names (e.g. scan numbers); not used for dense terms |
| `weights` | N×K loading matrix W; not used for dense terms |
| `cov` | K×K covariance Σ of the components; for a dense term, the N×N covariance itself |

How each kind contributes, in **linear** space:

- **multiplicative**: the components are log-scale factors (natural log). Point i is
  scaled by exp(Σ_k W_ik · ε_k) with ε ~ N(0, Σ). To first order:
  `C_term = D · W Σ Wᵀ · D`, where D = diag(intensity) (§4.4).
  This kind covers stitching scale factors and attenuator calibration.
- **additive**: the components add to the intensity, y_i += Σ_k W_ik · ε_k, so
  `C_term = W Σ Wᵀ`, in data units. The existing beam-footprint term is this kind, with
  K = 1, W = v and Σ = [[var_fwhm]].
- **dense**: a full N×N covariance supplied directly, in data units. This is a general
  fallback for correlations that fit neither form.

### 4.3 LSS stitching as a multiplicative term

There is one component per scan, in patch order, with K = number of scans. Scan 1 is
the anchor. When scan m is joined, LSS_Reader fits `fac_m` against the **already-merged
and already-scaled** data. So `fac_m` is scan m's absolute factor, and `facerr_m` is its
uncertainty *relative to the data it was matched to*. That data carries the earlier
scans' errors, which is where the chaining comes from.

Let ε be the vector of absolute log-scale errors, and let a_m be the mean of the weight
rows (§8.4) of the existing-data points used in scan m's factor fit (`overlap1` in
`oneDjoin`). Then ε_m = a_mᵀ ε + η_m, with η_m independent and of variance
δ_m² = (facerr_m / fac_m)². Built up one join at a time, with ε_1 = 0:

    Σ[m, j] = a_mᵀ Σ[:, j]                  for j < m
    Σ[m, m] = a_mᵀ Σ a_m + δ_m²

In the usual case, where scan m's overlap lies only on scan m−1's points, a_m = e_(m−1).
This then reduces to the cumulative random walk the user described:
Σ[k, l] = Σ_{m = 2..min(k,l)} δ_m².

W_ik = 1 for a point taken from scan k alone. An overlap point that LSS_Reader averaged
from two scans has 0.5 / 0.5 weights (§8.4).

### 4.4 Linear scale: model-based intensity

For a multiplicative term, D must be the **true** intensity, which is unknown.

- **Data-based**: D = diag(y), which is today's behaviour. It is biased. A point that
  fluctuated low gets a smaller variance, is trusted more, and pulls the fit down. This
  is the classic "Peelle's Pertinent Puzzle". Take two measurements, 1.5 and 1.0, each
  with 10 % independent error and a 20 % common normalisation error. GLS with a
  data-based covariance gives ≈ 0.88, which is below both.
- **Model-based (default) [call]**: D = diag(f(θ)). It is computed by outer iteration:
  1. C₀ is built from the data. Fit, giving θ̂₀.
  2. Rebuild C from f(θ̂₀). Refit, starting from θ̂₀.
  3. Repeat until every varying parameter moves by less than 0.1 of its standard error,
     or 5 rounds have run.

  A fit that reaches the 5-round limit without converging keeps its last result and adds
  a warning. The report states the number of rounds. Additive and dense terms do not
  depend on intensity in linear space.

`covariance_mode` selects between the two: `"model"` (default) or `"data"`. The mode is
reported with every fit.

### 4.5 Log scale

The residual is r = ln y − ln f (natural log; see the note below). The covariance of
ln y is:

    C_log = diag( (σ_meas / y)² )
          + Σ_mult  W Σ Wᵀ                                  (exact: no intensity)
          + Σ_add   E · W Σ Wᵀ · E,   E = diag(1 / I)
          + Σ_dense E · C_dense · E

The multiplicative terms are constant in log space. They are exact for one-hot weight
rows and first-order for averaged (0.5 / 0.5) rows. So for stitching-only data no
iteration is needed. Additive and dense terms carry an intensity I, which is the model
in `"model"` mode and the data in `"data"` mode. The model case uses the same outer
iteration as §4.4. The measurement term uses y. This is the same approximation the
existing `Log` scale makes.

*Note, fixed ahead of this work:* the no-covariance `Log` residual used to be
(log10 y − log10 f)·y/σ, which understates every residual by ln10, so χ² was about
1/5.3 of the Gaussian value and MCMC posteriors about 2.3× too wide. It is now
(ln y − ln f)·y/σ in `Fit.residual`, its Jacobian, `JointFit.residual` and the three
parallel log-prob mirrors (`pdb_logprob`, `pdb_joint_logprob`, `pdb_seed_worker`).
Best-fit values and lmfit's rescaled least-squares errors are unchanged by it; χ², χ²_red
and MCMC widths change. With that fix the covariance and no-covariance Log paths share one
definition, so their χ² are directly comparable and a mixed joint fit needs no special
weighting. `Log w/o error` is unweighted and keeps log10.

`Log w/o error` and `Linear w/o error` with a covariance attached: the covariance is
ignored and the fit runs unweighted as today, with a warning **[call]**.

### 4.6 What the covariance does and doesn't do

The covariance changes **how misfit is scored**. It adds no parameter to the model.
Misfit shaped like a whole-region offset of plausible size is cheap. Any other misfit is
penalised as usual. This is identical to fitting hidden per-region scale factors with a
prior Σ and profiling them out. That is a fact about the arithmetic and is not exposed as
parameters. A too-simple model is excused only where its error looks like a region-wide
offset within the stated uncertainty.

The hidden scale factors are therefore **never reported as results**. The only permitted
output is the optional WS2 diagnostic. For each multiplicative component it gives the
shift the fit leaned on, ε̂ = Σ Wᵀ D C⁻¹ (y − f), and its pull, ε̂_k / √Σ_kk. It is
labelled *model–reduction tension*, never *stitching quality*.

## 5. Masks, fit ranges, joint structure

- **Points excluded** by the fit range, the column mask or non-finite values remove the
  matching rows and columns of C. For a Gaussian this is exact marginalisation. The
  existing mask order holds: slice, then imask, then concatenate, then the finite-value
  mask. The mask is frozen once per `perform_fit` (CLAUDE.md rule 4).
- **Joint fits:** C is block-structured. The diagonal block for each dataset is its own C.
  A dataset with no covariance gets diag(yerr²) in linear space, or the log equivalent.
  Off-diagonal blocks are non-zero only through shared terms:
  `D⁽ᵃ⁾ W⁽ᵃ⁾ Σ W⁽ᵇ⁾ᵀ D⁽ᵇ⁾` for multiplicative terms, and the equivalent for additive terms.
- **Shared-term consistency:** every dataset carrying a shared term of the same name must
  give identical `labels` and `cov`. If they differ, the program does not refuse the fit.
  It treats them as separate private terms and warns **[call]**, which follows the
  error policy (§6.5).
- **Whitening:** factor C = L Lᵀ and fit L⁻¹ r.
  - The existing `cholesky_with_jitter` never fails. After 6 jitter attempts it floors the
    eigenvalues without saying so, and it returns the last jitter value even though that
    was not applied to the floored matrix.
  - The core module wraps the helper rather than changing it, so data-mode reproduction is
    untouched. The wrapper detects which path was taken, and reports the applied jitter or
    "eigenvalues floored" as a warning.
- **Analytic Jacobian** (`Fit.residual_jac`, used by XLayersPDB_XPR): it must be
  whitened in the same way, L⁻¹ · (∂r/∂θ), with the log-scale chain rule in Log mode.
  The two must stay consistent. The existing finite-difference comparison pattern tests
  this.

## 6. XModFit architecture

### 6.1 Core module

A new Qt-free root module, `covariance_core.py`, following the roadmap's global
constraints. It must not be placed in a `Functions/<category>/` folder (CLAUDE.md rule 1).
It provides:

- `DatasetCovariance`: x, σ_meas, a list of `CovTerm`, the source path, the loader name,
  and an optional `replaces_y` array (legacy formats only).
- `CovTerm`: the fields of §4.2.
- `load_covariance(path, raw_path=None) -> DatasetCovariance`: dispatches to registered
  loaders by content (the JSON `format` key) or extension. `raw_path` is needed only by
  the `*_c_factors.txt` loader.
- `DatasetCovariance.subset(index)`: the covariance restricted to a set of point indices,
  meaning rows of σ_meas and W, rows and columns of dense terms, and y. It is used for
  fit-range slicing and pre-sliced calls (§6.2).
- `assemble(blocks, intensities, scale, shared_terms) -> C`: the full matrix over the
  concatenated active points.
- `whiten`, the triangular solve, and the existing `cholesky_with_jitter`, moved here.

`joint_covariance.py` keeps its legacy readers. They become loaders producing
`DatasetCovariance`. `build_full_covariance` stays as a thin wrapper so that existing
harnesses and the bilayer reproduction keep working.

### 6.2 Single fits (`Fit_Routines.Fit`)

- **The covariance belongs to the dataset, not to the `Fit` object.** The GUI keeps one
  `self.fit` and moves it between datasets with `set_x`, in `runFit` (`xmodfit.py:1873`)
  and `update_plot` (`:6262`). So:
  - The covariance is stored with the dataset record in the GUI, keyed by dataset name,
    and is persisted from there.
  - `Fit.set_x(x, y, yerr, mask, cov=None)` takes the covariance that matches the arrays
    being passed. A `set_x` call without `cov` leaves the fit with no covariance, so moving
    to another dataset can never carry the previous one's covariance along.
  - `update_plot` passes pre-sliced [xmin, xmax] arrays. It must pass
    `cov.subset(window index)` with them, so the arrays and the covariance always cover the
    same points.
- With a covariance attached on an array-x dataset, `residual` and `residual_jac` use the
  whitened path from the core module.
- **Displayed numbers:** the Chi-Sqr, Red_Chi_Sqr and `Residuals_*` output shown by
  `update_plot` come from the same whitened objective the fit minimised, and are labelled
  as such. The plotted residual is the whitened one. `Fit_Routines.py` is an upstream-shared file, so
  the change is kept to one guarded branch in each method, and the no-covariance code is
  untouched.
- Dict-x (multi-curve) functions with a covariance: the covariance is ignored, with a
  warning **[call]**.
- `fiterr` / `stderr` come from the whitened Jacobian, so they reflect the covariance.
- **Error bars with a covariance attached (decided by the user, 2026-11-10): absolute,
  with χ²_red shown beside them.** lmfit's default `scale_covar=True` (never overridden in
  XModFit) multiplies the parameter covariance by the fit's χ²_red. For a nonlinear
  structural model with unmodelled error sources that is not a reliable error scale (the
  degrees of freedom are not N − p, and one common scale for every point is not
  justified), and it would partly undo what the covariance states. So, with a covariance:
  - least-squares fits run with `scale_covar=False`;
  - the profile confidence interval uses an absolute threshold, χ²_min + Δχ²(CL, 1 dof)
    (Δχ² = 1 at 68.3 %), in place of `setTargetChiSqr`'s χ²_red-scaled target **[call]**;
  - χ²_red is reported prominently next to the errors, with a warning line when it is
    well above 1, so a misfit is visible rather than folded into the error bars **[call:
    the threshold is the builder's, e.g. outside the central 99 % of χ²(nfree)/nfree]**.
  This matches what MCMC already does: weighted scales sample with `is_weighted=True`,
  which uses the residuals as absolute and never rescales by χ²_red. Without a
  covariance, nothing changes.

### 6.3 Joint fits (`JointFit`)

- `JointFit` replaces `_use_cov`, `_Sigma_logc`, `_var_fwhm` and `_use_footprint` with a
  list of per-dataset `DatasetCovariance` or `None`. It assembles one C from them through
  the core module.
- The §5a invariant holds: `residual` branches only on "any covariance attached?".
- Mixed attached and unattached datasets work, which fixes the crash.
- The old sharing rule becomes "legacy loaders mark their terms `shared`", so bilayer fits
  are expressed exactly as before.

### 6.4 MCMC

- The log-likelihood is −½ ‖L⁻¹ r‖².
- In `"model"` mode, C is frozen at the best-fit model for the run, so the log-det term is
  constant. If no prior fit exists, it uses `"data"` mode and warns **[call]**.
- The parallel picklable objectives (`pdb_logprob`, `pdb_joint_logprob`) do not get the
  covariance in this workstream. With a covariance they fall back to serial with a
  warning, as joint covariance MCMC already does.

### 6.5 Error policy (decided 2026-11-09)

Benign failures continue in a weaker mode and are reported. Only a case that needs a user
decision stops, and it asks. Nothing aborts the app. Every downgrade is recorded in the
result (`warnings`, the report, and later the WS3 library entry).

| situation | behaviour |
|---|---|
| covariance file's x doesn't match the dataset, **at load** (the user is present) | refuse to attach, with a message naming the mismatch |
| x mismatch **at fit time** (e.g. reloaded `.par` with changed data) | fit that dataset without its covariance, and warn |
| shared term with conflicting `cov` / `labels` | treat as private terms, and warn |
| `w/o error` fit scale with a covariance | ignore the covariance, and warn |
| C not positive definite | jitter or eigenvalue floor applied (§5), and warn |
| model-based iteration hits its round limit | keep the last result, and warn |
| dict-x function with a covariance | ignore the covariance, and warn |
| parallel MCMC requested with a covariance | run serial, and warn |
| malformed or unknown-version covariance file, at load | refuse to attach, with a message |

**What "fit without the covariance" means:**
- **v1 files:** exactly the no-covariance fit of the dataset as loaded, using the data
  file's y and its own error column. That is today's behaviour. The LSS total error
  already contains the stitching error, uncorrelated.
- **Legacy files** (`replaces_y`): the corrected data `y_corr` with diag(σ_meas²). The raw
  data-list y is a different, uncorrected dataset.

Tests stay strict: a deliberate fallback reports through `warnings`, never by printing a
traceback (the runner fails any harness that prints one).

### 6.6 Entry points, persistence, reporting

- **Scripts (primary until WS4):** `fit.attach_covariance(load_covariance(path))`, and
  in a joint spec the existing per-dataset `cov_path`.
- **GUI, minimal [call]:**
  - In the data list: "Load covariance…" and "Remove covariance" for the selected dataset,
    plus a marker on datasets with a covariance attached.
  - The joint panel's existing `covTable` picker keeps working, routed through the new
    loaders. A joint fit uses each dataset's attached covariance unless the panel overrides
    it.
  - Selecting the mode (`model`/`data`) is a fit option.
  - The full GUI is WS4. This minimal single-fit entry point is an exception to the GUI
    deferral in `14-roadmap.md`, because single fits have no route today.
- **Persistence:**
  - `.par` saves the covariance path, the raw path if any, and the mode.
  - `.jpar` keeps its existing `cov`/`raw` fields and adds the mode.
  - A saved file with **no** mode field was produced by the old engine, so it loads in
    `"data"` mode and the report notes it **[call]**. That way a reloaded bilayer fit is
    unchanged. New fits default to `"model"`.
  - A missing covariance file on reload follows the fit-time row of §6.5.
- **Report:** every fit with a covariance states the file, each term (name, kind, scope,
  K), the mode, the fit scale, the outer-iteration rounds, any jitter, and every
  downgrade. A downgraded fit says so on its first line. This follows the roadmap rule
  that every number states the objective it came from.

## 7. File formats and loaders

### 7.1 The interchange format (v1, written by LSS_Reader)

A JSON sidecar **[call]**: readable, versionable, and trivially extended to the future
combined file. The future file is the same schema plus a `data` block, so the switch to
it is one more loader returning the same object.

```json
{
  "format": "xmodfit-covariance",
  "version": 1,
  "producer": {"name": "LSS_Reader", "version": "<string>"},
  "data_file": "sample_ref.txt",
  "x": [0.0194, 0.0204, "... N values, full precision"],
  "y": ["... N values: the y written to data_file"],
  "sigma_meas": ["... N values, same units as y"],
  "terms": [
    {
      "name": "stitch",
      "scope": "dataset",
      "kind": "multiplicative",
      "labels": ["S12", "S13", "S14"],
      "weights": [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], "... N rows of K"],
      "cov": [[0, 0, 0], [0, 4e-4, 4e-4], [0, 4e-4, 9e-4]],
      "provenance": {"fac": [1.0, 0.0213, 0.0118], "facerr": [0.0, 4.3e-4, 2.6e-4]}
    }
  ]
}
```

Rules XModFit enforces at load:

- `format` and `version` are required. Unknown versions are refused (§6.5).
- `x` and `sigma_meas` have length N, which equals the dataset's length.
- x **and y** match the dataset point by point within a relative tolerance of 1e-4.
  LSS_Reader's text files carry 5 significant figures (`%.4e`). Checking y catches a
  sidecar paired with the wrong data. Examples are an `_rrf` file with a `_ref` sidecar,
  or a y column rescaled or re-expressed in the data dialog. Both would otherwise silently
  mismatch σ_meas and the data-unit terms.
- A `data_file` name that differs from the loaded file's name is a warning, not a refusal.
- `sigma_meas` > 0 and is finite.
- `weights` is N×K. `cov` is K×K, symmetric and positive semi-definite to within
  tolerance.
- `kind: dense` has `cov` N×N and no `weights`.
- `scope` defaults to `"dataset"`. `provenance` is informational only and never used in
  the fit.

**`sigma_meas` replaces the dataset's error column** for the weighted fit. The data file's
own error column is left as written, so no-covariance use is unchanged (§8.2).

### 7.2 Legacy loaders (bilayer workflow)

`*_corrected_cov.npz`, `*_c_factors.txt` + raw file, and the multi-dataset bundle
(`load_bundle_npz`, which yields one `DatasetCovariance` per dataset) map to:

- a multiplicative term `attenuator_cal`, `scope: shared`, with one-hot weights from
  `k_index` and `cov = Sigma_logc`;
- if `v`/`var_fwhm` are present, a **multiplicative** term `beam_fwhm`, `scope: shared`,
  with W = v / y_corr (N×1) and Σ = [[var_fwhm]] **[call]**:
  - `v` is the per-point sensitivity of the corrected R to the beam FWHM. The footprint
    correction is a division, so v = R·∂ln g/∂FWHM, which makes it multiplicative.
  - In `"data"` mode this reproduces today's var_fwhm·vvᵀ exactly. In `"model"` mode it
    removes the data-based bias, as for the attenuator term.
  - This assumes v was computed as stated. If a builder finds otherwise, keep it additive
    with W = v.
- `replaces_y = y_corr`, keeping today's behaviour that the sidecar supplies the corrected
  data.

The existing both-or-neither check on `v`/`var_fwhm` stays.

### 7.3 Future combined format

A data loader that returns `(x, y, yerr, DatasetCovariance | None)`. When it returns a
covariance, the dataset is attached automatically. Nothing downstream of
`DatasetCovariance` changes. The by-hand loading path stays.

## 8. LSS_Reader: what must change

Everything below refers to `mainwindow.py` in `chemmatcars/LSS_Reader`. The Pilatus
reflectivity patch path is `pilRefPatchData` → `pilRefPatchNext` → `oneDjoin(...,
mtype='merge')`, followed by the normalisation in `updateRefPlotData` and `saveRefData`.

### 8.1 What it does today

- Scans are sorted by qz and joined in sequence.
- In each overlap a cubic spline is drawn through the new scan. A constant `fac` is
  fitted to the ratio of the **already-merged, already-scaled** existing data to that
  spline, using the existing data's points in the overlap (`overlap1`; `curve_fit`,
  default `absolute_sigma=False`). `facerr` is √pcov. With `absolute_sigma=False`, pcov
  is rescaled by the reduced χ² of the ratios, so `facerr` reflects only their scatter.
- The new scan is scaled by `fac`. Its errors become √(σ²·fac² + y²·facerr²), which
  **folds in that join's scale error**.
- Overlap output points sit on the existing data's qz grid. Each is the average of the
  existing point and the spline of the scaled new scan, with error √(e₁² + e₂²)/2. The
  new scan's own overlap points are discarded.
- After patching, the curve is divided by a normalisation (the maximum below qc, or a
  user factor). `normerr = 0`. `_rrf.txt` additionally divides by R_F and shifts q by
  `qoff`.
- `_ref.txt` / `_rrf.txt` are 3-column `%.4e` text files.

### 8.2 Separate the measurement-only error

Carry a second error column, `σ_meas`, through the whole patch, normalisation and export
chain, alongside the existing total error:

- Scaling the new scan: σ_meas ← σ_meas · fac, **without** the facerr term.
- Averaged overlap points: σ_meas ← √(σ_meas,1² + σ_meas,2²) / 2.
- Normalisation: divide σ_meas by the same factor.
- Fresnel division (`_rrf`): divide σ_meas by R_F.
- The identical-scan merge (`pilRefSSMerge`), which happens before patching: combine as
  it does today. Those points stay in one region.

The existing total-error column is **unchanged**, so `_ref.txt` / `_rrf.txt` stay
byte-identical and every current workflow is unaffected. XModFit uses σ_meas only when a
covariance is attached.

### 8.3 Fit each factor against measurement-only errors

In `oneDjoin(mtype='merge')`, compute `facerr` with the following estimator
(**decided by the user, 2026-11-10**: inflate-only):

- Fit with ratio errors from **σ_meas**, using `absolute_sigma=True`. That gives
  δ_counting, the uncertainty that counting noise alone implies.
- If the reduced χ² of the ratios is above 1, inflate by √χ²_red (a Birge/PDG-style
  scale factor): facerr = facerr_abs · max(1, √χ²_red).

Why:
- `absolute_sigma=False` alone can make `facerr` far *smaller* than counting noise when
  χ²_red < 1, which is common with 3–6 overlap points. That shrinks Σ towards zero and C
  towards singular.
- `absolute_sigma=True` with the total errors would count the earlier joins' scale errors
  twice: once in those errors, and again through the chaining in §4.3.
- The inflation keeps the conservative behaviour when the overlap genuinely disagrees.

Why χ²_red is usable here, when it is not a safe variance estimator for the structural fit:
- The overlap fit has one parameter and is linear in it (the scan is multiplied by
  `fac`), so the degrees of freedom are exactly n − 1. The nonlinear-model problem with
  counting degrees of freedom does not arise.
- The "one common error scale" assumption is only made locally: a few adjacent points
  from one pair of scans, under the same counting process.
- Inflate-only bounds the damage of a broken assumption. A point with an underestimated
  error, or unmodelled disagreement, can only widen `facerr`. The estimator can never
  report less than counting noise.
- Residual caveat: the spline interpolation of the merged data onto the new scan's grid
  correlates neighbouring overlap points (§8.6), so the effective degrees of freedom are
  somewhat below n − 1. This makes χ²_red noisier but does not bias the floor.

### 8.4 Track per-point region weights

Carry an N×K weight matrix alongside the merged data, with K = number of scans in patch
order:

- Points of scan 1 get row e₁. Points taken from scan k alone get e_k.
- An averaged overlap point gets 0.5 · w(existing point) + 0.5 · e_k.
- Points dropped in the patch dialog (`pilRefPatchDrop`) drop their rows.
- After patching, every row sums to 1.

### 8.5 Build the region covariance

For every join (m = 2..K), record:
- `fac_m` and `facerr_m`;
- a_m, the mean of the weight rows of the existing-data points used in that join's factor
  fit (`overlap1`), taken **after** any points dropped in the patch dialog are removed.

Compute δ_m = facerr_m / fac_m and build Σ one join at a time with the §4.3 recursion, in
natural-log units. Scan 1, the anchor, has zero variance. The
normalisation is a global factor applied after patching, with `normerr = 0`. Its
uncertainty is not a term in v1 **[call]**, because it is equivalent to a free overall
scale, which fits normally either have or fix deliberately.

### 8.6 Known approximation (not addressed in v1)

The spline interpolation makes each averaged overlap point a combination of several
neighbouring new-scan points. Nearby overlap points are therefore slightly correlated
through counting noise. This is ignored, and the sidecar documents nothing for it.

### 8.7 Export

- Next to `<name>_ref.txt`, write `<name>_ref.cov.json`. Next to `<name>_rrf.txt`, write
  `<name>_rrf.cov.json`.
- Each sidecar is built from **exactly the arrays written to its data file**: the same x,
  y and σ_meas after every transform.
  - The `_rrf` file uses the qoff-shifted q.
  - There is an existing aliasing issue to fix. `self.RrfData` is the same array as the
    plotted `data`, so with the Qz² checkbox on, its x is squared in place, and the saved
    `_rrf.txt` can hold q². Copy the array before the Qz² plot transform.
- Both files carry the same stitching term.
- `x` and all numbers are written at full precision (JSON `repr`), not `%.4e`.
- For a single scan, or for the non-patch (Bruker / point-by-point) path, a sidecar is
  optional. If written, it has no terms.

### 8.8 Acceptance checks for the LSS_Reader change

These are run in the LSS_Reader repo on synthetic scans with known factors:

1. With the change, `_ref.txt` / `_rrf.txt` are byte-identical to before, apart from
   the q² aliasing fix, which affects only runs with the Qz² box checked.
2. σ_meas ≤ total error at every point, with equality on scan 1.
3. Every weight row sums to 1. Overlap rows are 0.5 / 0.5.
4. When every overlap lies only on the previous scan, diag(Σ) equals the cumulative
   Σ δ², and Σ[k,l] = Σ[min(k,l), min(k,l)]. Otherwise Σ follows the §4.3 recursion.
5. The sidecar's x and y equal the written data file's columns.
6. The sidecar loads in XModFit without warnings.

## 9. Testing (XModFit)

All tests are headless and run through `tests/run_all.py`. Each new check must be shown
to fail when its feature is reverted, as was done for WS0/WS1.

1. **No change without a covariance.** The full suite passes unchanged. The existing
   no-cov harnesses are the guard.
2. **Bilayer reproduction.** Run with legacy loaders in `"data"` mode, the existing GLS
   harnesses in this repo must reproduce their current numbers to 1e-12. Those are
   `validate_joint_gls.py` and `validate_joint_mask.py`, both on synthetic data. The real
   bilayer campaign runs on its own vendored engine, and the real-data validator was
   removed (audit A2.5), so no real-data rerun is required here. Covered as well:
   - a direct check that the new assembly, in data mode, reproduces
     `build_full_covariance` element-wise on a bundle-format fixture;
   - the `beam_fwhm` multiplicative conversion (§7.2).
3. **Peelle check.** On the textbook two-point case, `"data"` gives 0.882. `"model"`
   gives ≈ 1.154. With a fully common term, GLS reduces to the mean weighted by the
   independent errors, 0.15 and 0.10.
4. **Synthetic stitching recovery.**
   - Simulate a slab-model curve and split it into K overlapping scans.
   - Draw chained scale errors, and reduce with a Python re-implementation of the §8
     merge to produce y, σ_meas, W and Σ. Like LSS_Reader, it fits each factor against the
     already-merged data. Include a case where one overlap reaches back past the previous
     scan, so the §4.3 recursion is exercised.
   - Fit many seeds with and without the covariance.
   - In `"model"` mode, parameter pulls must have mean ≈ 0 and SD ≈ 1. The covariance-free
     fit must show the over-confidence the covariance removes.
   - Log-scale fits of the same data agree with linear `"model"` fits to within the
     first-order approximation.
5. **Single = one-dataset joint.** For the same data and covariance, `Fit` and a
   one-dataset `JointFit` give identical parameters and χ².
6. **Mixed joint.** One dataset with a covariance and one without, and no shared terms.
   - There must be no crash.
   - In Linear, χ² equals the sum of the two separate fits.
   - In Log, the same: the dataset without a covariance contributes its standalone Log χ²
     (§4.5).
7. **GUI single-fit path.** `set_x` without `cov` clears a covariance, and
   `update_plot`'s pre-sliced call with `cov.subset` gives the same χ² as the fit.
8. **Shared terms.** Two legacy datasets sharing `attenuator_cal` reproduce today's
   cross-block C. Two datasets with private stitching terms produce a block-diagonal C.
9. **Masks.** C assembled on the masked set equals the full C with rows and columns
   removed. A mask change rebuilds the factorisation.
10. **Jacobian.** The whitened analytic Jacobian equals the finite difference of the
   whitened residual, in both scales.
11. **Every §6.5 row:** the fit completes, the warning is present, and the report says so.
12. **Loader validation.** A hand-made v1 fixture loads. Each rule in §7.1 rejects a
    fixture that breaks it, with a clear message.

## 10. Relationship to other work

- **WS2:** builds the per-dataset χ² decomposition on the whitened residual. With shared
  terms that decomposition is not additive, and WS2 must state how it splits. The §4.6
  tension diagnostic is a WS2 optional item.
- **WS3:** the library entry stores the covariance file path, the mode and the downgrade
  warnings.
- **WS4:** the full covariance GUI.
- **bilayer-jointfit-analysis:** that campaign runs on its own vendored engine and is not
  changed by this. Whether its data-based fits should be redone in `"model"` mode is a
  scientific question for its owner, outside this spec.

## 11. Left to the builder

- Internal names and module layout beyond §6.1.
- How the outer iteration is wired into `perform_fit` for each fit method.
- Sparse versus dense storage of W.
- The exact wording of warnings and report lines, within §6.6.
- Plan granularity.

## 12. Calls made in this document (for the user to check)

1. Model-based covariance is the default in Linear. Data-based is opt-in for reproduction.
2. `w/o error` scales ignore an attached covariance, with a warning.
3. Dict-x (multi-curve) functions ignore an attached covariance, with a warning.
4. MCMC freezes C at the best-fit model, or uses data-based when no prior fit exists.
5. Conflicting shared terms degrade to private terms with a warning instead of refusing.
6. A JSON sidecar named `<name>_ref.cov.json` / `<name>_rrf.cov.json` is the v1
   interchange format. It carries y as well as x, so that mismatched pairings are caught.
7. **LSS_Reader's `facerr` estimator (§8.3), decided by the user 2026-11-10:**
   counting-noise uncertainty from σ_meas (`absolute_sigma=True`), inflated by √χ²_red
   when the overlap scatter exceeds it (inflate-only). This sets the size of Σ, so it is the largest single effect of the
   workstream.
8. The LSS normalisation uncertainty is not a term in v1.
9. A minimal single-fit GUI entry point (load/remove covariance, a marker, the mode
   option) is in scope, as an exception to the GUI deferral. The joint panel's existing
   picker is reused.
10. Outer iteration stops below 0.1 σ of parameter movement, or after 5 rounds.
11. Saved fits without a mode field reload in `"data"` mode.
12. The legacy `beam_fwhm` term is converted to multiplicative (W = v/y_corr).
13. (Superseded by the Log fix in 64babd9: every Log fit now uses correct log statistics.)
14. The covariance lives with the dataset, and `set_x` without `cov` clears it.
15. The fallback without a covariance: for v1 it is exactly today's fit, and for legacy
    files it is `y_corr` with σ_meas.
16. With a covariance, the profile confidence interval uses χ²_min + Δχ²(CL, 1 dof), and
    a χ²_red warning threshold is left to the builder (§6.2).