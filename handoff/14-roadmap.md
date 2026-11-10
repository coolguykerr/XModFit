# Roadmap: open issues, joint-fit reporting, and a joint-fit library

**Written 2026-11-02.** Prioritised plan for the work left open in `10-fix-list.md`,
`01-joint-fit.md` §6 and `11-audit-plan.md`, plus two things requested on 2026-11-02:

1. **The joint-fit best practices worked out in `../xpr-seeding-study`
   (`PLAN_JOINT_POSE.md`, `reports/JOINT_FIT_PLAN.md`) become what the program reports.**
   A number the user should not read directly (a raw Δ(−logL) between two joint fits, a
   bare AIC) is replaced on screen by the number they *should* read, computed for them.
2. **A joint-fit library**: store, list, reload and compare many joint fits of the same
   datasets — typically differing in which parameters were fixed, freed or shared.

This is a roadmap, not an implementation plan. Each workstream (WS) gets its own
task-by-task plan (`superpowers:writing-plans`) when it is picked up. WS2 and WS3 are new
designs, so their data model and interfaces are fixed here; the rest point at the fix-list
items that already specify them.

**Priority order:** WS0 → WS1 → WS2 → WS3 → WS5 → WS6 → WS4 → WS7 → WS8. WS0 blocks
everything. WS2 and WS3 can be built in parallel once WS2's core (§2.1–2.2) exists,
because WS3's comparisons call it.

**GUI work is deferred** (user, 2026-11-02). WS2, WS3, WS5 and WS6 are built and proven
headless and from scripts first. Their only user-visible output until then is text in the
existing fit report, via `JointFit._report()`, which needs no GUI change. Every
panel, table, button and dialog — WS2's buttons and comparison-text rebuild, WS3's
library panel, and all of WS4 — is collected into **WS4, a GUI redesign** done once the
features behave as wanted.

## Global constraints (apply to every workstream)

- Interpreter: `C:/Users/kerr/.conda/envs/cmc-py39/python.exe`. Headless: `QT_QPA_PLATFORM=offscreen`,
  stub `sys.modules['xraydb']`, `if __name__ == '__main__':` guard around any spawn Pool.
- New statistics and storage code is **Qt-free** and lives in repo-root modules next to
  `joint_covariance.py`, so it is testable headless and reusable from scripts.
- The §5a invariant stays: `JointFit.residual` branches only on "covariance attached?",
  never on "parameters shared?".
- Do not touch `julia/bakeoff/` or any path in `.githooks/pre-push`.
- Every new number shown to a user states **what objective it was computed from**
  (fit scale, covariance on/off, error rescaling on/off).
- `−logL = χ²/2` holds only when the minimised objective is a Gaussian χ²: scale `Linear`
  (with errors) or the covariance/GLS path. Under `Log`, `Log w/o error` or
  `Linear w/o error` the program reports the minimised objective by its own name and does
  **not** call it a likelihood.

---

## WS0 — Make the work verifiable (blocking)

Fix-list Phase 0 (`12-work-package-preaudit.md`), narrowed so it covers the joint path
first.

- **P-1 / A1.10** `check()` raises or counts failures, so a failing check fails the run.
- **P-2** one plain-script runner over every `validate_*.py` and `tests/recovered/*`
  (`pytest` is not in `cmc-py39`). Prints per-file PASS/FAIL and exits non-zero on any failure.
- **B2c + A2.5** synthetic cross-dataset GLS regression: extend `validate_joint_mask.py`
  with the cases `validate_jointfit.py` claimed (whitening identity with the footprint term
  on; N=1 no-cov ≡ stock `Fit`; 3-dataset GLS where cross-dataset blocks change the stderrs;
  two datasets with *different* masks and a non-trivial off-diagonal `Sigma_logc`). Then
  retire `validate_jointfit.py` and the El_Cat docstring examples (D2).
- **Baseline**: run everything and record the results in the work-package progress log.
- **Housekeeping, while here:** revert `.idea/` (A0.5); decide what happens to the 11
  uncommitted `julia/bakeoff/` edits (confirm they are in `../xpr-seeding-study`, then
  discard them here).

**Done when:** the runner exits 0 on a clean tree, with known failures listed explicitly,
and the joint engine has a synthetic GLS regression that does not need off-repo data.

**Status: Done 2026-11-04; baseline in `12-work-package-preaudit.md`.**

## WS1 — Joint-fit correctness (wrong results and crashes on the joint path)

Each item is specified in the fix list. All are small.

| item | what | tag |
|---|---|---|
| A1.4 | `_eval_chisqr` ignores per-dataset ranges and masks. **Superseded by WS2 §2.1**, which replaces it; until then route it through `imask`/`active_index` | wrong-result |
| A1.3 | seed workers ignore the frozen per-column mask | wrong-result |
| 01 §6.8 | a share-group silently takes value/bounds/vary from its **first** endpoint. Warn (list the disagreeing endpoints), and refuse when bounds disagree unless the user confirms | wrong-result |
| 01 §6.7 | `Linear w/o error` weights datasets by `1/max(data)`. Either remove the factor in the joint no-cov branch for that scale, or warn when it is selected for a joint fit | wrong-result |
| 01 §6.14 | `_primary_curve` uses only the first sub-curve. Warn when a dataset has more than one | silent |
| A1.1 | one module-level PDB model-name tuple, imported by the three sites (D1 has shrunk it to three names) | crash |
| B2a | `load_cov_npz` keeps `v`/`var_fwhm` when present, and `load_dataset_cov` returns them, so the footprint term reaches any sidecar-based fit (scripted now, GUI later). A loader fix, no GUI change | unreachable |
| B4 | `doFit` routing / mixed selection: **moved to WS4**, as it is GUI control flow | fragile |

**Done when:** each item has a regression check in the WS0 runner.

**Status: Done 2026-11-06 (A1.1, A1.3, A1.4, A1.17, B2a, 01 §6.2/3/6/7/8/14, plus A1.18 found during execution); full-run table in `12-work-package-preaudit.md`. Full run: 60 PASS, 0 FAIL, 0 KNOWN-FAIL, 1 TIMEOUT (`validate_xpr.py` at the 600 s default under a slow machine; PASS alone in 622 s).**

**Audit gaps closed 2026-11-09** (`10-fix-list.md` §A1a, AG.1-AG.8: Qt-slot tracebacks now fail a row, A1.4 call-site guard, W0.3 smoke rows given checks, §6.14 orchestration warning, N2 share-group keymap, N3/N4/N5). Full run at `3b58055`, one uninterrupted run: **61 PASS, exit 0** (`tests/_logs/gapfix_full_run.txt`).

## WS2 — The reporting contract: statistics the program computes for the user

> The general covariance workstream (docs 17 and 18) precedes WS2.

New module **`joint_fit_stats.py`** (Qt-free). It consumes a fitted `JointFit`. Every
function returns plain data plus a `format_*` text renderer, so the GUI, the `.jpar` and
scripts share one implementation.

### 2.1 Per-dataset decomposition (the core; WS3 depends on it)

```
per_dataset_objective(jf) -> list[DatasetTerm]
DatasetTerm: name, n_points, objective (float), chi2_red, window (xmin, xmax),
             mask_hash, scale, cov (bool), rescale (float|None)
```

- **No covariance:** the term is the dataset's slice of the residual vector actually
  minimised (same window, same mask, same scale), squared and summed. The terms sum to
  `result.chisqr` exactly; that identity is the test.
- **With covariance:** use the additive Mahalanobis split `t_i = r_iᵀ (C⁻¹ r)_i` over each
  dataset's rows. It sums exactly to the GLS χ². A single term can be negative, which is
  honest under correlation. See **Decision 2**.
- Replaces `_eval_chisqr`. That function evaluated a plain χ² on the scalar window, which
  is not the objective the fit used.

### 2.2 Comparability fingerprint

```
comparability_key(jf) -> dict  # datasets (path + hash of x,y,yerr over active points),
                               # per-dataset window + mask hash, fit scale, covariance
                               # source + hash, error-rescaling mode, model class, PDB file
```

Built on the same idea as `xmodfit._mcmc_fingerprint` (`xmodfit.py:2871`). Two fits are
compared only if their keys agree on everything except parameter state. Otherwise the
comparison is **refused**, naming the field that differs (e.g. "dataset 3 window 0.03:0.45
vs 0.03:0.50"). This is the guard that makes the calibrated number in §2.3 meaningful.

### 2.3 Calibrated comparison between two fits — the number to read instead of Δ(−logL)

```
compare_fits(a: FitRecord, b: FitRecord) -> Comparison
Comparison: d (per-dataset Δ, list), sum_d, sd_d, N, z = |sum_d| / (sd_d * sqrt(N)),
            delta_k (free-parameter difference), nested (bool),
            worst_dataset (name, d_i), objective_label
```

- `d_i = ½(t_i^A − t_i^B)` when the objective is a χ² (so `Σd` is Δ(−logL)). Under other
  scales it is the raw objective difference, labelled as such.
- **Headline shown to the user:** `Δ = Σd ± sd(d)·√N  (z = …)`, plus which fit is
  preferred and by how many datasets (sign count, e.g. "B preferred on 14/17").
- The bare Δ(−logL) is shown only in a secondary detail line, labelled "not a confidence
  measure".
- **Small-N rule:** with N < 4 datasets, `sd(d)` is not estimable usefully. Show the
  per-dataset `d_i` table and say that no calibrated z is available. Do not invent one.
- **Nested fits** (B is A with some parameters fixed or shared): also show Δk. A
  likelihood-ratio p-value is shown only with the caveat that the model is likely
  misspecified (`PLAN_JOINT_POSE.md` §4.1). The honest check is WS2 §2.6 LOO.

### 2.4 Per-dataset fit quality and compromise detection

For every joint fit, a table with one row per dataset:

- χ²_red;
- that dataset's own **individual** fit χ²_red (from `dataset_fits`) when present, over the
  same window and mask;
- the cost of sharing, `Δ_i = joint − individual`.

Flag rows where `Δ_i` is large relative to the corpus spread. This makes "a shared value
that fits nothing well" visible (hazard 4).

### 2.5 Diagnostics computed at the end of every joint fit

Shown in the fit report under a "Joint-fit diagnostics" heading:

1. **q-range spread.** If `max(qmax)/min(qmax) − 1` exceeds a tolerance (default 10%), say
   so and name the mode in use: *native windows* (the default, now labelled), or *common
   window* (one switch: intersect the windows). Anchor-and-verify is a workflow (§2.6), not
   a mode.
2. **Fisher information per dataset for each shared parameter.** Compute
   `I_ij = Σ_q (∂f/∂p_j)² / σ²` from the Jacobian the fit already has: lmfit `fjac` / a
   finite-difference column per shared master param, or the analytic one once WS5 lands.
   Show the table and each dataset's share of the total. This tells the user which
   datasets actually determine the shared parameters; point count and qmax are not a
   substitute.
3. **Error-bar heterogeneity.** The spread of per-dataset χ²_red. If max/min exceeds 3,
   offer — never default to — **per-dataset error rescaling** by `√χ²_red,i` from the
   individual fits. When it is on, it is recorded in the fingerprint and printed on every
   number (it hides misfit; `PLAN_JOINT_POSE.md` §3.4).
4. **Shared-parameter consistency.** The WS1 first-endpoint check, restated in the report.
5. **Degeneracy.** Correlation pairs above |0.9|, and the softest eigenvector of `JᵀJ`
   named in parameter terms (e.g. "d_HGi + 0.8·d_wat"). This is cheap and catches the
   bilayer-type degeneracies.
6. **What was minimised.** The objective label from the global constraints.

### 2.6 Optional heavier checks (functions now; buttons later, in WS4)

- **Leave-one-out joint** (arm J3): N refits, each omitting one dataset, warm-started from
  the full fit. Report the spread of each shared parameter and each dataset's influence.
  This is the honest uncertainty for shared parameters.
- **Anchor-and-verify**: choose an anchor subset **by Fisher information** (§2.5.2; the UI
  must not offer selection by fitted outcome). Fit shared parameters on the anchor, hold
  them, fit the rest with nuisances only, and score each held-out dataset against its own
  individual fit.
- **Sign-agreement pre-check** before grouping: for a proposed share-group, show whether
  the candidate subgroups individually prefer the same answer (`JOINT_FIT_PLAN.md` §2).

### 2.7 Replace the current model-comparison text (GUI side moves to WS4)

`computeModelComparison` (`xmodfit.py:7556`) currently headlines AIC/AICc/BIC on plain χ².
WS2 provides `format_joint_vs_individuals(jf, individual_fits) -> str` in
`joint_fit_stats.py`. Swapping the panel's text over to it is a WS4 item. The new
text is built on §2.1–2.4:

- **Headline:** the joint-vs-individuals calibrated comparison and the §2.4 table.
- **Moved down:** AIC/BIC, under "information criteria (assume a correct model; read with
  caution)". See **Decision 3**.

**Done when:** each §2.1–2.5 quantity has a headless test against a hand-computed synthetic
case. For example: two synthetic fits with a known per-dataset Δ must reproduce `Σd`, `sd`
and `z` exactly; the GLS decomposition must sum to `chisqr` to 1e-10; a refused comparison
must name the differing field.

## WS3 — Joint-fit library: store, list, reload, compare

### 3.1 Data model

```
FitRecord:
  id (uuid), name (user-editable), created (ISO time), notes (str)
  spec_jpar (str)          # the full .jpar text from joint_fit_io.format_jpar — the
                           # restorable definition, incl. sharing, fixed params, windows
  key (dict)               # comparability_key, WS2 §2.2
  result:                  # what lmfit returned, so nothing must be refit to compare
    method, scale, success, message, nfev, chisqr, redchi, ndata, nvarys
    params: {master_key: (value, stderr, vary, min, max, expr)}
    var_names, covar (list[list] | None)
    terms: list[DatasetTerm]                     # WS2 §2.1
    diagnostics: dict                            # WS2 §2.5 outputs
  parameter_state summary: per master key -> 'free' | 'fixed' | 'shared(n)' | 'expr'
```

`parameter_state` is what makes "which parameters were fixed in this one" a table column
rather than something you reconstruct from two `.jpar` files.

### 3.2 In memory first, disk second — comparing never requires a reload or a refit

Individual fits today live in memory as live objects: `self.dataset_fits`
(`xmodfit.py:541`) maps dataset name → `Fit`, each carrying its own model, parameters and
`result`. Nothing is written to disk unless a `.par` is saved. Joint fits currently have
only one slot, `self.jointFit`, which the next joint fit overwrites.

The library works the same way, with one addition:

- **In session**, each entry holds the **live `JointFit` object** (models, `master_params`,
  `result`) exactly as `dataset_fits` does. Viewing, re-plotting or continuing a stored fit
  is a pointer swap.
- **Each entry also carries the `FitRecord` snapshot** of §3.1, computed once when the fit
  finishes: parameter values and stderrs, covariance, per-dataset terms, fingerprint and
  diagnostics. **Comparisons read only the snapshot.** `compare_fits(a, b)` touches no
  model, so comparing fits loaded from disk costs the same as comparing fits made this
  session: no reload, no re-evaluation, no refit.
- **On disk** (across sessions), the snapshot is written as JSON beside the `.jpar`. The live
  object is not pickled: `JointFit` is a `QObject`, and B1 is why that is not attempted. On
  reopening a library, the snapshots load immediately, so the list and every comparison
  work. The live `JointFit` for an entry is rebuilt from its `.jpar` lazily, only when the
  user wants to view curves or refit it.
- Re-evaluating a stored fit is needed only to compare it on a **different** point set,
  and the fingerprint refuses that by design. If it is ever wanted, it is an explicit
  "re-score under the current windows" action that produces a *new* record, never a silent
  overwrite.

### 3.3 Storage backend

New module **`joint_fit_library.py`** (Qt-free):

```
FitLibrary(path=None)                  # path=None: in-memory only until saved
  .add(jf, name, notes='') -> id       # snapshots jf into a FitRecord, keeps jf live
  .live(id) -> JointFit                # rebuilt from .jpar on first call if not in memory
  .get(id) -> FitRecord   .list() -> list[FitSummary]
  .update(id, name=None, notes=None)   .remove(id)   .export(id, path)
  .compatible_with(key) -> list[id]    # records comparable to a given fit
```

Layout (recommended, see **Decision 1**):

```
<library>/index.json          # summaries only; rebuilt by scanning if missing or stale
<library>/<id>.jpar           # opens in the existing Load Joint Fit path unchanged
<library>/<id>.result.json    # FitRecord minus spec_jpar
```

Every record is a normal `.jpar`, so a library degrades to a folder of loadable files. It is
diff-able and needs no new dependency. Writes are atomic (temp file + rename). A record
whose dataset hash no longer matches the data on disk is listed as **stale**, not silently
compared.

Default location: next to the first dataset, as `<first-dataset-stem>.jfits/`, changeable
from the panel. Not under `Data/` in this repo, which `.gitignore` treats as test output.

### 3.4 GUI: a "Fit library" section (specified here, built in WS4)

- **Table columns:** name · date · #datasets · #free · #shared · differences from the
  selected reference (e.g. "+fixed: d_wat; −shared: protins") · χ²_red · objective label ·
  stale flag.
- **Store current fit** (prompts for a name). Optionally auto-store every completed joint
  fit as "unsaved", with a cap.
- **Load** restores spec and parameter values through the existing `loadJointFit` path.
- **Compare**:
  - 2 selected → WS2 §2.3 report;
  - ≥3 selected → ranked table against a chosen reference, one calibrated row per fit, with
    per-dataset `d_i` available on expansion;
  - incompatible selections are refused, with the reason.
- **Parameter diff** for two records: side by side, master key by master key, with value ±
  stderr and state (free/fixed/shared).
- Rename / notes / delete / export.

### 3.5 Scripting — the primary interface until WS4

`FitLibrary` and `compare_fits` are driven from a script running `JointFit` headless (the
`run_fit.py` style), so batch comparisons of "which parameters to fix" work before any GUI
exists. A `format_library_table(lib, reference=None) -> str` renderer gives the ranked
comparison as text.

**Done when:** headless tests round-trip a record (store → list → reload from disk →
identical snapshot, and `live(id)` reproduces `master_params` and spec), detect staleness
after a data edit, refuse an incompatible comparison, and reproduce a known `compare_fits`
result from snapshots alone, with no model evaluation.

## WS4 — GUI redesign (deferred: after WS2, WS3, WS5, WS6 behave as wanted)

One redesign rather than piecemeal panel edits. It collects:

- **from WS2:** the comparison text swapped to `format_joint_vs_individuals`; diagnostics
  shown in a structured view rather than only as report text; buttons for
  LOO / anchor-and-verify / sign check, run as background jobs with progress.
- **from WS3:** the library panel (§3.4), replacing the single `self.jointFit` slot with
  the library's in-session store.
- **the existing reachability gaps:**

| item | what |
|---|---|
| B3 | **subset sharing**: replace the share-across-all checkbox with a per-parameter scope control: *per dataset* · *all* · *subset…* (pick datasets). Plus a heterogeneous-key mode for "same physical layer at different rows" (`set_shared_endpoints`). The engine, spec and `.jpar` already support all of this |
| — | **priors** (`set_priors`) editable per master parameter, stored in `.jpar` and in library records |
| skill "stage 2-ii" | **verify** that editing a *shared* parameter in the stock tables fans out to every endpoint; fix if not |
| — | live per-iteration plotting under a joint fit (`fitCallback` is `Fit`-shaped) |
| 01 §6.12 | (deferred by user) fold the overlay controls into the fit-options panel |

## WS5 — Engine generalisations taken from the XPR work

In order of value to joint fits:

1. **Joint analytic Jacobian.** `JointFit.residual_jac` stacks each objective's `dydp`
   columns through `param_map`, summing columns for shared masters. It applies the same
   per-curve normalisation as `residual`, and under GLS returns `L⁻¹J`. Gate it on
   capability (duck-typed `dydp`/`parameter_columns`), keep the measured `nvarys ≥ 4` gate,
   and keep `expr` unsupported. Feeds WS2 §2.5.2 exact Fisher information.
2. **Generic derived-report keymap.** Pass each objective's full `{local_key: master_key}`
   to `derived_report`, instead of the hard-coded protins/theta/phi. (The three-key keymap
   was also *wrong* for spec/GUI share-groups; fixed 2026-11-09 as audit gap AG.5 by reading
   it off `param_map`. What remains here is the generalisation beyond the three keys.)
3. **Profile likelihood** per master parameter: pin one, refit the rest by continuation,
   and report the profile alongside the Hessian stderr.
4. **Hessian certification** of a joint endpoint (MIN / SADDLE / DEGENERATE, bound-aware,
   free block only) using Gauss-Newton `JᵀJ`. The classifier in `fastscan/modes.py` is
   already model-free.
5. **Pooled-endpoint joint search** (`JOINT_FIT_PLAN.md` §5.2): pool per-dataset scan
   endpoints, cluster by position only, evaluate the joint objective at each cluster centre
   with each dataset warm-started from its own endpoint, and polish the top few. This
   replaces a direct multistart on the joint objective. It depends on the fast scan
   accepting the engine's objective; the measured budget is 64 starts with K ≈ 18.

## WS6 — Joint MCMC completion

- **B2b** a whitened `JointLogProb`, so covariance/GLS joint MCMC runs in parallel and
  gets orientation seeding.
- **B7** `_seed_shared_axis` calls basin detection; plumb or delete `seed_uniform_frac` /
  `walkers_per_basin`; unify the two spawn pools.
- **Reporting:** a joint MCMC on shared parameters prints the WS2 §2.6 LOO spread next to
  the posterior width when one is available, and warns when none is
  (`PLAN_JOINT_POSE.md` §4.4: a narrow posterior can be narrow and wrong). Store MCMC
  summaries in library records.

## WS7 — The rest of the fix pass, the audit, and readiness to push

- **Remaining Tier A:**
  - A1.13 waters/heteroatoms counted as protein;
  - A1.14 re-enable the centre-of-mass step;
  - A1.15 viewer protins reference for XPR;
  - A1.2 `mkey` in `_box`;
  - A1.5–A1.9, A1.11, A1.12;
  - A2.1/A2.3 data cleanup;
  - A3 dead code (incl. retiring `joint_fit_orchestration.py` and `xmodfit_old.py`);
  - A4 merge-guard comments.
- **B8/D1** retire `_G`, `_H`, `_dev_smooth`. **B13** a PDB preparation entry point. **B9**
  rebase `XLayers_Beamfootprint` on current `XLayers` with a fixture.
- **Audit** the remaining claims in `11-audit-plan.md`. The 14 joint claims of `01` §7 are
  covered earlier by WS0/WS1.
- **Fast scan**: corpus acceptance runs, a live GUI run, the N1/K question at 8× solve cost.
- **Publishing:**
  - drop `7942458` (route B), after `sha256sum -c` in `../xpr-seeding-study`;
  - attribution for `ngl.js` and `julia/XPR/` (A3.15);
  - the **GenX / refnx prior-art check** before Axis 2 is described as novel;
  - reorganise and condense, then push (D4).

## WS8 — Low priority

- **B1** picklable objectives for arbitrary models at cores > 1.
- **B5** GID2D fit-and-interpret, the `window='engine'` question, `GID2DMapDialog`.
- **B6** structure-viewer improvements (A1.15, a wrong result, is in WS7, not here), NGL
  dataset dropdown for joint fits.
- **B10/B11** XPR stage 2/3 and clean-clone reproducibility of the Julia environment.
- **B12** upstream-owned items (dead `EMCEE_Widget.py`, `XFNTR_Plus` crash, `xraydb_bak/`).
  Raise these with the collaborators.

---

## Decisions needed before the workstream they block

1. **Library storage format (WS3).** Recommended: a directory of `.jpar` +
   `.result.json` + `index.json` (human-readable, every record loadable by the existing
   loader, no new dependency). Alternatives: one HDF5 file (`h5py` is already required for
   MCMC; compact, opaque) or SQLite (queryable; overkill at tens to hundreds of records).
2. **Per-dataset terms under GLS (WS2 §2.1/2.3).** Recommended: the additive Mahalanobis
   split, with z labelled "assumes dataset-level independence; approximate under
   cross-dataset covariance", and LOO offered as the check. Alternative: report no
   calibrated z for covariance fits at all.
3. **AIC/BIC (WS2 §2.7).** Recommended: keep them, demoted below the calibrated comparison
   with a caveat. Alternative: remove them from the display.
4. **Sequencing against D4.** D4 was fix package → audit → fast scan → reorganise → push.
   This roadmap puts WS2–WS4 (new joint features) **before** the full audit, with only the
   joint-path audit claims pulled forward into WS0/WS1. Confirm, or keep the audit first.
5. **Default q-range tolerance (WS2 §2.5.1)** and the χ²_red heterogeneity threshold
   (§2.5.3): proposed 10% and ×3. Both are user-settable.

## Review focus — failure modes no single workstream's tests own

- Comparing two library fits after the user edited a dataset file or changed a column
  limit: they must be refused or marked stale, never silently compared (WS3 staleness test
  + WS2 §2.2).
- A joint fit with 2–3 datasets: no calibrated z can be estimated, and the report must say
  so rather than print a z built on `sd` from two points (WS2 §2.3 small-N test).
- A `Log`-scale joint fit: no number may be labelled −logL or a likelihood (WS2 objective
  label test).
- Restoring a library record whose PDB template file has moved: `loadJointFit`'s
  missing-dataset prompt must cover the template too (WS3 reload test).
- Sharing a parameter whose endpoints have different bounds: warn before the fit, not after
  (WS1 §6.8 test).
