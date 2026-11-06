# WS1 — Joint-path correctness: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the known silent wrong-results and crashes on the joint-fitting path before
WS2 builds the reporting contract on top of it. The WS0 safety net must stay green
throughout.

**Architecture:** Small fixes in place, in the engine files that own each defect. One
new engine surface is introduced: **`JointFit.warnings`**, a list of plain-text warnings
collected while a joint fit is built and run, appended to the fit report. WS2's reporting
contract will extend it. Regression checks for every item go in one new validator,
`validate_joint_ws1.py`, except the seed-scan mask, which gets its own
`validate_seed_mask.py` because it needs a PDB model.

**Tech stack:** Python 3.9 (`cmc-py39`), numpy, vendored lmfit, PyQt5 offscreen.

**Spec:** `handoff/14-roadmap.md` §WS1. Item definitions: `handoff/10-fix-list.md` (A1.1,
A1.3, A1.4, A1.17, B2a) and `handoff/01-joint-fit.md` §6 (items 2, 3, 6, 7, 8, 14).
Baseline: `handoff/12-work-package-preaudit.md` "BASELINE (re-recorded 2026-11-04)".

## Global constraints

- Interpreter `C:/Users/kerr/.conda/envs/cmc-py39/python.exe`. Headless recipe:
  `handoff/09-validation-and-env.md` §3.
- **Before and after every task:** `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/run_all.py --quick`
  (~8.5 min). It must stay at **exit 0**, with no new FAIL and no new KNOWN-FAIL. The
  current KNOWN-FAILs are `smoke_fit` and `verify_jointmcmc` (A1.17); Task 7 removes them.
- New regression checks use `tests.harness` (`check`, `finish`). Register any new
  validator in `tests/run_all.py`'s `REGISTRY` only if it is slow or needs external data.
- The §5a invariant stays: `JointFit.residual` branches only on "covariance attached?".
- The no-covariance residual for scales `Linear`, `Log` and `Log w/o error` must not change
  numerically. `validate_joint_gls.py` G2 (N=1 equals stock `Fit`) is the guard.
- `nan_policy='raise'` everywhere. A mask drops points; it never writes NaN (CLAUDE.md rule 3).
- A mask is frozen once per `perform_fit` (CLAUDE.md rule 4).
- Merge cost: `xmodfit.py`, `Fit_Routines.py` and `lmfit/minimizer.py` are also edited
  upstream. Keep edits in those files minimal and local, with no reflowing.
- Do not touch `julia/bakeoff/`. No push, no history rewrite. Commit subjects carry the
  item ID. Trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- When an item is fixed:
  - mark it done in `10-fix-list.md`, with the date and commit;
  - update the matching `01-joint-fit.md` §6 item;
  - update any `§7` claim it settles, together with `11-audit-plan.md` (CLAUDE.md scope rule).

## Review focus

1. **Datasets seeded from their own individual fits** have different *values* for a shared
   parameter. That is normal and must only warn. Different *bounds* or *vary* must refuse.
   A rule that refuses on values would break every ordinary joint fit. → Task 3.
2. **A shared parameter whose bounds are ±inf on some endpoints and finite on others** must
   count as a bounds mismatch, compared with `inf == inf` semantics, not raise
   `TypeError` / `nan`. → Task 3.
3. **A column mask with `__masked_x__` models (GID2D)**, where `yfit` is already
   active-only. The seed-worker and chisqr mask code must not mask the fit twice
   (`yfit_masked` in `JointFit.residual`). → Tasks 5 and 6.
4. **A `.npz` covariance sidecar that has `v` but not `var_fwhm`, or the reverse.** It must
   fail loudly at load, not turn the footprint term half-on. → Task 2.
5. **Warnings must not pile up across re-fits:** `jf.warnings` is rebuilt by each
   `build_master_params` / `perform_fit`, not appended forever. → Task 3.

---

### Task 1: One PDB model-name list (A1.1)

**Files:**
- Modify: `lmfit/minimizer.py:94` — rename `_PDB_FUNCNAMES` to `PDB_FUNCNAMES` and keep the
  old name as an alias (`_PDB_FUNCNAMES = PDB_FUNCNAMES`), so its four internal uses are
  untouched.
- Modify: `xmodfit.py:2634` — `_PDB_MCMC_FUNCS = PDB_FUNCNAMES`, imported from `lmfit.minimizer`.
- Modify: `Fit_Routines.py:514` — `if funcname in PDB_FUNCNAMES:`.
- Test: `validate_joint_ws1.py` (create), section `T1`.

**Interfaces:**
- Produces: `lmfit.minimizer.PDB_FUNCNAMES: tuple[str, ...]`, the six current names. B8/D1
  (WS7) shrinks it to three when the variants are retired.

- [x] **Step 1: Confirm the three sites mean the same set.** `minimizer.py` uses it for angle
  wrapping, seeding and `CustomSampler`. `xmodfit.py:2658` uses it for backend choice.
  `Fit_Routines.py:514` uses it for the initial `fiterr` walker spread. If any site is found
  to intend a *different* set, stop and report it rather than unifying.
- [x] **Step 2: Write the failing check.**
  - `T1`: `Fit_Routines` and `xmodfit` (import `xmodfit` with the xraydb stub and an
    offscreen `QApplication`) both hold exactly `lmfit.minimizer.PDB_FUNCNAMES`;
  - `'XLayersPDB_XPR' in xmodfit.XModFit._PDB_MCMC_FUNCS`;
  - `'XLayersPDB_box'` is in the set `Fit_Routines` tests. Read it via a module constant
    the edit introduces (`Fit_Routines.PDB_FUNCNAMES`).
- [x] **Step 3: Run it.** `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" validate_joint_ws1.py`
  → FAIL.
- [x] **Step 4: Make the three edits.** Re-run → PASS. Run `--quick` → exit 0.
- [x] **Step 5: Commit** — `A1.1: one PDB model-name list, imported by all three sites`.

### Task 2: Footprint covariance term reachable from sidecars (B2a, 01 §6 items 2–3)

**Files:**
- Modify: `joint_covariance.py` `load_cov_npz` (~line 198) and `load_dataset_cov` (~line 275).
- Test: `validate_joint_ws1.py` section `T2`.

**Interfaces:**
- Produces: `load_cov_npz(path)` adds `"v"` (float array, length of `x`) and `"var_fwhm"`
  (float) to its dict **only when both keys are in the file**. If exactly one is present it
  raises `ValueError("<path>: footprint needs both 'v' and 'var_fwhm'")`.
  `load_dataset_cov(...)` passes `v` / `var_fwhm` through for the `.npz` route when present.
  The `_c_factors.txt` route is unchanged.

- [x] **Step 1: Write the failing checks (`T2`).** Into `tests/_logs/`, write three `.npz`
  files. They use the `validate_joint_mask.py` sidecar shape (`x`, `y_corr`,
  `sigma_meas`, `k_index`, `SF_LIST`, `Sigma_logc`) for 40 points.
  - **(a)** No footprint keys. `load_dataset_cov` must have neither `v` nor `var_fwhm`, and
    a 2-dataset `JointFit` built from it has `_use_footprint is False`.
  - **(b)** Both keys (`v = 0.3*exp(-x/0.5)`, `var_fwhm = 4e-4`). `_use_footprint is True`,
    and `jf._L @ jf._L.T` equals `build_full_covariance(..., v=..., var_fwhm=4e-4)` to
    relative `1e-12`.
  - **(c)** Only `v`. Raises `ValueError` naming the file.
- [x] **Step 2: Run it** → (b) and (c) FAIL.
- [x] **Step 3: Implement.** Re-run → PASS. Run `--quick` → exit 0.
- [x] **Step 4: Commit** — `B2a: .npz sidecars carry the shared-beam footprint term into the joint fit`.

### Task 3: Share-group mismatches and `JointFit.warnings` (01 §6.8)

**Files:**
- Modify: `Joint_Fit_Routines.py` — `build_master_params` (~line 330), `__init__`,
  `perform_fit` (to reset), `_report` (to print).
- Modify: `joint_fit_spec.py` `compile()` — pass `on_mismatch` through.
- Test: `validate_joint_ws1.py` section `T3`.

**Interfaces:**
- Produces:
  - `JointFit.warnings: list[str]`, reset to `[]` at the start of every
    `build_master_params` call.
  - `perform_fit` appends to it during the run, and `_report()` appends a block
    `"Joint-fit warnings:\n  - …"` when it is non-empty.
  - `class ShareMismatchError(ValueError)` in `Joint_Fit_Routines`.
  - `build_master_params(shared=None, groups=None, on_mismatch='raise')`, where
    `on_mismatch` is one of `'raise'` | `'first'`.
  - `JointFitSpec.compile(data, masks=None, on_mismatch='raise')`.
- Rule (Review focus 1–2). Within a share-group, compare every endpoint to the first:
  - **value differs** → one warning per group: `"shared <mkey>: endpoints start at different values (<ds>=<v>, …); using <first ds>=<v>"`.
  - **`min`, `max` or `vary` differs** → raise `ShareMismatchError` naming the master key
    and listing each endpoint's `(dataset, min, max, vary)`, unless `on_mismatch='first'`.
    In that case, append a warning saying the first endpoint's bounds were used.
    Compare `±inf` by equality; never `nan`.

- [x] **Step 1: Write the failing checks (`T3`).** Use the `Line` model from
  `validate_joint_gls.py`, with 3 datasets sharing `a`.
  - **(a)** Same bounds, different start values → builds; exactly 1 warning, naming all
    three datasets.
  - **(b)** Dataset 2 has `max=10` and the others `inf` → `ShareMismatchError` with
    `'a'` and the dataset name in the message.
  - **(c)** Same as (b) with `on_mismatch='first'` → builds; master `max == inf`; 1
    warning.
  - **(d)** `vary` differs → `ShareMismatchError`.
  - **(e)** Calling `build_master_params` twice → `len(jf.warnings)` does not grow.
  - **(f)** After `perform_fit` on (a), the report text contains `"Joint-fit warnings:"`.
- [x] **Step 2: Run** → FAIL.
- [x] **Step 3: Implement.** Re-run → PASS. Also run `validate_joint_gls.py`,
  `validate_joint_mask.py` and `validate_jointfit_gui.py`: their joint fits must still
  build, because they seed equal bounds. Then `--quick` → exit 0.
- [x] **Step 4: Commit** — `01§6.8: share-group bound/vary mismatches refuse, value mismatches warn`.

### Task 4: Warn on the two silent joint weightings (01 §6.7, §6.14)

**Files:**
- Modify: `Joint_Fit_Routines.py` `perform_fit` — add the `Linear w/o error` warning.
- Modify: `joint_fit_spec.py` `_primary_curve` (~line 291) and `compile()` — add the
  multi-curve warning, carried into `jf.warnings`.
- Test: `validate_joint_ws1.py` section `T4`.

**Interfaces:**
- Consumes: `JointFit.warnings` (Task 3).
- Produces:
  - `perform_fit(..., fit_scale='Linear w/o error')` on ≥2 datasets without covariance
    appends: `"fit scale 'Linear w/o error' divides each dataset by its own maximum, so datasets are weighted by 1/max(y); use 'Linear' (error-weighted) for a comparable joint objective"`.
  - `_primary_curve(dataset) -> (x, y, yerr, n_curves)`. `compile()` appends
    `"dataset <name> has <n> curves; only the first (<key>) enters the joint fit"` when
    `n_curves > 1`.
- **Decision X1 (recommended, pre-ruled):** warn, do not change the weighting. Changing it
  would alter results silently. WS2's objective label states the weighting explicitly.

- [x] **Step 1: Write the failing checks (`T4`).**
  - **(a)** 2-dataset no-cov fit under `Linear w/o error` → the warning is present.
  - **(b)** The same fit under `Linear` → no such warning.
  - **(c)** `JointFitSpec.compile` given a `data[name]` dict with two curve keys →
    warning present, and only the first curve's points are fitted.
- [x] **Step 2: Run** → FAIL. **Step 3: Implement.** Re-run → PASS. Then `--quick` → exit 0.
- [x] **Step 4: Commit** — `01§6.7/§6.14: warn on per-max weighting and on dropped sub-curves`.

### Task 5: Joint-vs-individual chi-square on the fitted points (A1.4, 01 §6.6)

**Files:**
- Modify: `Joint_Fit_Routines.py` — add `JointFit.active_index(i: int) -> np.ndarray`, and a
  module function `chisqr_on(func, x, y, yerr, idx, masked_x=False) -> tuple[float, int]`.
- Modify: `xmodfit.py` `_eval_chisqr` (~line 7535) and its two calls in
  `computeModelComparison` (~line 7593). Both models are evaluated on
  `jf.active_index(i)` for dataset `i`.
- Test: `validate_joint_ws1.py` section `T5`.

**Interfaces:**
- Produces:
  - `JointFit.active_index(i)` — integer indices into objective `i`'s full `x`:
    `arange(imin, imax+1)` filtered by the frozen `imask`, mirroring
    `Fit.active_index` (`Fit_Routines.py:186`).
  - `chisqr_on` — evaluates `func` on `x[imin_of(idx):imax_of(idx)+1]` (the full
    window, as `JointFit.residual` does), selects `idx`, and returns
    `(sum(((y-f)/yerr)²), len(idx))`. Error values below `1e-30` are guarded as in the
    current `_eval_chisqr`. If `masked_x` is set (the objective's `yfit_masked`), the model
    is evaluated on the active points only and not masked again (Review focus 3).
  - `_eval_chisqr(func, name, idx)` keeps returning `(chisqr, ndata)`.

- [x] **Step 1: Write the failing checks (`T5`).**
  - **(a)** Build a 2-dataset `JointFit` with per-dataset `ranges` and a mask on dataset 1.
    `active_index(1)` equals the hand-built `arange(i0, i1+1)[mask[i0:i1+1]]`.
  - **(b)** After a no-cov `Linear` fit, `chisqr_on(obj.func, ..., active_index(i))`, summed
    over datasets, equals `result.chisqr` to relative `1e-10`. The fit's per-dataset
    max-normalisation cancels under `Linear`, so the sums match.
  - **(c)** With a narrower range on one dataset, the count `ndata` equals
    `len(active_index)`, not the scalar-window count.
- [x] **Step 2: Run** → FAIL. **Step 3: Implement**, including the `xmodfit.py` call-site
  change. Re-run → PASS. Run `validate_jointfit_gui.py`, which drives
  `computeModelComparison`, then `--quick` → exit 0.
- [x] **Step 4: Commit** — `A1.4: model comparison evaluates both models on the joint fit's own active points`.

### Task 6: Seed scans honour the frozen column mask (A1.3)

**Files:**
- Modify: `lmfit/minimizer.py` parallel-seed block (~lines 1929–1956). Pass each dataset's
  window-restricted mask along with `xs`/`ys`/`es`:
  - single model: `fitobj.imask_for()`;
  - joint: `o["imask"]`, plus `o.get("yfit_masked")`.
- Modify: `pdb_seed_worker.py` (`_init`, `_cell_residual`, `map_cell_costs` — accept and
  apply `mask=None`), `pdb_joint_seed_worker.py` (`datas` entries gain the mask and the
  `yfit_masked` flag; `JointSeedScan.residual` applies them),
  `Joint_Fit_Routines.py` `_build_orientation_seed` (~line 715).
- Create: `validate_seed_mask.py`.

**Interfaces:**
- Consumes: `Fit.imask_for()` (`Fit_Routines.py:141`) and `obj["imask"]` / `obj["yfit_masked"]`.
- The rule, everywhere: evaluate the model on the full window, then select `fit[mask]`,
  `y[mask]`, `e[mask]` — the order `JointFit.residual` uses. Skip the fit-side selection
  when `yfit_masked`. `mask=None` must reproduce today's numbers exactly.

- [x] **Step 1: Write `validate_seed_mask.py`.**
  - Setup: `Data/1ubq.pdb`, `Data/i2_tim3_1000_test.txt`, a coarse orientation grid of at
    most 6×6 cells, `emcee_cores=2`, and a mask that drops a contiguous block of about 10
    points inside the window.
  - **(a) Single model:** costs from the parallel scan equal the serial scan's costs cell
    by cell, to relative `1e-9`, with the mask on.
  - **(b) Joint, both-shared, 2 datasets:** the same comparison.
  - **(c)** With the mask on, the parallel costs **differ** from a mask-off run. This
    proves the mask reached the workers.
  - **(d)** With mask=None, the costs equal those of the parent commit's code path. Assert
    against costs computed in-process with the mask argument omitted.
  - Guard the script with `if __name__ == '__main__':`. If it takes over 120 s, register it
    `slow`.
  - Read `tests/recovered/validate_parallel_seed.py` and `verify_joint_seed_worker.py`
    first for how the serial and parallel scans are driven.
- [x] **Step 2: Run** → (a) and (b) FAIL.
- [x] **Step 3: Implement.** Re-run → PASS. Run
  `tests/run_all.py --only seed --timeout 900` (the recovered seeding harnesses), then
  `--quick` → exit 0.
- [x] **Step 4: Commit** — `A1.3: parallel and serial seed scans apply the frozen column mask`.

### Task 6b (A1.18, added in execution)

Found by WS1 itself during Task 6, ruled in by the controller. With a column mask and
`emcee_cores>1`, the parallel MCMC *sampling* objectives (`PdbLogProb`, `JointLogProb`) used
unmasked data, a silently different likelihood from the serial path. Fixed in `bfdd234`
(workers take the frozen window mask plus a `__masked_x__` flag; seed workers and log-probs
share `apply_mask`/`prepare_datas`; `yfit_masked` is derived, not read from a stored flag) and
`6ffcff9` (check (j) no longer depends on git history: the parent lnprob values are frozen in
`tests/fixtures/seed_mask_parent_ref.json`). Regression checks: `validate_seed_mask.py` (g)-(k).

### Task 7: Bring the two stale harnesses up to date (A1.17)

**Files:** `tests/recovered/smoke_fit.py`, `tests/recovered/verify_jointmcmc.py`,
`tests/run_all.py` (`REGISTRY`), `handoff/10-fix-list.md`.

- [x] **Step 1: Confirm semantics before editing.** `GID2D.py:629` emits
  `chi2_per_N_reported_LINEAR`. Confirm it is the quantity `smoke_fit` meant by
  `chi2_per_N_reported`, read from `GID2D`'s code and history, not from the name.
  Similarly confirm that `_corner_render_data` (b953c3f) is a helper the method list in
  `verify_jointmcmc` must bind, and not a behaviour change.
  - If either is a real semantic change, keep it `known` and record why in A1.17. That is
    a finding, not a fix.
- [x] **Step 2: Apply the one-line updates.** Run each harness directly → PASS.
- [x] **Step 3: Drop both `known` entries.** Run `--quick` → exit 0 with **zero**
  KNOWN-FAIL rows.
- [x] **Step 4: Commit** — `A1.17: update smoke_fit and verify_jointmcmc to the current GID2D/MCMC API`.

### Task 8: Record WS1

**Files:** `handoff/10-fix-list.md`, `handoff/01-joint-fit.md` (§6, §7), `handoff/11-audit-plan.md`,
`handoff/12-work-package-preaudit.md` (progress log), `handoff/14-roadmap.md` (WS1 status).

- [x] **Step 1: Full run** in the background:
  `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/run_all.py`. Expected: exit 0
  with no KNOWN-FAIL rows.
- [x] **Step 2: Record.**
  - Mark these done with commit hashes: A1.1, A1.3, A1.4, A1.17, B2a, and 01 §6 items
    2, 3, 6, 7 (warned), 8 and 14 (warned).
  - Update the `01` §7 / `11` claims these settle: 01 #11 (masks compose identically) is
    now also covered for seed scans.
  - Add a "WS1" entry to the progress log with the full-run table.
  - Set the WS1 status line in the roadmap.
- [x] **Step 3: Commit** — `WS1: recorded`.

---

## Decisions (pre-ruled; listed so the user can overturn them)

- **X1 (Task 4):** `Linear w/o error` gets a warning, not a weighting change.
- **X2 (Task 3):** a difference in values only warns; a difference in bounds or `vary`
  refuses unless `on_mismatch='first'`. **Premise corrected during execution:** the original
  text claimed `run_joint_fit` already turned an exception into a message box. It did not
  (no `try`, no excepthook), so a `ShareMismatchError` would have reached a Qt slot and
  aborted the app. A minimal GUI guard was added in `9518ef9`: `ShareMismatchError` shows a
  message box on the Fit path and on both load paths (`.jpar` and `loadParIntoJoint`), with
  no retry dialog and no success claim. Fuller GUI handling stays in WS4.
- **X3 (Task 1):** the single list lives in `lmfit/minimizer.py`, where four of the uses
  already are, rather than in a new module.
- **Out of scope:** B4 (`doFit` routing; moved to WS4), A1.2 (`mkey` in `_box`; WS7 with
  B8), the WS0 deferred minors, which WS0's final review triaged as safe to leave.
