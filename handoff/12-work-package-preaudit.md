# Work package: pre-audit fixes

**Assigned to a dedicated implementation session. Orchestrated from session
`xmodjointfit-63`.** Opened 2026-09-04.

Everything here must land **before** the audit (`11-audit-plan.md`) and before any
reorganisation or refactoring. The audit cannot produce trustworthy verdicts against a
harness that cannot fail, and refactoring on top of known-broken code just moves the bugs
around.

Source of every item: `10-fix-list.md`. IDs match it — keep them in commit messages.

> **Line numbers in the fix list may already be stale.** Several were recorded before the
> commits of 2026-09-03/04, and every edit in this package shifts more. **Locate by symbol
> name, not by line number**, and correct any citation you find wrong as you go.

---

## Guardrails

**Do:**
- Work on `develop`. Themed commits, one per item or per tight group, with the item ID in
  the subject.
- Run `python -m py_compile` on everything you touch, every time.
- After Phase 0 exists, run the validator suite before and after each change.
- Use `C:\Users\kerr\.conda\envs\cmc-py39\python.exe` — never the default interpreter.
  Headless needs `QT_QPA_PLATFORM=offscreen`, an injected `xraydb` stub, and
  `if __name__ == '__main__':` in anything reaching a spawn Pool. Recipe:
  `09-validation-and-env.md`.
- When an item turns out to be wrong, or bigger than described, **stop and report it**
  rather than expanding scope. Several fix-list entries were written from a single
  agent's reading and have not been verified.

**Do not:**
- **Do not rewrite history.** No `rebase`, `filter-branch`, `commit --amend` on anything
  already committed. Two rewrites have already happened this week; a third from a second
  session would be a mess.
- **Do not push.** `origin` is upstream (`chemmatcars/XModFit`), not a fork.
  `.githooks/pre-push` guards this; do not disable it.
- **Do not touch the seeding study paths**: `julia/bakeoff/`, `handoff/seeding-study/`,
  `SEEDING_STUDY_TIM1.md`, `STAGE2_SEEDING_PLAN.md`, `TIM1_SEEDING_REPORT.md`,
  `PLAN_*.md`. They are gitignored and live in `../xpr-seeding-study` now. Commit
  `7942458` still holds the tracked half; removing it is the orchestrator's job.
- **Do not refactor, rename, reformat or reorganise** beyond what an item asks for. No
  reflowing, no import reordering, no type hints, no docstring rewrites except where an
  item names one.
- **Do not delete any `XLayersPDB` variant** (`_G`, `_H`, `_box`, `_dev_smooth`). That
  decision (D1) is open. Fix them in place.
- **Do not remove the footprint covariance capability.** Decision D2 is open.
- **Do not "fix" these deliberate designs** — each was a considered choice and each has a
  note saying so: theta/phi being unbounded for MCMC; `_DENSITY_STOPS` being an absolute
  rather than stack-relative colour map; the `'map2d'` choices value; `nan_policy='raise'`
  with points dropped rather than NaN'd.
- **Do not re-enable the centre-of-mass step** in `ED_profrot` (A1.14). It is switched
  off on purpose and its re-enabling is explicitly deferred by the user.

---

## Phase 0 — make the work verifiable (blocking; do first)

Nothing after this is trustworthy until these three land.

- [ ] **P-3 / A0.2 — recover the off-repo harnesses. DO THIS FIRST; they are in temp
      directories that get cleaned.** 21 files, confirmed present 2026-09-04:
      - 12 in `C:\Users\kerr\AppData\Local\Temp\claude\C--Users-kerr-Documents-GitHub-XModJointFit\22bf29f6-2dcf-4452-846a-74cdac330a9f\scratchpad\`
        (`verify_fingerprint`, `verify_jointmcmc`, `verify_jointwindow`,
        `verify_joint_mixed`, `verify_joint_neither`, `verify_joint_orient`,
        `verify_joint_parallel_scan`, `verify_joint_seed_worker`, `verify_reuse_wrap`,
        `verify_seedmode`, `verify_seed_gui`, `verify_seed_helpers`)
      - 9 in `...\c0878594-9211-4c15-b276-03aabd2cec97\scratchpad\`
        (`verify_banner_lifecycle`, `verify_detail_switch`, `verify_joint_4ds_parallel`,
        `verify_joint_detail_and_legend`, `verify_joint_parallel_edge`,
        `verify_joint_parallel_sample`, `verify_legend`, `verify_multifit_selection`,
        `verify_plotting_B`)

      Copy into `tests/` (which already exists, holding `tests/fixtures/`). **Exclude
      `verify_joint_parallel_scan.py`** — it asserts `sampler.pool is None`, the behaviour
      later deliberately flipped; `verify_joint_4ds_parallel.py` is its corrected
      replacement. Commit them **as found, unmodified**, in their own commit, before
      adapting anything: that commit is the provenance record. Fix paths/imports in a
      *second* commit. Expect some to fail — that is information, not a problem.

- [ ] **P-1 / A1.10 — make `check()` actually fail.** Every `validate_*.py` defines a
      local `check(name, cond, detail)` that prints PASS/FAIL and folds the result into a
      module-level `_OK` dict, with the verdict carried only by the exit code. Under
      pytest a failing check reports PASSED. Give the validators a shared helper that
      raises or accumulates a failure count and exits non-zero, and make sure a
      deliberately-broken assertion is observed to fail.

- [ ] **P-2 — add a runner.** `pytest` is **not installed** in `cmc-py39` (verified), so
      make it a plain script — `tests/run_all.py` — that runs each validator as a
      subprocess with the right interpreter and env, and prints one summary table plus a
      single non-zero exit on any failure. Do not add a pytest dependency. Note which
      validators need `offscreen`, which need out-of-tree data, and which are slow.

- [ ] **Phase 0 checkpoint — the BASELINE.** Run everything and report the results
      **before fixing anything**. This is the most valuable single artefact in the
      package: it separates "already broken" from "broken by this work". Record it in the
      progress log below. Known in advance: `validate_jointfit.py` was deleted with the
      bilayer campaign, and `validate_xpr.py` needs `XPR-main/test_data` from outside the
      repo.

## Phase 1 — correctness

Each item: fix, add or extend a regression check, confirm the suite still passes.

- [ ] **A1.1 — one source of truth for the PDB model-name list.** Three tuples disagree:
      `lmfit/minimizer.py` `_PDB_FUNCNAMES` (6 names), `xmodfit.py` `_PDB_MCMC_FUNCS`
      (3, dropping `_G`/`_H`/`_XPR`), and an inline tuple in
      `Fit_Routines.set_emcee_minimizer` (a different 3, dropping `_box`). Consequence:
      `_XPR`/`_G`/`_H` get a `CustomSampler` from the engine but a stock emcee
      `HDFBackend` from the GUI, and `CustomSampler.get_autocorr_time` then passes
      `angleindex=` into emcee's `integrated_time` → `TypeError` not caught by the
      `except` list in `Minimizer.emcee`. **Found independently by three agents; highest
      confidence and highest value in this package.** One constant, imported everywhere.
- [ ] **A1.2 — `mkey` used before assignment** in `XLayersPDB_G.py`, `_H.py`, `_box.py`:
      `self.__d__[mkey]` precedes `mkey`'s assignment, surviving only because `mkey` leaks
      from a loop that runs *only when `__fit__` is False*, and `Joint_Fit_Routines.py`
      sets `__fit__ = True`. So none can be joint-fitted. Hoist the assignment. **Fix, do
      not delete** (D1 open).
- [ ] **A1.3 — seed workers ignore the per-column fit mask.** `Fit.residual` applies the
      mask and `JointFit.residual` applies `obj["imask"]`, but `pdb_seed_worker`,
      `pdb_joint_seed_worker.JointSeedScan.residual` and
      `JointFit._build_orientation_seed`'s data slices all use bare `imin:imax`. Thread
      the frozen mask through. Add a regression: masked serial scan == masked parallel
      scan, and both differ from the unmasked result.
- [ ] **A1.4 — `_eval_chisqr` ignores per-dataset ranges and column masks**, so the
      joint-vs-individual model comparison is computed on mismatched point sets. Route it
      through the same `imask_for`/`active_index` path as the fit.
- [ ] **A1.5 — mask + `__masked_x__` breaks the 1-D plot.** `self.fit.yfit` is shorter
      than `self.fit.x[imin:imax+1]`, but `update_plot` and `fitCallback` still pass the
      full slice to `PlotWidget.add_data`, whose length check raises a modal dialog. The
      same slice feeds `output_params['Residuals']`. Slice `x` by `active_index()`.
- [ ] **A1.6 — drop the discarded `super()` call** in `CustomSampler.get_autocorr_time`.
      The same double-autocorrelation pattern was deliberately removed from
      `CustomBackend`; this copy was missed and costs a full-chain autocorrelation.
- [ ] **A1.7 — NGL viewer poll loop.** The 400 ms `QTimer` has no `hideEvent`/
      `closeEvent`, so it runs at 2.5 Hz for the whole session once opened, and each tick
      calls `func.update_parameters()` on the live model. Each reload also writes a
      ~1.3 MB temp HTML and leaks the last one. Stop on hide/close, make the poll
      read-only, clean up temp files.
- [ ] **A1.8 — `update_fixed_parameters` leaks private attributes** into saved files: the
      filter is `key[:2] != '__'`, so single-underscore names pass (`_last_xsig` reached a
      shipped `.xpar`). Skip single-underscore keys too.
- [ ] **A1.9 — `validate_xpr.py` has no `if __name__ == '__main__'` guard** (top-level
      body, `sys.exit` at the end). Currently harmless only because every emcee call
      passes `emcee_cores=1`. Add it.
- [ ] **A1.11 — `h5py` missing from the three newest requirements snapshots**
      (`requirements_07022026.txt`, `requirements_working_Mar2026.txt`,
      `requirements_xmodfit_viewer.txt`), so an env built from the "current" one cannot
      save an MCMC run. Add it, and collapse the five snapshots to one current file plus
      the two optional extras (viewer, xpr). Say in the commit which you designated
      current and why.
- [ ] **A1.13 — `pdb_cache.pdbread` counts waters as protein.** It iterates every atom
      `Bio.PDB` returns with no filter, so a structure straight from RCSB inflates
      electrons, radii and mass, and with them `protvol`, the coverage normalisation and
      the EDP. **Scope this narrowly: skip water residues only** (`HOH`, `WAT`, `DOD`,
      `TIP3`), and keep other heteroatoms — bound ions and ligands are real, and the
      structure viewer advertises showing them. Anything broader is B13, not this package.
      Regression: parse a raw structure with waters and a stripped copy, assert equal
      electron totals.
- [ ] **A1.12 (optional, low priority) — `XPR.jl` rejects `lo == hi`.** The workaround
      (`hi = lo + 1e-9`) left with the study, so nothing in-repo depends on this now. Fix
      only if free.

## Phase 2 — repo and data hygiene

- [ ] **A0.5 — revert `.idea/`.** `XModFit.iml` and `misc.xml` are modified only to
      encode one machine's absolute conda path. `git checkout -- .idea/`.
- [ ] **A2.1 — `Data/` cleanup.** Delete the untracked test outputs
      (`*_test_fit.txt`, `n1_tim3_1500_raw_test.txt`, `dppc_q2d_gid_indexed.xfit`).
      **The 13 personal `.pdb` structures need a user decision — do not delete them.**
      Instead list them with sizes and any in-repo references, and report; the user will
      say which go. Same for the three `Data/*.expr` files.
- [ ] **A2.3 — re-save `Data/dppc_q2d_gid_indexed.xpar`** from a Log-scale fit (it
      currently carries `#Fit Scale=Linear`, which the GID2D notes are explicit is wrong
      for this data), after A1.8 so it no longer leaks `_last_xsig`. If a headless
      Log-scale fit is impractical, report that and leave it; do not hand-edit the file.
- [ ] **A3.15 — attribution for vendored third-party code.** `Tools/StructureViewer/
      ngl.js` (NGL 2.4.0, 1.2 MB) and `julia/XPR/` (vendored from Elias Boshara's XPR).
      Add a short `NOTICE` or per-directory README with source, version and licence.
      Needed before any public push.

## Phase 3 — dead code, duplication, stale claims

Low risk individually; the value is that the later refactor starts from a smaller surface.

- [ ] **A3.1** Delete `xmodfit_old.py` (4,428 lines, imported by nothing — confirm first).
- [ ] **A3.2** Retire `joint_fit_orchestration.py`: effectively dead, its only caller is
      inside a triple-quoted block in `xmodfit.py`. `push_result_to_tables` has no caller;
      `_primary_curve` exists in three copies. Confirm before deleting.
- [ ] **A3.3** Collapse `gid_explorer.html` / `gid_template.html` — byte-identical but for
      one line, two 79 KB copies, and `write_browser_page` already generates one from the
      other.
- [ ] **A3.4** Remove the five stale private `parratt_numba` copies now that `XLayersPDB`
      imports the shared optimised one; drop `prange` under `parallel=False` in three.
      **Careful:** `XLayers_Beamfootprint`'s private copy returns a scalar where the
      shared one returns the `refc` array — that one is not a drop-in.
- [ ] **A3.5** Collapse the **seven** copies of the "is this dataset fitted?" predicate in
      `xmodfit.py`, in two forms, **two of which omit the `not hasattr(f,'objectives')`
      guard** — inconsistent, not merely redundant. `_dataset_is_fitted` already exists.
- [ ] **A3.6** De-duplicate the Tableau-10 palette (`PlotWidget.py` and `xmodfit.py`).
- [ ] **A3.7** `PlotWidget.remove_data` leaks `linestyle`/`fit`/`yerr` entries, and its
      `legendItem.removeItem(dname)` is a silent no-op for the same label-mismatch reason
      that forced `legendItem.clear()` in `Plot`.
- [ ] **A3.8** Delete dead symbols: `_viridis` in `NGLViewerDialog.py`;
      `fitter._ext_workers` (set, read nowhere); `ColumnLimitsDialog`'s unused `names`
      argument; and the `emcee()` params `seed_uniform_frac` / `walkers_per_basin`, which
      no caller sets — **either plumb them from the GUI or remove them, and say which**.
- [ ] **A3.9** De-duplicate the angleindex construction (5x in `xmodfit.py`, once in
      `minimizer.py`) and the validators' copy-pasted `Line` stubs and `check()` helpers
      (the latter falls out of P-1).
- [ ] **A3.11** `xmodfit.py` around line 1058: one very long line with a run of spaces
      where a continuation got flattened. Parses, but looks like a botched edit — restore
      the intended formatting.
- [ ] **A3.12** Fix stale docstrings and citations: `JointFit.set_emcee_minimizer` still
      says the orientation-seeded joint path is serial; `Joint_Fit_Routines.py` cites
      "xmodfit.py ~3246/3254" which no longer matches; `pdb_cache.py` claims every variant
      uses it (`_dev_smooth` does not); `XLayersPDB_XPR.py` and `pdb_geometry.py` still
      assert theta/phi do **not** transfer to XLayersPDB, which was retracted — they do,
      bit-identically to scipy `'zx'`; and four paths in `viewer_template.html` point at
      skill-package files absent here (`scripts/make_browser_viewer.py`,
      `references/*.md`, `assets/ngl.js` — note `scripts/` now exists for other reasons).
- [ ] **A4 — merge guards.** Add a short comment at each of these, since a careless
      upstream merge silently disables a feature rather than conflicting: the
      `self.xmin = {}` → `_LimitProxy` swap in `XModFit.__init__` (two lines; losing it
      disables column limits while still importing); `Fit.residual`'s mask block, which
      must stay **above** `maxd = np.max(data)`; and the covariance composition order
      (slice → `imask` → concatenate → `_cov_mask`) in both `_build_chol` and `residual`.

**Deferred out of this package** (do not start): A3.10 calling the extracted seeding
helpers — a genuine refactor, belongs to the refactor pass. A3.13 reconciling
contradictory recorded numbers — needs re-measurement, so it is audit work. A3.14
documenting arm D — left with the study. A1.14 — deferred by the user.

---

## Reporting protocol

The orchestrating session (`xmodjointfit-63`) monitors by reading this file and the git
log, so **the progress log below is the channel that matters** — it survives session
boundaries, which a chat message does not.

After each phase, and whenever something surprising turns up:

1. Append a dated entry to the progress log below: items done, items refused and why,
   anything discovered that is not in the fix list, and the current suite result.
2. Commit it (`Progress: phase N — <summary>`).
3. Additionally message session `xmodjointfit-63` via `SendMessage` with a short summary,
   if that tool is available to you. If it is not, the progress log alone is sufficient —
   do not block on it.

Report immediately, without waiting for a phase boundary, if: an item cannot be done as
described; a fix would change fitted numbers for existing saved analyses; you find
something in the "deliberate designs" list that looks genuinely wrong; or the Phase 0
baseline shows something already broken that the documentation claims works.

## Definition of done

Every unchecked box above is either checked or has a progress-log entry saying why not;
`python tests/run_all.py` runs from one command, genuinely fails when something is broken,
and its result is recorded; `py_compile` passes across the tree; nothing has been pushed;
history is unrewritten; and `10-fix-list.md` is updated so each item is marked done,
refused-with-reason, or deferred.

---

## Progress log

<!-- Append entries here. Newest last. Include the suite result each time. -->

### 2026-09-04 — package opened (orchestrator)

Scope compiled from `10-fix-list.md` Tier A plus the three hard preconditions from
`11-audit-plan.md` section 2. Verified while compiling: all 21 off-repo `verify_*.py`
harnesses are still on disk at the two paths listed in Phase 0, and `pytest` is not
installed in `cmc-py39` (so P-2 must be a plain script). No implementation work started.
