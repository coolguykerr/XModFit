# Fix list — before consolidation vs. involved to-dos

Assembled 2026-09-03 from the nine category documents in this directory. Nothing here has
been executed or verified by running code — see `11-audit-plan.md`.

Severity tags: **[wrong-result]** silently corrupts a number a scientist would report ·
**[crash]** raises · **[unreachable]** feature exists but cannot be used ·
**[hygiene]** no functional effect.

## Corrections applied, so they are not carried forward as defects

- `E` = 10.0 keV **is** the correct default and is set by the user per dataset
  (`XLayersPDB.py:150`, `XLayersPDB_XPR.py:107`, `XLayersPDB_dev_smooth.py:455` all
  default `E=10.0`). The 15.126 value belongs to one beam-footprint analysis campaign.
  The "wrong E in the GID2D fixture" finding is **struck**.
- The GLS/covariance core is **not** untested. `validate_joint_mask.py` is fully synthetic
  (a `Line` model with hand-built `cov` sidecars) and does exercise the covariance +
  footprint whitening path, including the one-dataset-without-`v` `zeros(nseg)` branch.
  Only the *real-data* validator is unrunnable.
- `Data/tim3_hmmm_ext_all_bestframe.pdb` (the default for every PDB model) is **already in
  upstream** `origin/develop`. Replacing it is a cleanliness choice, not a disclosure fix.

---

# Tier A — fix before consolidation

Small, local, low-risk: each is a single-file edit or a mechanical sweep. Doing these
first makes both the consolidation and the audit tractable.

## A0. Repo safety — before touching anything else

**A0.1 Commit the untracked work.** [highest-consequence item on this list]
Whole categories exist **only** in the working tree, in no commit: all of GID2D
(`Functions/GISAXS/GID2D.py`, `Functions/gid2d_core.py`, `Image2DWidget.py`,
`Tools/GrazingIncidence/*`), all of XPR/Julia (`Functions/xpr_backend.py`,
`Functions/XRR/XLayersPDB_XPR.py`, `Functions/pdb_geometry.py`, `julia/**`), all of the
orientation-seeding work (helpers, temperature, basins, inverse-CDF, parallel scans, every
joint seeding path, `anglepairs`), and most of the core-app work (the entire per-column
limits + mask engine, `DataListWidget`, the no-fit banner, all legend/overlay work, the
`PlotWidget` data-line toggle and `legend_formatter`, `Fit._canonicalize_result`,
`can_jac`/`residual_jac`). A single `git checkout .` destroys all of it.
*Fix:* themed commits on a branch, before any other change.

**A0.2 Recover the off-repo test harnesses.**
Roughly 17 `verify_*.py` harnesses exist only in per-session temp scratchpads: the seeding
set (`verify_seed_helpers`, `verify_seedmode`, `verify_seed_gui`, `verify_joint_orient`,
`verify_joint_neither`, `verify_joint_mixed`, `verify_joint_seed_worker`,
`verify_reuse_wrap`, `verify_fingerprint`, `verify_repro_full`, …) and the plotting set
(`verify_legend`, `verify_banner_lifecycle`, `verify_detail_switch`,
`verify_multifit_selection`, `verify_plotting_B`). These are the *only* coverage for the
seeding work and for core-app sub-features (b) and (c) — everything else there rests on
`validate_jointfit_gui.py` alone. *Fix:* move them into the repo (proposed `tests/`),
dropping the obsolete `verify_joint_parallel_scan.py`, which asserts `pool is None` — the
behaviour later deliberately flipped.

**A0.3 Purge MCMC backends; add a `.gitignore`.**
`Data/*.h5` is 872 MB across eight files (`p3e2test` 282, `A_Scan1MCMCnoprot` 239,
`i2test` 137, `htim3_A3_trial5_test` 115, `p3i4test` 44, `n1test` 28, `d6test` 27,
`inittest` 1.4). All are testing artefacts. There is no `.gitignore` at all.
*Fix:* delete the untracked ones; ignore `*.h5`, `*.xfit`, `__pycache__/`, `.idea/`,
`*.npz`, `.perr`, `mesh.txt`.

**A0.4 DONE 2026-09-03.** Rewrite `Data/d6test.h5` out of the local commits.
Done with `filter-branch --index-filter` over `origin/develop..HEAD`: all 16 commits now
carry upstream's 1,323,496-byte blob, the `a4fd04e` merge is preserved with both parents,
and a full-tree diff against the pre-rewrite tip shows only that file. Every local commit
hash changed. Backups `pre-h5-rewrite-bdb44ef` and `refs/original/refs/heads/develop`
still hold the 26.8 MB blob, so the pack has not shrunk until they are dropped and
`git gc --prune=now` runs. The file itself was then removed from the branch entirely
(A2.1). Original text follows. [irreversible once pushed]
Upstream has this file at 1.3 MB; our commit `d3e353e` replaced it with a 26.8 MB MCMC
backend. Pushing adds 27 MB to every clone permanently. *Fix:* restore upstream's blob
across the six local commits before any push.

**A0.6 The seeding study must not reach upstream — one step left.** [publishing risk]
The September 2026 global-optimum seeding study was research, not an XModFit feature. It
was **extracted to its own repository on 2026-09-04** (`../xpr-seeding-study`, still with
no remote configured), and `.gitignore` now covers its paths so the untracked half cannot
drift back in. That repository has since grown to 1829 tracked files / 19.3 MB, including
915 `.npz` — 703 endpoint archives under `julia/bakeoff/` with a `MANIFEST.sha256`, and
212 cost maps. The 703 were produced *here*, in this repo's gitignored
`julia/bakeoff/`, and were unversioned in either repo until 2026-09-06.

What remains: commit **`7942458`** ("Add XPR global-seeding bake-off harness, saved
endpoints and study write-ups", 114 files, 2.18 MB, 92 of them `.npz`) still carries the
tracked half in local history, **15 commits back** on `develop` as of 2026-09-06. Nothing
is pushed.

**Full inventory and the two verifications behind the recommendation:
`handoff/seeding-study/AUDIT-7942458.md`** (untracked, and hook-guarded, so it cannot be
pushed either). Re-checked 2026-09-06: (1) of the 15 commits from `7942458` to `HEAD`,
**none touches any of the 114 files**; (2) all **114 are preserved in the study repo** —
103 byte-identical, 11 differing only by line endings and later edits there, **0 absent**.
So dropping the commit loses no content. Three routes, from the study's own exclusion
manifest (now `../xpr-seeding-study/handoff/DO-NOT-COMMIT.md`):

- **A — leave history, never push `develop`.** Push only curated branches that exclude
  `7942458`. Now enforced mechanically by `.githooks/pre-push`, which refuses any push
  whose range touches study paths (verified: it correctly refuses `develop` today).
  Enable per clone with `git config core.hooksPath .githooks`.
- **B — drop the commit.** `7942458` is self-contained: no later commit touches any of
  its 114 files, so it rebases out without conflict. Now safe to do, since the study is
  preserved elsewhere -- note the rebase WOULD have deleted those files from disk.
  ```
  git branch backup-before-dropping-7942458 develop
  git rebase --onto 7942458^ 7942458 develop
  ```
  Costs: 15 commits get new hashes (a second rewrite after A0.4), and
  `pre-merge-backup-b3e1ab5`'s relationship to `develop` is invalidated.
- **C — separate repository.** DONE, as above.

**Recommended: B**, because the fold-in wants a clean linear `develop` for two-person
review and the only thing that made B risky (losing ~30 h of compute) no longer applies.
Deferred rather than done, because it rewrites history and the hook makes deferral safe.

One genuine XModFit feature is owed from the study: the fast-scan in that repository's
`PLAN_XMODFIT_FASTSCAN.md`. The recorded recommendation is to **reimplement** the
candidate physicality screen in `Functions/` root rather than import
`julia/bakeoff/seed_utils.py`, which is hardwired to the study's config format.

**A0.5 Revert `.idea/`.** `.idea/XModFit.iml` and `.idea/misc.xml` are modified only to
encode one machine's absolute conda path. *Fix:* `git checkout -- .idea/`.

## A1. Correctness fixes

**A1.1 One source of truth for the PDB model-name list.** [crash]
Three tuples disagree, all three uncommitted: `lmfit/minimizer.py:94` (6 names),
`xmodfit.py:2634` (3, drops `_G`/`_H`/`_XPR`), `Fit_Routines.py:491` (a different 3, drops
`_box`). Consequence: for `XLayersPDB_XPR`/`_G`/`_H` the engine builds a `CustomSampler`
while the GUI supplies a stock emcee `HDFBackend`; `get_autocorr_time` then passes
`angleindex=` into emcee's `integrated_time` and raises `TypeError`, which is not in the
`except` list at `minimizer.py:2300`. Found independently by three agents. XPR is under
active development, so this bites soonest. *Fix:* one module-level constant, imported by
all three sites.

**A1.2 `mkey` used before assignment in `_G`, `_H`, `_box`.** [crash]
`self.__d__[mkey]` precedes `mkey`'s assignment (`XLayersPDB_G.py:696` vs `:699`,
`_H.py:797` vs `:800`, `_box.py:523` vs `:526`). It survives only because `mkey` leaks from
a loop that runs *only when `__fit__` is False*, and `Joint_Fit_Routines.py:452` sets
`__fit__ = True` — so none of the three can be joint-fitted. *Fix:* hoist the assignment,
or drop the variants (D1 — if they go, this item disappears).

**A1.3 Seed workers ignore the per-column fit mask.** [wrong-result]
`Fit.residual` applies the mask (`Fit_Routines.py:291-296`) and `JointFit.residual` applies
`obj["imask"]` (`Joint_Fit_Routines.py:552-558`), but `pdb_seed_worker`,
`pdb_joint_seed_worker.JointSeedScan.residual` and `_build_orientation_seed`'s slices all
use bare `imin:imax`. The "parallel == serial bit-for-bit" validation is therefore only
true while no column limit is active — and the two features were never exercised together.
*Fix:* thread the frozen mask into both workers.

**A1.4 `_eval_chisqr` ignores per-dataset ranges and column masks.** [wrong-result]
The joint-vs-individual model comparison masks with the scalar Xrange only, so the two
chi2 values are computed on different point sets whenever `#FitRange=` or a column limit
is in use. *Fix:* route it through the same `imask_for`/`active_index` path as the fit.

**A1.5 Mask + `__masked_x__` breaks the 1-D plot.** [crash, modal dialog]
With both active, `self.fit.yfit` is shorter than `self.fit.x[imin:imax+1]`, but
`update_plot` (`xmodfit.py:6297`) and `fitCallback` still pass the full slice as `x` to
`PlotWidget.add_data`, whose length check (`PlotWidget.py:215`) pops a modal
"dimensions … not matching" dialog (`:270`). The same slice feeds
`output_params['Residuals']` (`:6301-6304`). Latent only because `GID2D` is the sole model
declaring `__masked_x__` and it renders in the 2D panel. *Fix:* slice `x` by
`active_index()` on both paths.

**A1.6 Drop the discarded `super()` call in `CustomSampler.get_autocorr_time:336`.**
[hygiene/perf] The identical double-autocorrelation pattern was deliberately removed from
`CustomBackend`; this copy was missed. Costs a full-chain autocorrelation per call.

**A1.7 NGL viewer poll loop never stops and has side effects.** [hygiene/perf]
The 400 ms `QTimer` has no `hideEvent`/`closeEvent`, so it runs at 2.5 Hz for the rest of
the session once opened, and each tick calls `func.update_parameters()` on the live model.
Each full reload also writes a ~1.3 MB temp HTML and leaks the last one. *Fix:* stop on
hide/close; make the poll read-only; clean up temp files.

**A1.8 `update_fixed_parameters` leaks private attributes into saved files.** [hygiene]
The filter is `key[:2] != '__'`, so single-underscore attributes pass — `_last_xsig` landed
in `Data/dppc_q2d_gid_indexed.xpar:23`. *Fix:* also skip single-underscore keys.

**A1.9 `validate_xpr.py` has no `if __name__ == '__main__'` guard.** [crash, latent]
Top-level body with `sys.exit` at `:382`. Harmless only because every emcee call passes
`emcee_cores=1` and all four spawn-`Pool` sites gate on `workers > 1`. Raise cores and it
fork-bombs.

**A1.10 `check()` never asserts in any validator.** [hides failures]
Verdicts ride the exit code, so under pytest a failing check reports PASSED. *Fix:* make
`check()` raise or accumulate a failure count; add one runner for all validators.

**A1.11 `h5py` missing from the three newest requirements snapshots.** [crash]
Absent from `requirements_07022026.txt`, `requirements_working_Mar2026.txt` and
`requirements_xmodfit_viewer.txt`; present only in upstream's two and
`requirements_01062026_old.txt`. An environment built from the "current" snapshot yields an
XModFit that cannot save an MCMC run. *Fix:* add it; collapse the five superseded snapshots
to one current file plus the two optional extras (viewer, xpr).

**A1.13 `pdb_cache.pdbread` has no water or heteroatom filter.** [wrong-result]
It iterates every atom `Bio.PDB` returns (`Functions/pdb_cache.py:76`), so a PDB
straight from RCSB has its crystallographic waters counted as protein: electrons, vdW
radii and mass all inflated, which propagates into `protvol`, the coverage normalisation
and the electron-density profile. Found 2026-09-03 while preparing the 1UBQ default,
which carries 58 HETATM waters against 602 protein atoms. Every structure in use so far
was hand-stripped, which is why it never showed. *Fix:* skip `HETATM`/`W` residues by
default, with an opt-in for genuine bound ions or ligands (the viewer already advertises
showing those, so the filter must not be unconditional).

**A1.14 Re-enable the centre-of-mass step in `ED_profrot` before the fold-in.**
[deliberate, must be reverted]
The CoM subtraction at `XLayersPDB.py:286-288` sits inside a string literal and the
rotation uses raw `coord` (`:294`), so today the model rotates about the coordinate
origin rather than the protein, and every template must arrive pre-centred (hence the
`*_center.pdb` / `*_com.pdb` naming, and hence `Data/1ubq.pdb` being mass-centred).

**This was intentional, not an oversight** -- confirmed by the author 2026-09-03: it was
commented out to compare against older code that did not centre. It **should be
commented back in for the final repo**. Deliberately not done yet.

When re-enabling: the fold is not cosmetic, it changes what `protins` means for any
template that was not already centred, so any saved parameter file from an off-centre
template will need refitting. The in-repo `.par` fixtures no longer constrain this (see
A2.6), but personal saved analyses will. The docstring at `:281` already describes the
intended behaviour, so it needs no change -- only the code below it.

**A1.12 `XPR.jl:262` still rejects `lo == hi`.** [hygiene]
The phase-0 fix the stage-2 plan calls for was never applied, so every bake-off harness
carries a `hi = lo + 1e-9` workaround. *Fix:* allow the degenerate bound; delete the hack.

**A1.15 The NGL viewer only understands the old top-referenced `protins`.** [wrong-result]
Logged 2026-10-26. The user has confirmed this is expected: the viewer was written
before XPR was integrated and hasn't been revisited since.
The page places the protein's top slice centre at `-sum(d[:-1]) + protins`
(`viewer_template.html:145-147`, `zref = ztop - zslice/2`), and `NGLViewerDialog.py`
passes `func.protins` through raw (`:209`, `:466`). That matches `XLayersPDB`. Since the
2026-09-08 XPR revision, however, `XLayersPDB_XPR.protins` is measured from the **mass
centroid** (`handoff/05-xpr-julia.md` §3.9). So for any XPR model, any post-2026-09-08 fit,
or any `configs_newconv/` value, the protein is drawn `maxh(theta, phi)` too deep: about
14-31 A depending on template and pose, so a bound protein appears to hang clear of the
film. Orientation is **not** affected: `rotRows` is the same extrinsic `'zx'`, `Rx(theta)Rz(phi)`, and
the membrane is at +z in both. No error is raised, the picture is simply wrong, and
`07-structure-viewer.md` §4 item 5 lists `_XPR` as supported, so nothing warns the user.
*Fix:* when `func` has `protins_top` (`XLayersPDB_XPR.py:475`), send
`func.protins_top()` to the page instead of `func.protins`, in both `scene_cfg_from_func`
and `_snapshot`. Show both values, labelled, in the read-only panel. Then check that the
two reference points really coincide: `maxh` is `max(r3·c + rad)` about the centroid,
while the page's `zref` adds `fslice` and subtracts `zslice/2`, and A1.14 changes the
rotation origin. Close it with audit claim 07 #5, run on an `_XPR` model
(`zprot[0]`/`sim_ref` against the page's placement). Stub `.par` files
(`stub_func_from_par`) carry no convention tag, so the standalone entry point needs a flag
or must assume old-convention input.

**A1.16 `validate_gid2d.py` hard-codes an off-machine raw map.** [hygiene]
Found by the WS0 baseline 2026-11-04. `RAW` (`validate_gid2d.py:25`) is
`T:\SURFACE\NX_school\2026sep\extracted_data\dppc_q2d_gid.txt`; when it is absent, section 6
recorded `check('raw map present', False, 'SKIPPED ...')`, a FAIL that hid the other checks.
*Done (WS0 final-review fixes):* when `RAW` is absent, section 6 now prints a NOTE and
regenerates a raw-format map into `tests/_logs/` from the tracked packed map (the way
`tests/recovered/repro_raw.py` does); `pack_map` output goes to `tests/_logs/` too, so the
harness never writes into tracked `Data/`. All 38 checks run and pass; the `known` entry
in `tests/run_all.py` is dropped.
*Remains:* the hard-coded `T:` path itself (still tried first when reachable), and that
without the T: file, section 6 packs a round-tripped raw map rather than the original.

**A1.17 Two recovered harnesses are stale.** [hygiene] **DONE 2026-11-06 (WS1 Task 7).**
`tests/recovered/smoke_fit.py` reads `scaler_parameters['chi2_per_N_reported']`, which GID2D
now emits as `chi2_per_N_reported_LINEAR` / `chi2_per_N_active_only_LINEAR` (KeyError at the
first print). `tests/recovered/verify_jointmcmc.py` binds a fixed list of XModFit methods
onto a `Fake`; `_draw_corner` now calls `_corner_render_data` (added in `b953c3f`), absent
from the list (AttributeError at section 3). Neither is an engine defect.
*Fix:* update the key names / add `_corner_render_data` to the bound list. Both are
registered `known` in `tests/run_all.py` until then.
*Done:* semantics confirmed, neither was a behaviour change. GID2D has emitted only the
`_LINEAR` keys since its first commit (`dad9f9d`); `chi2_raw`/`chi2_eff` are the padded-count
and active-only chi2/N the harness meant. Its two `chi2_frac_in_top_*` keys were renamed
`chi2_frac_top_*_px_LINEAR`, and `min_sigma_over_median_sigma` no longer exists (the report now
gives `median_sigma_over_I`, a different quantity; the harness only prints it).
`_corner_render_data` is a render-only stride thinning to `_CORNER_MAX_SAMPLES` rows;
the harness now binds it and sets the class constant on its `Fake`. Both `known` entries dropped.

## A2. Data folder

**A2.1 Remove testing artefacts.** The untracked `Data/*.h5` (A0.3), the 3.9 MB
`Data/dppc_q2d_gid_indexed.xfit`, and the `*_test*` outputs that are not fixtures.

**A2.2 Commit the three files that *are* required fixtures.**
`Data/dppc_q2d_gid_indexed.txt` (only regenerable from a `T:\` share),
`Data/i2_tim3_1000_test.txt`, `Data/i2_tim3_1000_test_fitparam.par`.

**A2.3 Re-save `Data/dppc_q2d_gid_indexed.xpar` as a good example.**
It carries `#Fit Scale=Linear` — the notes are explicit that this data must be fit in Log,
because with multiplicative errors one pixel held 72% of the Linear weight — plus the
leaked `_last_xsig`. As the only GID2D example in the repo it teaches the wrong workflow.
*Fix:* re-save from a Log-scale fit after A1.8. (The `E` complaint is struck.)

**A2.4 DONE 2026-09-03.** `Data/1ubq.pdb` (ubiquitin, waters stripped, mass-centred at
the origin) is now the default for all six models, the `_dev_smooth` demo, the joint
validator and the two `.par` fixtures. `Data/tim3_hmmm_ext_all_bestframe.pdb` and
`Data/d6test.h5` were removed from the branch with `git rm --cached` (kept on disk,
gitignored) in `c4300d0`. Two caveats now on the record: the parameter values in those
`.par` fixtures were fitted against the tim3 structure and no longer match the shipped
template, and the **i2_tim3 two-mode case that the multimodal seeding was validated
against is no longer reproducible from a clean clone** -- if that validation needs to be
repeatable, it needs a ubiquitin-based multimodal fixture built for the purpose.
`julia/bakeoff/run.py` and `run_seed.py` still name the tim3 file deliberately.
Removal from the branch is NOT un-publication: the structure has been in the public
upstream `develop` since 2024-09-24 (`6cf39f8`) and is in every existing clone.
Original text follows.

Every PDB model defaulted to
`./Data/tim3_hmmm_ext_all_bestframe.pdb` (`XLayersPDB.py:150`, `_XPR.py:107`,
`_dev_smooth.py:455`). Recommended replacement: **ubiquitin (PDB 1UBQ)** — 76 residues,
~600 atoms, universally recognised, and small enough to speed up every validator and JIT
warm-up. Not membrane-binding, which is fine for a smoke-test default and arguably clearer
than a structure implying a real system. *Fix:* add a stripped `Data/1ubq.pdb`, repoint the
three defaults, and remove the 13 untracked personal MD/crystal structures from `Data/`,
keeping only what a committed validator loads.

**A2.5 Drop the beam-footprint bilayer campaign from the repo.**
In-repo coupling is small: two docstring examples (`joint_covariance.py:18`,
`joint_fit_spec.py:55`) and `validate_jointfit.py`, which hard-codes
`El_Cat_Si_1_{DOPC,HBS_1,Air}_wtrans.txt` + `_c_factors.txt` at `:30-32` and `:210` and
cannot run anyway (its `BASE` defaults to the POSIX path `/mnt/project`). *Fix:* generalise
the two docstrings, delete `validate_jointfit.py`, and extend the already-synthetic
`validate_joint_mask.py` to carry the cross-dataset GLS cases the deleted validator
claimed. This keeps the *capability* and removes the *campaign*. See D2.

**A2.6 DONE 2026-09-03.** Remove the saved `.par` files from `Data/`.
They were per-analysis testing output, not source. `Data/i2_tim3_1000_test_fitparam.par`
was the only one any code consumed -- `validate_xlayerspdb_joint.py`'s "a Nov-2024 `.par`
must still load" test -- so it moved to `tests/fixtures/legacy_xlayerspdb_2024.par`
(byte-identical, tabs intact), which is where a genuine fixture belongs and which starts
the `tests/` directory A0.2 needs. `Data/d6_tim3_1000_test_fitparam.par` had no consumer
and was untracked with `--cached`. `.gitignore` now covers `Data/*.par`, `Data/*.xpar`
and `Data/*.jpar` so saved output cannot drift back in; `tests/fixtures/` is not ignored.
Still to remove under A2.1: `Data/dppc_q2d_gid_indexed.xpar` (untracked already), which
A2.3 wants re-saved from a Log-scale fit if it is kept as the GID2D example at all.

## A3. Dead code, duplication, stale claims

**A3.1** Delete `xmodfit_old.py` (4,428 lines, imported by nothing).
**A3.2** Retire `joint_fit_orchestration.py` — effectively dead; its only caller is inside
a triple-quoted block at `xmodfit.py:1798-1811`. `push_result_to_tables` has no caller;
`_primary_curve` exists in three copies (there, `joint_fit_spec.py`, `xmodfit_old.py`).
**A3.3** Collapse `gid_explorer.html` / `gid_template.html` — byte-identical except line
184 (`{}` vs `__CFG__`), two 79 KB copies, and `write_browser_page` already generates one
from the other.
**A3.4** Remove five stale private `parratt_numba` copies now that `XLayersPDB` imports the
shared optimised one; drop `prange` under `parallel=False` in three of them.
**A3.5** Collapse the **seven** copies of the "is this dataset fitted?" predicate
(`xmodfit.py:1212, 1548, 1609, 6350, 6401, 6696, 7573`), in two forms — two of them omit
the `not hasattr(f,'objectives')` guard, so they are not merely redundant but
inconsistent. `_dataset_is_fitted` already exists; use it.
**A3.6** De-duplicate the Tableau-10 palette (`PlotWidget.py:244`, `xmodfit.py:557`).
**A3.7** `PlotWidget.remove_data` leaks `linestyle`/`fit`/`yerr` entries, and its
`legendItem.removeItem(dname)` (`:436`) is a silent no-op for exactly the label-mismatch
reason that forced `legendItem.clear()` in `Plot`.
**A3.8** Delete dead symbols: `_viridis` (`NGLViewerDialog.py:119-125`),
`fitter._ext_workers` (`Joint_Fit_Routines.py:884`, read nowhere), the dead `names`
argument of `ColumnLimitsDialog`, and the unreachable `emcee()` params
`seed_uniform_frac` / `walkers_per_basin` (no caller sets them — plumb from the GUI or
remove).
**A3.9** De-duplicate the copy-pasted angleindex construction — 5x in `xmodfit.py`, once in
`minimizer.py`. And the four masking validators' copy-pasted `Line` stubs and `check()`.
**A3.10** Call the extracted seeding helpers: `build_orientation_grid` and
`seed_thetaphi_from_costmap` were extracted as reusable, but `Minimizer.emcee` still holds
inline copies. Caveat — the inline versions also fill `chi2mesh` and drive the refine loop,
so this is a real refactor, not a delete.
**A3.11** `xmodfit.py:1058` is one very long line with a run of spaces where a line
continuation got flattened. Parses fine; looks like a botched edit.
**A3.12 Fix stale docstrings and citations:** `JointFit.set_emcee_minimizer:900` still says
the orientation-seeded joint path is serial; `Joint_Fit_Routines.py:243` cites
"xmodfit.py ~3246/3254", which no longer matches; `pdb_cache.py`'s docstring claims every
variant uses it (`_dev_smooth` does not); ~~`XLayersPDB_XPR.py:54-56` and
`pdb_geometry.py:90-92` still assert theta/phi do **not** transfer to XLayersPDB~~ **(done
2026-09-06** -- both now state the equality to scipy `'zx'`, re-verified over 475 angle
pairs**)**; four paths in
`viewer_template.html:14,25` point at skill-package files absent here
(`scripts/make_browser_viewer.py`, `references/*.md`, `assets/ngl.js`).
**A3.13 Reconcile contradictory recorded numbers:** the sigma-floor pixel count (26 in the
notes and `pack_gid_map.py` vs 37 in the shipped file header); `sigma/I` median (0.41 vs
0.47); the `cov`-from-Patterson calibration (`phys_seed.py:3-11` r=0.969 / +/-1.06 vs
`run_multistart.py:128` LOO sd 1.23 / max 2.09); the XPR divergence count (2-then-3 in
`XPR.jl`, "two" then three listed in `julia/README.md:93`, four in
`STAGE2_SEEDING_PLAN.md:26`; actual 5 patches + the Ipopt dep).
**A3.14 Document arm D.** `run_armD.py`, `phys_seed.py`, `seed_utils.py` and 13
`armD-*.npz` files are dated 2026-09-03 — later than every design doc and memory note,
with no recorded verdict. Largest documentation gap in the fork.
**A3.15 Attribution for vendored third-party code:** `Tools/StructureViewer/ngl.js`
(NGL 2.4.0, 1.2 MB) and the vendored `julia/XPR/` need a license/provenance note before a
public push.

## A4. Merge guards — cheap now, expensive to discover later

These are the regions where a careless upstream merge silently disables a feature rather
than conflicting. Worth a comment in the code and a line in the merge checklist.

- **`self.xmin = {}` -> `_LimitProxy` swap in `XModFit.__init__`.** Two lines. If lost to a
  merge, the whole column-limits feature silently stops working while still importing.
- **`Fit.residual`'s mask block must stay ABOVE `maxd = np.max(data)`.** Below it, an
  excluded hot pixel still rescales the residual vector and `log10` of an excluded
  non-positive point reaches `nan_policy='raise'`.
- **Covariance composition order:** slice `[imin,imax]` -> apply `imask` -> concatenate ->
  apply `_cov_mask`, folded in symmetrically in *both* `_build_chol` and `residual`.
- **Highest-conflict functions** for the fold-in: `dataFileSelectionChanged`, `doFit`,
  `update_plot`, `plot_extra_param`, `create_plotDock`, and `Minimizer.emcee`.

---

# Tier B — involved to-dos (features that are not actually finished)

**B1. Picklable objectives for arbitrary models at cores>1.** [unreachable]
Only the PDB and joint paths swap in a picklable log-probability. Stock lmfit still builds
a pool whenever `workers > 1 and HAS_DILL` (`minimizer.py:1791`) and dill is installed, so
any *other* model at cores>1 hits the original `cannot pickle 'Fit' object` crash — the
objective is bound to a Qt `QObject` holding a `pyqtSignal`. Needs either a generic
picklable-objective factory over the existing `_ext_objective` hook, or a real separation
of the model from the Qt signal holder. Design work, then bit-for-bit validation per model
family.

**B2. Correlated-error / GLS fitting, end to end.** [unreachable + wrong-result risk]
Three gaps:
(a) the shared-beam **footprint** covariance term cannot be reached from the GUI at all —
`_use_footprint` needs `cov['v']`/`cov['var_fwhm']`, which only `load_bundle_npz` returns,
while the panel loads sidecars via `load_dataset_cov`, and `load_cov_npz` drops those keys
even when the `.npz` contains them;
(b) covariance/GLS joint **MCMC is still serial** — `JointLogProb` has no whitened
residual, so `_attach_joint_parallel` is gated on `not self._use_cov`;
(c) the real-data regression leaves with the campaign (A2.5), so the cross-dataset GLS
cases need rebuilding synthetically.
Do (c) first — it is the harness that makes (a) and (b) safe to touch.

**B3. Subgroup (arbitrary-subset) parameter sharing needs a UI.** [unreachable]
Engine, spec and `.jpar` all support sharing across an arbitrary subset of datasets; the
panel offers only a share-across-ALL checkbox, so today it requires a hand-written `.jpar`.
A headline capability of the fork that no user can reach.

**B4. `doFit` routing and the mixed-selection question.**
The joint **MCMC** route was deliberately hoisted above `doFit`'s individual-fit guards;
the joint **least-squares** route was not, and works only because
`clear_individual_selection` blocks signals and leaves `sfnames` stale. Separately, a Ctrl
overlay where both lists have selections silently runs a whole-spec joint fit and ignores
the individual selection. Both were deferred pending the planned overlay overhaul; settle
them together.

**B5. GID2D is not finished as an analysis.**
The fit-and-interpret step was never done (does one NN-tilted phase with `sig_t` free fit
both features?), and the tilt reading is a model choice the map does not support: at
`t=30.5` the model's (11) sits at q_xy 1.28 against the data's 1.356, and freeing `t`/`eps`
runs away to a0=17 A / t=60 deg. Also unresolved: `window='engine'` sets `active = err > 0`
and discards the q window, so the 1-D reductions, the chi2 and the model panel's
out-of-window view may not match their own docstrings. `GID2DMapDialog.py` is planned and
absent; GID explorer coupling is unbuilt (`cfg_from_func` reads only `func.E`).

**B6. Structure viewer: six catalogued improvements.**
Canned/side-on orthographic views (the default `autoView()` lands looking from above, the
wrong view for a layered interface), length-scale-normalised depth colours,
contrast-normalised slab colours, a responsive panel merging the layer list with the
legend, macOS compatibility, distance/protrusion measurements. **Warning on item 3:**
`_DENSITY_STOPS` is deliberately absolute — a stack-relative map made every slab colour
jump on any rho edit, and there is a 25-line comment saying so. Also merge the near-clone
`Tools/GrazingIncidence/GIDViewerDialog.py`, noting a shared base class cannot live under
`Tools/` because that loader turns every `.py` there into a menu item.

**B7. Seeding completeness.**
`_seed_shared_axis` (mixed sharing) never calls `_detect_orientation_basins`, so the
guaranteed-walkers-per-basin property does not hold there. `seed_uniform_frac` and
`walkers_per_basin` reach no caller. The two spawn pools (seeding + sampling) should be
unified into one. Of the four originally-planned improvements, inverse-CDF and temperature
landed; the sampler-move tuning never did.

**B8. XLayersPDB variant triage.**
Decide which of `_G`, `_H`, `_box`, `_dev_smooth` survive. Evidence: `_G`/`_H`/`_box` share
the `mkey` bug and have no `.par` or fixture anywhere, suggesting they were never fitted
with; `_dev_smooth` is the best-documented of the set but measured *worse* MCMC mixing than
plain `XLayersPDB` (tau ~384 vs 238, acceptance 0.075 vs 0.139, ~3x slower) on a single
caveated run. Keep `_G`'s exact `greens_grid_n` regardless — it is the natural reference for
checking `soft_area`. Also check `XLayersPDB_H.py:632`
(`layarea = 1 - Volfrac/xyarea` divides a sphere-segment volume by an area, where `_G`'s
identical line is correct because its `Volfrac` is an area — probably a missing `1/fslice`),
and fix `pdb_cache.pdbread` keying on filename only, so an in-place PDB edit is never
invalidated.

**B9. `XLayers_Beamfootprint` decision.** It is the *alternative* to the GLS approach —
footprint as fit parameters (`fwhm`/`sub_wth`) inside the model — not a component of it. It
forked a **pre-merge** `XLayers.py`, dropped `rrf`, and its private `parratt_numba` returns
a scalar `r` where the shared one returns the `refc` array (`:40`). With the campaign
leaving the repo it has no in-repo consumer: rebase it on the current `XLayers.py` and give
it a fixture, or drop it.

**B10. XPR stage 2 and 3 proper.** Stage 2 (global seeding returning a *list* of minima)
exists only as a research harness: no `CONFIG.json` is committed, `make_tim1_config.py` is
referenced at `run_multistart.py:10` but does not exist, data paths point outside the repo,
and `validate_xpr_seed.py` is specified but unwritten — so `julia/bakeoff/` cannot run on a
fresh clone. Stage 3 (feeding the minima to MCMC, possibly gradient-based HMC) is not
started. `julia/bakeoff/README.md` documents 6 of 15 scripts.

**B11. Reproducibility of the XPR/Julia environment.** `validate_xpr.py` needs
`XPR-main/test_data`, which lives as a *sibling* of the repo, so a collaborator cannot run
it. Julia 1.11.9 auto-downloads ~1 GB. The recorded environment traps (broken stdlib SSL in
`cmcars`, juliapkg keying its install dir on `CONDA_PREFIX`, a broken `tomli`) are handled
in `get_jl()` but only for the two known machines.

**B12. Upstream-owned decisions, to raise with the collaborators rather than fix here.**
`EMCEE_Widget.py` (upstream's independent non-lmfit MCMC, 780 lines) is imported by
nothing, and `Fit.log_probability`/`log_likelihood`/`log_prior` exist only to serve it — so
the repo carries two MCMC implementations, one dead. `Functions/XFNTR/XFNTR_Plus.py`
crashes with `KeyError: 'filepaths'` on load (one-line fix, deferred, fails identically on
clean upstream). `xraydb/` contains only `__pycache__` with seven orphan `.pyc` and no
`__init__.py`; `xraydb_bak/` is 10.3 MB, tracked upstream, referenced by nothing.

**B13. A PDB preparation step for the XLayersPDB family.**
Requested 2026-09-03. Every model in the family silently assumes its template has
already been prepared by hand, and there is nothing in the repo that does the preparing
or checks that it was done. Three separate assumptions, each of which currently fails
quietly:

- **waters and heteroatoms are already stripped** -- `pdb_cache.pdbread` counts every
  atom Bio.PDB returns (A1.13), so a raw RCSB download inflates electrons, radii and
  mass, and with them `protvol`, the coverage normalisation and the EDP;
- **coordinates are already centred** -- the model rotates about the origin (A1.14),
  so an off-centre template swings through an arc as theta/phi vary;
- **every element parses** -- `pdb_cache.pdbread` indexes the element table directly
  (`Functions/pdb_cache.py:83`) with only a special case for `CAL`, so an unusual atom
  name raises a bare `KeyError` from deep inside the parse.

Wanted: one preparation entry point -- CLI plus a call the model can make -- that strips
waters and heteroatoms with an **opt-in to retain genuine bound ions and ligands** (the
structure viewer already advertises showing those, so the filter must not be
unconditional), centres by mass-weighted CoM, optionally adds hydrogens, validates that
every element resolves, and reports atom count, total electrons, mass and bounding
extent so the result can be sanity-checked. Then either prepare-on-load or refuse to load
an unprepared file, rather than fitting one silently.

`Data/1ubq.pdb` was produced by exactly this procedure by hand, so the script is mostly
a matter of packaging what already had to be done once. That one-off is in the session
scratchpad as `mk_ubq.py` and is worth starting from. Relevant to the 13 personal
structures in `Data/` too: whichever survive A2.4 should go through the same step, so it
is knowable which are centred.

---

# Decisions needed

**D1. RESOLVED 2026-09-04 — keep `XLayersPDB_box` only.** `XLayersPDB_G`,
`XLayersPDB_H` and `XLayersPDB_dev_smooth` are retired. The surviving PDB family is
`XLayersPDB` (base), `XLayersPDB_box`, `XLayersPDB_XPR`. Consequences: A1.2's `mkey` bug
now needs fixing in `_box` only; both PDB-name tuples shrink to three entries, which
simplifies A1.1; `tests/recovered/compare_models.py` (XLayersPDB vs `_dev_smooth`) and the
variant loops in `validate_perdataset_fit.py` and `tests/recovered/test_phase1.py` need
updating. All three files stay recoverable from git history, so nothing is lost — but note
that `_G`'s exact `greens_grid_n` was the natural reference for checking `_box`'s
`soft_area`, and `_dev_smooth` carried a `protins_new = protins_old - fslice` registration
warning against old `XLayersPDB` fits. Both are worth reading before the files go.

**D2. RESOLVED 2026-09-04 — the capability stays; the specific analysis goes.**
The beam-footprint model and the ability to run that kind of joint fit are wanted. What
must not be in the repo is the particular analysis that was conducted — which was not
performed with XModFit's tools, so its specifics would only mislead. Concretely: keep
`XLayers_Beamfootprint`, keep `_use_footprint` and the `v`/`var_fwhm` plumbing in
`joint_covariance.py` and `Joint_Fit_Routines.py`, and keep B2a on the list as
"wire it up so it is reachable from the GUI". Remove per A2.5 the campaign specifics: the
`El_Cat_Si` / `TGP_Qz` docstring examples and `validate_jointfit.py`'s hard-coded dataset
names. This also settles **B9** — `XLayers_Beamfootprint` is kept, so it needs rebasing on
the current `XLayers.py` and a fixture of its own, not dropping.

**D3. RESOLVED 2026-09-03** — ubiquitin (1UBQ) adopted as the default template; the
unpublished tim3 structure and `d6test.h5` removed from the branch.

**A2.1 RESOLVED 2026-09-04.** The personal structures are **not** to be deleted from disk,
but none ship: users get one example template. Implemented as a `.gitignore` rule
(`Data/*.pdb` with `1ubq`/`Nd9`/`Pr4` exempted) rather than by deleting files, so adding
another needs a deliberate `git add -f`. No per-file decision needed.

**D4. DEFERRED 2026-09-04 — nothing is being pushed yet.** The sequence is: this fix
package -> the audit -> the fast-scan feature -> reorganisation and condensing -> only then
a push to `chemmatcars/XModFit`. XModJointFit exists precisely so the real repo was not
disturbed while this work happened. So the commit-strategy question is not live, and the
`.githooks/pre-push` guard stays in place for the duration. Revisit at the reorganisation.

**Also settled 2026-09-04:** `Data/d6test.h5` holds no meaningful result — it only proves
MCMC runs at all — so the 26.8 MB blob does not need recovering. See the note under A0.4
about why reclaiming the space is nonetheless not urgent.
