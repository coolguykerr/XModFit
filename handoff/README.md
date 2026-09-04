# XModJointFit change record — index

This directory documents everything this fork adds to XModFit, for the fold-in back
into `chemmatcars/XModFit` and for the Claude instances that will work there. Written
2026-09-03. One document per change category; each follows `_TEMPLATE.md`.

Read this file, then the category document you need. Do not assume a feature works
because it is documented: see **Verification status** below.

> **September 2026: the seeding study has been moved out.** A separate investigation into
> global-optimum seeding for XLayersPDB-class fits was carried out in this tree, because
> the XPR backend lives here. **It is not part of XModFit and must not reach
> `chemmatcars/XModFit`.** On 2026-09-04 it was extracted to its own repository,
> `../xpr-seeding-study` (177 files, including the 107 `endpoints_*.npz` run archives that
> are ~30 h of compute and the regression targets for anything built on it). Its four
> workstream handoffs, its write-ups and the original exclusion manifest went with it.
>
> Two things remain to be done here, tracked as **A0.6** in `10-fix-list.md`: commit
> `7942458` still holds the tracked half of the study (114 files, ~3 MB, mostly binary) in
> local history, and one genuine XModFit feature — the fast-scan described in that
> repository's `PLAN_XMODFIT_FASTSCAN.md` — still needs implementing here. Nothing has
> been pushed. `.githooks/pre-push` refuses any push carrying study paths; enable it with
> `git config core.hooksPath .githooks`.

---

## 1. Baseline — what "stock XModFit" means here

| | |
|---|---|
| upstream remote | `origin` = https://github.com/chemmatcars/XModFit (NOT a personal fork) |
| upstream baseline | `origin/develop`, tip `81e53c8` |
| local branch | `develop`, 19 commits ahead, 0 behind |
| upstream already merged in | yes — merge commit `f3335f9` (2026-08-05) |
| safety branches | `pre-merge-backup-b3e1ab5` (pre-commit tip, *pre-rewrite* hashes), `pre-h5-rewrite-bdb44ef` |

Local commits not in upstream, newest first. **Hashes below `013eabe` differ from those
recorded elsewhere in these documents**: `filter-branch` rewrote all of them on 2026-09-03
to drop a 26.8 MB `.h5` blob (fix-list A0.4), so `d3e353e` is now `7b37f15`, `c5f697d` is
`765754e`, `a4fd04e` is `f3335f9`, `babe1aa` is `5662813`.

```
edc8b80  Clear saved .par files out of Data/; add the 1UBQ generator and a PDB-prep to-do
8251c0a  Fix list: record the 1UBQ default, and two new PDB-handling defects
c4300d0  Remove d6test.h5 and the unpublished tim3 structure; ship 1UBQ as the default
a68f7a3  Note in the handoff index that the untracked work is now committed
4b56db6  Add handoff documentation: one change record per category, plus a fix list
3ca25a0  Add requirements snapshots for the viewer and the working environments
d662335  Add regression harnesses for the fit-mask, column-limit and joint-PDB paths
b953c3f  Per-column fit limits, multimodal MCMC seeding, parallel joint sampling, plotting
433fab7  Add picklable MCMC objectives and parallel seed-scan workers
7942458  Add XPR global-seeding bake-off harness, saved endpoints and study write-ups  <-- MUST NOT be pushed, see A0.6
a1a2a62  Add XLayersPDB_XPR: vendored Julia differentiable reflectivity model + bridge
dad9f9d  Add GID2D: 2D grazing-incidence map fitting, 2D Map tab and packing tools
013eabe  Add .gitignore for caches, MCMC backends and saved fit output
5662813  Fix MCMC confidence-interval workflow when run without a prior least-squares fit
f3335f9  Merge upstream chemmatcars/XModFit develop into joint-fit work
765754e  Joint-fit system: global/subgroup parameter sharing + covariance (GLS) fitting
1293395  MCMC Additions for Angles for XLayersPDB
c1815bb  MCMC Additions for Angles for XLayersPDB
7b37f15  MCMC Additions for Angles for XLayersPDB
```

Nothing has been pushed. The collaborators are working from `origin/develop`, i.e.
from the state *before* all of the above.

## 2. The change set has three buckets

> **Superseded 2026-09-03, later the same day.** Everything that was uncommitted or
> untracked has since been committed to `develop` in eight themed commits, `013eabe`
> through `a68f7a3` (hashes as rewritten; see section 1). The bucket statements
> throughout this directory — including every
> category document's section 2, and the many "untracked working-tree only" notes —
> describe the state at the time of writing, i.e. at `babe1aa`. They are still the right
> guide to *what was ours versus upstream's*, which is what the merge cares about; they
> are no longer a statement about risk of loss. Deliberately left out of those commits:
> the personal PDB structures and test outputs in `Data/`, `xmodfit_old.py`, the
> in-flight `Bnarrow` bake-off arm, and the two modified `.idea/` files.

Every category document records which bucket each of its files is in, because the
merge treats them very differently.

| bucket | how to see it | size |
|---|---|---|
| **committed** locally, not upstream | `git diff origin/develop...HEAD -- <path>` | +9,702 / -151 over 33 files |
| **uncommitted** edits to tracked files | `git diff -- <path>` | +4,002 / -357 over 13 files |
| **untracked** new files (never committed) | whole file is new | ~15,700 lines over ~50 source/doc files |

Reconciling the "+60k lines" figure: ~29,400 lines of that is Python/Julia/HTML source
and documentation. The rest is artefacts — `xmodfit_old.py` (a 4,428-line backup copy),
92 `julia/bakeoff/endpoints_*.npz` result arrays, 31 untracked `Data/` files (PDB
structures, `.h5` MCMC backends, datasets, `.xpar`/`.par` fit files), and `__pycache__`
trees. There was **no `.gitignore`** at the time, which is why they all showed up; one was
added in `013eabe`. See `09-validation-and-env.md` section 8 for the keep/ignore/drop
recommendation.

## 3. Category documents

| # | Document | Covers | Bulk of it lives in |
|---|---|---|---|
| 01 | `01-joint-fit.md` | joint/global multi-dataset fitting: parameter sharing, subgroups, covariance/GLS, per-dataset ranges, Joint Fit panel, `.jpar` | `Joint_Fit_Routines.py`, `joint_fit_*.py`, `joint_covariance.py` |
| 02 | `02-mcmc-stack.md` | emcee integration: vendored `lmfit/`, picklable parallel objectives, `.h5` backend + fingerprint, autocorrelation, corner plots, the two MCMC windows | `lmfit/*`, `pdb_logprob.py`, `pdb_joint_logprob.py`, `xmodfit.py` |
| 03 | `03-orientation-seeding.md` | (theta,phi) walker seeding for XLayersPDB MCMC: grid scan, temperature, basin detection, angle wrapping, parallel seed workers, joint seeding | `lmfit/minimizer.py`, `pdb_*seed_worker.py`, `Joint_Fit_Routines.py` |
| 04 | `04-pdb-xrr-models.md` | the XLayersPDB model family and its variants, beam-footprint layer model, shared PDB cache | `Functions/XRR/*`, `Functions/pdb_cache.py` |
| 05 | `05-xpr-julia.md` | the XPR Julia differentiable model, its Python bridge, and the Sobol/Ipopt/homometry global-seeding study | `julia/`, `Functions/xpr_backend.py`, `Functions/XRR/XLayersPDB_XPR.py` |
| 06 | `06-gid2d.md` | 2D (q_xy,q_z) GID map fitting, the 2D Map plot tab, map packing tools | `Functions/GISAXS/GID2D.py`, `Functions/gid2d_core.py`, `Image2DWidget.py`, `Tools/GrazingIncidence/` |
| 07 | `07-structure-viewer.md` | the NGL protein-in-layers 3-D viewer driven by fit parameters | `Tools/StructureViewer/` |
| 08 | `08-core-app.md` | per-column fit limits and the mask engine, per-dataset fit memory, plotting/legend/overlay work, fit-engine miscellany | `xmodfit.py`, `Fit_Routines.py`, `PlotWidget.py` |
| 09 | `09-validation-and-env.md` | the `validate_*.py` harnesses, conda environments and headless recipe, requirements snapshots, repo hygiene | repo root |
| 10 | `10-fix-list.md` | what to fix before consolidating, and the involved to-dos where a feature is unfinished | — |
| 11 | `11-audit-plan.md` | the deferred audit: the registry of unverified claims and how to test each | — |
| 12 | `12-work-package-preaudit.md` | the pre-audit fix package handed to an implementation session: phases, guardrails, progress log | — |

Two research write-ups that `05-xpr-julia.md` summarises — `STAGE2_SEEDING_PLAN.md` and
`SEEDING_STUDY_TIM1.md` — **moved to `../xpr-seeding-study` on 2026-09-04** along with the
rest of the study, so its references to them as "at the repo root" now point there. It
says which of their sections are superseded.

## 4. Verification status — read this before trusting anything

These documents were written by reading the code and the development notes. They were
**not** produced by running the code. An audit pass is deliberately deferred and will
work from section 7 ("Claims to verify") of each document, consolidated in
`11-audit-plan.md`.

Rough confidence by category, as recorded by the chats that built each one:

- **Validated headless, and exercised live by the user:** joint least-squares fitting,
  individual XLayersPDB MCMC including the new seeding, the plotting/overlay work.
- **Validated headless, never clicked through live:** several joint-MCMC GUI paths, the
  per-dataset fit ranges in a real 2-dataset joint fit.
- **Research-grade, not a user-facing feature:** everything under `julia/` and the
  bake-off harness.
- **Known incomplete:** GID2D's fit-and-interpret step; every one of the six catalogued
  structure-viewer improvements; covariance/GLS joint MCMC is still serial.

## 5. Conventions that will bite you

Three hard-won rules that are not visible from the code, repeated here because breaking
any of them fails silently:

1. **Every `.py` in a `Functions/<category>/` folder is treated as a fittable model
   class named after the file.** Never put a helper module there — it will be listed as
   a model and crash the function list. Shared helpers go in `Functions/` root
   (`pdb_cache.py`, `gid2d_core.py`, `pdb_geometry.py`, `xpr_backend.py` are all there
   for this reason).
2. **Fixed-parameter cell text is `eval()`ed.** A `choices` value that happens to be a
   valid Python name comes back as that object, not the string — this is why the GID2D
   mode is `'map2d'` and not `'map'`.
3. **Headless testing needs the `cmc-py39` interpreter**, `QT_QPA_PLATFORM=offscreen`,
   an injected `xraydb` stub, and an `if __name__ == '__main__':` guard in any script
   that reaches a spawn multiprocessing Pool. Details in `09-validation-and-env.md`.
