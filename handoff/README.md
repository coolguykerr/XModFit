# XModJointFit change record — index

This directory documents everything this fork adds to XModFit, for the fold-in back
into `chemmatcars/XModFit` and for the Claude instances that will work there. Written
2026-09-03. One document per change category; each follows `_TEMPLATE.md`.

Read this file, then the category document you need. Do not assume a feature works
because it is documented: see **Verification status** below.

---

## 1. Baseline — what "stock XModFit" means here

| | |
|---|---|
| upstream remote | `origin` = https://github.com/chemmatcars/XModFit (NOT a personal fork) |
| upstream baseline | `origin/develop`, tip `81e53c8` |
| local branch | `develop`, 6 commits ahead, 0 behind |
| upstream already merged in | yes — merge commit `a4fd04e` (2026-08-05) |
| safety branch | `pre-merge-backup-b3e1ab5` = the pre-commit tip |

Local commits not in upstream:

```
babe1aa  Fix MCMC confidence-interval workflow when run without a prior least-squares fit
a4fd04e  Merge upstream chemmatcars/XModFit develop into joint-fit work
c5f697d  Joint-fit system: global/subgroup parameter sharing + covariance (GLS) fitting
b3e1ab5  MCMC Additions for Angles for XLayersPDB
d955174  MCMC Additions for Angles for XLayersPDB
d3e353e  MCMC Additions for Angles for XLayersPDB
```

Nothing has been pushed. The collaborators are working from `origin/develop`, i.e.
from the state *before* all of the above.

## 2. The change set has three buckets

> **Superseded 2026-09-03, later the same day.** Everything that was uncommitted or
> untracked has since been committed to `develop` in eight themed commits, `0296ec3`
> through `b2b76ac`. The bucket statements throughout this directory — including every
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
trees. There is **no `.gitignore`**, which is why they all show up. See
`09-validation-and-env.md` section 8 for the keep/ignore/drop recommendation.

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
| 10 | `10-merge-and-consolidation.md` | the fold-in plan: conflict surface, ordering, and what to consolidate or drop first | — |
| 11 | `11-audit-plan.md` | the deferred audit: the registry of unverified claims and how to test each | — |

Two research write-ups predate this directory and are **not** duplicated here:
`STAGE2_SEEDING_PLAN.md` and `SEEDING_STUDY_TIM1.md` at the repo root (both untracked).
`05-xpr-julia.md` summarises them and says which of their sections are superseded.

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
