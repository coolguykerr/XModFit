# Audit plan

Written 2026-09-03, amended 2026-09-04 after the seeding study was extracted.

The nine category documents in this directory were written by reading code and
development notes, **not** by running anything. Each ends with a section 7 recording what
its author could not confirm. This is the plan for working through those, in the order
that puts the highest-consequence claims first.

**138 recorded claims** across the nine documents (07 gained #12 on 2026-10-26, for fix-list A1.15):

| doc | claims | doc | claims |
|---|---|---|---|
| `01-joint-fit.md` | 14 | `06-gid2d.md` | 13 |
| `02-mcmc-stack.md` | 12 | `07-structure-viewer.md` | 12 |
| `03-orientation-seeding.md` | 20 | `08-core-app.md` | 15 |
| `04-pdb-xrr-models.md` | 12 | `09-validation-and-env.md` | 15 |
| `05-xpr-julia.md` | 25 | | |

Defects already identified (as opposed to claims needing checking) live in
`10-fix-list.md`. This document is about establishing what is *true*; that one is about
what to *change*. They cross-reference by item ID.

---

## 1. Ordering principle

Audit in order of what a wrong answer costs:

1. **Publishing risk** — irreversible once it happens.
2. **Silently wrong numbers** — a scientist reports a value that is wrong and nothing
   complained. Worst realistic outcome for this codebase.
3. **Crashes** — loud, annoying, cheap to find, cheap to fix.
4. **Performance and measurement claims** — the recorded speedups and comparisons that
   decisions were based on.
5. **Feature claims never exercised live** — mostly GUI paths.

Not "in document order", and not "easiest first".

## 2. Preconditions

Before any of this is worth starting:

- **P-1. The harness has to actually fail.** `check()` in every `validate_*.py` does not
  assert; the verdict rides the exit code, so under pytest a failing check reports PASSED
  (fix-list A1.10). Nothing below is trustworthy until this is fixed. **Do this first.**
- **P-2. There must be a runner.** No `tests/`, `conftest.py`, `pytest.ini` or Makefile
  existed; `tests/fixtures/` now exists but nothing runs the suite. Needs one entry point
  that runs all validators and reports a single pass/fail.
- **P-3. Recover the off-repo harnesses** (A0.2). ~17 `verify_*.py` scripts live only in
  per-session temp scratchpads and are the *only* coverage for the seeding work and for
  core-app sub-features (b) and (c). Some are cited as proof in the notes for claims this
  audit is meant to check — so without them, those claims have no evidence at all.
  Exclude `verify_joint_parallel_scan.py`, which asserts the opposite of current
  behaviour.
- **P-4. Environment.** `C:\Users\kerr\.conda\envs\cmc-py39\python.exe`,
  `QT_QPA_PLATFORM=offscreen`, an injected `xraydb` stub, and `if __name__ == '__main__'`
  in anything that reaches a spawn Pool. XPR work additionally needs `juliacall` and
  `XPR-main/test_data`, which sits *outside* the repo. Full recipe:
  `09-validation-and-env.md`.

## 3. P0 — publishing risk

**Resolve before any `git push`, regardless of everything else.**

| # | Item | State |
|---|---|---|
| P0.1 | Seeding study out of the fold-in path (fix-list A0.6) | study extracted to `../xpr-seeding-study`; `.gitignore` and `.githooks/pre-push` in place; commit `7942458` still in local history, 15 commits back — decision B pending, and **now unblocked**: `handoff/seeding-study/AUDIT-7942458.md` inventories all 114 files and verifies (2026-09-06) that no later commit touches any of them and that all 114 are preserved in the study repo, 0 absent |
| P0.2 | `Data/d6test.h5` 26.8 MB blob out of history (A0.4) | done, `filter-branch`; backups still hold the blob until `gc` |
| P0.3 | Unpublished tim3 structure out of the branch (A2.4) | done — but it remains in upstream's *published* history since 2024-09-24, which a branch deletion cannot undo |
| P0.4 | Vendored third-party code has attribution (A3.15) | not done — `ngl.js` (NGL 2.4.0, 1.2 MB) and `julia/XPR/` |
| P0.5 | No personal/unpublished data left in `Data/` (A2.1) | 13 structures + test outputs still untracked-but-present; needs a keep/drop decision |

Verification for P0.1, to be run before every push until B is done:

```bash
git diff --stat origin/develop..<branch> -- julia/bakeoff \
    SEEDING_STUDY_TIM1.md STAGE2_SEEDING_PLAN.md   # must print nothing
```

Run today it prints **114 files changed, 3035 insertions** — that is the expected state
until B is done, not a fault in the check. The 114 files of `7942458` are still in
`HEAD`'s tree and still tracked (112 under `julia/bakeoff/` plus the two root write-ups),
because "extracted to its own repository" copied them out; it never removed them here.
`.gitignore` covers the paths only for the *untracked* half. So the command is a genuine
gate: it must print nothing before a push, and it will only do that once `7942458` is
dropped or the branch is curated.

`.githooks/pre-push` is the belt to this braces — it scans the commit range with
`git log --name-only`, so it also catches a branch that adds and then deletes study
files, which the tip-only diff would miss.

One dependency runs the other way, and it is why B is deferrable but not indefinitely:
**the 703 endpoint archives the study's published numbers rest on were produced in this
repo's gitignored `julia/bakeoff/`.** They were copied into the study repo on 2026-09-06
with per-file SHA-256 verification, so `julia/bakeoff/` here is now disposable — and note
that route B's rebase *will* delete these 114 files from disk. Before doing it, confirm
`sha256sum -c julia/bakeoff/MANIFEST.sha256` passes in `../xpr-seeding-study`, since that
repo has no remote and is the sole copy.

## 4. P1 — claims where being wrong corrupts a reported number

These are the audit. Everything else is housekeeping by comparison.

| # | Claim to test | From | How |
|---|---|---|---|
| P1.1 | The covariance composition order (slice -> `imask` -> concatenate -> `_cov_mask`) is symmetric in `_build_chol` and `residual`, so GLS whitening stays aligned | 01, 08 | `validate_joint_mask.py` covers this synthetically; extend to the cross-dataset cases the deleted real-data validator claimed (A2.5). Assert `len(r) == len(_cov_mask)` fails loudly when broken. **2026-11-04:** `validate_joint_gls.py` adds, all on SYNTHETIC data: G1 = whitening identity with the footprint term ON (not the real El_Cat covariance the original claim named); G3 = engine stderr vs the analytic GLS covariance for a LINEAR model; G4 = two datasets with DIFFERENT masks and an off-diagonal Sigma_logc, whitened residual equal to an independently built C_active Cholesky solve on the full x-window; G4b = the same with per-dataset fit windows cutting into both masked regions, so the slice -> `imask` order under covariance IS now covered (reference = per-dataset window AND mask, then the independently built full C restricted to those rows; `_cov_mask` all True). `validate_joint_mask.py` covers the no-covariance mask/window cases and the covariance length guard, but NOT the order with a narrowed window. Claim 01 #2 is verified on synthetic data only; claim 01 #4 is PARTLY CONFIRMED: engine stderr matches the redchi-scaled GLS covariance (lmfit `scale_covar=True`) for a linear model; whether reported GLS errors should be scaled or unscaled is open -- a WS2 reporting-contract decision; MCMC comparison untested. |
| P1.2 | Seed workers ignore the per-column mask, so "parallel == serial" is false with a column limit active (A1.3) | 03 | Build a masked case, run the scan serial vs parallel, diff cell costs. Expected to FAIL — this is a confirmation, not a check |
| P1.3 | `_eval_chisqr` compares joint vs individual on mismatched point sets when per-dataset ranges are used (A1.4) | 01 | Two datasets with different `#FitRange=`, compare its chi2 against one computed on `active_index()` |
| P1.4 | GID2D must be fit in Log scale; one pixel holds 72% of the Linear weight | 06 | `GID2D._weight_report` on the shipped map; check the 72.2% and the `L_dom` 144 -> 313 A shift |
| P1.5 | `window='engine'` sets `active = err > 0` and discards the q window, so reductions/chi2 may not match their docstrings | 06 | Compare `redchi` and the 1-D reductions between `window='model'` and `'engine'` on a windowed map. Recorded: identical params, `redchi` 0.131 -> 0.271 = exactly n/n_active |
| P1.6 | XPR `cov_norm='protvol'` removes the dimensional inconsistency in `area_slices_sphere_seg!`, and `xb.grad` is exact with `cov` raw | 05 | Reference log-likelihoods (2box -1214.1600, 3box -600.6398, 4box -62408.5320, protein+2box -466.4557) to 1e-4; finite-difference the gradient off the grid-alignment kinks. **The gradient half is DONE 2026-09-08**: AD vs central differences is at the round-off floor for every parameter, and the only remaining kink is `sum(d)` on a 0.5 A node (`handoff/05-xpr-julia.md` 3.9, claim 26). Rebaselined onto the 2026-09-08 upstream |
| P1.7 | theta/phi transfer between XPR and XLayersPDB (bit-identical to scipy `'zx'`); `cov` and `protins` do not | 05 | **PARTLY REFUTED 2026-09-04** (`xpr-seeding-study/TIM4_STAGE0_INVENTORY.md` §6b, mTIM4 corpus). `protins` **does** transfer: offset -0.01 A / -0.17 A on two datasets, below the 0.5 A grid. `cov` transfers **through a documented deterministic conversion**, not "not at all": `'approx'` wants `cov = frac*cov_scale(theta,phi)` (63-90 for this template), `'exact'` wants `cov = frac`. theta/phi confirmed independently here - XPR's theta optimum lands within 2 deg of the archived value and 180-theta is worse by 9-13x. So the correct statement is **all four pose coordinates transfer; absolute `logL` does not** (model discretization differs by 0.17-0.36% in R, which is 0.25-0.6x these data's error bars and therefore 20-42% in chi2). The two source docstrings that asserted the retracted theta/phi claim (`Functions/pdb_geometry.py:xpr_rotation_matrix` and `Functions/XRR/XLayersPDB_XPR.py`) were **fixed 2026-09-06** (A3.12) |
| P1.8 | `pdb_cache.pdbread` counts waters as protein (A1.13) | new | Parse a raw vs stripped structure, diff electron totals. Expected to FAIL |
| P1.9 | The MCMC fingerprint hashes `y[active_index()]`, so a backend from an unmasked run is rejected for a masked one | 08 | Round-trip an `.h5` across a mask change |
| P1.10 | Angle canonicalisation is a relabelling only — chi2 and residuals untouched | 05, 08 | Fit with an out-of-window start, compare pre/post `chisqr`. Recorded caveat: a reflected theta flips the sign of its correlations |

## 5. P2 — crash claims

| # | Claim | From | How |
|---|---|---|---|
| P2.1 | The three disagreeing PDB-name lists give `_XPR`/`_G`/`_H` a `CustomSampler` with a stock `HDFBackend`, raising an uncaught `TypeError` at `minimizer.py:2300` (A1.1) | 02, 03, 04 | Start a short MCMC on `XLayersPDB_XPR` through the GUI path. Found by three agents independently; highest prior of anything here |
| P2.2 | `_G`/`_H`/`_box` cannot survive a joint fit — `mkey` used before assignment, masked by a loop that only runs when `__fit__` is False (A1.2) | 04 | Compile each into a two-dataset joint fit |
| P2.3 | Mask + `__masked_x__` pops a modal dimension-mismatch dialog (A1.5) | 08 | A masked GID2D fit through `update_plot` |
| P2.4 | Non-PDB models at cores>1 still hit `cannot pickle 'Fit' object` (B1) | 02 | Any non-PDB model, `emcee_cores=4` |
| P2.5 | `_box`/`_G`/`_H` leave `fiterr` scalar, so `minimizer.py:2077` `fiterr[ind]` raises when only one of theta/phi varies | 03 | Vary theta only |
| P2.6 | `XFNTR_Plus.py` raises `KeyError: 'filepaths'` on load, identically on clean upstream | 02, B12 | Load it. Deferred by the user — confirm, do not fix |

## 6. P3 — measured claims the design decisions rest on

Lower stakes, but these are the numbers that justified the architecture, and several were
already found to be stated against a superseded implementation.

- Seeding speedup ~6.2x on 32 cores; sampling 70.7 -> 13.8 ms/step (~5x). Measure with
  the delta of two step counts, never total wall time — pool spawn plus numba JIT is a
  one-time ~15-25 s cost that made a 40-step test look slower.
- The "~0.05 s constant per tau call" figure was measured on the *windowed* live-tau
  variant, which was later reverted to full-chain plus a throttled cadence. Per-call cost
  now grows with N and only the cadence bounds the total — **the recorded number does not
  describe the shipped code** (02, audit hook). Re-measure.
- `XLayersPDB` vs `_dev_smooth` mixing (tau ~238 vs ~384, acceptance 0.139 vs 0.075):
  single run, tau in the "chain too short" regime, and `_dev_smooth` forced off its
  recommended `dz`. Treat as unresolved; it is the main evidence in the B8 triage.
- Analytic Jacobian: 261 -> 41 model calls, gated at `nvarys >= 4` because a loss was
  measured below that.
- `projected_area` matches XPR's Green's-theorem union area to 0.07-0.31% at `fslice=0.5`.

## 7. P4 — never exercised live

Requires a human at the GUI; produce a click-through checklist rather than a script.

- The joint MCMC window end to end: subset corner plot, summary heatmap, CI table, pop-out.
- Per-dataset `#FitRange=` in a real two-dataset joint fit.
- Parallel joint sampling at cores>1 on a real `XLayersPDB` joint fit (the synthetic
  per-step benchmark is meaningless — `SynPDB.y()` is trivial, so IPC dominates).
- The structure viewer: rendering, hover-through-slab picking, camera persistence across
  reload, whether `AA_ShareOpenGLContexts` is actually required, and macOS at all.
- GID2D's 2D Map tab against a freshly packed map.

## 8. Delegation

- **Subagents** can do P1, P2, P3 and all of section 2: they are scriptable, headless, and
  each produces a verdict plus evidence. One agent per category document, each given that
  document's section 7 and told to report CONFIRMED / REFUTED / INCONCLUSIVE with the
  command and output. They must not edit source — the audit establishes facts; fixes are a
  separate pass against `10-fix-list.md`.
- **The user** is needed for P4, and for the D1/D2/D4 decisions, which change what is even
  in scope: dropping `_G`/`_H`/`_box` (D1) deletes P2.2 and P2.5 outright.

## 9. Exit criteria

The audit is done when every section 7 claim is marked CONFIRMED, REFUTED or
INCONCLUSIVE-with-a-reason in its own document; every REFUTED claim has a `10-fix-list.md`
entry; P0 is fully closed; and the validator suite runs from one command and genuinely
fails when something is broken.

## 10. Known blockers

- P-1 through P-3 are hard blockers. Auditing against a harness that cannot fail, or
  against claims whose evidence scripts no longer exist, produces false confidence — which
  is worse than no audit.
- `validate_jointfit.py` was deleted with the bilayer campaign (A2.5); its cross-dataset
  GLS coverage must be rebuilt synthetically before P1.1 can be finished.
- The i2_tim3 two-mode case that the seeding work was validated against is no longer
  reproducible from a clean clone (A2.4), so several claims in doc 03 have no in-repo
  fixture. Either rebuild a multimodal fixture with the 1UBQ default or accept those as
  permanently INCONCLUSIVE.
- XPR claims need `XPR-main/test_data` from outside the repo, and the study harness cannot
  run from a clean clone at all (no `CONFIG.json`, missing `make_tim1_config.py`) — so P1.6
  and P1.7 are reproducible only on the original machine until B11 is addressed.
- **Amended 2026-09-10.** On the original machine the harnesses *can* run again. The
  2026-09-10 upstream integration broke all 32 of them on the sizing API
  (`alloc_bounds_for` now returns a dict; `load()` dropped `max_sig`/`slab_headroom`),
  which for part of that day made **no** XPR fitting possible in either repo. They were
  migrated the same day and one real fit was run end to end to confirm it. Full record in
  `05-xpr-julia.md` §3.10.1. Three consequences for this audit:
  1. **The fingerprint to check provenance against is `2402cb195df0ce6f`**, not the
     `432da5d0da1b10fe` recorded elsewhere on the day — that one was taken
     mid-integration. Any claim keyed to the wrong value will look like a mismatch.
  2. **The 32 harness files are freshly modified.** The change is mechanical and verified,
     but it is *not* in any commit yet, so a diff-based audit will see churn across
     `julia/bakeoff/` and `tests/recovered/check_aproj.py` that postdates every category
     document.
  3. **No number in any document was recomputed** and no completed analysis was re-run, so
     every recorded result still refers to the revision it was measured under — and by
     the fingerprint rule those archives still cannot be pooled with anything produced
     from here on.
