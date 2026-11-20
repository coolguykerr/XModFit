# XModJointFit — orientation for Claude

This is a working fork of [chemmatcars/XModFit](https://github.com/chemmatcars/XModFit)
carrying a large body of unmerged work: joint (global) multi-dataset fitting, an
overhauled MCMC stack with protein-orientation walker seeding, a Julia differentiable
reflectivity backend, 2D GID map fitting, a 3-D structure viewer, and a per-column
fit-limit engine.

**Start here: [`handoff/README.md`](handoff/README.md).** It defines the baseline, splits
the change set into committed / uncommitted / untracked, and indexes one document per
category. Read the category document before editing code in that area — each one records
the design decisions, the measured numbers, and the dead ends, none of which are
recoverable from the code.

## State of the repo

- **Remotes (since 2026-11-20):** `origin` is the shared public fork
  [coolguykerr/XModFit](https://github.com/coolguykerr/XModFit), where this work lives and
  where collaborators push; `upstream` is
  [chemmatcars/XModFit](https://github.com/chemmatcars/XModFit), the original project.
  **Never push to `upstream`.** The joint-fit work is on `develop`; upstream's own `develop`
  is merged in via `f3335f9`. Changes go back upstream only as pull requests from the fork.
- Before the fork, `develop`'s history was rewritten once to drop the seeding-study commit
  (`handoff/10-fix-list.md` A0.6). Hashes in `handoff/` are the new ones; a hash that does
  not resolve is from the old history, kept only in a local backup.
- Check `git status --porcelain -uall` before assuming a file is under version control.
- A `.gitignore` was added in `0296ec3` covering caches, `.h5` MCMC backends, saved
  `.xpar`/`.par` output and the seeding study. Files still showing as untracked are
  mostly personal structures and test output in `Data/`; do not "clean them up" without
  checking `handoff/09-validation-and-env.md` — a few are fixtures the validators need.
- Run `git config core.hooksPath .githooks` in a fresh clone: `.githooks/pre-push`
  refuses to push any commit carrying XPR seeding-study paths (research material kept in
  the separate `xpr-seeding-study` repository, never in this one). See
  `handoff/10-fix-list.md` A0.6.
- Upstream collaborators work from stock `upstream/develop`. Anything you change in a file
  that upstream also touches raises the merge cost; `handoff/10-merge-and-consolidation.md`
  tracks that surface. Work added on the fork should stay minimally intrusive for the
  same reason.

## Running and testing

Use the conda environment, not the default interpreter:

```bash
"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" xmodfit.py
"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/run_all.py --quick   # whole safety net, ~8.5 min (full run ~33 min)
```

For headless work set `QT_QPA_PLATFORM=offscreen`, inject a stub `sys.modules['xraydb']`
before importing `xmodfit` (the local `xraydb/` directory has no source), and guard any
script that reaches a spawn multiprocessing Pool with `if __name__ == '__main__':`.
Full recipe and the list of `validate_*.py` harnesses: `handoff/09-validation-and-env.md`.

## Rules that fail silently if you break them

1. **`Functions/<category>/` is a model namespace.** Every `.py` in a category folder is
   loaded as a fittable model class named exactly after the file. A helper module placed
   there crashes the function list. Shared helpers belong in `Functions/` root — that is
   why `pdb_cache.py`, `gid2d_core.py`, `pdb_geometry.py` and `xpr_backend.py` live there.
2. **Fixed-parameter cells are `eval()`ed** (`update_plot`, `fixedParamChanged`). A
   `choices` value that is a valid Python name arrives as that object, not as a string.
   Never use a builtin, keyword or literal as a choices value.
3. **`nan_policy='raise'` is on every least-squares path.** Exclude points by dropping
   them from the residual, never by writing NaN.
4. **A fit mask must be frozen once per `perform_fit`.** lmfit's `leastsq` requires a
   constant residual length; recomputing a mask inside `residual()` is a latent crash.
5. **Angles (theta/phi) are deliberately unbounded** for MCMC, and folded only locally
   inside the model. Reported values are canonicalised after the fit — do not "fix" an
   out-of-window angle by adding bounds without reading `handoff/03-orientation-seeding.md`.

## Scope discipline

The documentation in `handoff/` was written from the code and the development notes but
**not** by running it. Section 7 of each document lists claims that are still unverified;
`handoff/11-audit-plan.md` is the plan for checking them. If you confirm or refute one,
update both the category document and the audit plan.
