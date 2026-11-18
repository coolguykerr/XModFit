# Validation, environments and repository hygiene — XModJointFit change record

**Category:** the 11 root-level `validate_*.py` harnesses, the two conda environments and
the headless recipe, the seven `requirements_*.txt` snapshots, and the non-source
artefacts sitting in the working tree ahead of the fold-in.
**Status:** Research harness (the validators); Production (the environments — the user
runs the GUI from them daily); repository hygiene is **unaddressed** (there is no
`.gitignore` at all).
**Owner chat(s):** `cmc-py39-env.md`, `functions-category-folder-rule.md`,
`column-fit-limits.md`, `gid2d-function.md`, `xpr-julia-backend.md`,
`homometry-seeding-for-xpr.md`
**Last touched:** validators 2026-07-29 .. 2026-09-01; `requirements_xpr.txt` 2026-08-31;
artefacts accumulating continuously.

## 1. Why this exists

XModFit has no test suite: upstream `origin/develop` ships no `tests/`, no `conftest.py`, no
`pytest.ini`, and no CI. Everything in this fork touches the fit engine — the residual vector,
the covariance whitening, the MCMC data fingerprint — where a mistake produces a *plausible*
number rather than an exception. The `validate_*.py` scripts are the substitute: each was
written alongside one specific change, to pin down the invariant that change could silently
break (residual length constant across iterations, whitening rows aligned with the
concatenated residual, a `.h5` backend not being reused for a different data window). They
are root-level bare scripts so they run from a bare checkout with nothing installed.

The environment section exists because the harnesses are not portable: XModFit is a PyQt5
desktop app that must be coaxed into running without a display, several of its imports need
packages absent from one of the two working conda environments, and the newest model
(`XLayersPDB_XPR`) drags in a 1 GB Julia runtime whose install location depends on whether
the environment happens to be *activated*. The hygiene section exists because the fork has
accumulated ~870 MB of untracked artefacts in a repository with **no `.gitignore`** — the
single biggest obstacle to a clean fold-in, since a naive `git add -A` would put 886 MB of
MCMC backend files into upstream's history permanently.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `validate_joint_gls.py` | 324 | new (WS0, 2026-11-04) | GLS core, synthetic: replaced the retired real-data `validate_jointfit.py` (whitening identity, N=1 == stock Fit, cross-cov stderrs) |
| `validate_jointfit_gui.py` | 362 | new (`c5f697d`) + **uncommitted** (+43/-5) | joint-fit GUI: seeding, `.jpar` round-trip, overlay selection, model comparison |
| `validate_perdataset_fit.py` | 199 | new (`c5f697d`) | per-dataset fit memory + shared PDB read cache |
| `validate_fit_mask.py` | 199 | new-untracked | `Fit` per-point mask: freeze invariant, `_window_indices`, dict-x branch |
| `validate_joint_mask.py` | 236 | new-untracked | mask on the JOINT path incl. covariance + shared-footprint composition order |
| `validate_column_limits.py` | 251 | new-untracked | per-column limits store, `xmin`/`xmax` proxies, `.xpar`/`.jpar` round-trip |
| `validate_masked_x.py` | 145 | new-untracked | `__masked_x__` opt-in, GID2D `window='engine'` honest dof |
| `validate_xlayerspdb_joint.py` | 260 | new-untracked | regression: real XLayersPDB joint fit still works; MCMC fingerprint unchanged |
| `validate_gid2d.py` | 343 | new-untracked | GID2D physics: sum rule, quadrature, tilt spread, pack/load — pure numpy, no Qt |
| `validate_gid2d_gui.py` | 365 | new-untracked | 2D Map tab wiring, exact q ticks, wrong-file diagnosis, Tools packer |
| `validate_gixos.py` | 249 | new (2026-11-18) | GIXOS model: GUI listing, horizon zeros, pickle, single/masked/joint/joint+cov parameter recovery (doc 22) |
| `validate_xpr.py` | 382 | new-untracked | XPR Julia backend: 3 reference logL, coverage, unit cube, analytic Jacobian |
| `requirements.txt` | 102 pkgs | upstream, **unmodified** | Oct-2024 pinned snapshot (has `h5py`) |
| `requirements_latest.txt` | 102 pkgs | upstream, **unmodified** | same package set, unpinned |
| `requirements_01062026_old.txt` | 164 pkgs | new-untracked | conda `pip freeze` incl. `file:///` local pins; superset of the rest |
| `requirements_working_Mar2026.txt` | 156 pkgs | new-untracked | 01062026 minus conda pins, plus `sympy`/`mpmath` |
| `requirements_07022026.txt` | 154 pkgs | new-untracked | Mar2026 minus `sympy`/`mpmath` |
| `requirements_xmodfit_viewer.txt` | 155 pkgs | new-untracked | 07022026 **plus `PyQtWebEngine`** (the NGL viewer) |
| `requirements_xpr.txt` | 6 pkgs | new-untracked | additive: `juliacall`==0.9.28 + its transitive deps, heavily commented |
| `.idea/XModFit.iml`, `.idea/misc.xml` | — | upstream, **modified** | IDE interpreter path changed to a machine-local absolute path |

There is **no `CLAUDE.md`** in the repo (the orchestrator will write it) and **no
`.gitignore`** anywhere except `.idea/.gitignore`, which only covers `/shelf/` and
`/workspace.xml`.

## 3. How it works

### 3.1 The shape every validator shares

All eleven are **bare scripts**, not pytest tests. The common skeleton:

1. `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` before any Qt import
   (`validate_perdataset_fit.py:32`).
2. `os.chdir(ROOT)` + `sys.path.insert(0, ROOT)`, and for engine-level ones also
   `Functions/`, `Functions/XRR/`, `Functions/GISAXS/` (`validate_masked_x.py:16-17`) —
   exactly what `xmodfit.py` does at runtime. **`os.chdir` is load-bearing**: the
   `Functions/<category>` discovery walk and every `Data/...` path assume root is cwd.
3. A conditional `xraydb` stub (`validate_perdataset_fit.py:38-49`). See 3.3.
4. Modal-dialog neutering: `QMessageBox`, `QProgressDialog`, `FitResultDialog` replaced with
   no-ops, patched on the `xmodfit` module *and* on `PyQt5.QtWidgets`, because
   `joint_fit_panel` does its own `from ... import QMessageBox`
   (`validate_jointfit_gui.py:44-55`).
5. A local `check(name, cond)` printing `PASS`/`FAIL` and ANDing into a module flag;
   `main()` returns `0`/`1`, called under an `if __name__ == '__main__'` guard.

Two deviations matter:

- **`validate_xpr.py` has no `if __name__ == '__main__'` guard** — the whole script is
  top-level statements ending in `sys.exit(1 if FAIL else 0)` (`validate_xpr.py:382`). It
  gets away with it today only because both of its emcee calls pass `emcee_cores=1`
  (`validate_xpr.py:219`, `:243`), and every spawn `Pool` in this fork is gated on
  `workers > 1` (`lmfit/minimizer.py:2110`, `:2159`; `pdb_seed_worker.py:90`). Raise those
  cores and the script re-executes in each child. This is the one structural defect in the
  set.
- **`validate_gid2d.py` never imports Qt at all** — it is pure numpy/numba against
  `Functions/gid2d_core.py` and `Functions/GISAXS/GID2D.py`, so no `QApplication` and no
  offscreen platform is needed. It is also the only one whose test bodies are named
  `test_*` (`validate_gid2d.py:53-332`), called explicitly from the `__main__` block.

### 3.2 Why they are not pytest-compatible as written

Three independent blockers, all cheap to fix but all real:

1. Filenames are `validate_*.py`, not `test_*.py` — default collection skips them.
2. Only `validate_gid2d.py` has `test_*`-named functions; the other ten put everything in
   one `main()`.
3. **`check()` does not `assert`.** It records a failure in a module global and the exit
   code carries the verdict. Under pytest a failing `validate_gid2d.py::test_sum_rule`
   would be reported as **PASSED**. Do not run these under pytest without first converting
   `check` to raise.

Neither conda environment has `pytest` installed at all, so nothing is silently relying on it.

### 3.3 The headless recipe, and why each piece is needed

- **`QT_QPA_PLATFORM=offscreen`.** Every validator that constructs `XModFit()` sets this
  itself via `setdefault`; export it anyway when driving them from a wrapper, since it must
  be set before the first Qt import.
- **`QApplication.instance() or QApplication(sys.argv)`** must exist before `Fit_Routines`
  is used, because `Fit` is a `QObject` with a `functionCalled` signal.
- **`addData(fnames=[f1, f2])` with two or more files.** A single file takes an interactive
  `exec_()` path that blocks forever headless; several validators load the *same* file twice
  purely to dodge this (`validate_column_limits.py:69`).
- **The `xraydb` stub.** `Chemical_Formula.py:2` does `from xraydb import XrayDB` at module
  scope and `xmodfit` imports it transitively, so a missing `xraydb` breaks *construction* of
  the whole app even for a fit that never touches an X-ray calculation (ten modules import
  it: `Functions/ASAXS/*`, `Functions/FormFactors/*`, `Functions/XFNTR/*`,
  `Tools/Calculators/*`). It is **absent in cmc-py39**, **installed in cmcars** (4.4.7) —
  hence the conditional stub. The stub's `__getattr__` returns `lambda *a: 0`, so any
  harness that starts asserting on an X-ray number must run in cmcars instead.
- **`if __name__ == '__main__'` whenever a spawn `Pool` can be reached.** Windows has no
  `fork`; `get_context('spawn')` re-imports `__main__` in each child, so without the guard
  the child re-runs the whole harness — for the orientation seed scan, an unbounded fan-out.
  The Pool sites: `lmfit/minimizer.py:2125` (parallel PDB MCMC), `:2164` (generic
  `_ext_objective`, joint MCMC), `pdb_seed_worker.py:96` and
  `pdb_joint_seed_worker.py:125` (parallel `(theta,phi)` seed scan). All gated on
  `workers > 1` / `_seed_workers > 1`, so a serial run is safe either way.

### 3.4 The `xraydb` / `xraydb_bak` situation — verified

The notes' claim is correct and slightly worse than stated:

- `xraydb/` contains **exactly seven `.pyc` files under `__pycache__/` and nothing else** —
  no `__init__.py`, no source. `cpython-39` bytecode for a deleted source tree.
- `xraydb_bak/` is the full source (9 files incl. a **10.3 MB `xraydb.sqlite`**) and is
  **tracked in `origin/develop`** — the rename is upstream's, not this fork's. **Nothing in
  the repo references it.**
- Because `xraydb/` has no `__init__.py` it is only a *namespace portion*, so Python keeps
  scanning `sys.path` and pip's `xraydb` in cmcars still wins. That is why cmcars works
  despite the directory. Pure garbage, not an active trap — but one
  `touch xraydb/__init__.py` away from becoming one.

## 4. How to use it — the audit runbook

### 4.0 Prerequisites

```
$CMCPY  = "C:\Users\kerr\.conda\envs\cmc-py39\python.exe"                    # py3.9.19
$CMCARS = "C:\Users\kerr\AppData\Local\anaconda3\envs\cmcars\python.exe"     # py3.9.25
```

Run every command **from the repo root** (`C:\Users\kerr\Documents\GitHub\XModJointFit`).
Every validator `os.chdir`es there anyway, but the vendored `lmfit/` only shadows pip's
`lmfit 1.2.2` in cmcars because the script's directory is first on `sys.path` — run one from
elsewhere and you silently lose `customsampler`/`custombackend`.

The shell here is `powershell.exe`, so `&&` and `VAR=x cmd` are parse errors; use
`& $CMCPY validate_fit_mask.py`. Bash-tool form:
`"/c/Users/kerr/.conda/envs/cmc-py39/python.exe" validate_fit_mask.py`.

### 4.1 The table

Runtimes are **estimates** (see section 7, claim V1) — nothing here was executed.

| Validator | Covers | Needs GUI/Qt? | Data required | Present? | Interpreter | Est. time |
|---|---|---|---|---|---|---|
| `validate_fit_mask.py` | `Fit` mask freeze invariant, `_window_indices` on negative bounds, dict-x, 4 fit scales | `QApplication` only, no widget | none (synthetic `Line` model) | n/a | cmc-py39 | < 20 s |
| `validate_joint_mask.py` | joint mask composition order in `_build_chol` + `residual`, cov + shared-footprint branch | `QApplication` only | none (synthetic) | n/a | cmc-py39 | < 30 s |
| `validate_masked_x.py` | `__masked_x__`, GID2D `window='engine'` vs `'model'` dof | `QApplication` only | `Data/dppc_q2d_gid_indexed.txt` | **yes, untracked** | cmc-py39 | 1-3 min (two real GID2D fits) |
| `validate_gid2d.py` | GID2D physics + `gid2d_core` selfcheck | **no Qt at all** | `Data/dppc_q2d_gid_indexed.txt`; optional raw at `T:\SURFACE\...` (skips) | packed yes; raw **no** | cmc-py39 | 1-3 min |
| `validate_gid2d_gui.py` | 2D Map tab, image `output_params`, exact q ticks, wrong-file banner, Tools packer | **full offscreen widget** | `Data/dppc_q2d_gid_indexed.txt` | **yes, untracked** | cmc-py39 | 1-3 min |
| `validate_column_limits.py` | limits store, `xmin`/`xmax` proxies, badge, `.xpar`/`.jpar` round-trip | **full offscreen widget** | none (writes its own 5-column file to `%TEMP%`) | n/a | cmc-py39 | 30-60 s |
| `validate_perdataset_fit.py` | per-dataset fit memory, shared PDB cache, per-dataset PDB persistence | **full offscreen widget** | `Data/*.pdb` (any, for cache); `Data/Pr4.pdb` + `Data/Nd9.pdb` (else SKIP) | yes, tracked | cmc-py39 | 1-2 min |
| `validate_jointfit_gui.py` | joint GUI: seeding, `.jpar` save/load-with-missing-dataset, `.par`-into-joint, overlay selection, CI button, model comparison | **full offscreen widget** | none (synthetic, `%TEMP%`) | n/a | cmc-py39 | 1-2 min |
| `validate_xlayerspdb_joint.py` | real XLayersPDB individual + joint fit; `0.99*xmin` unification measured on real data; MCMC fingerprint unchanged | **full offscreen widget** | `Data/i2_tim3_1000_test.txt`, `Data/d6_tim3_1000_test.txt`, `Data/1ubq.pdb`, `tests/fixtures/legacy_xlayerspdb_2024.par` | **yes** (all 4 tracked) | cmc-py39 | 2-5 min (numba JIT + 2 real fits, `maxiter=12`) |
| `validate_joint_gls.py` | GLS whitening identity (footprint ON), N=1 JointFit == stock `Fit`, engine stderr vs the redchi-scaled analytic GLS covariance, different masks with cross blocks (G4) and with per-dataset windows (G4b); replaced the real-data `validate_jointfit.py` on 2026-11-04 | synthetic; creates a real `QApplication` (offscreen) but no widget | none | n/a | cmc-py39 | ~1 s |
| `validate_xpr.py` | 3 published XPR logL, projected area, derived coverage, EDP, grad/hess, leastsq, emcee, angle canonicalisation, unit cube, analytic Jacobian | `QApplication` only | `../XPR-main/test_data/` (`2box.rrf`, `3box.rrf`, `protein_2box.rrf`, `protein.pdb`) | **yes, but OUTSIDE the repo** | cmc-py39 or cmcars (needs `juliacall`; both have it) | **5-15 min** (Julia session ~7 s, JIT ~8 s, cold Hessian ~34 s, an orientation seed scan, and a leastsq-with/without-Jacobian bake-off) |

### 4.1b The runner: `tests/run_all.py`

`tests/run_all.py` runs every `validate_*.py`, every `tests/recovered/*.py` and
`tests/test_*.py` as its own subprocess (own interpreter per the REGISTRY, `cwd` = repo
root, offscreen), logs each to `tests/_logs/`, and prints a status / seconds / note table.
Exit code 1 on any FAIL or TIMEOUT, 2 when `--only` matches nothing. A row that exits 0
but whose log contains `Traceback (most recent call last)` is FAIL, with a note naming the
exception: pyqtgraph replaces `sys.excepthook` when xmodfit is imported, so an exception in a
Qt slot is printed and the process carries on (WS0/WS1 audit N1). `tests/harness.py` also
installs a counting excepthook, so such an exception is a failed check. **The runner is
the gate:** a harness run directly with `python file.py` that does not use `tests.harness`
(the bare-`assert` ones) can still exit 0 on a slot exception; only the runner's log scan
catches it. SKIP rows are scanned too, and a `multiprocessing.Pool` worker's exception is
seen only if the parent prints it. Registry fields:
`interp`, `external`, `slow`, `known` (expected to fail, with a fix-list ID), `smoke`
(assertion-free script), `timeout`, `expected_tracebacks` (True, or the exception type
every traceback must end with, optionally module-qualified; only `test_joint_abort` uses
it, and the row note counts the allowed tracebacks).
Flags: `--only SUBSTR`, `--timeout SEC`, `--dir DIR`, `--registry PATH`, and `--quick`,
which skips every `slow` and `external` entry. **`--quick` is the before-and-after-every-change
command for WS1** (about 8.5 min); the full run takes about 33 min. The recorded baseline is in
`handoff/12-work-package-preaudit.md`.

### 4.2 Copy-pasteable commands

```
# --- no data dependency, run these first (fastest signal) ---
& $CMCPY validate_fit_mask.py
& $CMCPY validate_joint_mask.py
& $CMCPY validate_column_limits.py
& $CMCPY validate_jointfit_gui.py
& $CMCPY validate_perdataset_fit.py

# --- need Data/dppc_q2d_gid_indexed.txt (present, untracked) ---
& $CMCPY validate_gid2d.py
& $CMCPY validate_gid2d_gui.py
& $CMCPY validate_masked_x.py

# --- needs the four tim3 fixtures (present) ---
& $CMCPY validate_xlayerspdb_joint.py

# --- needs the sibling XPR-main checkout; slow ---
& $CMCPY validate_xpr.py
& $CMCPY validate_xpr.py "C:\path\to\XPR-main\test_data"     # explicit path form

# validate_joint_gls.py (synthetic; replaced validate_jointfit.py on 2026-11-04)
& $CMCPY validate_joint_gls.py

# --- ancillary selfchecks (not validate_* but referenced by the notes) ---
& $CMCPY Functions\gid2d_core.py        # geometry selfcheck, >50 assertions
```

### 4.3 Broken / dangling references — flag list

| Validator | Reference | Problem |
|---|---|---|
| `validate_jointfit.py:23` | `BASE = os.environ.get("ELCAT_BASE", "/mnt/project")` | POSIX container path; the script was written inside a Linux sandbox and never re-homed for Windows |
| `validate_jointfit.py:30-32` | six `El_Cat_Si_1_*` files | **absent from the repo and from `Data/`** (retired 2026-11-04; its role is taken by the synthetic `validate_joint_gls.py`) |
| `validate_gid2d.py:25` | `RAW = r"T:\SURFACE\NX_school\2026sep\extracted_data\dppc_q2d_gid.txt"` | hardcoded network drive. When absent, section 6 regenerates a raw-format map into `tests/_logs/` from the tracked packed map (A1.16), so `pack_map` is exercised on a round-tripped file |
| `validate_xpr.py:26` | `../XPR-main/test_data` | outside the repo; present on this machine, will not exist for a collaborator. Overridable via `argv[1]` |
| `validate_masked_x.py:36`, `validate_gid2d_gui.py:37`, `validate_gid2d.py:26` | `Data/dppc_q2d_gid_indexed.txt` | present but **untracked**, and only regenerable from the unavailable `T:\` raw file. If it is not committed, three validators become unrunnable for everyone else |
| `validate_xlayerspdb_joint.py:40-43` | `Data/i2_tim3_1000_test.txt`, `Data/d6_tim3_1000_test.txt`, `Data/1ubq.pdb`, `tests/fixtures/legacy_xlayerspdb_2024.par` | all four tracked |
| `validate_jointfit.py:165` | `local = local[2:] if False else local` | dead line — a debugging leftover in the T2 name-matching block (and `local` itself is never used) |

**There is no runner that executes all of them.** No `Makefile`, `tox.ini`, `noxfile.py`,
`pytest.ini`, `setup.cfg`, `pyproject.toml`, `conftest.py`, or shell wrapper. The only
scripts at the root are upstream's `build_doc.{bat,sh}` and `pyinstaller_compile.*`.
Writing a `run_validators.ps1` that loops the list in 4.2 and reports exit codes is the
single cheapest thing the audit pass could do first.

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `requirements.txt` | whole file | none — untouched | n/a |
| `requirements_latest.txt` | whole file | none — untouched | n/a |
| `.idea/XModFit.iml` | `orderEntry jdkName` | `"cmcars"` -> `"C:\Users\kerr\AppData\Local\anaconda3\envs\cmcars"` | **rewrite, and unintended** |
| `.idea/misc.xml` | `ProjectRootManager project-jdk-name` | same substitution | **rewrite, and unintended** |
| `xraydb_bak/` (9 files, 10.3 MB) | — | none from this fork; pre-existing upstream dead weight | n/a |

Everything else in this category is a **new root-level file** with zero upstream contact:
the 11 validators, the 5 new `requirements_*` snapshots, `requirements_xpr.txt`.

The two `.idea/` edits are the only accidental modifications to upstream-tracked files in
this whole category. They encode one developer's absolute interpreter path and will conflict
with any other contributor's IDE config forever. **Recommendation: `git checkout -- .idea/`
before the fold-in, and add `.idea/` to `.gitignore` (which also means `git rm -r --cached
.idea` upstream, a separate conversation with the maintainer).**

## 6. Known limitations, gotchas and TODOs

### 6.1 Environments

| | `cmc-py39` | `cmcars` |
|---|---|---|
| path | `C:\Users\kerr\.conda\envs\cmc-py39\python.exe` | `C:\Users\kerr\AppData\Local\anaconda3\envs\cmcars\python.exe` |
| Python | 3.9.19 | 3.9.25 |
| numpy / scipy | **1.21.6** / 1.7.3 | 1.22.4 / 1.7.3 |
| numba | 0.55.1 | 0.57.1 |
| PyQt5 | 5.15.10 | 5.15.9 |
| PyQtWebEngine | **absent** | **5.15.6** |
| xraydb | **absent** | **4.4.7** |
| lmfit (pip) | absent (vendored `lmfit/` only) | 1.2.2 installed **and** shadowed by vendored `lmfit/` |
| h5py | 3.11.0 | 3.14.0 |
| juliacall / juliapkg | 0.9.28 / 0.1.26 | 0.9.28 / 0.1.26 |
| numdifftools | 0.9.41 | **absent** |
| periodictable | absent | 1.6.1 |
| mendeleev | 0.17.0 | 0.14.0 |
| pytest | absent | absent |

Which to use for which category:

- **Everything except the NGL viewer: `cmc-py39`.** It is the environment the headless
  recipe was developed against, and the `xraydb` stub exists precisely for it.
- **`Tools/StructureViewer/` (NGL) needs `cmcars`** — it is the only environment with
  `PyQtWebEngine`, and `requirements_xmodfit_viewer.txt` is its snapshot.
- **XPR/Julia works in both** (both have `juliacall` 0.9.28) but `cmcars` carries the SSL
  problem below.
- **Anything that asserts on a real X-ray number needs `cmcars`** (real `xraydb`); in
  `cmc-py39` the stub silently returns `0`.

`numpy 1.21.6` in cmc-py39 is a hard constraint on new code: `np.trapz`, not
`np.trapezoid`; no `np.exceptions`; no NEP-50 casting rules.

### 6.2 Environment traps (all recorded as CLAIMS in `xpr-julia-backend.md`; not re-tested here)

1. **`cmcars` has broken stdlib SSL.** OpenSSL 3.5.7 rejects a malformed cert in the Windows
   store, so `ssl.create_default_context()` raises `SSLError: [ASN1: NOT_ENOUGH_DATA]`;
   `SSL_CERT_FILE` does not help (`load_default_certs` reads the Windows store first).
   `juliapkg` pings julialang-s3 on every startup, so `xpr_backend.get_jl()` retries once
   with `PYTHON_JULIAPKG_OFFLINE=yes`. Any other stdlib-urllib HTTPS in cmcars still fails.
   cmc-py39 (OpenSSL 3.0.13) is fine on the same machine.
2. **`juliapkg`'s install dir depends on activation.** It keys on `CONDA_PREFIX`, so an
   *activated* env gets its own ~1 GB Julia copy under `$CONDA_PREFIX/julia_env` while the
   same interpreter called by absolute path uses the shared depot. `get_jl()` now defaults
   `PYTHON_JULIAPKG_PROJECT` to the shared depot. **For the audit: invoke by absolute path,
   do not activate.**
3. **`cmcars` had a broken `tomli`** (dist-info present, package dir gone) so `juliapkg` died
   on `ModuleNotFoundError`. Fixed with `--force-reinstall --no-deps`; both envs now say 2.4.1.
4. **`juliacall` kills the Qt process silently with exit status 0** if it auto-loads its
   IPython extension — which it does whenever `IPython` is in `sys.modules`, and pyqtgraph
   puts it there. Needs `PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=no` before the import.
5. **Upstream XPR segfaults rather than raising** when its slab arrays are exceeded; the
   vendored copy takes `max_box_length`/`max_sig`/`slab_headroom`. A harness that changes
   layer thicknesses without sizing these crashes the interpreter instead of failing a check.

### 6.3 Requirements snapshots — which is current, and which should survive

By content (mtimes disagree with the filenames, so trust the content relations):

```
requirements.txt (Oct 2024, upstream, pinned)  ~  requirements_latest.txt (same set, unpinned)
        |  (numpy 1.26 -> 1.22, scipy 1.10 -> 1.7, PyQt5 5.15.10 -> 5.15.9, drops the
        |   dash/plotly/bokeh/flask/seaborn block, adds lmfit + periodictable)
requirements_01062026_old.txt (164)  -- superset; includes 10 conda `@ file:///` local pins
        |  (-conda pins, -h5py, +sympy, +mpmath)
requirements_working_Mar2026.txt (156)
        |  (-sympy, -mpmath)
requirements_07022026.txt (154)
        |  (+PyQtWebEngine)
requirements_xmodfit_viewer.txt (155)   <-- the most complete usable snapshot
```

`requirements_xpr.txt` is orthogonal and additive (6 packages) with a useful comment on *why*
`juliacall` is pinned: juliapkg writes the PythonCall version into the shared Julia project as
an exact `=x.y.z` compat bound, so two Python envs on different juliacall versions re-resolve
on every startup.

**Two problems worth acting on:**

- **`h5py` is missing from `requirements_07022026.txt`, `requirements_working_Mar2026.txt`
  and `requirements_xmodfit_viewer.txt`** (present only in `requirements.txt`,
  `requirements_latest.txt`, `requirements_01062026_old.txt`). MCMC `.h5` backends are a
  headline feature of this fork, so installing from the "current" snapshot yields an
  environment that cannot save an MCMC run.
- The five new snapshots are `pip freeze` dumps of one machine, `spyder-kernels` and
  `argon2-cffi` included. Archaeology, not dependency declarations.

**Fold-in recommendation:** keep upstream `requirements.txt` / `requirements_latest.txt`
untouched; keep `requirements_xpr.txt`; replace the other five with **one**
`requirements_viewer.txt` containing just `PyQtWebEngine` (same "extras for one optional
feature" pattern); add `h5py` to upstream `requirements.txt` if it is not already effectively
required there.

### 6.4 Validator gaps

- **The GLS/covariance core now has a synthetic regression** (`validate_joint_gls.py` G1-G4b,
  2026-11-04; the real-data `validate_jointfit.py` is retired). What is still open: the
  real El_Cat covariance, and a comparison of `result.covar` stderrs against a covariance-aware
  MCMC posterior.
- No validator covers `Tools/StructureViewer/` (the NGL viewer) at all.
- No validator runs a **parallel** (`emcee_cores > 1`) MCMC, so none of the four spawn-`Pool`
  paths is exercised. Covariance/GLS joint MCMC is still serial (per the notes) and untested.
- `validate_gid2d.py`'s `pack_map` path runs on a map regenerated from the packed file when the `T:\` raw file is absent (A1.16), not on the original raw data.

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| V1 | The runtime estimates in section 4.1 | this document (extrapolated from `maxiter`, dataset sizes and the notes' timings) | run each validator once under `Measure-Command`; record real times back into 4.1 |
| V2 | All nine runnable validators currently **pass** | nothing — none has been run since the changes were made | run 4.2 top to bottom; expect at least `validate_jointfit.py` to fail on missing data |
| V3 | `cmc-py39` can run every non-viewer validator, i.e. the `xraydb` stub is sufficient everywhere | `cmc-py39-env.md` (verified in Aug 2026, before GID2D/XPR existed) | run `validate_gid2d_gui.py` and `validate_xpr.py` in cmc-py39 and check nothing reaches a real `xraydb` call |
| V4 | The vendored `lmfit/` shadows pip `lmfit 1.2.2` in cmcars when cwd is the repo root | inference from `sys.path` ordering | `& $CMCARS -c "import lmfit; print(lmfit.__file__)"` from the repo root and from `C:\` |
| V5 | `cmcars` SSL is broken and `SSL_CERT_FILE` does not fix it | `xpr-julia-backend.md`, 2026-08-31 | `& $CMCARS -c "import ssl; ssl.create_default_context()"` |
| V6 | `tomli` is now healthy in cmcars | same note (a fix that was applied) | `& $CMCARS -c "import tomli; print(tomli.__file__)"` |
| V7 | juliapkg's 1 GB lands in the shared depot when the interpreter is called by absolute path | same note | `& $CMCPY -c "import Functions.xpr_backend as xb; print(xb.julia_project_dir())"` plus `du` on both candidate dirs |
| V8 | `numpy 1.21.6` in cmc-py39 forces `np.trapz` | `gid2d-function.md` | `& $CMCPY -c "import numpy;print(numpy.__version__, hasattr(numpy,'trapezoid'))"` |
| V9 | `validate_xpr.py`'s missing `__main__` guard is currently harmless (no Pool at `emcee_cores=1`) | this document, from reading the three gates | set `emcee_cores=2` in a scratch copy and confirm it fan-outs; then add the guard |
| V10 | `Data/dppc_q2d_gid_indexed.txt` cannot be regenerated without the `T:\` share | `gid2d-function.md` + `validate_gid2d.py:25` | ask the user whether the raw map still exists anywhere; if yes, archive it |
| V11 | The `El_Cat_Si_1_*` datasets still exist on the user's machine outside the repo | inference — `validate_jointfit.py` was clearly run once against them | `Get-ChildItem -Recurse -Filter "El_Cat_Si_1_*"` over the user's data drives |
| V12 | The published XPR log-likelihoods still reproduce to 1e-3 | `xpr-julia-backend.md` | `& $CMCPY validate_xpr.py` — this is exactly what its first checks do. **Rebaselined 2026-09-08** onto Elias Boshara's updated upstream: `-1214.1600`, `-600.6398`, `-62408.5320` (4-box, new) and `-466.4557` for protein+2box at `protins = 13.358030329245691 - maxh`. The protein number did NOT move — that equality is what says the insertion redefinition is a reparameterisation and not a change of physics. Ran green 2026-09-08. See `handoff/05-xpr-julia.md` 3.9 |
| V13 | `Data/*.h5` MCMC backends are all reproducible / disposable, i.e. dropping them loses no unpublished result | not asserted anywhere | ask the user before deleting; 886 MB includes runs that may back a figure |
| V14 | `h5py`'s absence from the three newest snapshots is an omission, not a deliberate removal | this document | ask the user; check whether an MCMC save was ever attempted in an env built from those files |
| V15 | Nothing imports `xmodfit_old.py` | verified by `grep -rn xmodfit_old --include=*.py` (only a handoff `.md` mentions it) | re-run the grep after the fold-in; safe to drop |

## 8. Consolidation candidates

### 8.1 What not to commit — the artefact inventory

No `.gitignore` exists, so `git status` currently shows **867 MB of untracked material**
across ~90 entries. Grouped:

| Group | Count / size | Recommendation | Reason |
|---|---|---|---|
| `Data/*.h5` MCMC backends (`A_Scan1MCMCnoprot`, `p3e2test`, `i2test`, `htim3_A3_trial5_test`, `p3i4test`, `n1test`, `inittest`) | 7 files, **886 MB** | **gitignore + move out of the repo** | emcee HDF5 chains; regenerable, machine-specific, and irreversible once in git history. Confirm with the user first (claim V13) |
| `Data/d6test.h5` | 1 file, 28 MB, **already tracked** | flag to upstream | added by upstream commit `03474cd`, not by this fork — evidence the pattern is already established and should be stopped |
| `Data/*.pdb` untracked research structures (`4FQZ_stripped*`, `7M37vertical...`, `IBAR_2ykt...`, `StateB/CAlignedForXR`, `Tim4-orig-cen`, `hTIM3From7M3YTrial5`, `hTIM4_MD`, `tim1_orientation1_bf`, `tim1_orig`, `tim3_pre-md`, `tim3mirrornumbered`) | 13 files, ~1.9 MB | **drop / move to the user's data tree** | none is referenced by any validator or module; they are one campaign's inputs. `Pr4.pdb`, `Nd9.pdb`, `tim3_hmmm_ext_all_bestframe.pdb` are already tracked and *are* fixtures — keep those |
| **Test fixtures that must be committed**: `Data/dppc_q2d_gid_indexed.txt` (2.3 MB), `Data/i2_tim3_1000_test.txt` (4.6 kB), `Data/i2_tim3_1000_test_fitparam.par` (1.7 kB) | 3 files, 2.3 MB | **KEEP — `git add` these** | four validators hard-code them; the GID2D map is not regenerable without the `T:\` share |
| `Data/*_fit.txt` model-output dumps, `Data/*.xfit` (3.9 MB) | 3 files | **gitignore** | saved fit curves, pure output |
| `Data/*.xpar`, `Data/*.par` other than the fixture above | ~2 files | gitignore the pattern, `git add -f` the one fixture | fit-parameter files are per-session state |
| `Data/*.expr` (`orientationmod`, `phimod`, `thetamod`, Jul 2024) | 3 files, small | drop | saved parameter-expression files from an old XLayersPDB session; nothing reads them |
| `Data/n1_tim3_1500_raw_test.txt` | 5 kB | keep if a validator will use it, else drop | currently referenced by nothing |
| `julia/bakeoff/endpoints_*.npz` | **92 files, 2.4 MB** | **gitignore** (keep `results_p3i2.txt` + `README.md` + the `.py`/`.jl` drivers) | saved multistart endpoint arrays from a bake-off whose conclusions are already written up in `STAGE2_SEEDING_PLAN.md`; regenerable from `run_multistart.py` |
| `xmodfit_old.py` | 243 kB, 4,428 lines | **drop** | backup of the pre-fork main window. **Nothing imports it** (only `handoff/README.md` mentions it). Also a grep hazard: every search for a symbol in `xmodfit.py` returns a stale duplicate hit |
| `.perr` | 960 B | **drop + gitignore `*.perr`** | an MCMC parameter-error report saved with an empty filename (the dialog appends the extension). Accident |
| `mesh.txt` | 8.6 kB, mtime **2024-11-11** | **drop + gitignore** | predates this work; no `.py` reads or writes it. Orphan output |
| `__pycache__/` | **18 directories** (`.`, `Functions/*`, `Tools/*`, `lmfit/`, `julia/bakeoff/`, `xraydb/`) | **gitignore `__pycache__/` + `*.py[cod]`** | `Functions/XRR/__pycache__` alone is 7.6 MB |
| `xraydb/` | 78 kB, `__pycache__` only | **drop** | seven orphan `.pyc` for a deleted tree (see 3.4) |
| `xraydb_bak/` | 9 files, 10.3 MB, **tracked upstream** | raise with the maintainer, do not touch unilaterally | upstream's dead copy; referenced by nothing |
| `.idea/` | 6 tracked files, **2 currently modified** | **`git checkout -- .idea/`** now; propose `.gitignore` + `git rm -r --cached` upstream | the modification is a machine-local absolute interpreter path (section 5) |
| `Tools/GrazingIncidence/`, `SEEDING_STUDY_TIM1.md`, `STAGE2_SEEDING_PLAN.md` | 6 files | **keep** — source / research write-ups | covered by `06-gid2d.md` and `05-xpr-julia.md` |

### 8.2 Proposed `.gitignore` (new file at the repo root)

```
__pycache__/
*.py[cod]
.idea/
build/
dist/
*.egg-info/
# fit/MCMC outputs and per-session state
*.h5
*.perr
*.xfit
Data/*_fit.txt
# bake-off result arrays
julia/bakeoff/*.npz
# stray outputs
mesh.txt
```

Then add the three fixtures explicitly:
`git add Data/dppc_q2d_gid_indexed.txt Data/i2_tim3_1000_test.txt Data/i2_tim3_1000_test_fitparam.par`.
Already-tracked files (incl. `Data/d6test.h5`) are unaffected by `.gitignore`. `*.h5` as a
blanket rule is blunt — if `.h5` ever becomes a legitimate *input* format, narrow it to
`Data/*.h5`.

### 8.3 Consolidation of the validators themselves

| Item | Recommendation | Risk |
|---|---|---|
| `validate_fit_mask.py` + `validate_joint_mask.py` + `validate_masked_x.py` + `validate_column_limits.py` | **merge into one `validate_masking.py`** (four scripts, one feature, four copies of the same `Line` stub and `check()` helper) | low — they are independent `main()`s; merging is mechanical. Do it *after* the audit run so the pre/post comparison stays honest |
| the `Line` model stub | duplicated verbatim in `validate_fit_mask.py:30-52` and `validate_joint_mask.py:32-50` | trivial |
| `check()` + xraydb stub + dialog neutering | duplicated in 8 of 11 validators | extract to a `validate_common.py` at the root — but **not** into `Functions/<category>/`, which would register it as a fittable model |
| `validate_jointfit.py` | **either supply the `El_Cat_Si_1_*` data (preferred) or rewrite T1/T2/T3 against synthetic data.** T1 (the whitening identity) needs no real data at all — it builds a random residual | medium: T3's value is that it is real data; a synthetic rewrite loses the cross-dataset stderr comparison |
| `validate_jointfit.py:165` dead line | delete | none |
| `validate_xpr.py` missing `__main__` guard | **add the guard** — wrap the body in `def main()` | none, and it removes a latent process fan-out |
| `check()` never asserting | convert to `assert` (or raise) so the scripts become pytest-collectable, then rename to `test_*.py` and add a `pytest.ini` | low, but it changes fail-fast behaviour: today a validator reports *all* failures in one run, which is more useful during a port |
| no all-validators runner | **add `run_validators.ps1`** looping section 4.2 and summarising exit codes | none; highest value per minute |
| `xmodfit_old.py` | drop | none (V15) |
| the five superseded `requirements_*` snapshots | drop, keep one `requirements_viewer.txt` + `requirements_xpr.txt` | low; note the `h5py` omission first (6.3) |
