# WS0 — Test safety net: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A test suite that can fail, a single command that runs all of it, a synthetic
regression for cross-dataset covariance (GLS) joint fitting that needs no off-repo data, and
a recorded baseline. Everything after WS0 is measured against that baseline.

**Architecture:** One tiny shared helper, `tests/harness.py`, that every harness uses for
`check()`. One subprocess runner, `tests/run_all.py`, with a registry of per-harness
metadata (interpreter, external data, slow, known failure). One new synthetic validator,
`validate_joint_gls.py`, which replaces the unrunnable real-data `validate_jointfit.py`. No
engine code changes in WS0.

**Tech stack:** Python 3.9 (`cmc-py39`), numpy, lmfit (vendored), PyQt5 offscreen,
`subprocess`. No pytest: it is not installed and none is added.

**Spec:** `handoff/14-roadmap.md` §WS0. Item definitions: `handoff/12-work-package-preaudit.md`
Phase 0 (P-1, P-2, P-3b, baseline) and `handoff/10-fix-list.md` (A1.10, B2c, A2.5, A0.5).
Headless recipe: `handoff/09-validation-and-env.md` §3.

## Global constraints

- Interpreters: `C:/Users/kerr/.conda/envs/cmc-py39/python.exe` (default) and
  `C:/Users/kerr/AppData/Local/anaconda3/envs/cmcars/python.exe` (only where a harness needs
  real `xraydb` or the XPR/Julia environment).
- Headless: `QT_QPA_PLATFORM=offscreen`, `os.chdir(ROOT)` (load-bearing), a conditional
  `xraydb` stub, and `if __name__ == '__main__':` around anything that can reach a spawn Pool.
- Guardrails from `12-work-package-preaudit.md`:
  - Do **not** touch engine code (`Joint_Fit_Routines.py`, `joint_covariance.py`,
    `Fit_Routines.py`, `xmodfit.py`, `lmfit/`). The one exception is the A2.5 docstring edits
    in Task 4.
  - Do not touch `julia/bakeoff/` except as Task 6 specifies.
  - No refactoring, reformatting or import reordering beyond what a step names.
  - No history rewrites, no push.
- Harnesses that fail for a **real** reason are recorded, not forced green. A WS0 task may
  change a test's *mechanics* (paths, fixtures, guards, `check()`). It must not change a
  test's *expectations*. The one exception: a numeric expectation tied to the removed tim3
  structure is converted to the mechanical assertion it was standing in for, and the
  conversion is noted in the commit message.
- Commit subjects carry the item ID (`P-1`, `P-2`, `P-3b`, `B2c`, `A2.5`, `A0.5`).
- Exit-code convention: **0 pass · 1 fail · 77 skip** (missing external data or environment).

## Review focus

1. **A harness that blocks forever.** The usual cause is a single-file `addData` taking the
   interactive `exec_()` path headless. The runner must kill it at the timeout and report
   `TIMEOUT`, not hang the suite. → Task 2, `hang.py` self-test.
2. **A harness that spawns child processes** (an unguarded spawn Pool). On timeout the
   runner must kill the whole process tree, not only the parent. → Task 2,
   `hang_child.py` self-test.
3. **Non-ASCII output on a Windows pipe.** `±`, `≡` and `—` appear in harness prints. Under
   cp1252 they raise `UnicodeEncodeError`, which looks like a test failure. The runner sets
   `PYTHONIOENCODING=utf-8`. → Task 2, `unicode.py` self-test.
4. **A harness that passes vacuously.** Zero checks run (e.g. an early return), and it exits
   0. `finish()` must return 1 when no check ran and `skip()` was not called. → Task 1
   test.
5. **A stale known-failure list.** A harness registered as a known failure that now passes
   must be reported as `KNOWN-FAIL NOW PASSES`, so the list gets pruned rather than hiding a
   fix. → Task 2, `fixed.py` self-test.

---

### Task 1: Shared check helper (P-1 / A1.10)

**Files:**
- Create: `tests/harness.py`
- Create: `tests/selftest/harness_cases.py` (a driver script the test runs as a subprocess)
- Test: `tests/test_harness.py`

**Interfaces:**
- Produces, in `tests/harness.py`:
  - `check(name: str, cond, detail: str = '') -> bool` — prints `PASS - name | detail` or
    `FAIL - name | detail`, counts it, returns `bool(cond)`. If env `XMF_CHECK_RAISE=1`, a
    failing check raises `AssertionError(name)` after printing.
  - `check_close(name: str, got: float, want: float, tol: float) -> bool` — `abs(got-want) <= tol`;
    prints in `validate_xpr.py`'s existing format (`'%-42s got %13.4f  want %13.4f  %s'`),
    then delegates the counting to `check`.
  - `skip(reason: str) -> NoReturn` — prints `SKIP - reason`, `sys.exit(77)`.
  - `finish() -> int` — prints `ALL PASS (n checks)` and returns 0, or
    `FAILURES PRESENT (k of n)` and returns 1. With **zero checks run** it prints
    `NO CHECKS RAN` and returns 1.
  - `counts() -> tuple[int, int]` — `(n_checks, n_failures)`.
- Importable from any harness after `sys.path.insert(0, ROOT)` as `from tests.harness import ...`.
  `tests/` stays a namespace package: do not add `__init__.py`, which would change how the
  recovered scripts import.

- [ ] **Step 1: Write the failing test.** `tests/test_harness.py` runs
  `tests/selftest/harness_cases.py <case>` as a subprocess with `sys.executable` and asserts
  on its exit code and output:

```python
CASES = {            # case -> (expected exit code, substring expected in stdout)
    'all_pass':   (0,  'ALL PASS (2 checks)'),
    'one_fail':   (1,  'FAILURES PRESENT (1 of 2)'),
    'no_checks':  (1,  'NO CHECKS RAN'),
    'skipped':    (77, 'SKIP - no data'),
    'raise_mode': (1,  'AssertionError'),     # run with XMF_CHECK_RAISE=1
    'close_fail': (1,  'FAIL'),               # check_close(…, 1.0, 1.5, 0.1)
}
```

  The test prints a PASS/FAIL line per case and exits 0 only if every case matches. It is
  itself a harness, so it uses its own minimal local counter, not `tests.harness`.
- [ ] **Step 2: Run it and confirm it fails.**
  `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/test_harness.py`
  Expected: non-zero exit (`ModuleNotFoundError: tests.harness` inside the cases).
- [ ] **Step 3: Implement `tests/harness.py` and `tests/selftest/harness_cases.py`.**
  The case script calls the helper in the six ways the table names. Module-level counters,
  no classes needed.
- [ ] **Step 4: Run the test and confirm it passes.** Same command. Expected: exit 0, six
  `PASS` lines.
- [ ] **Step 5: Commit** — `P-1: add tests/harness.py, a check() that can fail`.

### Task 2: The runner (P-2)

**Files:**
- Create: `tests/run_all.py`
- Create: `tests/selftest/fixtures/{ok.py, bad.py, skip.py, hang.py, hang_child.py, unicode.py, fixed.py}`
- Create: `tests/selftest/registry.json` (`{"tests/selftest/fixtures/fixed.py": {"known": "self-test"}}`)
- Test: `tests/test_run_all.py`
- Modify: `.gitignore` (add `tests/_logs/`)

**Interfaces:**
- Consumes: the exit-code convention from Task 1.
- Produces:
  - CLI: `python tests/run_all.py [--quick] [--only SUBSTR] [--timeout SECONDS] [--dir DIR ...] [--registry PATH]`
    - Default discovery: root `validate_*.py`, `tests/recovered/*.py`, `tests/test_*.py`.
      `--dir` replaces that discovery (used by the self-test).
  - `REGISTRY: dict[str, dict]`, keyed by path relative to `ROOT`. Keys per entry, all
    optional:
    - `interp`: `'cmc-py39'` | `'cmcars'`
    - `external`: str — what is needed and from where
    - `slow`: bool
    - `known`: str — reason for a known failure
    - `timeout`: int
  - A summary table, one row per harness: `status · seconds · interp · note`.
    - Statuses: `PASS`, `FAIL`, `SKIP`, `TIMEOUT`, `KNOWN-FAIL`, `KNOWN-FAIL NOW PASSES`.
  - Exit code 1 if any `FAIL` or `TIMEOUT`, else 0. `KNOWN-FAIL` and
    `KNOWN-FAIL NOW PASSES` do not fail the run, but the latter prints a line asking for the
    registry entry to be removed.
  - Per-harness full output in `tests/_logs/<path-with-slashes-as-__>.log`.

- [ ] **Step 1: Write the fixtures and the failing test.** Fixtures, one behaviour each:

  | fixture | behaviour |
  |---|---|
  | `ok.py` | exit 0 |
  | `bad.py` | exit 1 |
  | `skip.py` | exit 77 |
  | `hang.py` | `time.sleep(3600)` |
  | `hang_child.py` | spawns `python -c "import time; time.sleep(3600)"`, writes the child's PID to the path in env `XMF_CHILD_PID_FILE`, then sleeps |
  | `unicode.py` | prints `'± ≡ —'`, exit 0 |
  | `fixed.py` | exit 0, but registered `known` in the self-test registry |

  `tests/test_run_all.py` invokes
  `run_all.py --dir tests/selftest/fixtures --timeout 5 --registry tests/selftest/registry.json`
  and asserts:
  - the per-fixture statuses, in the table's order (`ok`, `bad`, `skip`, `hang`,
    `hang_child`, `unicode`, `fixed`), are
    `PASS, FAIL, SKIP, TIMEOUT, TIMEOUT, PASS, KNOWN-FAIL NOW PASSES`;
  - the runner's exit code is 1;
  - the PID in `XMF_CHILD_PID_FILE` is no longer alive after the run
    (`tasklist /FI "PID eq <pid>"` shows no match);
  - total wall time is under 60 s.
- [ ] **Step 2: Run it and confirm it fails.** Expected: non-zero (`run_all.py` missing).
- [ ] **Step 3: Implement `tests/run_all.py`.**
  - `subprocess.Popen` with `cwd=ROOT` and `creationflags=CREATE_NEW_PROCESS_GROUP`, and env
    `QT_QPA_PLATFORM=offscreen` and `PYTHONIOENCODING=utf-8`.
  - On timeout: `taskkill /T /F /PID <pid>`.
  - Interpreters map `'cmc-py39'`/`'cmcars'` to the two paths in Global constraints; the
    default is `cmc-py39`.
  - `--registry` loads JSON in place of the built-in `REGISTRY`.
  - `--quick` skips `slow` entries and entries with an `external` requirement.
- [ ] **Step 4: Run the test and confirm it passes.**
  `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/test_run_all.py` → exit 0.
- [ ] **Step 5: Fill the real `REGISTRY` for what is known today.**
  - `validate_jointfit.py`: `known` "El_Cat data not in repo; retired in Task 4".
  - `validate_xpr.py`: `external` "XPR-main/test_data, sibling of the repo", `interp`
    `cmcars`, `slow`.
  - `validate_fastscan.py`: `slow` and `interp` `cmcars`, if it reaches Julia (check its
    header).
  - Leave everything else empty; Task 7 fills in the rest from evidence.
- [ ] **Step 6: Record the as-found run.**
  `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/run_all.py > tests/_logs/AS-FOUND.txt`.
  This is before any harness is migrated. Do not commit the log. Paste its summary table
  into the `12-work-package-preaudit.md` progress log under a dated "as-found" heading.
- [ ] **Step 7: Commit** — `P-2: add tests/run_all.py with timeout, tree-kill and a known-failure registry`
  (includes `.gitignore` and the progress-log entry).

### Task 3: Move the root validators onto the shared helper (P-1, continued)

**Files:**
- Modify: `validate_column_limits.py`, `validate_fit_mask.py`, `validate_gid2d.py`,
  `validate_gid2d_gui.py`, `validate_joint_mask.py`, `validate_jointfit_gui.py`,
  `validate_masked_x.py`, `validate_perdataset_fit.py`, `validate_xlayerspdb_joint.py`,
  `validate_xpr.py`, `validate_fastscan.py`, `validate_fastscan_gui.py`
- Not `validate_jointfit.py` (Task 4 retires it).

**Interfaces:**
- Consumes: `check`, `check_close`, `finish`, `skip` from Task 1.

- [ ] **Step 1: For each file, a mechanical edit only.**
  - Delete the local `check` and its `_OK`/`FAIL` state, and import from `tests.harness`.
  - Replace the final verdict (`return 0 if _OK["ok"] else 1`, or `sys.exit(1 if FAIL else 0)`)
    with `finish()`.
  - `validate_xpr.py` uses `check_close`. Its missing-data `sys.exit('No XPR test data…')`
    becomes `skip('No XPR test data at …')`.
  - Grep each file for any other read of `_OK`/`FAIL` before deleting it. Where a file
    defines `check(name, cond)` without `detail`, the helper's signature is a superset, so
    call sites do not change.
  - A file with no local `check` (the `validate_fastscan*.py` pair may be such files):
    replace only its verdict line with `finish()` if it uses a pass/fail flag. If it already
    exits 0/1 by its own counting, leave it alone.
- [ ] **Step 2: Prove a failure is now observable.** For two files (`validate_joint_mask.py`
  and `validate_gid2d.py`), temporarily flip one `check` condition to `False` and run the
  file directly. It must exit 1 and print `FAILURES PRESENT`. Revert the flip. Record that
  this was done in the commit message.
- [ ] **Step 3: Byte-compile and run.** `python -m py_compile <each file>`, then
  `python tests/run_all.py --only validate_`. Every file must show the same status it had in
  `AS-FOUND.txt`, apart from `SKIP` replacing `FAIL` for missing external data. Any other
  status change means the migration changed behaviour: fix it before committing.
- [ ] **Step 4: Commit** — `P-1: root validators use tests.harness`.

### Task 4: Synthetic cross-dataset GLS regression; retire the real-data validator (B2c, A2.5)

**Files:**
- Create: `validate_joint_gls.py`
- Delete: `validate_jointfit.py` (`git rm`)
- Modify: `joint_covariance.py` and `joint_fit_spec.py` — docstrings only. Replace the
  `El_Cat_Si` / `TGP_Qz` examples (near `joint_covariance.py:18` and `joint_fit_spec.py:55`)
  with generic names (`wafer_A_buffer.txt`, …).
- Modify: `tests/run_all.py` `REGISTRY` — drop the `validate_jointfit.py` entry.

**Interfaces:**
- Consumes:
  - `tests.harness`;
  - `JointFit.add_dataset(name, funcClass=, x=, y=, yerr=, cov=, mask=)`;
  - `JointFit.build_master_params(groups=)` and `JointFit.share_all(name)`;
  - `JointFit._set_ranges(xmin, xmax)`, `JointFit.residual(params, scale)`,
    `JointFit.perform_fit(xmin, xmax, fit_scale=, fit_method=, maxiter=)`;
  - `JointFit._L`, `JointFit._cov_mask`;
  - `joint_covariance.build_full_covariance(y_corr, sigma_meas, k_index, Sigma_logc, v=None, var_fwhm=0.0)`;
  - stock `Fit_Routines.Fit(funClass, x, y=, yerr=)`.
  - Reuse the `Line` model pattern from `validate_joint_mask.py` (y = a·x + b, `a`/`b` in
    `params`); copy the class rather than import it.
- Fixed data for every test: `x = np.linspace(0.1, 2.0, 40)`, true `a = 3.0`, `b = 0.5`,
  `e = 0.05`, noise `np.random.default_rng(0)`. Two scale-factor segments
  (`k_index = (x > 1.0).astype(int)`) with
  `Sigma_logc = [[4e-4, 2e-4], [2e-4, 4e-4]]`.
  Footprint, where on: `v = 0.3*exp(-x/0.5)`, `var_fwhm = 4e-4`.

- [ ] **Step 1: Write `validate_joint_gls.py` with four tests, all failing or erroring
  until the expectations are met.** (The engine already exists, so "failing" here means
  run before review and watch each test print its numbers.)
  - **G1 `whitening_identity_with_footprint`** — 2 datasets, covariance and footprint on.
    Build `C` independently with `build_full_covariance` over the concatenated sidecar
    arrays. Assert:
    - `norm(jf._L @ jf._L.T - C) / norm(C) < 1e-12`;
    - for the residual `r = d - f` at the start values,
      `|sum((L⁻¹r)²) - rᵀC⁻¹r| / (rᵀC⁻¹r) < 1e-10`, with `L⁻¹r` taken from `jf.residual`.
  - **G2 `n1_nocov_equals_stock_fit`** — 1 dataset, no covariance, `a` and `b` free, for
    `fit_scale` in `('Linear', 'Log')`. Fit with stock `Fit` and with `JointFit`, then
    assert:
    - `chisqr` relative difference `< 1e-8`;
    - each parameter's relative difference `< 1e-6`;
    - each stderr's relative difference `< 1e-5`.
  - **G3 `gls_covar_matches_analytic`** — 3 datasets, `a` shared (`share_all('a')`), `b` per
    dataset. Two variants:
    - **cross**: every dataset uses `k ∈ {0,1}` of the 2×2 `Sigma_logc`, so there are
      cross-dataset blocks;
    - **block**: dataset *i* uses `k ∈ {2i, 2i+1}` of a 6×6 block-diagonal `Sigma_logc`
      built from the same 2×2 block, so there are no cross-dataset blocks.

    For each variant, compute the analytic GLS covariance of the linear model,
    `(XᵀC⁻¹X)⁻¹`, scaled by `result.redchi`, because lmfit's default `scale_covar=True`
    and `JointFit` does not override it (`Joint_Fit_Routines.py`, `Minimizer(` calls).
    Assert:
    - stderr(`a`) from `result.params` matches `sqrt` of the analytic diagonal to relative
      `< 1e-4`, in both variants;
    - stderr(`a`) differs between the variants by more than 5%.

    This replaces the old T3's "the stderrs changed" with "the stderrs are right". It also
    partly answers audit claim `01` #4.
  - **G4 `different_masks_with_cross_blocks`** — 2 datasets in the **cross** variant above.
    Mask points 5–11 out of A and points 20–24 out of B, with the footprint on. Build
    `C_active` by restricting the independently built full `C` to the active rows, in the
    composition order slice → `imask` → concatenate → `_cov_mask`. Assert:
    - `len(jf.residual(...)) == 80 - 7 - 5`;
    - the whitened residual equals `solve_triangular(chol(C_active), r_active, lower=True)`
      to relative `< 1e-10`.
- [ ] **Step 2: Run it.** `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" validate_joint_gls.py`.
  Expected: `ALL PASS (n checks)`.
  - If G3's analytic match fails, **stop and report it**. It would mean lmfit's GLS
    covariance is not what `01-joint-fit.md` §1 claims, and that is a finding for WS1, not
    something to adjust the tolerance for.
- [ ] **Step 3: Retire `validate_jointfit.py`** and apply the two docstring generalisations.
  `git grep -n "El_Cat\|TGP_Qz"` over `*.py` must return nothing outside `xmodfit_old.py`.
- [ ] **Step 4: Run the suite.** `python tests/run_all.py --only validate_joint`.
  Expected: `validate_joint_gls.py` `PASS`, `validate_joint_mask.py` unchanged, and no entry
  for `validate_jointfit.py`.
- [ ] **Step 5: Commit** — `B2c/A2.5: synthetic cross-dataset GLS regression replaces validate_jointfit.py`.

### Task 5: Adapt the recovered harnesses (P-3b)

**Files:**
- Modify: `tests/recovered/*.py`, in the batches below.
- Modify: `tests/run_all.py` `REGISTRY`, with an entry for every harness that cannot pass
  today (with the reason).
- Modify: `tests/recovered/README.md` — a status column per harness.

**Interfaces:**
- Consumes: `tests.harness`, `REGISTRY` fields, `Data/1ubq.pdb`,
  `tests/fixtures/legacy_xlayerspdb_2024.par`.

Allowed edits, and only these:

1. absolute and scratch paths → `ROOT`-relative;
2. `Data/tim3_hmmm_ext_all_bestframe.pdb` → `Data/1ubq.pdb`;
3. the old `.par` path → `tests/fixtures/legacy_xlayerspdb_2024.par`;
4. add `if __name__ == '__main__':` where a spawn Pool is reachable;
5. local `check` → `tests.harness`, and a final `finish()`;
6. a numeric expectation that only held for the tim3 fit → the mechanical assertion it
   stood for (e.g. "chisqr equals 1234.5" → "chisqr is finite and the fit succeeded"),
   noted in the commit;
7. a script that has no assertions at all (several `repro_*`, `smoke_*`) gets **no**
   invented checks. Register it with `known: 'smoke: passes if it exits 0'`, or leave it
   unregistered if it exits 0.

Batches, one commit each, in this order:
- (a) `22bf29f6` — joint MCMC and seeding: `verify_joint_*`, `verify_seed*`, `verify_reuse_wrap`,
  `verify_fingerprint`, `verify_jointmcmc`, `verify_jointwindow`, `repro_*`;
- (b) `6f65f848` — PDB MCMC and pickling;
- (c) `c0878594` — plotting, overlay, legend;
- (d) `d63b5d0d` — environment and GUI construction;
- (e) `7068116c` — GID2D and column limits;
- (f) `85efc5b2` — XPR (`interp` `cmcars`, `external` XPR-main where it applies);
- (g) `1fc38b3e` — `smoke_merge`.

The joint batch goes first, because WS1–WS3 lean on it.

- [ ] **Step 1 (per batch): run the batch as found.**
  `python tests/run_all.py --only <batch pattern>`. Note each failure's cause from its log.
- [ ] **Step 2 (per batch): apply only the allowed edits.** Re-run.
- [ ] **Step 3 (per batch): register what still fails** in `REGISTRY` with the real cause
  ("asserts serial sampling; behaviour flipped in b953c3f", "needs a 2-mode fixture; tim3
  case not reproducible from a clean clone (fix-list A2.4)", …). Do not edit engine code to
  make a harness pass. If a failure looks like a genuine engine bug, add it to
  `10-fix-list.md` under a new ID and register the harness `known` with that ID.
- [ ] **Step 4 (per batch): commit** — `P-3b(<batch>): adapt recovered harnesses; N pass, M registered`.

### Task 6: Housekeeping (A0.5 and the stray `julia/bakeoff/` edits)

**Files:** `.idea/XModFit.iml`, `.idea/misc.xml`; the 11 modified files under `julia/bakeoff/`.

- [ ] **Step 1: `.idea/`.** Apply **Decision W1** below. If revert: `git checkout -- .idea/`.
  If keep-local: `git update-index --skip-worktree .idea/XModFit.iml .idea/misc.xml`.
- [ ] **Step 2: `julia/bakeoff/`.** For each of the 11 modified files, diff the working copy
  against `../xpr-seeding-study/julia/bakeoff/<same file>`.
  - If identical, ignoring line endings (`git diff --no-index --ignore-cr-at-eol`):
    `git checkout -- julia/bakeoff/<file>`.
  - If they differ, leave the file as is and list it in the report. **Do not** copy in
    either direction; the study repo has no remote and is the sole copy.
- [ ] **Step 3: Commit** only if Step 1 changed tracked state. Step 2 produces no commit.

### Task 7: The baseline (Phase 0 checkpoint)

**Files:** `handoff/12-work-package-preaudit.md` (progress log), `handoff/14-roadmap.md` (WS0
status line), `tests/run_all.py` (`REGISTRY`, final).

- [ ] **Step 1: Full run.**
  `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/run_all.py > tests/_logs/BASELINE.txt`.
  Expected: exit 0. Every non-`PASS` row is either `SKIP` with an `external` reason or
  `KNOWN-FAIL` with a registered reason.
- [ ] **Step 2: Quick run timing.** `tests/run_all.py --quick`. Record its wall time. It is
  the command WS1 onwards will run before and after every change.
- [ ] **Step 3: Record.**
  - In the progress log: a dated "BASELINE" entry containing the summary table, the quick-run
    wall time, the list of `KNOWN-FAIL` reasons, and any new fix-list IDs Task 5 raised.
  - In `14-roadmap.md` WS0: `Done <date>; baseline in 12-work-package-preaudit.md`.
  - Tick P-1, P-2, P-3b and the checkpoint in the work package.
- [ ] **Step 4: Commit** — `WS0: baseline recorded`.

---

## Decisions for the implementer to get from the user before the task that needs them

- **W1 (Task 6): `.idea/`.** Revert to upstream's files (fix-list A0.5; PyCharm then needs
  the interpreter re-selected locally), **or** keep the local interpreter path and mark the
  two files `skip-worktree` so they stop showing as modified. Recommended: skip-worktree —
  same clean status, nothing on this machine breaks.
- **W2 (Task 5): smoke scripts.** Recommended: an assertion-free script counts as passing if
  it exits 0, and is labelled `smoke` in the table rather than given invented checks.
