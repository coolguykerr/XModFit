# General covariance fitting (XModFit side): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A covariance file can be attached to any single-curve dataset and every fit
function honours it, in single and joint fits, in Linear and Log, with the model-based
(unbiased) covariance by default. Fits without a covariance are unchanged.

**Architecture:** One new Qt-free root module, `covariance_core.py`, owns the data model
(`DatasetCovariance`, `CovTerm`), the loaders, assembly of C over the active points,
factorisation and whitening, and the model-based outer iteration. `Fit` and `JointFit`
each gain one guarded covariance branch that calls into it. `joint_covariance.py` keeps
its public functions as thin wrappers. The GUI gets a minimal single-fit entry point; the
joint panel's existing picker is routed through the new loaders.

**Tech stack:** Python 3.9 (`cmc-py39`), numpy, scipy (`linalg.solve_triangular`,
`stats.chi2`, `interpolate.CubicSpline`), vendored lmfit, PyQt5 offscreen.

**Spec:** `handoff/17-spec-general-covariance.md` (approved 2026-11-10). It is the
authority; section numbers below (§n) refer to it. The LSS_Reader side (spec §8) is a
separate task in that repo and is **not** in this plan, except for a Python reference
re-implementation used as a test fixture (Task 6).

## Global constraints

- Interpreter `C:/Users/kerr/.conda/envs/cmc-py39/python.exe`. Headless recipe:
  `handoff/09-validation-and-env.md` §3 (`QT_QPA_PLATFORM=offscreen`, stub
  `sys.modules['xraydb']` before importing `xmodfit`, `if __name__ == '__main__':` guard
  around anything that reaches a spawn Pool).
- **Before and after every task:** `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/run_all.py --quick`
  (~8.5 min) exits 0 with no new FAIL. Baseline at plan time: 54 PASS / 7 SKIP (`9cddb21`).
- New checks use `tests.harness` (`check`, `check_close`, `skip`, `finish`). New root
  `validate_*.py` files are auto-discovered; register in `tests/run_all.py` `REGISTRY`
  only if slower than 120 s or needing external data. Each new check is shown to fail
  when its feature is reverted (spec §9); record the revert you used in the commit body.
- `covariance_core.py` lives at the repo root, never in `Functions/<category>/`
  (CLAUDE.md rule 1), and imports neither PyQt5 nor `xmodfit`.
- `nan_policy='raise'` everywhere. Points are excluded by dropping rows/columns, never
  by writing NaN (rule 3). The mask is frozen once per `perform_fit` (rule 4). Mask
  order: slice `[imin, imax]` → imask → concatenate → finite mask (§5).
- **No change without a covariance**, numerically or in report text (anti-goal §3).
  `validate_joint_gls.py` G2 and the existing no-cov harnesses are the guard.
- The §5a invariant: `JointFit.residual` branches only on "any covariance attached?".
- Hidden per-region scale factors are never reported as fitted results (§4.6).
- Every downgrade continues the fit, appends a warning to the fit's `warnings`, and
  appears in the report; a downgraded fit's report says so on its first line (§6.5).
  A deliberate fallback never prints a traceback (the runner fails a harness that does).
- With a covariance attached: least squares runs with `scale_covar=False`; χ²_red is
  reported next to the errors with a warning when it falls outside the central 99 % of
  χ²(nfree)/nfree, i.e. outside `[chi2.ppf(0.005, nfree)/nfree, chi2.ppf(0.995, nfree)/nfree]`.
- Outer iteration: stop when every varying parameter moves less than **0.1 × its
  stderr**, or after **5** model-based rounds; on the limit keep the last result and warn.
- v1 load-time match tolerance: x and y within **rtol 1e-4** point by point.
- Merge cost: `xmodfit.py`, `Fit_Routines.py`, `lmfit/minimizer.py` are upstream-shared.
  One guarded branch per method, no reflowing, no renames of existing names.
- Do not touch `julia/bakeoff/`. No push, no history rewrite. Commit only the task's
  files (`git add <paths>`). Trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review focus

1. **The GUI moves one `Fit` between datasets.** Load a covariance on dataset A, select
   B, fit B: B must fit with no covariance, and A must still have its own when reselected.
   → Task 7, test `G2`.
2. **A fit window that cuts through a region, plus a column mask, plus a y ≤ 0 point in
   Log.** The C actually factorised must equal the full C with exactly the excluded rows
   and columns removed, and the non-positive point must be dropped before any log, not
   raise. → Task 5, test `S4`.
3. **A `_ref` sidecar loaded against the `_rrf` data file** (same x, different y). Refused
   at load, with a message naming y and the first mismatching index. → Task 1 `C1`
   (`y_mismatch`), Task 7 `G1`.
4. **Fits saved before this work** (`.xpar` / `.jpar` with a covariance path but no mode
   line) reload in `"data"` mode and reproduce the old numbers; the report says why.
   → Task 7 `G4`, Task 8 `J3`.
5. **A covariance attached to a dataset whose model is fitted in `Linear w/o error`, or a
   multi-curve (dict-x) dataset.** The fit runs exactly as it would without a covariance
   and the report's first line says the covariance was ignored and why. → Task 5 `S5`.

---

### Task 1: Data model and the v1 JSON loader

**Files:**
- Create: `covariance_core.py`
- Create: `validate_covariance.py` (sections `C1`, `C2`; later tasks add sections)
- Create: `tests/fixtures/cov_v1/` with `good.cov.json` and its data file `good_ref.txt`
  (N = 12, K = 3, one 0.5/0.5 overlap row), generated by a small script committed as
  `tests/fixtures/cov_v1/make_fixtures.py`.
- Modify: `handoff/README.md` (index line for doc 18), `handoff/14-roadmap.md` (one line:
  the covariance workstream, docs 17/18, precedes WS2).

**Interfaces:**
- Produces (all in `covariance_core`):
  - `class CovarianceError(ValueError)` — every load-time refusal raises this, with a
    message naming the rule and, for array mismatches, the first bad index.
  - `@dataclass CovTerm`: `name: str`, `kind: str` (`'multiplicative'|'additive'|'dense'`),
    `cov: np.ndarray` (K×K, or N×N for dense), `weights: np.ndarray | None` (N×K),
    `labels: list[str] | None`, `scope: str = 'dataset'` (`'dataset'|'shared'`),
    `provenance: dict = field(default_factory=dict)`.
  - `@dataclass DatasetCovariance`: `x: np.ndarray`, `sigma_meas: np.ndarray`,
    `terms: list[CovTerm]`, `y: np.ndarray | None = None` (v1: the y the sidecar was built
    for; used only for matching), `replaces_y: np.ndarray | None = None` (legacy only),
    `source: str = ''`, `loader: str = ''`, `data_file: str | None = None`.
    Methods: `n -> int`; `subset(index: np.ndarray) -> DatasetCovariance` (rows of x,
    y, replaces_y, sigma_meas and every W; rows and columns of dense `cov`; K×K `cov`
    untouched); `match_problems(x, y=None, rtol=1e-4) -> list[str]` (empty = matches; y
    compared only when `self.y` is not None); `y_for_fit(y_dataset) -> np.ndarray`
    (`replaces_y` if set, else `y_dataset`); `without_terms() -> DatasetCovariance`
    (same, `terms=[]`; used by the legacy fallback).
  - `load_covariance(path: str, raw_path: str | None = None) -> DatasetCovariance` —
    dispatches through a registry: `register_loader(name: str, accepts: Callable[[str], bool], load: Callable[[str, str|None], DatasetCovariance])`.
    Task 1 registers `'xmodfit-covariance-v1'` (accepts a JSON file whose `format` is
    `"xmodfit-covariance"`); Task 3 registers the legacy loaders. An unrecognised file
    raises `CovarianceError`.

- [ ] **Step 1: Write the fixture generator and failing tests.** `make_fixtures.py`
  writes `good_ref.txt` (3 columns, `%.4e`, like LSS_Reader) and `good.cov.json` (full
  precision, spec §7.1 schema, one `stitch` multiplicative term). `C1` loads it and
  checks every field; then, for each §7.1 rule, it mutates a copy in a temp dir and
  asserts `CovarianceError` with a message containing the given word:

  | case | mutation | message contains |
  |---|---|---|
  | `no_format` | delete `format` | `format` |
  | `bad_version` | `version: 2` | `version` |
  | `len_sigma` | drop one `sigma_meas` | `sigma_meas` |
  | `neg_sigma` | one `sigma_meas` = 0 | `sigma_meas` |
  | `w_shape` | `weights` N×(K−1) | `weights` |
  | `cov_asym` | `cov[0][1] += 1e-3` | `symmetric` |
  | `cov_not_psd` | `cov = [[1,2,0],[2,1,0],[0,0,1]]` | `positive` |
  | `dense_w` | dense term with `weights` | `dense` |
  | `bad_kind` | `kind: "mult"` | `kind` |

  `C2` (matching and subset): `match_problems(x_file, y_file)` is empty for the
  `%.4e` data file; `y × 1.01` gives one problem mentioning `y` and index `0`
  (`y_mismatch`); x shifted by one point gives a problem mentioning `x`.
  `subset(np.array([2,3,4]))` returns n = 3, W rows 2..4, and the same K×K cov.
  A `data_file` differing from the loaded name is **not** an error at this level
  (Task 7 warns).
- [ ] **Step 2: Run, verify they fail** — `python validate_covariance.py` → FAIL (`ModuleNotFoundError: covariance_core`).
- [ ] **Step 3: Implement the dataclasses, registry, v1 loader and validation.**
  Symmetry tolerance `max|cov − covᵀ| ≤ 1e-12·max|cov|`; PSD: smallest eigenvalue
  ≥ −1e-10·max eigenvalue. `scope` defaults to `"dataset"`; `provenance` is kept, never
  used.
- [ ] **Step 4: Run, verify pass**; quick suite still exit 0.
- [ ] **Step 5: Commit** — `covariance_core.py validate_covariance.py tests/fixtures/cov_v1/ handoff/README.md handoff/14-roadmap.md`, subject `COV T1: covariance data model and v1 JSON loader`.

---

### Task 2: Assembly, factorisation, whitening, outer iteration

**Files:**
- Modify: `covariance_core.py`
- Modify: `validate_covariance.py` (sections `C3`–`C7`)

**Interfaces:**
- Consumes: Task 1 dataclasses.
- Produces:
  - `@dataclass Block`: `cov: DatasetCovariance | None` (already restricted to the active
    points), `y: np.ndarray` (data in the residual), `yerr: np.ndarray` (used only when
    `cov is None`), `intensity: np.ndarray` (D: the data in `"data"` mode, the model in
    `"model"` mode).
  - `assemble(blocks: list[Block], scale: str) -> tuple[np.ndarray, list[str]]` —
    C over the concatenation, in block order; returns warnings (shared-term conflicts).
    `scale` is `'Linear'` or `'Log'`. Formulas, per spec §4.2/§4.5:

    | | Linear | Log (residual ln y − ln f) |
    |---|---|---|
    | measurement | σ_meas² (or yerr² if `cov is None`) | (σ_meas/y)² (or (yerr/y)²) |
    | multiplicative | D W Σ Wᵀ D | W Σ Wᵀ |
    | additive | W Σ Wᵀ | E W Σ Wᵀ E, E = diag(1/intensity) |
    | dense | C_dense | E C_dense E |

    Shared terms (`scope='shared'`, same `name`) are one set of random variables: build
    a stacked loading matrix over all blocks carrying the term and add `G Σ Gᵀ`
    (cross-blocks included). If their `labels` or `cov` differ (`np.allclose`, rtol
    1e-12, atol 0), each block's copy is treated as private and one warning names the
    term and the datasets.
  - `needs_intensity(blocks, scale) -> bool` — True when any term's contribution depends
    on D (any multiplicative in Linear; any additive or dense in Log).
  - `factorize(C) -> tuple[np.ndarray, list[str]]` — wraps
    `joint_covariance.cholesky_with_jitter` without changing it: plain Cholesky first;
    on failure call the helper and decide whether it floored eigenvalues
    (`‖L Lᵀ − (C + jitter·I)‖ > 1e-8‖C‖`). Warning texts name the jitter value or say
    "eigenvalues floored".
  - `whiten(L, r) -> np.ndarray` and `whiten_jac(L, J) -> np.ndarray` (lower
    `solve_triangular`, `check_finite=False`).
  - `outer_iterate(fit_once: Callable[[], R], rebuild: Callable[[R], None], max_rounds: int = 5, tol: float = 0.1) -> tuple[R, int, bool]`
    — `fit_once()` runs one fit from the current parameters and returns an lmfit-like
    result with `.params`; `rebuild(result)` re-assembles C from the model at
    `result.params`. Round 0 is the data-based fit; up to `max_rounds` model rounds
    follow; converged when every varying parameter's |Δvalue| < tol·stderr (stderr None
    or 0: relative change < 1e-6). Returns `(last_result, model_rounds, converged)`.

- [ ] **Step 1: Write failing tests.**
  - `C3` Peelle (spec §9.3): measurements y = (1.5, 1.0), σ_meas = (0.15, 0.10), one
    shared-by-both multiplicative term, W = [[1],[1]], Σ = [[0.04]]; model f = μ.
    Using `assemble` + `factorize` + `whiten` and the closed-form GLS of a constant:
    data mode → μ = 0.8823529 (to 1e-6); `outer_iterate` with model mode → 1.1538462
    (to 1e-6), equal to the σ-weighted mean (1.5/0.0225 + 1/0.01)/(1/0.0225 + 1/0.01).
  - `C4` blocks: two blocks with private `stitch` terms → C block-diagonal (off-block
    max |·| == 0). Same term marked `shared` with identical cov → off-block equals
    D_a W_a Σ W_bᵀ D_b to 1e-14. Shared with different cov → block-diagonal **and** one
    warning naming the term.
  - `C5` mask = submatrix: assemble on `subset(idx)` equals the full C indexed
    `[np.ix_(idx, idx)]` to 1e-14, for linear and log, idx non-contiguous.
  - `C6` factorize: a PD C → no warning, `L Lᵀ == C`; a rank-deficient C (Σ = 1e2·11ᵀ,
    σ_meas tiny) → one warning; force the floor path with an indefinite matrix → warning
    contains `floored`.
  - `C7` log formulas: for a one-hot multiplicative term, log C equals
    diag((σ/y)²) + W Σ Wᵀ exactly; for an additive term it equals E(WΣWᵀ)E with
    E = 1/intensity.
- [ ] **Step 2: Run, verify fail** (`AttributeError: assemble`).
- [ ] **Step 3: Implement.** Symmetrise C once at the end (`0.5(C+Cᵀ)`). Store W
  dense (spec §11 leaves sparse to the builder; N ≲ 10³ here).
- [ ] **Step 4: Run, verify pass**; quick suite exit 0.
- [ ] **Step 5: Commit** — subject `COV T2: assemble, factorize, whiten, outer iteration`.

---

### Task 3: Legacy loaders into the core

**Files:**
- Modify: `covariance_core.py` (legacy conversion + registration)
- Modify: `joint_covariance.py` (keep every public function; `load_dataset_cov` and
  `build_full_covariance` unchanged in signature and output)
- Modify: `validate_covariance.py` (section `C8`)

**Interfaces:**
- Consumes: Task 1, Task 2 (`assemble`).
- Produces:
  - `from_legacy_meta(meta: dict, x: np.ndarray, source: str, loader: str) -> DatasetCovariance`
    — `meta` is today's cov dict (`y_corr, sigma_meas, k_index, Sigma_logc`, optional
    `v, var_fwhm`). Builds term `attenuator_cal` (multiplicative, `scope='shared'`,
    one-hot W from `k_index`, `cov=Sigma_logc`, labels = `SF_LIST` as strings) and, if
    `v` present, `beam_fwhm` (multiplicative, shared, W = (v / y_corr)[:, None],
    `cov=[[var_fwhm]]`) per spec §7.2. `replaces_y = y_corr`; `y = None`.
  - Registered loaders `'legacy-npz'` (`*.npz` that is not a bundle) and
    `'legacy-c-factors'` (`*_c_factors.txt`; needs `raw_path`, else `CovarianceError`
    naming `raw_path`). Both go through the existing `joint_covariance` readers.
  - `load_bundle(path: str, datasets: list[str] | None = None) -> dict[str, DatasetCovariance]`
    (via `joint_covariance.load_bundle_npz`; the different-`var_fwhm` refusal stays).
  - The both-or-neither `v`/`var_fwhm` check stays where it is (`load_cov_npz`).

- [ ] **Step 1: Write failing test `C8`.** Build a 2-dataset bundle-format `.npz`
  fixture in a temp dir (reuse `validate_joint_gls.py`'s X, K2, SIG2, V_FP, VAR_FWHM
  values). For each dataset and the concatenation: `assemble(..., 'Linear')` with
  `intensity = y_corr` (data mode) equals `joint_covariance.build_full_covariance(...)`
  element-wise to 1e-12 relative (spec §9.2), with and without the footprint term, and
  the cross-dataset block is present. Also: the npz and c_factors+raw routes give equal
  `DatasetCovariance` arrays; a c_factors file without `raw_path` raises
  `CovarianceError`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** Do not change `cholesky_with_jitter` or
  `build_full_covariance`.
- [ ] **Step 4: Run, verify pass**; quick suite exit 0 (`validate_joint_ws1.py` T2 still
  covers the v/var_fwhm refusal).
- [ ] **Step 5: Commit** — subject `COV T3: legacy covariance formats as core loaders`.

---

### Task 4: `JointFit` on the core

**Files:**
- Modify: `Joint_Fit_Routines.py` (`__init__`, `add_dataset`, `clear`, `_build_chol`,
  `residual` cov branch, `perform_fit`, `set_emcee_minimizer` cov gate, `_report`)
- Modify: `validate_joint_gls.py` (pass `covariance_mode='data'`; G3 stderr expectation
  becomes absolute — drop the `* redchi`; see Ruling R1)
- Modify: `validate_joint_mask.py` (pass `covariance_mode='data'`)
- Modify: `validate_covariance.py` (sections `J1`, `J2`)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `JointFit.add_dataset(..., cov=None)` accepts a `DatasetCovariance`, a legacy dict
    (converted with `from_legacy_meta`), or None. The objective dict keeps key `"cov"`
    (now a `DatasetCovariance | None`); `"y"` is `cov.y_for_fit(y)`; `"yerr"` stays the
    dataset's own error column (used when that dataset has no covariance or degrades).
  - `JointFit.covariance_mode: str = 'model'` (`'model'|'data'`), and
    `perform_fit(..., covariance_mode=None)` overriding it for one call.
  - Read-only properties kept for existing callers: `_use_cov` (any objective has a
    covariance; used by `lmfit/minimizer.py:1947`, `xmodfit.py:7212`,
    `pdb_joint_logprob`) and `_use_footprint` (any term named `beam_fwhm`; used by
    `validate_joint_ws1.py` T2, `validate_joint_mask.py`). `_Sigma_logc`, `_var_fwhm`,
    `_register_shared_sigma`, `_register_shared_var_fwhm` are removed; the
    different-`Sigma_logc` case now degrades to private terms with a warning (spec §5).
  - Attributes `_L`, `_cov_mask`, `_jitter` keep their names and meaning (harnesses read
    them).
  - `JointFit.cov_info: dict` after each fit: `mode`, `scale`, `rounds`, `converged`,
    `factor_warnings`.
  - `covariance_core.report_block(covs: list[tuple[str, DatasetCovariance | None]], info: dict, warnings: list[str], result) -> str`
    — the `Covariance:` report block and the χ²_red line; Task 5 reuses it.

- [ ] **Step 1: Write failing tests.**
  - `J1` mixed joint (spec §9.6): dataset A with a v1-style `DatasetCovariance`
    (private `stitch` term), dataset B without; no shared params. Linear and Log: no
    crash; joint χ² equals A's one-dataset covariance fit χ² + B's no-cov stock-`Fit` χ²
    to 1e-8 relative; B's parameters equal its stock fit to 1e-6.
  - `J2` Log: a dataset with a y ≤ 0 point inside the window and a covariance — the fit
    completes (no `ValueError` from `nan_policy`), the point is not in `_cov_mask`'s
    active rows, and the report lists the scale as `Log`.
  - Existing `validate_joint_gls.py` G1–G4b and `validate_joint_mask.py` pass in
    `'data'` mode to their current tolerances (spec §9.2).
- [ ] **Step 2: Run, verify fail** (J1 crashes in `_build_chol` today).
- [ ] **Step 3: Implement.**
  - `_build_chol(intensities=None)`: per objective build a `Block` from the slice →
    imask view (`cov.subset(active index)`); data mode uses the fit data as intensity,
    model mode the current model; concatenate; finite mask (`np.isfinite` of y and
    σ, σ > 0, and y > 0 in Log) → `_cov_mask`; `assemble` on the masked subset;
    `factorize`; collect warnings as run warnings.
  - `residual` cov branch: r = d − f (Linear) or ln d − ln f (Log, on the finite-masked
    rows); `whiten`. `w/o error` scales take the existing no-cov branch, with a run
    warning (§6.5) — the branch condition stays "covariance attached and scale is
    weighted".
  - `perform_fit`: least-squares family builds `Minimizer(..., scale_covar=False)` when
    any covariance is active; in model mode with `needs_intensity`, wrap the minimise
    call in `outer_iterate`; emcee freezes C at `self.result.params` (model mode) and
    stays serial (existing gate).
  - `_report`: `report_block(...)` — a `Covariance:` block (per dataset: source file, loader, each term's
    name/kind/scope/K; then mode, scale, model rounds, jitter/floor, the χ²_red line and
    its 99 % warning). Any downgrade puts `DEGRADED FIT:` + the reasons as the report's
    first line.
- [ ] **Step 4: Run, verify pass**; quick suite exit 0.
- [ ] **Step 5: Commit** — subject `COV T4: JointFit assembles C through covariance_core`.
  Commit body lists Ruling R1.

---

### Task 5: Single-dataset `Fit` with a covariance

**Files:**
- Modify: `Fit_Routines.py` (`set_x`, new `attach_covariance`, `residual` and
  `residual_jac` one guarded branch each, `perform_fit`, `set_emcee_minimizer`, `_report`)
- Modify: `validate_covariance.py` (sections `S1`–`S6`)

**Interfaces:**
- Consumes: Tasks 1–4 (S1 compares against `JointFit`).
- Produces:
  - `Fit.set_x(x, y=None, yerr=None, mask=None, cov=None)` — `cov=None` clears any
    covariance (spec §6.2). A non-None `cov` goes through `attach_covariance`.
  - `Fit.attach_covariance(cov: DatasetCovariance | None) -> None` — raises
    `CovarianceError` if `cov.n != len(self.x)` or `match_problems(self.x, self.y)` is
    non-empty (array-x only; dict-x stores it and the fit ignores it, see below).
  - `Fit.cov`, `Fit.covariance_mode = 'model'`, `Fit.warnings: list[str]` (reset by
    each `perform_fit`), `Fit.cov_info` (as `JointFit.cov_info`).
  - `Fit.perform_fit(..., covariance_mode=None)`.
  - `Fit.chi2_stat_for_ci() -> tuple[str, float]` — `('chisqr', result.chisqr)` with a
    covariance active in the last fit, else `('redchi', result.redchi)` (consumed by
    Task 7's confidence-interval code).

- [ ] **Step 1: Write failing tests.**
  - `S1` single = one-dataset joint (spec §9.5): same data and covariance, Linear and Log,
    model mode: parameters equal to 1e-8 relative, χ² equal to 1e-10 relative.
  - `S2` Jacobian (spec §9.10): a harness model with `dydp` (exponential, as in the
    Log-fix check); whitened `residual_jac` vs central finite differences of the
    whitened `residual`, max relative error < 1e-6, Linear and Log. Build the
    perturbed parameter set from a fixed snapshot each time (`residual` writes values
    back into `fit_params`).
  - `S3` `set_x` without `cov` after one with → `fit.cov is None` and the residual
    equals the stock no-cov residual exactly.
  - `S4` window + mask + Log (Review focus 2): x window cutting through region 2, a
    column mask excluding 3 interior points, one y ≤ 0 point; the factorised C equals
    the full Log C restricted to the expected active indices (1e-14); the fit completes.
    Then the pre-sliced call pattern used by `update_plot`:
    `set_x(x[i0:i1], ..., cov=cov.subset(np.arange(i0, i1)))` + `residual` gives the
    fit's final χ² to 1e-10 (spec §9.7). Finally change the column mask and refit: the
    factorised L has the new active size and again equals the restricted C (spec §9.9,
    "a mask change rebuilds the factorisation").
  - `S5` the §6.5 rows that live in `Fit` (each: fit completes, warning present, report
    first line starts `DEGRADED FIT:` except where noted):
    `Linear w/o error` and `Log w/o error` (covariance ignored; residual equals the
    stock one exactly); dict-x model with a covariance (ignored); x changed after attach
    (`fit.x = x * 1.01` before `perform_fit`: fit without covariance; for a legacy cov
    the fallback is `y_corr` with diag σ_meas² via `without_terms()`, spec §6.5); model
    rounds forced to hit the limit (`max_rounds=1` with a strongly nonlinear start);
    an indefinite term (floor warning); `fit_method='emcee'`, `emcee_cores=4` →
    sampling runs serial (assert the minimiser received `workers=1`) with a warning.
  - `S6` no change without a covariance: for each of the four scales, a fit with
    `cov=None` reproduces a stored baseline (computed in the test from a fresh stock
    `Fit` before any covariance code path is touched) bit-for-bit in parameters, χ² and
    report text.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
  - `residual`: at the top of the array-x branch, `if self._cov_active(fit_scale): return self._cov_residual(fit_scale)` (a private method that builds the `Block`,
    reusing a cached L keyed on `(fit_scale, n_active, mode)` that `set_x`,
    `attach_covariance` and every outer-iteration round invalidate). Called directly
    (as `update_plot` does) it builds C from the current model in model mode.
  - `residual_jac`: same guard; base Jacobian −∂f/∂p (Linear) or −(1/f)∂f/∂p (Log) on
    the finite-masked rows, then `whiten_jac`.
  - `perform_fit`: decide applicability once (weighted scale, array-x, x still matches
    → else downgrade with warning); `scale_covar=False` on every `Minimizer` built;
    model mode with `needs_intensity` → `outer_iterate`; emcee → force `workers=1`
    (sampling and seed scan) with a warning when `emcee_cores > 1`, and freeze C at
    `self.result.params`.
  - `_report`: Task 4's `report_block` and the same `DEGRADED FIT:` rule. With no
    covariance attached the report is byte-identical to today.
  - emcee with no prior least-squares result in model mode: freeze C in data mode and
    warn (spec §6.4).
- [ ] **Step 4: Run, verify pass**; quick suite exit 0.
- [ ] **Step 5: Commit** — subject `COV T5: single-dataset fits honour an attached covariance`.

---

### Task 6: Synthetic stitching recovery (statistical acceptance)

**Files:**
- Create: `tests/lss_reference.py` — a Python re-implementation of the spec §8 merge, used
  only as a fixture (not run by the runner: it does not match `tests/test_*.py`).
- Create: `validate_cov_stitching.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 5.
- Produces: `lss_reference.reduce(scans: list[dict], drop: dict | None = None) -> dict`
  with keys `x, y, sigma_meas, sigma_total, weights (N×K), Sigma (K×K), fac, facerr, a`.
  Each scan dict has `x, y, sigma` (counting). Per join, per spec §8: overlap = existing
  points inside the new scan's x range; cubic spline (`CubicSpline`) of the new scan at
  those x; `fac` = weighted mean ratio existing/spline with ratio errors from σ_meas,
  `facerr = facerr_abs · max(1, √χ²_red)` (spec §8.3); scale the new scan; average
  overlap points (0.5/0.5 weight rows); σ_meas without facerr; σ_total with it; a_m =
  mean weight row of the overlap points; Σ by the §4.3 recursion. Also
  `lss_reference.to_cov(red) -> DatasetCovariance` (one `stitch` term).

- [ ] **Step 1: Write the harness.** Model: harness-local slab curve
  R(q) = scale · R_F(q) · |1 + (Δρ/ρ_s)(e^{iqd} − 1)|² · e^{−q²σ²}, parameters d, Δρ, σ,
  scale (fixed q_c, Fresnel R_F) — cheap and nonlinear in d and σ. Truth: q 0.02–0.5,
  K = 4 overlapping scans of ~45 points, counting σ = 1 % of R, true per-scan log scale
  errors N(0, 0.03²) with scan 1 exact. Seeds: 200, `np.random.default_rng(seed)`.
  - `T6a` reduction sanity: every weight row sums to 1; overlap rows are 0.5/0.5; in the
    previous-scan-only layout diag(Σ) equals the cumulative Σδ² and
    Σ[k,l] = Σ[min,min] (spec §8.8 checks 3–4, on the reference).
  - `T6b` recursion case: scan 3's x range starts inside scan 1, so a_3 mixes rows of
    scans 1 and 2; Σ equals an independent Monte-Carlo estimate of the region-error
    covariance over 4000 draws to within 10 % elementwise on the diagonal.
  - `T6c` pulls, model mode, Linear: for d and σ, (θ̂ − θ_true)/stderr over 200 seeds has
    |mean| < 0.25 and SD in [0.85, 1.15].
  - `T6d` the covariance-free fit with σ_meas only shows the over-confidence: pull SD
    > 1.5 for at least one of d, σ. Print (do not assert) the same for σ_total.
  - `T6e` Log vs Linear model mode: mean over seeds of |θ̂_log − θ̂_lin| / stderr < 0.2.
- [ ] **Step 2: Run.** If a seed's fit fails, the harness records it and fails `T6c`
  only if more than 2 of 200 fail. If runtime exceeds 120 s, register it in `REGISTRY`
  as `slow`.
- [ ] **Step 3: Verify revert sensitivity.** Temporarily make `assemble` ignore terms →
  `T6c` fails; restore.
- [ ] **Step 4: Commit** — subject `COV T6: synthetic stitching recovery harness`.

---

### Task 7: Single-fit GUI entry point, persistence, confidence intervals

**Files:**
- Modify: `xmodfit.py` — `create_dataDock` (~937: two buttons next to "Remove Files"),
  `removeData` (4274), `doFit` (1873 `set_x`), `update_plot` (6262 `set_x`, residuals
  6293-6313), the fit-options area (mode combo next to `fitScaleComboBox`),
  `setTargetChiSqr` (2304), `calcErrPushButtonClicked` / `checkMinMaxErrLimits`
  (2346-2470), `saveParameters` (4929) / `loadParameters` (5005) and the fit-result
  header writer (~1960-2090).
- Create: `validate_covariance_gui.py` (sections `G1`–`G5`)

**Interfaces:**
- Consumes: Task 1 `load_covariance`, `CovarianceError`; Task 5 `Fit.set_x(cov=)`,
  `Fit.covariance_mode`, `Fit.chi2_stat_for_ci()`.
- Produces:
  - `XModFit.covariances: dict[str, DatasetCovariance]`, keyed by data key
    (`'<n><>fname'`), deleted in `removeData`.
  - `XModFit.attach_covariance_to(data_key: str, path: str, raw_path: str | None = None) -> bool`
    — the scripted and dialog path share it: loads, refuses multi-curve datasets and any
    `match_problems` with a `QMessageBox.warning` naming the problem (returns False);
    warns (status bar + message) when `data_file` differs; marks the list item (bold
    font + tooltip `covariance: <file>`; the item **text** is the data key and must not
    change).
  - `XModFit.remove_covariance_from(data_key: str) -> None`.
  - `.xpar` header lines (written only when a covariance is attached, read by the
    existing substring chain): `#Covariance=<path>`, `#Covariance Raw=<path>`,
    `#Covariance Mode=<model|data>`. A file with `#Covariance=` but no mode line loads
    in `"data"` mode and appends the note "saved before covariance modes existed;
    reloaded in data mode" to `Fit.warnings`. A missing covariance file on load → warn,
    continue without (spec §6.5 fit-time row).

- [ ] **Step 1: Write failing tests** (headless `XModFit` instance, as in
  `validate_jointfit_gui.py`; two datasets from the Task 1 fixture).
  - `G1` `attach_covariance_to` with the matching sidecar → True, item bold. With a
    y-mismatched sidecar (Review focus 3) → False, the patched `QMessageBox.warning`
    received a message containing `y`.
  - `G2` Review focus 1: attach to A, select B, `doFit` → B's `self.fit.cov is None`;
    reselect A, `doFit` → `self.fit.cov` is A's.
  - `G3` `update_plot` on A after a fit: displayed Chi-Sqr equals `self.fit.result.chisqr`
    to 1e-10, and `output_params['Residuals']['y']` equals the whitened residual.
  - `G4` save `.xpar` with a covariance; reload → covariance reattached, mode restored.
    Strip the mode line → reload in `"data"` mode with the note in the report (Review
    focus 4).
  - `G5` confidence interval with a covariance: `setTargetChiSqr` sets
    `targetchisqr = chisqr_min + chi2.ppf(CL, 1)` (68.3 % → +1.0 to 1e-3) and the profile
    compares `result.chisqr`; without a covariance both are unchanged (assert today's
    formula on the same fit).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** `doFit` passes `cov=self.covariances.get(fname)` only on the
  array-x branch; `update_plot` passes `cov.subset(np.arange(_pi0, _pi1))`; the overlay
  `set_x` at 6377 passes none (plotting only). Residuals with a covariance: the whitened
  residual vector against `x[active]`. The mode combo writes `self.fit.covariance_mode`
  and is disabled when the selected dataset has no covariance. CI code reads
  `self.fit.chi2_stat_for_ci()` and uses chisqr + Δχ² when it returns `'chisqr'`.
- [ ] **Step 4: Run, verify pass**; quick suite exit 0 (the GUI harnesses in
  `tests/recovered/` are the guard for unchanged no-cov behaviour).
- [ ] **Step 5: Commit** — subject `COV T7: load/remove covariance for single fits; absolute CI`.

---

### Task 8: Joint panel, spec, orchestration, `.jpar`; documentation

**Files:**
- Modify: `joint_fit_spec.py:267-268`, `joint_fit_orchestration.py:94-99` (call
  `covariance_core.load_covariance(path, raw_path)` and `match_problems`; a mismatch at
  compile time follows the fit-time row: dataset fitted without covariance + warning)
- Modify: `joint_fit_panel.py:413-425` (file filter adds `*.cov.json`; when an entry has
  no `cov_path`, the panel pre-fills it from `XModFit.covariances` for that dataset —
  spec §6.6 "uses each dataset's attached covariance unless the panel overrides it")
- Modify: `joint_fit_io.py` (a file-level `#CovMode: <model|data>` line; absent → `"data"`
  plus the same note as Task 7)
- Modify: `xmodfit.py` joint launch (7172, 7241: pass the mode)
- Modify docs: `handoff/01-joint-fit.md` (covariance section → points to 17/18),
  `handoff/08-core-app.md` (single-fit covariance entry point), `handoff/17` §2 status
  line ("implemented in 18, commits …"), `handoff/14-roadmap.md` (mark done).
- Modify: `validate_covariance_gui.py` (sections `J1`–`J3`)

**Interfaces:**
- Consumes: Tasks 1–4, 7.

- [ ] **Step 1: Write failing tests.**
  - `J1` a spec with one dataset carrying a v1 sidecar and one without compiles and fits
    (Linear), matching Task 4 `J1`'s χ².
  - `J2` panel pre-fill: attach a covariance to dataset A in the GUI, open the joint
    panel → A's covTable row shows the file; overriding it in the spec wins.
  - `J3` `.jpar` round trip keeps the mode; a `.jpar` written without `#CovMode:` reloads
    in `"data"` mode with the note (Review focus 4); a legacy bundle-format joint fit
    reproduces its pre-change parameters to 1e-12 (fixture from Task 3).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**, then update the docs listed above. Each doc edit states
  what was verified and by which check.
- [ ] **Step 4: Full suite** — `tests/run_all.py` (full, ~33 min) exits 0.
- [ ] **Step 5: Commit** — subject `COV T8: joint panel and .jpar on the new loaders; docs`.

---

## Rulings made while planning (for the user to check)

- **R1 — joint covariance stderr changes.** `validate_joint_gls.py` G3 asserts
  `stderr = √((XᵀC⁻¹X)⁻¹ · redchi)`, i.e. today's joint covariance fits rescale by
  χ²_red. The approved rule (spec §6.2) is absolute errors whenever a covariance is
  attached, so G3's expectation drops the `· redchi`. Best-fit values and χ² still
  reproduce to 1e-12 in data mode; only stderr moves, by √χ²_red. If wrong: one
  keyword (`scale_covar`) per minimiser call.
- **R2 — display reuses the last round's factor (changed by the final review, M1).**
  `update_plot` calls `residual` outside `perform_fit`. `Fit` now reuses the factor the
  fit's final round used when the active rows match (same scale, mode, x, data, σ and
  terms), as `JointFit` keeps its `_L`, so the displayed χ² equals `result.chisqr` in
  model mode too (`validate_covariance_gui.py` G3, `validate_covariance.py` F6); otherwise
  it falls back to building C from the current state (in model mode from the model at the
  current parameters). Originally "display uses the current model", accepted because a
  converged fit differs from the last round's C by less than the 0.1σ tolerance.
- **R3 — multi-curve datasets refuse a covariance at load** (the user is present), in
  addition to the spec's fit-time "ignore with warning" for dict-x models.
- **R4 — the test model for Task 6 is a harness-local single-slab curve**, not
  `XLayers`, so 200 seeds × outer iterations stay within the quick-suite budget.
