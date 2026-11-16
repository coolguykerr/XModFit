# WS2 Joint-Fit Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every joint fit reports numbers computed from its exact objective: an exact per-dataset split of χ² and of −2 ln L (log-determinant included, exact also in `model` mode), a guarded and calibrated comparison between two fits, end-of-fit diagnostics, leave-one-out refits, and opt-in per-dataset error rescaling.

**Amended 2026-11-11 (user):** the exact likelihood replaces re-scoring under a common covariance (spec §3.5, §4.1). Task 3 is now the key only, and the distance-to-optimum diagnostic is in Task 8.

**Architecture:** A new Qt-free root module `joint_fit_stats.py` holds everything that reads a fit. It works from an immutable `JointSnapshot` taken once at the end of `JointFit.perform_fit`; the split and the comparison read only the snapshot, and the diagnostics and LOO also use the live `JointFit` through a quiet evaluation path that has no side effects. `Joint_Fit_Routines.py` gains only what the snapshot needs, the opt-in rescaling, and one diagnostics block at the end of `_report()`.

**Tech Stack:** Python 3.9 (`C:/Users/kerr/.conda/envs/cmc-py39/python.exe`), numpy, scipy (`solve_triangular`, `scipy.stats.chi2`), lmfit, the existing `covariance_core`.

**Spec:** `handoff/20-spec-ws2-reporting.md` (approved 2026-11-11). Section numbers below (§3.2 etc.) refer to it. Background: `handoff/14-roadmap.md` §WS2, `handoff/17-spec-general-covariance.md`.

## Global Constraints

- Interpreter: `C:/Users/kerr/.conda/envs/cmc-py39/python.exe`; headless: `QT_QPA_PLATFORM=offscreen`.
- `joint_fit_stats.py` lives in the repo root, never in a `Functions/<category>/` folder (CLAUDE.md rule 1). It imports no Qt.
- `nan_policy='raise'` stays on every least-squares path; exclude points by dropping rows, never by writing NaN (CLAUDE.md rule 3).
- The covariance mask is frozen once per `perform_fit` (rule 4). Nothing in this plan recomputes `_cov_mask` outside `_build_chol`'s first build.
- Angles stay unbounded (rule 5). Nothing here adds bounds.
- `_eval_chisqr` and `computeModelComparison` in `xmodfit.py` are **not** touched. No GUI change anywhere (spec §2 non-goals). `Fit_Routines.py` (upstream-shared) is not touched.
- With rescaling off (the default) and the diagnostics block aside, every fit result is bit-identical to today: same `result.params`, `chisqr`, `nfev`, `Niter`, emitted signals.
- Error policy: degrade and report. A diagnostic that fails prints `diagnostics unavailable: <reason>` and never alters the fit or the rest of the report. Tests stay strict.
- Labels (§3.4), verbatim in meaning: `χ² (Linear, weighted by the error column)`; `GLS χ² (Linear, covariance <mode>)`; `χ² of ln y (Log, σ_ln = σ/y)`; `GLS χ² of ln y (Log, covariance <mode>)`; `unweighted sum of squares of log10 y` (not a likelihood); `unweighted sum of squares of y / max(y), per dataset` (not a likelihood); `data` mode with a covariance appends `; covariance built from the measured data; approximate`; a rescaled fit prefixes `rescaled ` and appends `(errors rescaled per dataset by √χ²_red, s = …; error scales fitted per dataset; approximate)`; a `model`-mode emcee snapshot's label adds `; sampled with C frozen at the prior fit, not the exact likelihood`.
- Exact likelihood (spec §3.5): `nll2 = χ² + log det + n ln 2π`, with `log det` from the Cholesky diagonal of C(θ̂) (`2 Σ ln L_jj`), never from `np.linalg.det`. Everything in §3–§5 uses C at the fitted model (`SnapDataset.intensity`); `fit_intensity` is used only to reproduce `result.chisqr` and compute `fit_gap`.
- Thresholds (all keyword arguments with these defaults): q-range spread `0.10`; heterogeneity ratio `3.0`; compromise flag `median + 3·1.4826·MAD` and `> 2`; correlation `|ρ| > 0.9`; small-N `N < 4` → no z; rescale convergence `1 %` or `5` rounds.
- Tests: every test is shown to fail when its feature is reverted (state in the commit message how you checked). Harnesses use `tests.harness.check/check_close/finish`, live in the repo root as `validate_*.py` (auto-discovered by `tests/run_all.py`).
- Commits: `git add <paths>` only (never `Data/*`, `xmodfit_old.py`, `julia/bakeoff/*`); never push; message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A dataset with zero rows left** (every active point dropped by the covariance mask, e.g. all y ≤ 0 in Log): its row range is empty; the split gives `n_points=0`, `sequential=marginal=0.0`, `chi2_per_point=None`, and the report prints "no points", never NaN or ZeroDivisionError. Test P9 (Task 2).
2. **A failed Hessian** (`result.covar is None`, stderrs `None`): degeneracy says "no parameter covariance", LOO influence is in raw units with a note, AIC/BIC still print. Tests D6 (Task 6), L4 (Task 9).
3. **A `__masked_x__` model with a column mask** (`yfit_masked=True`): the snapshot's `f_i` must not be masked twice. Test P8 (Task 2).
4. **A host that has `functionCalled` connected** (the GUI plots from it): snapshot, Fisher, joint-vs-individual and LOO emit nothing. Test Q2 (Task 1), and the emission counter in D7, I4, L3.
5. **One-dataset and two-dataset joint fits:** no z, Fisher shares are 1.0, `leave_one_out` refuses N < 2 with a message instead of fitting an empty set. Tests M4 (Task 4), F3 (Task 8), L5 (Task 9).

---

## File map

| file | responsibility | tasks |
|---|---|---|
| `Joint_Fit_Routines.py` | quiet evaluation, row ranges, prior rows, last blocks, model blocks; rescaling loop; snapshot at end of fit; diagnostics block in `_report` | 1, 5, 6 |
| `joint_fit_stats.py` (new) | snapshot, split, exact likelihood, labels, key, comparison, information criteria, diagnostics, joint-vs-individual, Fisher, tension, distance to optimum, LOO | 2–4, 6–9 |
| `joint_fit_io.py`, `joint_fit_spec.py` | `#RescaleErrors:` line and `JointFitSpec.rescale_errors` | 5 |
| `validate_joint_stats.py` (new) | tests Q, P, K, M, R | 1–5 |
| `validate_joint_diag.py` (new) | tests D, I, F, L | 6–9 |
| `handoff/14-roadmap.md`, `handoff/20-…`, `handoff/README.md` | status updates | 10 |

Shared test fixtures, defined once at the top of `validate_joint_stats.py` and imported by `validate_joint_diag.py` (`from validate_joint_stats import Line, make_jf, shared_cov, private_cov`):

- `Line`: copy of `validate_joint_gls.py:44` (`y = a*x + b`, params `a`, `b`).
- `X = np.linspace(0.1, 2.0, 40)`, `A_TRUE, B_TRUE, E = 3.0, 0.5, 0.05`, data `A_TRUE*X + B_TRUE + E*rng.standard_normal(40)` from `np.random.default_rng(seed)`.
- `private_cov(y, k)`: `cc.DatasetCovariance(X, E*ones, [cc.CovTerm('stitch', 'multiplicative', np.diag([0, 4e-4]), weights=onehot(X > 1.0), scope='dataset')], y)`.
- `shared_cov(y)`: same but the term is `CovTerm('beam', 'multiplicative', np.array([[9e-4]]), weights=np.ones((40, 1)), labels=['beam'], scope='shared')`, identical on every dataset, so it couples them.
- `make_jf(nds, cov=None, share='a', scale_err=None, seed=0)`: `nds` datasets `D0..`, `a` shared via `build_master_params(groups=[jf.share_all('a')])` when `share`, `b` free per dataset, `vary=True` set on the master parameters. `scale_err[i]` multiplies dataset i's true noise (the yerr column stays `E`).

---

### Task 1: Quiet evaluation, row ranges, prior rows (`JointFit`)

**Files:**
- Modify: `Joint_Fit_Routines.py` (`__init__` ~143, `_build_chol` 627–721, `residual` 726–799, `perform_fit` 1243–1401)
- Test: `validate_joint_stats.py` (create; sections Q)

**Interfaces:**
- Produces:
  - `JointFit.quiet()` — context manager. Inside it `residual()` does not increment `Niter` and does not emit `functionCalled`. On exit (also on exception) restores `Niter`, and for every objective its model's `x`, `__fit__`, the values of `func.params[k]` and `func.__dict__[k]` for every local key in `param_map`, and `obj["yfit_masked"]`.
  - `JointFit.row_ranges() -> list[tuple[int, int]]` — per objective in fit order, half-open `(start, stop)` into the residual vector minimised, prior rows excluded. Covariance path (`self._use_cov` and the last fit scale in `('Linear','Log')`): segment counts of `_cov_mask`. Otherwise: `len(self.active_index(k))`. Raises `RuntimeError` on the covariance path when `_cov_mask` is `None`.
  - `JointFit.prior_rows(params) -> np.ndarray` — `[(params[k].value - mu)/sig for k,(mu,sig) in self._priors.items()]`, `float`, length 0 without priors. `residual` uses it (same values as today).
  - `JointFit._last_blocks: list[covariance_core.Block] | None` — the masked blocks passed to `cc.assemble` in the most recent `_build_chol` (their `intensity` is the C actually used). Reset to `None` where `perform_fit` resets `_L`.
  - `JointFit._fit_scale`, `JointFit._fit_method` — recorded at the start of `perform_fit`.
  - `JointFit.model_blocks(params, fit_scale) -> tuple[list[cc.Block], list[str]]` — the masked blocks `_build_chol(fit_scale, model=True)` would pass to `cc.assemble` at `params` (model intensity, with the data substituted on active rows where it is non-finite or, in Log, ≤ 0), plus the substitution notes. Evaluated inside `quiet()`. It does **not** touch `_L`, `_cov_mask`, `_cov_warnings`, `_jitter` or any model's state. Refactor `_build_chol` so both share one private block builder; `_build_chol`'s behaviour stays bit-identical. Requires `_cov_mask` (raises `RuntimeError` without it).

- [ ] **Step 1: Write the failing tests** (`validate_joint_stats.py`, with the fixtures above)

```python
def test_q1_quiet_restores():          # Q1
    jf = make_jf(2); jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    n0 = jf.Niter; before = [(o['func'].x.copy(), o['func'].a, o['func'].params['a'].value)
                             for o in jf.objectives]
    p = jf.result.params.copy(); p['a'].value += 1.0
    with jf.quiet():
        jf.residual(p, 'Linear')
    check('Q1 Niter unchanged', jf.Niter == n0)
    check('Q1 models restored', all(np.array_equal(b[0], o['func'].x) and b[1] == o['func'].a
                                    and b[2] == o['func'].params['a'].value
                                    for b, o in zip(before, jf.objectives)))

def test_q2_no_emission():             # Q2
    jf = make_jf(2); hits = []
    jf.functionCalled.connect(lambda *a: hits.append(1))
    jf.perform_fit(X[0], X[-1], fit_scale='Linear'); n = len(hits)
    with jf.quiet():
        jf.residual(jf.result.params, 'Linear')
    check('Q2 no functionCalled inside quiet()', len(hits) == n)

def test_q3_row_ranges():              # Q3: no cov, private cov + Log y<=0 row, window + mask
    for case in ('nocov', 'cov_log_drop', 'window_mask'):
        jf = ...  # per case; 'cov_log_drop' sets one y of D1 to -1 and fits Log
        r = jf.residual(jf.result.params, scale)
        rr = jf.row_ranges()
        check('Q3 %s: ranges tile the data rows' % case,
              rr[0][0] == 0 and all(a[1] == b[0] for a, b in zip(rr, rr[1:]))
              and rr[-1][1] == len(r) - len(jf.prior_rows(jf.result.params)))
    # cov_log_drop: D1's range is one row shorter than len(active_index(1))

def test_q4_priors_unchanged():        # Q4
    jf = make_jf(2); jf.set_priors({'d0_b': (0.5, 0.1)}); jf.perform_fit(X[0], X[-1])
    r = jf.residual(jf.result.params, 'Linear')
    check('Q4 prior rows are the tail', np.array_equal(r[-1:], jf.prior_rows(jf.result.params)))

def test_q5_last_blocks():             # Q5: model mode, Linear, private cov
    jf = make_jf(2, cov=private_cov); jf.covariance_mode = 'model'; jf.perform_fit(X[0], X[-1])
    C, _ = cc.assemble(jf._last_blocks, 'Linear')
    check_close('Q5 last blocks rebuild L', np.linalg.cholesky(C), jf._L, rtol=1e-12)

def test_q6_model_blocks():            # Q6: model mode, Linear, private cov
    jf = make_jf(2, cov=private_cov); jf.covariance_mode = 'model'; jf.perform_fit(X[0], X[-1])
    state = (jf._L.copy(), jf._cov_mask.copy(), list(jf._cov_warnings), jf.Niter)
    blocks, notes = jf.model_blocks(jf.result.params, 'Linear')
    check('Q6 no state touched', np.array_equal(state[0], jf._L) and
          np.array_equal(state[1], jf._cov_mask) and state[2] == jf._cov_warnings
          and state[3] == jf.Niter)
    jf.push_params(jf.result.params); jf._build_chol('Linear', model=True)
    check_close('Q6 equals _build_chol(model=True)',
                np.linalg.cholesky(cc.assemble(blocks, 'Linear')[0]), jf._L, rtol=1e-12)
    # a model returning NaN on one active row: that block's intensity equals y there, one note
```

- [ ] **Step 2: Run and see them fail**

Run: `QT_QPA_PLATFORM=offscreen "C:/Users/kerr/.conda/envs/cmc-py39/python.exe" validate_joint_stats.py`
Expected: FAIL — `AttributeError: 'JointFit' object has no attribute 'quiet'`.

- [ ] **Step 3: Implement** the Interfaces above. `quiet()` with `contextlib.contextmanager` and a `self._quiet` flag (initialised `False`) checked in `residual`. In `_build_chol`, assign `self._last_blocks = masked` just before `cc.assemble`.

- [ ] **Step 4: Run the new tests, then the existing joint harnesses**

Run: `... validate_joint_stats.py` → all Q PASS. Then `... tests/run_all.py --only joint` → no new FAIL.

- [ ] **Step 5: Commit** — `git add Joint_Fit_Routines.py validate_joint_stats.py`; `git commit -m "WS2 T1: JointFit quiet evaluation, row ranges, prior rows, last and model blocks"`

---

### Task 2: Snapshot, split and labels (`joint_fit_stats.py`)

**Files:**
- Create: `joint_fit_stats.py`
- Modify: `Joint_Fit_Routines.py` (`perform_fit`: take the snapshot on all three return paths)
- Test: `validate_joint_stats.py` (sections P)

**Interfaces:**
- Consumes: Task 1 (`quiet`, `row_ranges`, `prior_rows`, `_last_blocks`, `model_blocks`, `_fit_scale`, `_fit_method`).
- Produces:
  - `@dataclass(frozen=True) SnapDataset`: `name: str`, `x, y, yerr: np.ndarray` (active rows), `active_index: np.ndarray`, `index_hash: str`, `window: tuple[float, float]`, `rows: tuple[int, int]`, `used: np.ndarray` (bool over active rows: kept by `_cov_mask`, all True off the covariance path), `r: np.ndarray` (on used rows: d−f Linear, ln d − ln f Log, the residual rows themselves for `w/o error`), `f: np.ndarray` (model on used rows), `fit_intensity: np.ndarray | None` (from `_last_blocks`: the C the fit was scored under), `intensity: np.ndarray | None` (C at θ̂: from `jf.model_blocks(result.params, scale)` when `cov_mode == 'model'`, else equal to `fit_intensity`; both `None` off the covariance path), `cov: DatasetCovariance | None` (as used, subset to used rows; `None` when ignored), `cov_source: str | None`, `cov_hash: str | None`, `model_class: str`, `model_files: dict`.
  - `@dataclass(frozen=True) JointSnapshot`: `datasets: tuple[SnapDataset, ...]`, `fit_scale: str`, `cov_in_use: bool`, `cov_mode: str | None` (`None` unless `cov_in_use`), `z: np.ndarray` (data rows whitened under C(θ̂): `L_exact⁻¹ r`), `z_fit: np.ndarray` (data rows of the residual the fit minimised, from the quiet residual), `logdiag: np.ndarray` (`ln diag(L_exact)` per data row; off the covariance path `ln σ` per row, σ the weight the residual used; empty for `w/o error`), `prior: np.ndarray`, `jitter: float` (of the exact factorisation), `method: str`, `nvarys: int`, `ndata: int`, `chisqr: float` (`result.chisqr`), `params: dict[str, float]`, `vary: dict[str, bool]`, `stderr: dict[str, float | None]`, `covar: np.ndarray | None`, `var_names: tuple[str, ...]`, `param_map: dict[str, tuple[tuple[int, str], ...]]`, `warnings: tuple[str, ...]`, `label: str | None` (`'at posterior medians'` for emcee, plus the frozen-C clause of the Global Constraints when the run used a frozen `model` C; `'aborted fit; not the minimised value'` when `cov_info['aborted']`), `rescale: dict | None` (`None` until Task 5, which defines its keys).
  - `snapshot(jf) -> JointSnapshot` — one quiet residual at `jf.result.params` (gives `z_fit`), one `model_blocks` call in `model` mode, one exact factorisation (gives `z`, `logdiag`); no model left changed.
  - `raw_parts(jf, params) -> list[tuple[np.ndarray, np.ndarray]]` — per dataset `(r_i, f_i)` on used rows at `params`, unwhitened, assembled as `residual` assembles them, inside `quiet()`. `snapshot` uses it; Task 8 reuses it.
  - `rebuild_L(snap, which='exact') -> np.ndarray | None` — `cc.factorize(cc.assemble(blocks, scale)[0])[0]` over `Block(d.cov, y_used, yerr_used, d.intensity)` per dataset (`which='fit'`: `d.fit_intensity`); `None` off the covariance path.
  - `@dataclass DatasetTerm`: `name, n_points: int, sequential: float, marginal: float, chi2_per_point: float | None, logdet_seq: float | None, logdet_marg: float | None, nll2_seq: float | None, nll2_marg: float | None, window, index_hash, scale: str, cov: bool, cov_mode: str | None, rescale_s: float | None`. `logdet_seq = 2 Σ logdiag[rows]`; `logdet_marg = ln det (L Lᵀ)_ii` via `np.linalg.slogdet` of the block (or its own Cholesky); `nll2_* = term + logdet_* + n ln 2π`; all `None` for `w/o error`.
  - `per_dataset_objective(snap) -> tuple[list[DatasetTerm], dict]`; extras keys `prior: float`, `coupling: float`, `order: list[str]`, `objective_label: str`, `likelihood: bool`, `nll2: float | None` (Σ `nll2_seq`), `fit_gap: float | None` (`Σ sequential + prior − snap.chisqr`; `None` unless `cov_mode == 'model'` and `cc.needs_intensity`).
  - `objective_label(fit_scale: str, cov_in_use: bool, cov_mode: str | None, rescaled: dict | None = None) -> tuple[str, bool]` — the §3.4 table; `(label, is_likelihood)`.
  - `marginal_terms(snap, L=None) -> list[float]` — `m_i = r_iᵀ (L Lᵀ)_ii⁻¹ r_i` (after any rescale division, Task 5); equals `sequential` off the covariance path.
  - `JointFit.last_snapshot: JointSnapshot | None` — set at the end of `perform_fit`; a snapshot failure leaves it `None` and appends a `_RunWarning('diagnostics unavailable: snapshot failed: <reason>')`.

- [ ] **Step 1: Write the failing tests**

```python
P1  no covariance, Linear and Log, priors on: terms, ex = per_dataset_objective(snapshot(jf))
    sequential == marginal == hand sum of ((y-f)/E)**2 (Linear) / ((ln y - ln f)*y/E)**2 (Log)
    per dataset (hand f from result params); sum(sequential) + ex['prior'] == result.chisqr (rel 1e-12)
P2  private covariance, mixed fit (D0 cov, D1 none), data mode: sequential == marginal (1e-10),
    coupling == 0 (abs 1e-10 * chisqr), sum + prior == result.chisqr (1e-10)
P3  shared covariance, 3 datasets: sum + prior == chisqr (1e-10); terms[0].sequential ==
    terms[0].marginal (1e-10); marginal_i == r_i @ solve(C_ii, r_i) with C from an
    independent np build of diag(E^2) + D w Σ w^T D (1e-10); coupling == sum(seq) - sum(marg);
    a second jf with datasets in reverse order: same total (1e-10), different terms[0].sequential
P4  model mode, Linear, private cov: Σ z_fit² + prior reproduces result.chisqr (1e-10) and
    so does the rebuild_L(snap, 'fit') path, although f != fit_intensity (assert
    max|f - fit_intensity| > 0: the final round used the previous C); snap.datasets[i].intensity
    == f_i on used rows; rebuild_L(snap) equals chol(assemble(jf.model_blocks(...)))
    (1e-12); extras['fit_gap'] == Σ sequential + prior − result.chisqr and is not None
P12 exact likelihood, shared covariance, 3 datasets, Linear and Log, data and model mode:
    Σ logdet_seq == np.linalg.slogdet(C)[1] and extras['nll2'] ==
    -2 * scipy.stats.multivariate_normal(mean=f_cat, cov=C).logpdf(y_cat) (Linear; Log on
    ln y with the Log C) to 1e-10 rel, C built independently in numpy; terms[0].nll2_seq
    == terms[0].nll2_marg (1e-10)
P13 no covariance, Linear: logdet_seq_i == Σ ln(E²) over the dataset's rows; 'Linear w/o
    error': all nll2/logdet fields None and extras['nll2'] None; data mode: fit_gap None
    and the label contains 'covariance built from the measured data'
P5  labels: objective_label for all six §3.4 rows; 'w/o error' rows return is_likelihood False
P6  emcee (emcee_walkers=10, emcee_steps=20, Linear, no cov): snap.label ==
    'at posterior medians'; no identity asserted
P7  aborted model-mode fit (a functionCalled slot sets jf.fit_abort = True once Niter
    exceeds the round-0 evaluation count, measured from an identical unaborted fit):
    snap.label == 'aborted fit; not the minimised value'
P8  __masked_x__ model + column mask: snapshot f has the used length; identity holds (1e-12)
P9  Log, covariance on (so the mask drops rows instead of logging them), every y of D1
    set to -1 and d1_b fixed: D1's rows are empty -> n_points 0, sequential 0.0,
    chi2_per_point None; per_dataset_objective and format_diagnostics do not raise
P10 snapshot has no side effects: Niter, emission count, every model's x and a/b unchanged
P11 jf.last_snapshot is set after perform_fit (leastsq, model-mode, emcee paths); forcing
    snapshot() to raise (monkeypatch) leaves result unchanged and adds the warning
```

For P9 build D1 with y negative on all points except where it would be needed; the dataset stays registered (the mask drops its rows).

- [ ] **Step 2: Run and see them fail** — `ModuleNotFoundError: joint_fit_stats`.

- [ ] **Step 3: Implement** `joint_fit_stats.py` per the Interfaces. Notes the signatures leave open:
  - Recover `r` and `f` per dataset the way `residual` assembles them (slice `[imin, imax]`, then `imask`, honour `yfit_masked`, then the dataset's `_cov_mask` segment on the covariance path). `SnapDataset.r` is always unwhitened (d−f, ln d − ln f, or for `w/o error` scales the residual rows themselves); the weighted rows are `snap.z[rows]`. Off the covariance path `sequential = marginal = Σ z_i²`.
  - `sequential_i = Σ z[rows_i]²`. Except in `model` mode with `cc.needs_intensity`, `z` **is** `z_fit` (the residual taken under `quiet()`) and `logdiag` comes from `jf._L`, never from a recomputation. Only that case factorises C(θ̂) afresh.
  - `marginal_terms` and `logdet_marg` use the same L as `z` (the exact one).
  - Hashes: `hashlib.sha1` over `np.ascontiguousarray(a, float).tobytes()`; `cov_hash` over every term's `kind`, `scope`, `labels`, `cov` and `weights`, and `sigma_meas` (never the intensities; see Task 3).
  - `model_files = dict(getattr(func, 'filepaths', {}) or {})`.
  - In `perform_fit`, a private `_take_snapshot()` called just before each `return self._report(), …`.

- [ ] **Step 4: Run** `validate_joint_stats.py` → Q and P PASS; `tests/run_all.py --only joint` → no new FAIL.

- [ ] **Step 5: Commit** — `"WS2 T2: JointSnapshot, sequential/marginal split, exact likelihood, labels"`

---

### Task 3: Comparability key

**Files:** Modify `joint_fit_stats.py`; Test `validate_joint_stats.py` (sections K)

There is no re-scoring under another fit's covariance (amended spec §4.1): each fit is scored under its own C(θ̂), which Task 2 already does.

**Interfaces:**
- Consumes: Task 2 (`JointSnapshot`).
- Produces:
  - `class NotComparable(ValueError)` — message names the first differing field, e.g. `"dataset 3 window 0.030:0.450 vs 0.030:0.500"`.
  - `comparability_key(snap) -> dict` — keys: `fit_scale`; `datasets`: list of `{name, data_hash (x,y,yerr), window, index_hash, cov: None | (cov_source, cov_hash, cov_mode)}` in order; `rescale: None | (mode, inflate_only)`.
  - `check_comparable(a, b) -> None` — raises `NotComparable`.
  - The `cov_hash` in the key covers everything that defines the likelihood surface (every term's `kind`, `scope`, `cov`, `weights`, `labels`, and `sigma_meas`), but **not** the intensities, which legitimately differ between `model`-mode fits.

- [ ] **Step 1: Tests**

```python
K1  each key field changed alone on b (window of one dataset, one y value, a mask point,
    fit scale Linear->Log, covariance detached, cov_mode model->data, dataset order,
    rescale on) -> check_comparable raises NotComparable and the message contains the
    field name ('window', 'data', 'index', 'scale', 'covariance', 'mode', 'order', 'rescale')
K2  allowed to differ: model class (Line vs Quad: y = a x^2 + b), priors, fit method,
    a parameter fixed in b -> check_comparable does not raise
K3  two model-mode fits of the same data and covariance file from different starting
    points (different θ̂, so different intensities): check_comparable does not raise;
    changing one term's Σ, or one sigma_meas value, raises naming 'covariance'
```

- [ ] **Steps 2–5:** run (fail: `ImportError comparability_key`), implement, run (Q, P, K pass), commit `"WS2 T3: comparability key"`.

---

### Task 4: `compare_fits`, nesting, information criteria

**Files:** Modify `joint_fit_stats.py`; Test `validate_joint_stats.py` (sections M)

**Interfaces:**
- Consumes: Task 2 (`per_dataset_objective`, `nll2_seq`), Task 3 (`check_comparable`).
- Produces:
  - `@dataclass Comparison`: `names: list[str]`, `d: list[float]`, `sum_d: float`, `sd_d: float | None`, `N: int`, `z: float | None`, `preferred: str | None` (`'A'`/`'B'`), `sign_count: tuple[int, int]` (datasets favouring the preferred fit, N), `delta_k: int | None`, `nested: str | None` (`'A in B'`, `'B in A'`, `None`), `lr_p: float | None`, `worst_dataset: str`, `objective_label: str`, `likelihood: bool`, `delta_chi2: float`, `delta_logdet: float` (`sum_d == ½(delta_chi2 + delta_logdet)` for weighted objectives; `delta_logdet = 0.0` for unweighted), `prior_difference: float`, `model_differences: list[str]`, `notes: list[str]`.
  - `compare_fits(a, b) -> Comparison` — calls `check_comparable`; for weighted objectives (including rescaled) `d_i = ½(nll2_seq_i^A − nll2_seq_i^B)`, each snapshot's own terms; for unweighted `d_i = s_i^A − s_i^B`. `sd_d = np.std(d, ddof=1)`; `z = |Σd| / (sd_d·√N)`; `z = None` and the note `"no calibrated z: too few datasets to estimate the spread"` when `N < 4`. Positive Σd prefers B. A rescaled vs unrescaled pair is already refused by the key.
  - `nesting(a, b) -> str | None` — `'B in A'` when every endpoint set of a B master is a union of A masters' endpoint sets, every endpoint free in B is free in A, and per dataset `model_class` and `model_files` agree; and symmetrically.
  - `lr_p`: for `'B in A'`, `scipy.stats.chi2.sf(max(0, −2·Σd), Δk)` (Σd = ½(nll2_A − nll2_B), so −2Σd = nll2_B − nll2_A, the LR statistic of the nested B against A) with `Δk = a.nvarys − b.nvarys`; only for likelihood objectives that are not rescaled.
  - `information_criteria(snap) -> dict` — `{'aic', 'bic', 'k', 'n', 'form'}`; weighted objectives: `Φ + 2k`, `Φ + k·ln n` with `Φ = extras['nll2'] + extras['prior']`, `form='known errors'` (rescaled: `k = nvarys + N` datasets); unweighted: `n·ln(RSS/n) + 2k`, `n·ln(RSS/n) + k·ln n`, `form='unknown error scale'`; `n = Σ n_points`, `k = nvarys`. Rescaled fits add `note='conditional on the fitted s'`.
  - `format_comparison(c, ic_a=None, ic_b=None) -> str` — headline `Δ = Σd ± sd·√N (z = …)`, preferred fit and sign count (`"B preferred on 14/17 datasets"`); the coupling sentence when either snapshot has coupling ≠ 0: `"datasets treated as independent after conditioning on shared error terms in the listed order"`; the per-dataset `d` table; the detail line with bare Δ(−logL) labelled `"not a confidence measure"`, followed by its parts `"(Δχ²/2 = …, Δ log det/2 = …)"` whenever `delta_logdet != 0`; nesting and Δk and LR p with the caveat `"the model is likely misspecified; leave-one-out is the honest check"`; the prior difference on its own line; then, last, `"information criteria (assume a correct model; read with caution)"` with AIC/BIC when given.

- [ ] **Step 1: Tests**

```python
M1  synthetic snapshots (dataclasses.replace on a real snapshot to set z and logdiag per
    dataset so nll2^A, nll2^B are known): sum_d, sd_d, z, sign_count reproduce hand
    values exactly (1e-12)
M2  the formatted text contains 'not a confidence measure', the sign-count sentence and
    the label
M3  shared covariance: the coupling sentence is present; private covariance: absent
M4  N = 3 (and N = 1): z is None, the 'no calibrated z' note is in the text, no 'z =' in it
M5  Line vs Quad on the same data: compared, nested None, model_differences names both
M6  a = b-with-'a'-unshared (A has d0_a, d1_a; B shares a): nesting == 'B in A',
    delta_k == 1, lr_p == chi2.sf(-2*sum_d, 1) (1e-12), and -2*sum_d >= 0
M7  information_criteria: Linear weighted -> aic == nll2 + prior + 2k exactly, and
    aic_A - aic_B == chisqr_A - chisqr_B + 2Δk (data mode: log det cancels, 1e-10);
    'Linear w/o error' -> the n ln(RSS/n) form; format puts the IC heading after the
    headline
M8  constructed Linear model-mode pair (two fits whose models differ by 20 % in scan-2
    intensity, private multiplicative term): delta_logdet != 0; sum_d ==
    ½(delta_chi2 + delta_logdet) (1e-12); sum_d equals ½ of the difference of two
    independent multivariate_normal logpdfs, each with its own C(θ̂) (1e-10); the
    '(Δχ²/2 = …, Δ log det/2 = …)' parts are in the text
M9  Log, multiplicative-only covariance, model mode, two different fits: delta_logdet == 0
    (abs 1e-10), since C_log does not depend on the intensity
```

- [ ] **Steps 2–5:** run (fail), implement, run, commit `"WS2 T4: calibrated comparison, nesting, known-error AIC/BIC"`.

---

### Task 5: Opt-in per-dataset error rescaling (§6.3)

**Files:**
- Modify: `Joint_Fit_Routines.py` (`__init__`, `residual`, `perform_fit`, `_report`), `joint_fit_stats.py` (snapshot `rescale`, labels), `joint_fit_io.py` (`format_jpar` ~210, `parse_jpar` ~284), `joint_fit_spec.py` (`JointFitSpec.__init__` ~147, `compile` ~313)
- Test: `validate_joint_stats.py` (sections R)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `JointFit.rescale_errors: str | None = None` (`None` or `'per_dataset'`), `JointFit.rescale_inflate_only: bool = False`, `JointFit.rescale_s: list[float] | None` (current factors, one per objective; `None` when off), `JointFit.rescale_info: dict` (`{'rounds': int, 'converged': bool, 'nu': list[float]}`).
  - In `residual`, when `rescale_s` is set: on the covariance path each dataset's segment of `r` is divided by `s_i` **before** whitening (equivalent to `C → S C S`); off it, each dataset's weighted residual segment is divided by `s_i`. Prior rows are never scaled.
  - `rescale_factors(snap, inflate_only=False) -> tuple[list[float], list[float]]` in `joint_fit_stats.py` — `(s, nu)` with `nu_i = n_i − p·n_i/N`, `s_i = √(m_i^unscaled / nu_i)` where `m_i^unscaled` is the marginal term with the current `s` undone; `max(1, s_i)` when `inflate_only`. A dataset with `nu_i ≤ 0` keeps `s_i = 1` and adds a note.
  - Loop in `perform_fit` after the normal fit (all least-squares paths, including after `outer_iterate`): snapshot → factors → set `rescale_s` → refit warm-started from `result.params` (model mode: the whole `outer_iterate` again) → repeat until `max |Δs_i/s_i| < 0.01` or 5 rounds. Requested with a `w/o error` scale or `emcee`: not applied, `_Downgrade` warning `"error rescaling needs a weighted least-squares fit; not applied"`.
  - `JointSnapshot.rescale = {'mode': 'per_dataset', 'inflate_only': bool, 's': tuple, 'nu': tuple, 'rounds': int, 'converged': bool}` or `None`; `objective_label` with it returns the rescaled label (Global Constraints) and `is_likelihood=False`.
  - Log-determinant with rescaling (spec §6.3): chol(S C S) = S·L for diagonal S > 0, so the snapshot's `logdiag` on dataset i's rows gains `ln s_i`, and `logdet_seq_i` gains exactly `2 n_i ln s_i`. Off the covariance path σ becomes σ·s_i. Task 2's fields carry it; nothing else changes.
  - `.jpar`: `#RescaleErrors: per_dataset` or `#RescaleErrors: per_dataset inflate_only`, written only when on, read into `settings['rescale_errors']`; `JointFitSpec.rescale_errors` / `rescale_inflate_only` copied onto the `JointFit` by `compile`. (Wiring the GUI's save/load to pass the setting through is WS4; the fitted `s` are in the report and the snapshot, and their storage across sessions is WS3.)
  - `_report` prints, when on: `"Errors rescaled per dataset by √χ²_red (opt-in): s = [...], ν = [...], rounds, converged"` and, beside it, the unrescaled χ²/n per dataset.

- [ ] **Step 1: Tests**

```python
R1  off by default: a fit with rescale_errors=None gives bit-identical result.params,
    chisqr, nfev and Niter to a fit on a JointFit that never had the attribute set
    (construct both; compare with ==)
R2  two datasets, true noise E and 2E, yerr column E, a shared: s ≈ (1, 2) within 15 %;
    sum(nu) == N - p (1e-12)
R3  shared covariance: the C implied (S C S from rebuild_L of the snapshot and s) is PSD
    (min eig >= -1e-10 * max) and equals np.diag(sv) @ C @ np.diag(sv) (1e-12), sv the
    per-row factor
R4  inflate_only with true noise E/2 on D0: s_0 == 1.0 exactly
R5  label starts with 'rescaled ' and is_likelihood False; comparability_key has the
    rescale field; a rescaled vs unrescaled compare raises NotComparable naming 'rescale'
R6  .jpar round trip: format_jpar -> parse_jpar keeps 'per_dataset inflate_only';
    a .jpar without the line parses rescale_errors None; JointFitSpec.compile sets it
R7  'Linear w/o error' with rescale on: the fit is the unrescaled fit and the report starts
    with 'DEGRADED FIT:' naming error rescaling
R8  rescaled, shared covariance: logdet_seq_i − (the same snapshot's logdet with s undone)
    == 2 n_i ln s_i (1e-10); Σ logdet_seq == slogdet(S C S) (1e-10); two rescaled fits
    compare (no NotComparable), with lr_p None; information_criteria k == nvarys + N
```

- [ ] **Steps 2–5:** run (fail), implement, run Q–R plus `tests/run_all.py --only joint` and `--only covariance`, commit `"WS2 T5: opt-in per-dataset error rescaling by sqrt(chi2_red)"`.

---

### Task 6: Diagnostics block in the report (§6.3 report, §6.4, failure safety)

**Files:**
- Modify: `joint_fit_stats.py`, `Joint_Fit_Routines.py` (`_report` 1432–1492)
- Test: `validate_joint_diag.py` (create; sections D)

**Interfaces:**
- Consumes: Tasks 2, 4, 5.
- Produces:
  - `diagnostics(jf, snap, individual_fits=None, *, qrange_tol=0.10, hetero_ratio=3.0, rho_tol=0.9, fisher=True, tension=True, optimum=True) -> dict` with keys `objective` (label, mode, order when coupling ≠ 0), `quality` (rows `{name, n, chi2_per_point}`; Task 7 adds individual columns), `heterogeneity` (`{min, max, ratio, flagged}`), `qrange` (`{qmax: list, spread, flagged, mode: 'native windows'}`), `shared_warnings` (the `jf.warnings` entries starting with `"shared "`), `degeneracy` (`{pairs: [(p, q, rho)], softest: [(name, coef)]}` or `{'unavailable': 'no parameter covariance (the Hessian failed)'}`), `information_criteria`, `rescale` (or `None`), `fisher`, `tension`, `optimum_distance` (Task 8 fills all three; `None` here), `unavailable: list[tuple[str, str]]`. `objective` also carries `nll2` and, in `model` mode, `fit_gap`.
  - Each section computed in its own `try`; an exception appends `(section, '<ExcType>: <msg>')` to `unavailable`, never propagates.
  - Degeneracy: correlation from `snap.covar` over `snap.var_names`; pairs with `|ρ| > rho_tol`; softest direction = eigenvector of the correlation matrix with the largest eigenvalue, coefficients scaled so the largest |coef| is 1, terms with |coef| < 0.1 omitted, written `"d_HGi + 0.8·d_wat"` (coefficients are in stderr units because the correlation matrix is the stderr-normalised covariance).
  - `format_diagnostics(diag) -> str`, starting `"Joint-fit diagnostics:"`, one line per unavailable section: `"  diagnostics unavailable (<section>): <reason>"`.
  - `_report()` appends `'\n\n' + format_diagnostics(diagnostics(self, self.last_snapshot))` at the very end, inside `try`; if `last_snapshot` is `None` or this raises, it appends `"Joint-fit diagnostics:\n  diagnostics unavailable: <reason>"`. The first line (`DEGRADED FIT:` or lmfit's) is unchanged.

- [ ] **Step 1: Tests**

```python
D1  the report of a 3-dataset fit ends with a 'Joint-fit diagnostics:' block containing the
    objective label and one quality row per dataset; everything before it is byte-identical
    to the report text produced with diagnostics monkeypatched to return {} and
    format_diagnostics to return '' (strip the block before comparing)
D2  heterogeneity: true noise E, E, 4E -> flagged with 'error bars or model adequacy differ
    between datasets'; E, E, E -> not flagged
D3  q-range: windows 0.1:2.0 and 0.1:1.6 -> spread 0.25 flagged with 'native windows';
    0.1:2.0 twice -> not flagged
D4  shared_warnings: endpoints of shared 'a' starting at different values -> the
    'shared a: endpoints start at different values' text is restated in the block
D5  degeneracy: model y = (a + c)·x + b with a and c both free (exactly degenerate pair) ->
    pair (a, c) with |rho| > 0.9 and softest direction naming a and c with coefficients of
    the same sign ('a + 1·c' or equivalent)
D6  result.covar forced to None: degeneracy shows the 'no parameter covariance' line;
    information criteria still printed
D7  a section forced to raise (monkeypatch the degeneracy helper): result.params, chisqr
    and the report text before the block unchanged; the 'diagnostics unavailable
    (degeneracy)' line present; emission count and Niter unchanged across _report()
D8  snapshot forced to fail (T2's P11 path): the report still returns, with the
    'diagnostics unavailable' line
```

- [ ] **Step 2–4:** run (fail), implement, run D plus the whole `tests/run_all.py --quick`. Any harness comparing full report text: update it deliberately so it ignores the trailing diagnostics block, and list each such change in the commit message.

- [ ] **Step 5: Commit** — `"WS2 T6: joint-fit diagnostics block (quality, heterogeneity, q-range, degeneracy)"`

---

### Task 7: Joint vs individual fits (§6.1)

**Files:** Modify `joint_fit_stats.py`; Test `validate_joint_diag.py` (sections I)

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 6.
- Produces:
  - `individual_terms(jf, snap, individual_fits: dict[str, object]) -> list[dict]` — per joint dataset `{name, joint_m, indiv_m, joint_nll2, indiv_nll2, delta, flagged, unavailable: str | None}`, with `delta = joint_nll2 − indiv_nll2` (marginal, −2 ln L units; equal to `joint_m − indiv_m` when C_ii does not depend on the model). The individual `Fit` supplies `fit.func` and `fit.result.params`; its fit scale is `fit.fitter.userargs[0]` when present. Unavailable with a reason when: no entry, `fit.result is None`, scale unknown, or scale ≠ `snap.fit_scale`.
  - The individual model is evaluated once at the joint dataset's active x on its used rows, with its params from `fit.result.params` written in and the model's `x`, params and scalar attributes restored afterwards (same save/restore as `JointFit.quiet`; factor it into a shared helper `_saved_model_state(func, keys)` in `joint_fit_stats.py`). The individual `Fit` object is never mutated.
  - Each side is scored under its own `C_ii` (spec §6.1): the joint side under the snapshot's exact marginal block; the individual side, in `model` mode with `cc.needs_intensity`, under `cc.assemble([Block(d.cov, y, yerr, inten)], scale)` with `inten` the individual model on used rows and the data substituted where it is non-finite (or ≤ 0 in Log), the rule of `_build_chol` lines 667–684. Otherwise both sides use the snapshot's block. Each `nll2 = r C_ii⁻¹ r + ln det C_ii + n ln 2π`.
  - `flagged = delta > median(delta) + 3·1.4826·MAD(delta) and delta > 2` over the available rows.
  - `diagnostics(..., individual_fits=...)` adds `indiv_chi2_per_point`, `delta`, `flagged` to each `quality` row.
  - `format_joint_vs_individuals(jf, individual_fits, snap=None) -> str` — the full table plus a calibrated comparison over `d_i = ½ delta_i` with the §5 machinery (small-N rule included), labelled `"joint vs individual fits (marginal terms)"`.

- [ ] **Step 1: Tests**

```python
I1  3 datasets with a common true a: individual Fits (Fit_Routines.Fit on Line, fitted
    with the same scale and no cov) -> delta_i >= -1e-8 for all, none flagged
I2  D2 generated with a = 4.0 instead of 3.0, a shared: D2 flagged; the others not
I3  an individual fitted in Log while the joint is Linear -> that row unavailable with
    'fit scale' in the reason; a missing entry -> 'no individual fit'
I4  side effects: the individual Fit's func.x, params and result unchanged; jf emission
    count and Niter unchanged
I5  private covariance on D0: indiv_m for D0 equals r_indiv @ solve(C_00, r_indiv) with
    C_00 from the joint's blocks (1e-10)
I6  format_joint_vs_individuals with 3 datasets: no z, the small-N sentence present
I7  model mode, Linear, private multiplicative cov on D0, individual θ different from the
    joint: indiv_nll2 equals the hand value with C_00 assembled from the individual f
    (1e-10), and that C_00 differs from the joint's block; delta == joint_nll2 − indiv_nll2
```

- [ ] **Steps 2–5:** run (fail), implement, run, commit `"WS2 T7: joint vs individual fits on one covariance"`.

---

### Task 8: Fisher shares, model–reduction tension, distance to the exact optimum (§6.2, §6.4.5, §6.4.6)

**Files:** Modify `joint_fit_stats.py`; Test `validate_joint_diag.py` (sections F)

**Interfaces:**
- Consumes: Tasks 1 (`model_blocks`, `quiet`), 2 (`raw_parts`, snapshot), 6.
- Produces:
  - `fisher_shares(jf, snap, step=None) -> dict` — `{'params': [shared free masters], 'shares': {name: {param: float}}, 'coupled': bool}`. A shared master is one whose `param_map` entry spans more than one dataset. Forward-difference Jacobian of each dataset's unwhitened `r_i` (used rows) w.r.t. every free master, inside `jf.quiet()`, one evaluation per free master; default `step = 1e-6·max(|v|, stderr or 1, 1e-8)` [builder may refine; test F1 pins accuracy]. `I_i = J_iᵀ C_ii⁻¹ J_i` with `C_ii` the snapshot's marginal block (off the covariance path `diag((yerr·s_i)²)` in Linear and `diag((yerr·s_i/y)²)` in Log, with `s_i = 1` unless rescaled); `share_i[p] = I_i[p,p] / Σ_j I_j[p,p]`. When coupling ≠ 0, `coupled=True` and the formatter adds `"the joint information is smaller than the sum: shared errors are counted once"`. In `model` mode with `cc.needs_intensity` the result carries `'mean_information_only': True` and the formatter adds the footnote `"mean information only; the information the covariance carries about the parameters is omitted"` (spec §6.2). After the call every model and `jf.result` are as before.
  - `tension(snap) -> list[dict]` — per dataset and multiplicative term (private or shared): `eps = Σ_t Gᵀ C⁻¹ r` over the whole fit (C the snapshot's full C, G the term's loading from `cc` with the stored intensity, scaled by `S⁻¹` when rescaled), `pull = eps_k / √Σ_kk` (Σ_kk = 0 components, e.g. the anchor, are omitted). Rows `{dataset, term, label, eps, pull}`. Formatted under the heading `"model–reduction tension"`, never "stitching quality", never as a fitted scale factor.
  - `exact_objective(jf, snap, params) -> float` — Φ(θ) = −2 ln L(θ) + Σ prior_rows(θ)² at `params`: `raw_parts` for r, `jf.model_blocks(params, scale)` for C(θ), `cc.factorize`, `2 Σ ln diag L`, all with the snapshot's rescale factors applied as in Task 5. Quiet, with no state left changed. At `snap`'s parameters it equals `extras['nll2'] + extras['prior']` (1e-10).
  - `optimum_distance(jf, snap, step=None) -> dict | None` — `None` unless `snap.cov_mode == 'model'` and `cc.needs_intensity`. Central-difference gradient g of `exact_objective` over `snap.var_names` (default step as in `fisher_shares`); `δ = −½ · snap.covar @ g`; returns `{'delta': {name: δ_p}, 'in_stderr': {name: δ_p/stderr_p}, 'worst': (name, value), 'fit_gap': extras['fit_gap']}`. With `snap.covar is None` it returns `{'unavailable': 'no parameter covariance (the Hessian failed)'}`. Formatted as `"the fitted point is the reweighting fixed point; the exact-likelihood optimum is estimated <worst value> stderr away (<name>); fit gap Δχ² = <fit_gap>"`, plus `"; the WS6 exact optimiser should settle it"` when `|worst| > 0.3`.
  - `diagnostics` fills `fisher`, `tension` and `optimum_distance` (the first two skipped when their flag is False; tension `None` when no multiplicative term exists; `optimum_distance` costs `2·nvarys` model evaluations plus as many factorisations, so it gets its own flag `optimum=True`).

- [ ] **Step 1: Tests**

```python
F1  linear joint model (shared a, per-dataset b), private covariance: per dataset
    I_i from fisher equals the analytic X_iᵀ C_ii⁻¹ X_i (design columns x, 1) for the
    shared 'a' within 1e-5 relative
F2  block-diagonal: shares for 'a' sum to 1 (1e-12); shared covariance: coupled True and
    the sentence present
F3  one-dataset joint fit: no shared master -> 'params' empty, no exception
F4  tension on a synthetic set whose scan-2 data were scaled by exp(0.03) with Σ_22 = 4e-4:
    the scan-2 pull is within 0.5 of 0.03/0.02 = 1.5; text contains
    'model–reduction tension' and not 'stitching quality'
F5  no side effects: emission count, Niter, model x and params, result unchanged (also
    across exact_objective and optimum_distance)
F6  exact_objective at the snapshot's params == nll2 + prior (1e-10); model mode, Linear,
    private multiplicative cov with Σ large enough that C moves visibly (e.g. 0.05²):
    optimum_distance's δ agrees in sign and within 20 % with θ* − θ̂ from
    scipy.optimize.minimize(lambda v: exact_objective(jf, snap, params_from(v)), θ̂,
    method='Nelder-Mead', xatol 1e-10) for every free parameter whose |θ* − θ̂| >
    0.05 stderr
F7  data mode and no-covariance fits: optimum_distance is None and no line is printed;
    model mode with covar None: the 'unavailable' dict; model mode: the Fisher footnote
    'mean information only' is present, and absent in data mode
```

- [ ] **Steps 2–5:** run (fail), implement, run, commit `"WS2 T8: Fisher shares, model-reduction tension, distance to the exact optimum"`.

---

### Task 9: Leave-one-out (§7)

**Files:** Modify `joint_fit_stats.py`; Test `validate_joint_diag.py` (sections L)

**Interfaces:**
- Consumes: Tasks 1, 2.
- Produces:
  - `@dataclass LOOResult`: `omitted: list[str]`, `estimates: dict[str, list[float | None]]` (per shared master, one per omitted dataset; `None` where the master has no endpoint left), `spread: dict[str, float]` (SD over the available estimates), `influence: dict[str, list[float | None]]` (shift / full-fit stderr; raw shift with `units='raw'` when stderr is `None`), `units: dict[str, str]`, `failures: list[tuple[str, str]]`, `notes: list[str]`.
  - `leave_one_out(jf, workers=1, fit_kws=None) -> LOOResult` — N refits, each on a fresh `JointFit` that carries the same models (deep-copied), data, yerr, masks, covariance objects, windows (`ranges` from the original objectives' `imin/imax` → x values), priors restricted to remaining masters, share groups and `covariance_mode`, `rescale_errors`; warm-started from `jf.result.params`; fit with `jf._fit_scale` and `jf._fit_method` (emcee → least squares with a note). Omitting a dataset drops its rows of every covariance by construction (its block is not added), which is exact marginalisation. A refit that raises is a `failures` row, not an exception. `workers > 1` may use a process pool; `workers=1` is the tested path. `jf` is never mutated (deep-copy the models before anything touches them).
  - N < 2 → `ValueError("leave-one-out needs at least 2 datasets")`.
  - `format_loo(res) -> str`.
  - Not called from `perform_fit` or `_report`.

- [ ] **Step 1: Tests**

```python
L1  linear model, shared a, per-dataset b, no covariance, 4 datasets: each LOO estimate of
    'a' equals the closed-form weighted least-squares solution on the remaining datasets
    (solve the normal equations directly) within 1e-6 relative
L2  same with a private covariance on every dataset: closed-form GLS on the remaining
    blocks within 1e-6
L3  jf unchanged: result.params values, chisqr, Niter, emission count, every model's x
    and parameters, last_snapshot identity
L4  result.covar None on jf: influence in raw units, units['a'] == 'raw', note present
L5  one dataset: ValueError with 'at least 2'
L6  a refit forced to raise (monkeypatch Minimizer.minimize for the 2nd refit): one
    failures row, the other estimates present
```

- [ ] **Steps 2–5:** run (fail), implement, run, commit `"WS2 T9: leave-one-out refits on copies"`.

---

### Task 10: Docs and the full suite

**Files:** Modify `handoff/14-roadmap.md` (WS2 status), `handoff/20-spec-ws2-reporting.md` (implementation record: commits, any rulings), `handoff/README.md` (row 21 status); `tests/run_all.py` registry only if a new harness needs `slow`.

- [ ] **Step 1:** Run `"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/run_all.py` (full, in the background; ~33 min). Expected: no FAIL or TIMEOUT beyond the known ones listed in `handoff/09-validation-and-env.md`.
- [ ] **Step 2:** Record in doc 20 a `## 13. Implementation record` section: commit range, the test ids per section, and every deviation from this plan with its reason. Mark WS2 done in the roadmap, and note the two pieces left to WS4 (GUI rescale control and `.jpar` pass-through in `xmodfit.py`; the comparison panel swap) and to WS3 (storing `s` and snapshots). Record the distance-to-optimum value on the bilayer bundle if a model-mode fit of it is run, since it decides whether the WS6 exact optimiser is needed (roadmap WS6).
- [ ] **Step 3: Commit** — `git add handoff/14-roadmap.md handoff/20-spec-ws2-reporting.md handoff/README.md`; `"WS2 T10: record the implementation"`.
