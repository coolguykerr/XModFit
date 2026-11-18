# GIXOS model (from upstream) — XModJointFit change record

**Category:** upstream model brought in: `Functions/GIXOS/GIXOS.py`
**Status:** Working, validated headless (`validate_gixos.py`); never clicked through live, never fitted to real data here
**Owner chat(s):** side session started 2026-11-18 from the main XModJointFit session
**Last touched:** 2026-11-18

## 1. Why this exists

Upstream added a GIXOS (grazing-incidence X-ray off-specular scattering) model. It
computes the diffuse intensity along `q_z` at a fixed in-plane offset (`twotheta`) from a
liquid surface. The model is a Parratt layer stack (same `mpar` table as `XLayers`)
multiplied by capillary-wave terms (`eta` from `temp` and `tension`), a Vineyard
transmission factor in the exit angle `beta`, and `qxy^(eta-2)`. The output is
`yscale * (R/R_F) * T(beta) * eta/q_z^2 / qmax^eta * qxy^(eta-2) * cos(beta)`. It is the
XModFit way to get an EDP from GIXOS data. This fork brings it in so it can be fit
single, jointly with shared film parameters (for example the same film at several
tensions), and with a covariance attached.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `Functions/GIXOS/GIXOS.py` | 244 | new (upstream file + 1 line) | the model; a new `GIXOS` category holding only this file |
| `Functions/XFNTR/XFNTR_Plus.py` | +1 | modified (upstream file) | same `self.filepaths = {}` fix, see section 3 |
| `validate_gixos.py` | 249 | new | headless regression, auto-discovered by `tests/run_all.py` (no registry entry) |
| `handoff/22-gixos.md` | — | new | this document |

**Provenance:** upstream commit `681f84c` ("add GIXOS fitting function. change xraydb to
xraydb_bak", 2026-11-18). **It is on `origin/main`, not `origin/develop`.** `origin/develop`
is still at `81e53c8`, with nothing new since this fork's baseline. `681f84c` sits on
the `main` line (parent `2a1bbf5`), so `git log 81e53c8..origin/develop` shows nothing.

## 3. How it works

**Bring-in method: the single file was checked out from `origin/main`**
(`git checkout origin/main -- Functions/GIXOS/GIXOS.py`, then unstaged). Merging was not
an option. The commit is on `main`, which this fork has never tracked, and merging `main`
would pull in its whole history, including the XFNTR line. The rest of `681f84c` was left
out on purpose:

- `xraydb/ -> xraydb_bak/` rename: this tree already has `xraydb_bak/` (an earlier
  upstream rename, see doc 09 section 3.4), and the local `xraydb/` is a source-less
  namespace portion. The rename changes nothing here.
- `.idea/XModFit.iml`, `.idea/misc.xml`: IDE interpreter paths.

**The one change (`GIXOS.py:86`): `self.filepaths = {}`.** Selecting the upstream file
in the GUI raises `KeyError: 'filepaths'` in `update_fixed_parameters`
(`xmodfit.py:5901`). The `except` in `functionChanged` (`xmodfit.py:5831`) turns that
into a "Some syntax error in the function still exists" dialog. Under the offscreen
platform that modal dialog **segfaults** the process. Upstream's own `xmodfit.py`
(`origin/develop:xmodfit.py:3585`) indexes `params['filepaths']` without a guard as well,
so **the model is broken the same way in upstream's GUI**. It is the same defect as
`XFNTR_Plus.py` (fix-list B12), which got the same one-line fix in the same change at the
user's request. The fix follows the convention every other
model uses, and it was preferred over guarding `xmodfit.py`, a high-conflict file. It is
worth offering upstream.

Checked and **not** changed:

- **Category-folder rule:** no helper module. `parratt_numba` is defined inside the model
  file, so `Functions/GIXOS/` holds exactly one `.py`, and the function list shows
  exactly `['GIXOS']`. There is no `__init__.py`. Import works as a namespace package
  (`Functions.GIXOS.GIXOS`), as the validator's GUI step confirms.
- **`choices`:** `{'fix_sig': [True, False]}`. These eval to the bools the model wants,
  the same as `XLayers`.
- **Imports:** `lmfit`, `numpy`, `numba` (0.55.1 in cmc-py39), `math`, `cmath`,
  `functools`. No `xraydb`.
- **Joint fitting / pickling:** the model is model-agnostic for `JointFit`.
  `push_params` writes scalar attributes and Parameters. Instances pickle, including
  after a joint fit (`lru_cache` lives on the class, not the instance).
- **Per-column fit limits and covariance:** both live in `Fit`/`JointFit`; the model
  needs nothing from them.

## 4. How to use it

GUI: category **GIXOS**, function **GIXOS**. The layer table is the `XLayers` one: row 0
is the top phase (`d=0`) and the last row is the subphase. `fix_sig=True` ties every
interface roughness to `sig_001`. Fixed parameters: `E` (keV), `dz`, `alpha` (incident
angle, deg), `twotheta` (in-plane angle, deg), `temp` (deg C), `tension` (mN/m). Fit
parameters: `qoff`, `yscale`, `qmax`, plus the layer table. Fit in **Log** scale; the
data span 4+ decades.

Joint fit across tensions: set each dataset's `tension` as a fixed parameter, share the
film rows, and keep `yscale` per dataset. **`push_params` clears local exprs, so the
`fix_sig` tie does not survive into a joint fit.** Tie the roughnesses with one share
group instead, as `validate_gixos.py:build_joint` does with `sig_film`.

```
"C:/Users/kerr/.conda/envs/cmc-py39/python.exe" validate_gixos.py   # 36 checks, ~4 s (2026-11-18 quick suite: 60 PASS, 7 SKIP)
```

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `Functions/GIXOS/GIXOS.py` (upstream `main`) | `GIXOS.__init__` | `+ self.filepaths = {}` after `self.choices` | yes, one line; a later checkout or merge of upstream's file conflicts only if upstream edits the same line, and drops the fix silently if the file is re-checked-out |

| `Functions/XFNTR/XFNTR_Plus.py` (upstream `30507b7`) | `XFNTR_Plus.__init__` | `+ self.filepaths = {}` after `self.choices` | yes, one line |

No other upstream file was touched.

## 6. Known limitations, gotchas and TODOs

- **Zero below the horizon.** For `q_z <= k0 sin(alpha)` (0.0088 Å⁻¹ at 10 keV, 0.1°) the
  model returns exactly 0 (`beta <= 0`, transmission 0). A Log-scale fit on such points
  takes `log10(0)`. Exclude them with the x/column limits. Not tested what the residual
  does there.
- **First call is about 2 s** (numba JIT, `parallel=True, cache=True`). After that a call
  takes 0.5 ms. The JIT cache goes to `Functions/GIXOS/__pycache__/` (gitignored).
- **`parallel=True` numba inside spawn-Pool MCMC workers** has not been exercised.
  Upstream de-parallelised numba elsewhere for MCMC parallelism (memory
  `merge-with-upstream-develop`). Watch for oversubscription with `emcee_cores > 1`.
- `alpha`, `twotheta`, `temp`, `tension` are fixed parameters, not fit parameters, so
  they cannot be shared or fitted in a joint fit.
- `parratt_numba` here duplicates the one in `XRR/XLayers.py`. It returns `(ref, r)` with
  the last `r`, not the `refc` array; that is harmless because the caller ignores it.
- The `qoff` handling (applied to `x`, removed again for `R_F`, and fed into `del_alpha`)
  is upstream physics that nobody here has reviewed.
- No `Function_Details/GIXOS/` help, so the Details menu shows "help files do not exist".

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| G1 | Upstream's GUI shows the same `KeyError: 'filepaths'` dialog for GIXOS | code reading of `origin/develop:xmodfit.py:3585` | select GIXOS on a clean upstream checkout |
| G2 | The model's physics (Vineyard factor, `eta`, `qxy` exponent) matches the GIXOS literature | not checked | compare against the published GIXOS formalism the upstream author used |
| G3 | Parallel MCMC (`emcee_cores > 1`) works with the `parallel=True` numba kernel | not run | short 2-core emcee run on synthetic data |

## 8. Consolidation candidates

| Item | Recommendation | Risk |
|---|---|---|
| local `parratt_numba` vs `XRR.XLayers.parratt_numba` | keep as upstream ships it; importing the shared one would diverge from upstream | nil now |
| `self.filepaths = {}` | offer upstream for both `GIXOS.py` and `XFNTR_Plus.py` | nil |
