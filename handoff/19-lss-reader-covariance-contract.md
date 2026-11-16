# Handoff: what LSS_Reader must write so XModFit can fit with the stitching covariance

**Written 2026-11-11, for the agent that will rework `chemmatcars/LSS_Reader`.** That
agent's work is broader than this. This document is the one piece of it that XModFit
depends on: the **output contract** (§2), the arithmetic that fills it (§3), a map of the
current code with the traps in it (§4), and the acceptance tests (§5). Everything else in
LSS_Reader is that agent's to redesign.

- The XModFit side is finished and tested (`XModJointFit` `develop`, `b55bf41..c884afa`).
  XModFit reads this file today. Nothing in LSS_Reader writes it yet.
- The rationale (why a covariance, why these estimators, the statistics) is in
  `17-spec-general-covariance.md` §4.3, §7.1 and §8. Where this document and spec 17 §8
  differ, **this one wins**: it was written later and checked against LSS_Reader's code.
- Line numbers below are for LSS_Reader `mainwindow.py` at `a861d79` (5412 lines). They
  will drift as soon as the rework starts, so the function names are the stable reference.

## 1. What problem this solves, in one paragraph

LSS_Reader joins Pilatus reflectivity scans one after another. Each new scan is scaled by
a factor `fac_m` fitted in its overlap with the data merged so far. Today the factor's
uncertainty `facerr_m` is folded into that scan's per-point error bars, as if it were
independent from point to point. It isn't. Every point of scan m shares the same error
in `fac_m`, and through the chaining it shares the errors of every earlier factor too. A
structural fit that treats them as independent is overconfident: in XModFit's synthetic
test, a roughness pull has SD 5.3 instead of 1. XModFit now fits with the correct
covariance if it is told three things per dataset: the counting-only error of every point
(`sigma_meas`), which scan(s) each point came from (`weights`), and the covariance of the
per-scan log scale errors (`Sigma`). This document specifies that file.

## 2. The contract (fixed: XModFit enforces it)

### 2.1 Files

- Next to every `<name>_ref.txt`, write `<name>_ref.cov.json`.
- Next to every `<name>_rrf.txt`, write `<name>_rrf.cov.json`.
- Each sidecar describes **exactly the arrays in its data file**, after every transform
  (normalisation, sorting, q offset, Fresnel division), point for point, in the same order.
- The two sidecars of one reduction share one stitching term (same `weights`, same `cov`).
  Only `x`, `y` and `sigma_meas` differ between them.
- `_ref.txt` / `_rrf.txt` keep their 3-column format, but their error column becomes the
  **corrected** total error and their y uses the corrected factors (§2.5, §3.4).
- Every reduction also writes **`<name>_ref_legacy.txt` / `<name>_rrf_legacy.txt`**,
  byte-identical to what today's code writes as `_ref.txt` / `_rrf.txt` (§2.5). They have
  no sidecar.

### 2.2 Schema (version 1)

```json
{
  "format": "xmodfit-covariance",
  "version": 1,
  "producer": {"name": "LSS_Reader", "version": "<git describe, or any string>"},
  "data_file": "sample_rrf.txt",
  "x": [0.019412345678, 0.020412345678],
  "y": [0.98123456789, 0.97123456789],
  "sigma_meas": [0.0123456789, 0.0113456789],
  "terms": [
    {
      "name": "stitch",
      "scope": "dataset",
      "kind": "multiplicative",
      "labels": ["S12", "S13", "S14"],
      "weights": [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]],
      "cov": [[0, 0, 0], [0, 4e-4, 4e-4], [0, 4e-4, 9e-4]],
      "provenance": {"fac": [1.0, 0.0213, 0.0118], "facerr": [0.0, 4.3e-4, 2.6e-4],
                     "facerr_counting": [0.0, 3.9e-4, 2.6e-4], "chi2_red": [null, 1.21, 0.64],
                     "a": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                     "fac_legacy": [1.0, 0.0212, 0.0118], "facerr_legacy": [0.0, 2.1e-4, 1.9e-4]}
    }
  ]
}
```

| field | required | meaning |
|---|---|---|
| `format` | yes | the literal `"xmodfit-covariance"` |
| `version` | yes | the integer `1`. XModFit refuses any other value |
| `producer` | no | informational |
| `data_file` | no, but write it | base name of the paired data file. A mismatch with the file the user loads is a warning in XModFit, not a refusal |
| `x` | yes | N values: the q column of the data file (the q-offset-shifted q for `_rrf`) |
| `y` | yes for LSS_Reader (the loader tolerates absence) | N values: the intensity column of the data file. XModFit checks it, which catches a sidecar paired with the wrong file |
| `sigma_meas` | yes | N values, same units as `y`: the counting-only error (§3.1). Finite and > 0 at every point |
| `terms` | yes for a patched reduction | list of terms. One term, `stitch`, as below |

The `stitch` term:

| field | value |
|---|---|
| `name` | `"stitch"`. Term names must be unique within a file |
| `scope` | `"dataset"` (it is private to this reduction; never `"shared"`) |
| `kind` | `"multiplicative"`: the components are natural-log scale errors |
| `labels` | K strings, one per scan in patch order. Use `"S<scan number>"` |
| `weights` | N rows of K: the region weights of §3.3. Every row sums to 1 |
| `cov` | K×K: Σ of §3.4, in natural-log units squared. Symmetric, positive semi-definite. Row and column 0 (the anchor scan) are zero |
| `provenance` | optional and free-form. XModFit never uses it in the fit. Recommended keys as shown, so a reduction can be audited from the file alone |

### 2.3 What XModFit checks when the file is loaded (`covariance_core._v1_load`, `match_problems`)

A failed check refuses to attach the covariance and shows the user the message. So all of
these must hold:

- `x` finite. `sigma_meas` has length N and is finite and > 0 everywhere.
- `weights` is exactly N×K and finite. `labels`, if present, has K entries.
- `cov` is K×K, finite, symmetric to `|C − Cᵀ| ≤ 1e-12·max|C|`, and its smallest
  eigenvalue is ≥ `−1e-10 ×` its largest.
- No two terms share a name.
- `x` **and** `y` equal the dataset's columns point by point within a relative tolerance of
  `1e-4` (`atol = 0`). The data files carry 5 significant figures (`%.4e`, relative
  rounding ≤ 5e-5), so the sidecar must be built from the same arrays that are written,
  not from an earlier or later stage.
- Unknown top-level or term keys are ignored. That is the only safe way to extend the
  file without a version change (§2.6).

### 2.4 What XModFit does with it

- With the covariance attached, `sigma_meas` **replaces** the data file's error column in
  the weighted fit. The data file's own (total) error column is used only when no
  covariance is attached, which is why it must be the covariance's diagonal (§3.4): the
  best a fit that ignores the correlations can be given.
- The dataset covariance is `C = diag(sigma_meas²) + D·W·Σ·Wᵀ·D` in Linear scale, with
  D = diag(model intensity) by default, and `C_log = diag((sigma_meas/y)²) + W·Σ·Wᵀ` in
  Log scale. So `Σ` must be in natural-log units: Σ_mm ≈ (facerr_m / fac_m)² and
  accumulated (§3.4).
- XModFit's fit range and column masks select rows of `x`, `y`, `sigma_meas` and
  `weights`. A sidecar therefore never needs to know which q-range the user will fit.

### 2.5 Two outputs: corrected and legacy (decided by the user, 2026-11-11)

Each reduction is computed **twice from the same raw scans**, and both are written:

| files | factors from | error column | sidecar | purpose |
|---|---|---|---|---|
| `_ref.txt`, `_rrf.txt` (+ `.cov.json`, or the combined file) | the §3.2 fit | corrected: √ of the covariance diagonal (§3.4) | yes | the final, correct output |
| `_ref_legacy.txt`, `_rrf_legacy.txt` | today's fit (total errors, `absolute_sigma=False`) | today's, exactly (trap T6) | none | regression test, and the comparison study |

Why both:
- **Regression testing.** The legacy files must stay byte-identical to the pre-change
  code on the same inputs, through all of this agent's other LSS_Reader changes. That is
  the cheapest proof that a refactor did not change the reduction. The only permitted
  difference is the q² aliasing fix (§4, trap T8), which affects only runs saved with the
  Qz² box checked; it applies to both outputs.
- **The comparison study.** Once LSS_Reader is amended, the user will fit many datasets
  processed both ways to measure the impact of the corrected errors: legacy files without
  a covariance (the old method), corrected files without a covariance (diagonal only),
  and corrected files with the covariance. So the two outputs must come from the same
  scans, drops and settings, in one run.

Why today's column is not kept as the main output: it is wrong in two ways. Its `facerr`
comes from an estimator that can fall below counting noise (§3.2), and each scan's points
carry only their own factor's uncertainty, although a scan joined late inherits the
uncertainty of every earlier factor in the chain (§3.4). Its error bars are too small,
increasingly so at high q.

Rules:
- The two chains never share mutable state. Each starts from its own copy of the scans
  (trap T3), and the legacy chain is today's code path, kept unchanged.
- The legacy chain is a protected regression target: do not "fix" anything in it other
  than T8.
- Writing the legacy files may become a setting later; for now it is always on.
- The corrected and legacy `y` differ slightly (the two factor fits weight the 3–6
  overlap ratios differently). That is expected and must not be "fixed".
- If the rework wants to change the text format of the corrected files (more columns, a
  header, more digits), that is a separate decision (the legacy files never change): XModFit's loaders read these files and the
  sidecar's `y` check depends on the written precision. Raise it with the user first.

### 2.6 Extending the file

- Adding **optional keys** (top level, or inside a term or `provenance`) needs no version
  bump. XModFit ignores keys it does not know.
- Changing the meaning, units or shape of an existing key, or making a new key required,
  is version 2. XModFit refuses unknown versions, so it needs a matching loader first.
- **More error sources** can be added as further terms with distinct names, all
  `scope: "dataset"`, using the three kinds XModFit already supports:
  - `multiplicative`: a per-region log-scale error (e.g. an attenuator calibration per
    scan): `weights` N×K, `cov` K×K, natural-log units;
  - `additive`: an error that adds to the intensity in data units (e.g. a background
    level per scan): `weights` N×K, `cov` K×K, units of y²;
  - `dense`: a full N×N covariance in data units, no `weights`.

  The normalisation uncertainty is deliberately **not** a term in v1. It is a free
  overall scale, which a fit either fits or fixes deliberately.
- **A combined file (data and covariance in one) is allowed** (user, 2026-11-11), and
  this agent may choose to write it instead of, or as well as, the sidecar. To keep the
  XModFit side small it must be a **v1 file plus a `data` block**, so XModFit's existing
  covariance loader reads it unchanged (it ignores unknown keys) and the x/y match is
  exact by construction:

  ```json
  {
    "format": "xmodfit-covariance", "version": 1,
    "x": [...], "y": [...], "sigma_meas": [...], "terms": [...],
    "data": {
      "yerr": [...],
      "columns": {"Qz": "x", "R": "y", "R_err": "yerr"},
      "extra": {"<name>": [...]},
      "meta": {"Energy": 10.0, "Qc": 0.0217, "Qoff": 0.0, "kind": "rrf"}
    }
  }
  ```

  - `data.yerr` (N, required): the total error column, the one a fit without the
    covariance uses (§2.4). `x` and `y` are the top-level arrays, not repeated.
  - `data.columns` (optional): display names for x, y, yerr.
  - `data.extra` (optional): further N-length columns, shown as extra data columns.
  - `data.meta` (optional): scalars and strings, shown as the file's metadata.
  - Name it `<name>_ref.json` / `<name>_rrf.json`, one file per reduction output, at
    full precision (the `%.4e` rounding of T9 no longer applies to it).
  - Until XModFit has the loader (below), also write the text files and sidecars; users'
    other tools read the text files with `loadtxt`, so dropping them is the user's call.

  **XModFit side, not yet built** (a small follow-up: one loader branch and tests):
  `Data_Dialog.readData` reads a `.json` with a `data` block into its usual
  `{'meta', 'data'}` structure; `addData` accepts `*.json` and, when the file has
  `terms`, calls `attach_covariance_to(data_key, fname)` on the same file; saved `.par` /
  `.jpar` files store that one path for both. Nothing downstream of `DatasetCovariance`
  changes (spec 17 §7.3).

## 3. The arithmetic

Notation: scans in patch order 1..K (scan 1 is the anchor, the lowest q). The merged data
after join m has points indexed i. "Existing data" in join m is the merged data from
joins 1..m−1; "new scan" is scan m.

Everything in §3.1–3.4 is the **corrected chain**. The legacy chain (§2.5) is today's
code, unchanged: its factor fit, its scaling and its total-error propagation (the middle
column below).

### 3.1 Carry the counting-only error, `sigma_meas`

Start each scan with `sigma_meas =` its own error column (after background subtraction
and the identical-scan merge, which combine as today). Then:

| step | legacy total error (unchanged) | `sigma_meas` (corrected chain) |
|---|---|---|
| scale new scan by `fac` | `√(σ²·fac² + y_raw²·facerr²)` | `σ_meas · fac` (no `facerr` term) |
| averaged overlap point | `√(e₁² + e₂²) / 2` | `√(σ_meas,1² + σ_meas,2²) / 2` |
| identical-scan merge (`pilRefSSMerge`) | `√(Σσ²) / n` | the same formula on `sigma_meas` |
| normalisation by `norm` | `/ norm` | `/ norm` |
| Fresnel division (`_rrf`) | `/ R_F(q + qoff)` | `/ R_F(q + qoff)` |

The corrected chain carries only `sigma_meas`; its total error is built at the end from
the covariance (§3.4), not propagated step by step.

The new scan's error is interpolated at the existing points with the same interpolator
the legacy code uses for the total error (`interp1d(..., kind='cubic')`). A cubic interpolant of
an error column can overshoot below zero next to a sharp change. If `sigma_meas` at any
interpolated point is ≤ 0 or non-finite, use linear interpolation at that point and note
it in `provenance`. XModFit refuses a file with a non-positive `sigma_meas`.

### 3.2 Fit each factor against counting errors, inflate-only (decided by the user, 2026-11-10)

In the merge step, the factor fit is a one-parameter weighted mean of the ratios
`r_i = y_existing,i / spline_new(x_i)` over the existing points in the overlap. Today it
is `curve_fit(polynomial, x, r, p0=[1], sigma=r_err)` with `absolute_sigma=False`, and
`r_err` is built from the **total** errors. Change it to:

1. Build `r_err` from **`sigma_meas`** of both sides, the same formula as today
   (`r·√((σ₁/y₁)² + (σ₂/f)²)`).
2. Fit with `absolute_sigma=True`. That gives `fac` (unchanged estimator: the weighted
   mean) and `facerr_counting = (Σ 1/r_err²)^(-1/2)`.
3. `chi2_red = Σ ((r − fac)/r_err)² / (n − 1)` for n ≥ 2 overlap points; take 1 when n = 1.
4. `facerr = facerr_counting · max(1, √chi2_red)`.

Why this exact estimator (one line each; spec 17 §8.3 has the full argument):
- `absolute_sigma=False` alone can return a `facerr` far below counting noise when the 3–6
  overlap points happen to scatter little, which drives Σ towards zero.
- Total errors with `absolute_sigma=True` count the earlier joins' scale errors twice:
  once inside those errors, and again through the chaining of §3.4.
- Inflating when χ²_red > 1 keeps the conservative behaviour when the overlap genuinely
  disagrees. The estimator can never report less than counting noise.

In the corrected chain this fit's `fac` scales the new scan (`y·fac`, `sigma_meas·fac`)
and its `facerr` gives δ_m = facerr/fac. The legacy chain keeps today's fit for its own
`fac` and `facerr`. Record both chains' `fac` and `facerr` in `provenance`
(`fac_legacy`, `facerr_legacy`), so the comparison study can see how far they differ.
(Running two chains resolves a conflict inside spec 17, whose §8.2 keeps the total
column unchanged while its §8.3 changes the `facerr` that column is built from.)

### 3.3 Region weights, `W` (N×K)

Carry W alongside the merged data, row for row:

- Points of scan 1 get the row e₁. A point taken from scan m alone gets e_m.
- An averaged overlap point of join m gets `0.5·W[existing point] + 0.5·e_m`. (The spline
  of the new scan is built from scan m's points only, so its row is e_m.) A point already
  averaged in an earlier join can be averaged again: its row then has three non-zero
  entries, e.g. 0.25 / 0.25 / 0.5.
- New-scan points beyond the existing data are appended with row e_m.
- New-scan points inside the overlap are discarded, as today, and get no row.
- Points the user drops in the patch dialog drop their rows (§4, trap T4).
- Every row sums to 1. That is an acceptance test.

### 3.4 The scan covariance, `Σ` (K×K, natural-log units)

For each join m = 2..K, record:
- `fac_m` and `facerr_m` (the §3.2 value);
- `a_m` = the mean of the W rows of the existing-data points used in that join's factor
  fit (the `overlap1` set), taken **after** any patch-dialog drops.

Then, with δ_m = facerr_m / fac_m and Σ = 0 to start (scan 1 is the anchor), for
m = 2..K in order:

```
Sigma[m, j] = Sigma[j, m] = a_m · Sigma[:, j]      for j < m
Sigma[m, m]               = a_m · Sigma · a_m + delta_m**2
```

(0-based code: anchor index 0, loop m = 1..K−1, `a_m` a length-K vector that is zero
from index m on.) When each overlap lies only on the previous scan's own points,
`a_m = e_(m−1)` and this reduces to a random walk: `Σ[k, l] = Σ_{m=2..min(k,l)} δ_m²`.

**The corrected total error** (the error column of `_ref.txt` / `_rrf.txt`, and
`data.yerr` of a combined file) is the square root of the covariance's diagonal in data
units, using the measured y:

```
sigma_total_i = sqrt( sigma_meas_i**2 + y_i**2 * (W[i] @ Sigma @ W[i]) )
```

computed after normalisation, sorting and Fresnel division, from the same `y`,
`sigma_meas` and W rows that go in the sidecar. (Σ is in log units, so normalisation and
Fresnel division do not change it.) Testable: `sigma_total ≥ sigma_meas` everywhere, with
equality on scan 1's points outside any overlap.

A working reference implementation of §3.1–3.4 is `tests/lss_reference.py` in
XModJointFit (`reduce()` and `_sigma()`, 110 lines), for the corrected chain. It differs
from LSS_Reader in one way: it interpolates the new scan's error linearly (`np.interp`)
where LSS_Reader uses a cubic `interp1d`. It does not compute the corrected total error;
that is the one-line formula above.

Its spline of the data (`CubicSpline`, not-a-knot) matches `interp1d(kind='cubic')`.

### 3.5 Single scans and non-patch paths

- A single Pilatus scan (`pilRefPatchData`, one scan selected) or the Bruker
  point-by-point path: a sidecar is optional. If written, it has `sigma_meas` equal to the
  error column and `terms: []`. There is no factor, so the corrected and legacy outputs
  are identical; write the legacy files anyway, so every reduction has both.
- If any join fails or the dialog is abandoned, write no sidecar rather than a partial one.

## 4. The current code, and its traps

The Pilatus patch path is `pilRefPatchData` (3693) → identical-scan dialog
(`pilRefSameScan` 3847, `pilRefSSMerge` 3891) → `pilRefPatch` (3836) →
`pilRefPatchPlot` (3913, shows the `mtype='fac'` preview) ↔ `pilRefPatchDrop` (3983) →
`pilRefPatchNext` (4005, calls `oneDjoin(..., mtype='merge')` 4014) → on the last join,
`self.refData = list(self.pilrefdata)` (3916) → `updateRefPlotData` (4109,
normalisation, sort, Fresnel) → `saveRefData` (4200).

| # | where | trap |
|---|---|---|
| T1 | `oneDjoin` 4015–4020 | Truncates both inputs to their first **3 columns** on entry. A `sigma_meas` column added to the arrays is silently thrown away here. |
| T2 | `pilRefPatchData` 3739/3771 | Per-scan arrays in `self.scanqzframe` have **7 columns** `[qz, sig−bg, err, sig, sigerr, lbg, rbg]`. Column 3 is not free. Single-scan display relies on `len(data[0]) != 3` (4164). Carry `sigma_meas` and W in a parallel structure, or restructure deliberately. |
| T3 | `oneDjoin` 4057–4058 | Writes the scaled y and error **into `data2` in place**, which is `self.scanqzframe[...]` (the comment says so). Re-running a join on the same scan scales it twice. With two chains (§2.5) this is fatal: the second chain would start from scans the first already scaled. Each chain must work on its own copy of every scan, or the rework should stop mutating. |
| T4 | `pilRefPatchDrop` 3983 | Index semantics: non-negative indices delete from the **new scan**; negative indices delete from the existing data, which is scan 1's frame when `pilreforder == 0` (3993) and the merged `self.pilrefdata` otherwise (3995). Drops are sorted by \|index\| descending. W and `sigma_meas` rows must be deleted with the same indices. `a_m` is computed after the drops. |
| T5 | `oneDjoin` 4028–4030 | `overlap1 = data1x >= data2x[0]` (all existing points above the new scan's start) is the set used for the factor and for `a_m`. `overlap2 = data2x <= data1x[-1]` are the new scan's points that are discarded. The cubic `interp1d` raises if `data1x` runs past `data2x[-1]`, so the existing data never extends beyond the new scan. |
| T6 | `oneDjoin` 4056 | The total error uses `data2y` **after** it was scaled, divided by `fac²`, i.e. `y_raw²·facerr²`. Keep this expression exactly in the **legacy** chain, for byte identity (§2.5). The corrected chain does not use it (§3.4). |
| T7 | `updateRefPlotData` 4118, 4121, 4154 | Runs on every UI refresh, starting from a copy of `self.refData` (`np.array`). It **sorts** (`argsort` at 4121 when `refpatchindex == 0`, and again at 4154 for `normRefData`). Apply the same permutation to `sigma_meas` and W. Normalisation (4123–4129 max below qc; 4139–4143 user factor) divides y and errors, so divide `sigma_meas` too. Store the un-normalised `sigma_meas`/W with `refData` and recompute each refresh; do not normalise stored state in place. |
| T8 | `updateRefPlotData` 4155–4163 | **A display option corrupts the saved `_rrf.txt`.** With R/RF checked, the code adds `refQoff` to column 0 of the local `data`, divides columns 1–2 by R_F, and stores `self.RrfData = data`: the same array object, not a copy. With the **Qz²** box also checked, the next lines square `data[:,0]` in place for the plot's x axis (4161–4163), which squares `RrfData`'s q too. `saveRefData` then writes q² into the q column of `_rrf.txt`, with nothing in the file to show it. (`normRefData`, written to `_ref.txt`, is safe: `data[np.argsort(...)]` at 4155 is a copy.) Fix: `self.RrfData = data.copy()` before the Qz² transform, or square a separate plot array. Apply it to both outputs; it is the only permitted change to the legacy files. Note that `_rrf.txt`'s q column is `q + refQoff` by design and `_ref.txt`'s is not; the sidecar's `x` must follow each file. Also note `RrfData` is only refreshed while R/RF is checked, so a stale `RrfData` can be saved after the box is cleared (T10). |
| T9 | `saveRefData` 4214–4221 | The Pilatus branch writes `%.4e` with `np.savetxt`. Write the legacy files through exactly this code, and the corrected files with the same format. Build each sidecar from the very arrays passed to `savetxt` (after T7/T8), written with full precision (`json.dump` of Python floats, i.e. `repr`). The Bruker branch (4202–4213) writes `str()` floats and a fourth info column; it is not a patch path (§3.5). |
| T10 | state lifetime | `self.refData`, `self.normRefData` and `self.RrfData` persist across reductions. A sidecar must never be written from a previous reduction's W/Σ, and the legacy and corrected files of one save must come from the same reduction. Tie the covariance state to the reduction that produced `refData`, and at save time refuse to write a sidecar (and say so) if its N differs from the arrays being saved. |
| T11 | `pilRefPatchPlot` 3924, 3957 | A "no overlap" pair shows a warning and, on the first join, closes the dialog; on later joins it does not, and `Next` would still call the merge. Under §3.5, no sidecar in that case. |

## 5. Acceptance tests (run in LSS_Reader, headless, on synthetic scans)

Build K = 3–5 synthetic scans from a known curve (a Fresnel-like decay times a slab
form factor is enough) with known true factors, Poisson-like counting errors and 3–6
point overlaps. Include one case whose overlap reaches back past the previous scan's own
points into an earlier overlap, so `a_m` is not one-hot. Drive the reduction functions
directly; the Qt dialogs are not needed if the merge logic is factored out (recommended).

1. **Legacy byte identity.** `_ref_legacy.txt` and `_rrf_legacy.txt` are byte-identical
   to the pre-change code's `_ref.txt` / `_rrf.txt` on the same inputs and drops, with the
   Qz² box off. With it on, only the q² aliasing fix differs. Keep a set of pre-change
   outputs as fixtures (from the unmodified `a861d79` code, synthetic and a few real
   reductions) and run this test after every change to LSS_Reader, not only this one.
2. **`sigma_meas` and the corrected error.** `sigma_meas` is finite and > 0. The corrected
   error column equals `√(σ_meas² + y²·W_i Σ W_iᵀ)` recomputed from the sidecar, to the
   file's `%.4e` rounding (relative 1e-4). It is ≥ `sigma_meas` everywhere, equal on scan
   1's non-overlap points. (It is usually, but not always, larger than the legacy column
   on late scans; do not test that pointwise.)
3. **Weights.** Every row sums to 1. Overlap rows of a simple chain are exactly 0.5/0.5.
   N matches the data file.
4. **Σ.** Symmetric, PSD, row/column 0 zero. For a simple chain, `diag(Σ)` equals the
   cumulative `Σ δ²` and `Σ[k, l] = Σ[min(k,l), min(k,l)]`. Otherwise it equals the §3.4
   recursion computed independently from the recorded `a_m` and δ_m.
5. **Oracle.** Copy `tests/lss_reference.py` into the LSS_Reader tests and change its
   error interpolation to `interp1d(kind='cubic')`. On the same synthetic scans:
   - merged `x` and `weights` agree exactly;
   - the §3.2 values (`fac`, `facerr_counting`, `chi2_red`, `facerr`, δ_m, `a_m`),
     `Sigma`, `sigma_meas` and merged `y` of the corrected chain agree to 1e-10 relative.
6. **The sidecar loads in XModFit without a warning or refusal.** With the
   `cmc-py39` interpreter (`C:/Users/kerr/.conda/envs/cmc-py39/python.exe`) and the
   XModJointFit repo root on `sys.path`:

   ```python
   import numpy as np, covariance_core as cc
   cov = cc.load_covariance('sample_rrf.cov.json')
   d = np.loadtxt('sample_rrf.txt')
   assert not cov.match_problems(d[:, 0], d[:, 1])
   ```

   for both `_ref` and `_rrf`, including a run with drops, a run with the Qz² box on, and
   a run with the normalisation-factor box on.
7. **Statistical sanity (optional, recommended once).** Repeat the reduction over a few
   hundred noise seeds: the spread of `ln fac_m` about the truth matches `√Σ_mm`, and the
   inflation leaves Σ at or above the counting-only `Σ_abs` (XModJointFit's
   `validate_cov_stitching.py` measured the inflated Σ 6–17 % above the true scatter, by
   design).

## 6. Not this agent's concern

- XModFit's handling of the file: loading, masking, model/data mode, the GUI, `.xpar`
  persistence. All done.
- Whether the bilayer campaign's older covariance files (`.npz`, `_c_factors.txt`, the
  bundle) are regenerated through this path. Those keep loading through XModFit's legacy
  loaders.

## 7. Git and repository notes

- `origin` of `LSS_Reader` is the upstream `chemmatcars/LSS_Reader`. Nothing should be
  pushed without the user saying so.
- At `a861d79` the working tree has untracked output files (`_q2d_gid.txt`,
  `_qxy_gid.txt`, `_qz_gid.txt`, `__pycache__/`). They are not part of this work.
