"""
joint_covariance.py  --  covariance / sidecar helpers for XModJointFit
======================================================================

Pure-Python (numpy only, NO PyQt) so the numerical core can be unit-tested
head-less and reused by both the GUI (`Joint_Fit_Routines.JointFit`) and any
offline validation script.

Axis-2 (correlated-error) joint fitting needs, for every data point in every
dataset:

    * y_corr      -- the *corrected* intensity  (y_raw * c_k)
    * sigma_meas  -- the measurement-only sigma (err_raw * c_k)
    * k_index     -- which scale-factor segment (0..K-1) the point belongs to

plus one shared K x K log-space covariance of the correction factors,
``Sigma_logc`` (identical across datasets stitched with the same attenuator
calibration -- e.g. the three El_Cat_Si_1 measurements share a byte-identical
``*_c_factors.txt``).

The full data covariance over the *concatenation* of all datasets is then

    C_ij = delta_ij * sigma_meas_i^2  +  y_corr_i * y_corr_j * Sigma_logc[k_i, k_j]

This is the exact formula used by the single-dataset scale-factor-correction
patch (`scale-factor-correction/scripts/xmodfit_covariance_patch.py`); the only
new thing here is that i and j may live in *different* datasets, so the
off-diagonal ``Sigma_logc[k_i,k_j]`` terms become genuine cross-dataset blocks.

Sidecar formats supported
--------------------------
1. ``*_c_factors.txt`` (human-readable, produced by
   ``scale-factor-correction/scripts/correct_scale_factors.py``)::

       # correction factors and log-space covariance
       # SF   c   sigma(log c)
       <SF_0>  <c_0>  <sigmalogc_0>
       ...  (K rows) ...
       # Sigma_logc matrix
       <K x K rows>

   Combined with the raw 4-column data file (x, y_raw, err_raw, sf) this fully
   determines y_corr / sigma_meas / k_index.

2. ``*_corrected_cov.npz`` (the .npz sidecar) -- keys ``x, y_corr,
   sigma_meas, k_index, c, Sigma_logc, SF_LIST``. Loaded directly.
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  Low-level readers
# --------------------------------------------------------------------------- #
def load_rrf(path):
    """Load a whitespace/tab-separated 3- or 4-column reflectivity file.

    Returns dict with keys x, y, err and (if present) sf.  Handles CRLF and
    stray blank lines.  Comment lines (starting '#') are skipped.
    """
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"{path}: no numeric rows found")
    ncol = len(rows[0])
    if any(len(r) != ncol for r in rows):
        raise ValueError(f"{path}: ragged columns (expected {ncol} everywhere)")
    arr = np.array(rows, dtype=float)
    out = {"x": arr[:, 0], "y": arr[:, 1]}
    if ncol >= 3:
        out["err"] = arr[:, 2]
    if ncol >= 4:
        out["sf"] = arr[:, 3]
    return out


def load_c_factors(path):
    """Parse a ``*_c_factors.txt`` sidecar.

    Returns dict: SF_LIST (K,), c (K,), sigma_logc (K,), Sigma_logc (K,K).
    """
    with open(path, "r") as fh:
        raw = [ln.rstrip("\n").rstrip("\r") for ln in fh]

    sf, c, slc = [], [], []
    matrix_rows = []
    in_matrix = False
    for ln in raw:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "sigma_logc matrix" in s.lower() or "sigma matrix" in s.lower():
                in_matrix = True
            continue
        parts = s.replace(",", " ").split()
        vals = [float(p) for p in parts]
        if not in_matrix:
            # (SF, c, sigma(log c)) rows
            sf.append(vals[0])
            c.append(vals[1])
            slc.append(vals[2] if len(vals) > 2 else 0.0)
        else:
            matrix_rows.append(vals)

    SF_LIST = np.array(sf, dtype=float)
    c = np.array(c, dtype=float)
    sigma_logc = np.array(slc, dtype=float)
    K = len(SF_LIST)
    if matrix_rows:
        Sigma_logc = np.array(matrix_rows, dtype=float)
        if Sigma_logc.shape != (K, K):
            raise ValueError(
                f"{path}: Sigma_logc is {Sigma_logc.shape}, expected ({K},{K})"
            )
    else:
        # no explicit matrix -> diagonal from sigma(log c)
        Sigma_logc = np.diag(sigma_logc ** 2)
    return {
        "SF_LIST": SF_LIST,
        "c": c,
        "sigma_logc": sigma_logc,
        "Sigma_logc": Sigma_logc,
    }


def load_bundle_npz(path, datasets=None):
    """Load the multi-dataset ``xrr_corrected_bundle`` npz appendix.

    That bundle stores, for a whole back-to-back experiment corrected *globally*
    (one shared attenuator calibration + one shared beam), keys of the form
    ``<dataset>__<field>`` with fields ``x, q_nominal, R_corr, sigma_meas, v,
    k_index, sigma_marginal, c_fp, sf, scalars`` (``scalars = [A, qoff,
    var_fwhm, fwhm, fwhm_err]``) plus the shared globals ``Sigma_logc, c,
    SF_LIST``.

    Returns ``(metas, shared)`` where ``metas`` is an ordered dict
    ``{dataset_name: meta}`` -- each ``meta`` is exactly the per-dataset cov dict
    ``JointFit.add_dataset(cov=...)`` expects: ``x, y_corr (= R_corr),
    sigma_meas, k_index, v, var_fwhm, SF_LIST, Sigma_logc`` -- and ``shared`` is
    ``{'Sigma_logc', 'c', 'SF_LIST', 'var_fwhm'}`` for reference/validation.

    ``datasets`` optionally restricts/orders the returned datasets (e.g. one
    substrate's subset); default is every dataset present, in file order.
    """
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    all_names = []
    for k in keys:
        if "__" in k:
            nm = k.split("__", 1)[0]
            if nm and nm not in all_names:
                all_names.append(nm)
    if datasets is None:
        datasets = all_names
    else:
        missing = [d for d in datasets if d not in all_names]
        if missing:
            raise KeyError(f"{path}: datasets not in bundle: {missing}")

    Sigma_logc = np.asarray(z["Sigma_logc"], dtype=float)
    SF_LIST = np.asarray(z["SF_LIST"], dtype=float) if "SF_LIST" in z else None
    var_fwhm_seen = []
    metas = {}
    for nm in datasets:
        def g(field):
            key = f"{nm}__{field}"
            if key not in z:
                raise KeyError(f"{path}: missing '{key}'")
            return z[key]
        scalars = np.asarray(g("scalars"), dtype=float)
        var_fwhm = float(scalars[2])
        var_fwhm_seen.append(var_fwhm)
        metas[nm] = {
            "x": np.asarray(g("x"), dtype=float),
            "y_corr": np.asarray(g("R_corr"), dtype=float),
            "sigma_meas": np.asarray(g("sigma_meas"), dtype=float),
            "k_index": np.asarray(g("k_index")).astype(int),
            "v": np.asarray(g("v"), dtype=float),
            "var_fwhm": var_fwhm,
            "SF_LIST": SF_LIST,
            "Sigma_logc": Sigma_logc,
        }
    shared = {"Sigma_logc": Sigma_logc, "SF_LIST": SF_LIST,
              "c": np.asarray(z["c"], dtype=float) if "c" in z else None,
              "var_fwhm": float(np.mean(var_fwhm_seen)) if var_fwhm_seen else 0.0}
    if var_fwhm_seen and not np.allclose(var_fwhm_seen, var_fwhm_seen[0]):
        raise ValueError(
            "datasets carry different var_fwhm; the shared-beam footprint term "
            "assumes ONE beam FWHM across the whole corrected bundle.")
    return metas, shared


def load_cov_npz(path):
    """Load a ``*_corrected_cov.npz`` sidecar into the common cov dict."""
    z = np.load(path, allow_pickle=True)
    return {
        "x": np.asarray(z["x"], dtype=float),
        "y_corr": np.asarray(z["y_corr"], dtype=float),
        "sigma_meas": np.asarray(z["sigma_meas"], dtype=float),
        "k_index": np.asarray(z["k_index"]).astype(int),
        "SF_LIST": np.asarray(z["SF_LIST"], dtype=float),
        "Sigma_logc": np.asarray(z["Sigma_logc"], dtype=float),
        "c": np.asarray(z["c"], dtype=float) if "c" in z else None,
    }


# --------------------------------------------------------------------------- #
#  Per-dataset covariance metadata
# --------------------------------------------------------------------------- #
def segment_index(sf_col, SF_LIST, atol_rel=1e-6):
    """Map each row's scale-factor value to its index in SF_LIST (0..K-1).

    Matching is by nearest SF_LIST entry within a relative tolerance; raises
    if any point cannot be matched (guards against a mismatched sidecar).
    """
    sf_col = np.asarray(sf_col, dtype=float)
    SF_LIST = np.asarray(SF_LIST, dtype=float)
    k_index = np.empty(len(sf_col), dtype=int)
    for i, v in enumerate(sf_col):
        # closest in log space (scale factors span orders of magnitude)
        with np.errstate(divide="ignore"):
            d = np.abs(np.log(SF_LIST) - np.log(v)) if v > 0 else np.abs(SF_LIST - v)
        j = int(np.argmin(d))
        ref = SF_LIST[j]
        if ref == 0:
            ok = (v == 0)
        else:
            ok = abs(v - ref) <= atol_rel * abs(ref) + 1e-30
        if not ok:
            raise ValueError(
                f"scale factor {v!r} (row {i}) not found in SF_LIST={SF_LIST.tolist()}"
            )
        k_index[i] = j
    return k_index


def build_dataset_cov_meta(rrf, cfac):
    """Given raw data (dict from load_rrf, must have 'sf') and a c-factor dict
    (from load_c_factors), return the per-point covariance metadata:

        y_corr, sigma_meas, k_index   (all length N),
    together with the shared SF_LIST / Sigma_logc echoed back for convenience.

    y_corr     = y_raw * c[k]
    sigma_meas = err_raw * c[k]
    """
    if "sf" not in rrf:
        raise ValueError("raw data has no 4th (scale-factor) column; cannot "
                         "attach scale-factor covariance")
    SF_LIST = cfac["SF_LIST"]
    c = cfac["c"]
    k_index = segment_index(rrf["sf"], SF_LIST)
    ck = c[k_index]
    y_corr = rrf["y"] * ck
    err = rrf.get("err", np.ones_like(rrf["y"]))
    sigma_meas = err * ck
    return {
        "y_corr": y_corr,
        "sigma_meas": sigma_meas,
        "k_index": k_index,
        "SF_LIST": SF_LIST,
        "Sigma_logc": cfac["Sigma_logc"],
        "c": c,
    }


# --------------------------------------------------------------------------- #
#  Full concatenated covariance + Cholesky
# --------------------------------------------------------------------------- #
def load_dataset_cov(sidecar_path, x=None, raw_path=None, atol_x=1e-8):
    """Unified entry point: return per-dataset covariance meta (y_corr,
    sigma_meas, k_index, SF_LIST, Sigma_logc, c) from either sidecar flavour.

    * ``*.npz``  -- self-contained (correct_scale_factors.py output). Loaded
      directly; if ``x`` is given it is checked against the sidecar's x.
    * ``*_c_factors.txt`` -- needs the raw 4-column data (x, y, err, sf) to
      recover per-point y_corr / sigma_meas / k_index. Supply ``raw_path`` (the
      original stitched data file); its 4th column is required.
    """
    p = str(sidecar_path)
    if p.endswith(".npz"):
        z = load_cov_npz(p)
        if x is not None and (len(z["x"]) != len(x)
                              or not np.allclose(z["x"], x, atol=atol_x, rtol=1e-9)):
            raise ValueError("cov .npz x-axis does not match dataset x-axis")
        return {
            "y_corr": z["y_corr"], "sigma_meas": z["sigma_meas"],
            "k_index": z["k_index"], "SF_LIST": z["SF_LIST"],
            "Sigma_logc": z["Sigma_logc"], "c": z["c"],
        }
    # c_factors text route
    cfac = load_c_factors(p)
    if raw_path is None:
        raise ValueError(
            f"{p}: a *_c_factors.txt sidecar needs the raw 4-column data file "
            "(pass raw_path=...) to recover per-point y_corr/sigma_meas/k_index."
        )
    rrf = load_rrf(raw_path)
    if x is not None and (len(rrf["x"]) != len(x)
                          or not np.allclose(rrf["x"], x, atol=atol_x, rtol=1e-9)):
        raise ValueError("raw data x-axis does not match dataset x-axis")
    return build_dataset_cov_meta(rrf, cfac)


def build_full_covariance(y_corr, sigma_meas, k_index, Sigma_logc,
                          v=None, var_fwhm=0.0):
    """Assemble the full N x N data covariance over a concatenation of points.

        C_ij =  delta_ij * sigma_meas_i^2                       (measurement)
              + y_corr_i * y_corr_j * Sigma_logc[k_i, k_j]      (scale-factor)
              + var_fwhm * v_i * v_j                            (shared beam)

    The first two terms are the scale-factor-correction model.  The third,
    optional term is the **shared-beam footprint** contribution: a single beam
    FWHM (variance ``var_fwhm``) was used to footprint-correct every dataset, so
    its uncertainty couples ALL points that carry a footprint sensitivity ``v``
    -- within AND across datasets.  Because ``var_fwhm`` is one shared scalar and
    ``v`` is the concatenated per-point sensitivity, this term is exactly the
    rank-1 update ``var_fwhm * outer(v_all, v_all)`` (see the footprint-correction
    skill and the ``xrr_corrected_bundle`` covariance formula
    ``C = diag(sm^2) + var_fwhm*outer(v,v) + outer(R,R)*Sigma_logc[k,k]``).

    ``v`` / ``var_fwhm`` default to no footprint term (backward-compatible with
    the scale-factor-only assembly).  Per the footprint-correction discipline the
    footprint *amplitude* error is absorbed by a free per-dataset ``yscale`` and
    is deliberately NOT in this covariance; only the beam-shape (FWHM) error is.

    All inputs are already concatenated across datasets and sliced to the fit
    range, in the SAME order the residual vector will use.
    """
    y_corr = np.asarray(y_corr, dtype=float)
    sigma_meas = np.asarray(sigma_meas, dtype=float)
    k_index = np.asarray(k_index).astype(int)
    Sigma_logc = np.asarray(Sigma_logc, dtype=float)

    C = np.outer(y_corr, y_corr) * Sigma_logc[np.ix_(k_index, k_index)]
    if v is not None and var_fwhm:
        v = np.asarray(v, dtype=float)
        if v.shape != y_corr.shape:
            raise ValueError(
                f"footprint sensitivity v has shape {v.shape}, expected "
                f"{y_corr.shape} (concatenated in residual order)")
        C += float(var_fwhm) * np.outer(v, v)
    C[np.diag_indices_from(C)] += sigma_meas ** 2
    C = 0.5 * (C + C.T)  # symmetrize against round-off
    return C


def cholesky_with_jitter(C, base_jitter=1e-10, max_tries=6):
    """Cholesky factor of C, adding progressively larger diagonal jitter if
    the raw matrix is not numerically PD.  Returns (L, jitter_used)."""
    C = 0.5 * (C + C.T)
    mean_diag = float(np.mean(np.diag(C))) if C.size else 1.0
    jitter = 0.0
    for attempt in range(max_tries):
        try:
            L = np.linalg.cholesky(C + jitter * np.eye(C.shape[0]))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter = base_jitter * mean_diag * (10.0 ** attempt)
    # last resort: eigenvalue floor
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, 1e-12 * mean_diag, None)
    Cpd = (V * w) @ V.T
    L = np.linalg.cholesky(0.5 * (Cpd + Cpd.T))
    return L, jitter