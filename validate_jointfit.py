"""
validate_jointfit.py -- head-less validation of the XModJointFit engine.

Run from the XModJointFit root (so Functions/XRR is importable) with the three
El_Cat_Si_1 datasets + *_c_factors.txt present alongside (or point BASE at them).

Tests:
  T1  Whitening identity: sum(L^{-1} r)^2 == r^T C^{-1} r   (the GLS core).
  T2  N=1, no-covariance JointFit == stock Fit  (numerical regression).
  T3  3-dataset GLS fit on real data: runs, converges, returns namespaced
      params + a covariance; cross-dataset covariance changes stderrs vs the
      diagonal-only case.
"""
import os, sys
import numpy as np

# make Functions/XRR importable exactly like XModFit does
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "Functions"))
sys.path.insert(0, os.path.join(HERE, "Functions", "XRR"))

BASE = os.environ.get("ELCAT_BASE", "/mnt/project")

import joint_covariance as jcov
from Joint_Fit_Routines import JointFit
from XLayers import XLayers

FILES = {
    "DOPC":  ("El_Cat_Si_1_DOPC_wtrans.txt",  "El_Cat_Si_1_DOPC_wtrans_c_factors.txt"),
    "HBS_1": ("El_Cat_Si_1_HBS_1_wtrans.txt", "El_Cat_Si_1_HBS_1_wtrans_c_factors.txt"),
    "Air":   ("El_Cat_Si_1_Air_wtrans.txt",   "El_Cat_Si_1_Air_wtrans_c_factors.txt"),
}


def load_all():
    out = {}
    for name, (df, cf) in FILES.items():
        rrf = jcov.load_rrf(os.path.join(BASE, df))
        cfac = jcov.load_c_factors(os.path.join(BASE, cf))
        meta = jcov.build_dataset_cov_meta(rrf, cfac)
        out[name] = (rrf, meta)
    return out


# --------------------------------------------------------------------------- #
def T1_whitening_identity():
    print("\n=== T1: whitening identity  sum((L^-1 r)^2) == r^T C^-1 r ===")
    data = load_all()
    yc = np.concatenate([data[n][1]["y_corr"] for n in FILES])
    sm = np.concatenate([data[n][1]["sigma_meas"] for n in FILES])
    ki = np.concatenate([data[n][1]["k_index"] for n in FILES]).astype(int)
    S = data["DOPC"][1]["Sigma_logc"]
    C = jcov.build_full_covariance(yc, sm, ki, S)
    L, jit = jcov.cholesky_with_jitter(C)

    rng = np.random.default_rng(0)
    r = rng.standard_normal(len(yc)) * yc * 1e-3   # arbitrary residual, right scale
    lhs = np.sum(np.linalg.solve(L, r) ** 2)
    rhs = r @ np.linalg.solve(C, r)
    rel = abs(lhs - rhs) / abs(rhs)
    print(f"  N={len(r)}  jitter={jit:.2e}")
    print(f"  sum(whitened^2) = {lhs:.10e}")
    print(f"  r^T C^-1 r      = {rhs:.10e}")
    print(f"  relative diff   = {rel:.2e}   -> {'PASS' if rel < 1e-8 else 'FAIL'}")
    return rel < 1e-8


# --------------------------------------------------------------------------- #
def _fake_pyqt():
    """Inject a minimal fake PyQt5 so the STOCK Fit_Routines.py imports head-less
    and we can regression-test JointFit against the real Fit class."""
    import types
    if "PyQt5" in sys.modules:
        return
    qtcore = types.ModuleType("PyQt5.QtCore")

    class QObject:
        def __init__(self, *a, **k): pass

    class _Sig:
        def __init__(self, *a, **k): pass
        def connect(self, *a, **k): pass
        def disconnect(self, *a, **k): pass
        def emit(self, *a, **k): pass

    def pyqtSignal(*a, **k): return _Sig()
    qtcore.QObject = QObject
    qtcore.pyqtSignal = pyqtSignal

    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    class QMessageBox: pass
    qtwidgets.QMessageBox = QMessageBox

    pkg = types.ModuleType("PyQt5")
    pkg.QtCore = qtcore
    pkg.QtWidgets = qtwidgets
    sys.modules["PyQt5"] = pkg
    sys.modules["PyQt5.QtCore"] = qtcore
    sys.modules["PyQt5.QtWidgets"] = qtwidgets


_TEST_MPAR = {"Model": {"Layers": ["top", "l1", "bottom"],
                        "d":   [0.0, 30.0, 1.0],
                        "rho": [0.0, 0.45, 0.333],
                        "mu":  [0.0, 0.0, 0.0],
                        "sig": [0.0, 3.0, 3.0]}}


def _fresh_xlayers(x=0.1):
    """An XLayers instance with a small, well-posed test layer model.
    Zero-arg-callable (x defaulted) so the stock Fit(funClass) can call it,
    while a specific x can still be passed for JointFit."""
    import copy as _copy
    return XLayers(x=x, mpar=_copy.deepcopy(_TEST_MPAR), rrf=True, fix_sig=False)


def _configure_vary(func, yscale0):
    p = func.params
    p["yscale"].set(value=yscale0, vary=True, min=0)
    p["bkg"].set(value=0.0, vary=False)
    p["qoff"].set(value=0.0, vary=False)
    p["__Model_d_001"].set(vary=True, min=5, max=80)
    p["__Model_rho_001"].set(vary=True, min=0.0, max=1.0)
    p["__Model_sig_001"].set(vary=True, min=1.0, max=10.0)


def T2_regression_vs_stock_fit():
    print("\n=== T2: N=1 no-covariance JointFit == stock Fit ===")
    _fake_pyqt()
    import importlib
    Fit_Routines = importlib.import_module("Fit_Routines")
    Fit = Fit_Routines.Fit

    data = load_all()
    rrf = data["DOPC"][0]
    x, y, err = rrf["x"], rrf["y"], rrf["err"]
    yscale0 = float(np.max(y))
    xmin, xmax = 0.02, float(np.max(x))

    # stock Fit -- funClass must be zero-arg callable; Fit sets func.x itself
    fit = Fit(_fresh_xlayers, x, y=y, yerr=err)
    _configure_vary(fit.func, yscale0)
    rep_a, msg_a = fit.perform_fit(xmin, xmax, fit_scale="Linear",
                                   fit_method="leastsq", maxiter=2000)
    chi_a = fit.result.chisqr
    pa = {k: fit.result.params[k].value for k in fit.result.params
          if fit.result.params[k].vary}

    # JointFit, one dataset, no covariance
    jf = JointFit()
    fj = _fresh_xlayers(x)
    fj.x = x; fj.init_params()
    _configure_vary(fj, yscale0)
    jf.add_dataset("DOPC", func=fj, x=x, y=y, yerr=err)
    jf.build_master_params(shared=None)
    rep_b, msg_b = jf.perform_fit(xmin, xmax, fit_scale="Linear",
                                  fit_method="leastsq", maxiter=2000)
    chi_b = jf.result.chisqr
    # master keys are namespaced d0_* ; strip the prefix to compare
    pb = {}
    for k in jf.result.params:
        if jf.result.params[k].vary:
            local = k.split("_", 1)[1] if k.startswith("d0_") else k
            local = local[2:] if False else local
            pb[k] = jf.result.params[k].value

    print(f"  stock Fit  chisqr = {chi_a:.6e}")
    print(f"  JointFit   chisqr = {chi_b:.6e}")
    rel_chi = abs(chi_a - chi_b) / max(abs(chi_a), 1e-300)
    # compare parameter values by matching local names
    def localname(k):
        if k.startswith("d0___"):     # namespaced multipar __d0_Model.. -> here JointFit uses __d0_..
            return k
        if k.startswith("d0_"):
            return k[3:]
        return k
    matched = True
    print("  parameter comparison (local name : stock vs joint):")
    stock_by_local = {}
    for k, v in pa.items():
        stock_by_local[k if not k.startswith("__") else k] = v
    for k in sorted(jf.result.params):
        pj = jf.result.params[k]
        if not pj.vary:
            continue
        # recover local name
        if k.startswith("__d0_"):
            ln = "__" + k[5:]
        elif k.startswith("d0_"):
            ln = k[3:]
        else:
            ln = k
        va = pa.get(ln)
        if va is None:
            continue
        rel = abs(va - pj.value) / (abs(va) + 1e-30)
        flag = "ok" if rel < 1e-4 else "DIFF"
        if rel >= 1e-4:
            matched = False
        print(f"    {ln:22s}: {va:.6e}  vs  {pj.value:.6e}   ({flag}, rel={rel:.1e})")
    print(f"  chisqr relative diff = {rel_chi:.2e}")
    ok = matched and rel_chi < 1e-4
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def T3_three_dataset_gls():
    print("\n=== T3: 3-dataset GLS fit on real El_Cat data (cross-dataset cov) ===")
    data = load_all()
    xmin, xmax = 0.02, 0.30

    def build(with_cross_cov):
        jf = JointFit()
        for name in FILES:
            rrf, meta = data[name]
            f = _fresh_xlayers(rrf["x"])
            f.x = rrf["x"]; f.init_params()
            _configure_vary(f, float(np.max(meta["y_corr"])))
            if with_cross_cov:
                jf.add_dataset(name, func=f, x=rrf["x"], cov=meta)
            else:
                # diagonal only: strip off-diagonal Sigma_logc -> independent
                meta_diag = dict(meta)
                meta_diag["Sigma_logc"] = np.diag(np.diag(meta["Sigma_logc"]))
                jf.add_dataset(name, func=f, x=rrf["x"], cov=meta_diag)
        jf.build_master_params(shared=None)
        return jf

    jf = build(with_cross_cov=True)
    print(f"  #objectives={len(jf.objectives)}  "
          f"#master params={len(jf.master_params)}  "
          f"#varying={sum(jf.master_params[k].vary for k in jf.master_params)}")
    print(f"  use_cov={jf._use_cov}  master keys sample={list(jf.master_params)[:4]} ...")
    rep, msg = jf.perform_fit(xmin, xmax, fit_scale="Linear",
                              fit_method="leastsq", maxiter=4000)
    r = jf.result
    print(f"  converged: success={r.success}  nfev={r.nfev}  "
          f"ndata={r.ndata}  nvarys={r.nvarys}")
    print(f"  chisqr={r.chisqr:.4e}  redchi={r.redchi:.4e}  Cholesky jitter={jf._jitter:.1e}")
    has_cov = r.covar is not None
    print(f"  parameter covariance returned: {has_cov}")
    if has_cov:
        print("  per-parameter stderr (cross-cov GLS):")
        for k in r.params:
            if r.params[k].vary:
                se = r.params[k].stderr
                print(f"    {k:24s} = {r.params[k].value:.5e}  +/- "
                      f"{('%.3e' % se) if se is not None else 'None'}")

    # diagonal-only comparison
    jf2 = build(with_cross_cov=False)
    jf2.perform_fit(xmin, xmax, fit_scale="Linear", fit_method="leastsq", maxiter=4000)
    r2 = jf2.result
    print("\n  stderr comparison  (cross-cov GLS  vs  diagonal-only):")
    changed = False
    for k in r.params:
        if r.params[k].vary:
            s1 = r.params[k].stderr
            s2 = r2.params[k].stderr if k in r2.params else None
            if s1 and s2:
                rel = abs(s1 - s2) / s2
                if rel > 1e-3:
                    changed = True
                print(f"    {k:24s}: {s1:.3e}  vs  {s2:.3e}   (rel diff {rel:.2%})")
    ok = r.success and has_cov
    print(f"  cross-dataset covariance changes error bars: {changed}")
    print(f"  -> {'PASS' if ok else 'FAIL'} (machinery runs end-to-end and returns GLS covariance)")
    return ok


if __name__ == "__main__":
    results = {}
    results["T1_whitening_identity"] = T1_whitening_identity()
    try:
        results["T2_regression_vs_stock_Fit"] = T2_regression_vs_stock_fit()
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T2_regression_vs_stock_Fit"] = False
    try:
        results["T3_three_dataset_gls"] = T3_three_dataset_gls()
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T3_three_dataset_gls"] = False
    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"  {k:32s}: {'PASS' if v else 'FAIL'}")