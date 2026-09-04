import os, sys, types, traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Stub xraydb (not installed in cmc-py39; only used for X-ray calcs, not construction)
xrd = types.ModuleType("xraydb")
class XrayDB:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        return lambda *a, **k: None
xrd.XrayDB = XrayDB
sys.modules.setdefault("xraydb", xrd)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./Functions"))

results = []
def check(label, fn):
    try:
        fn()
        results.append((label, True, ""))
        print("PASS:", label)
    except Exception as e:
        tb = traceback.format_exc()
        results.append((label, False, tb))
        print("FAIL:", label, "->", repr(e))

# 1. Resolved conflict modules
def imp_biphasic():
    import importlib
    m = importlib.import_module("Functions.XRR.XLayers_Biphasic")
    assert hasattr(m, "parratt_numba"), "parratt_numba not importable in Biphasic"
    m.XLayers_Biphasic  # class present
check("import Functions.XRR.XLayers_Biphasic", imp_biphasic)

def imp_pdb():
    import importlib
    m = importlib.import_module("Functions.XRR.XLayersPDB")
    assert hasattr(m, "parratt_numba")
    assert hasattr(m, "pdbread"), "pdbread cache import missing"
    m.XLayersPDB
check("import Functions.XRR.XLayersPDB", imp_pdb)

def imp_fitroutines():
    import importlib
    m = importlib.import_module("Fit_Routines")
    m.Fit
    assert hasattr(m.Fit, "set_emcee_minimizer")
    assert hasattr(m.Fit, "log_probability"), "origin's log_probability missing"
check("import Fit_Routines (Fit + merged MCMC methods)", imp_fitroutines)

def imp_xmodfit():
    import importlib
    m = importlib.import_module("xmodfit")
    assert hasattr(m, "XModFit")
    assert hasattr(m.XModFit, "fitErrorCallback")
    assert hasattr(m.XModFit, "perform_post_sampling_tasks")
check("import xmodfit (XModFit class)", imp_xmodfit)

# 2. Construct XModFit offscreen
def construct():
    from PyQt5.QtWidgets import QApplication
    import xmodfit
    app = QApplication.instance() or QApplication(sys.argv)
    w = xmodfit.XModFit()
    assert w is not None
check("construct XModFit() offscreen", construct)

# 3. Numeric: shared parratt_numba runs on the resolved import path
def run_parratt():
    import numpy as np
    from Functions.XRR.XLayersPDB import parratt_numba
    q = np.linspace(0.01, 0.3, 50)
    d   = np.array([0.0, 20.0, 0.0])
    rho = np.array([0.0, 0.5, 0.333])
    beta= np.array([0.0, 0.0, 0.0])
    ref, refc = parratt_numba(q, 1.24, d, rho, beta)
    assert ref.shape == q.shape
    assert np.all(np.isfinite(ref))
    assert refc.dtype == complex or np.iscomplexobj(refc), "refc should be complex (bugfix intact)"
check("numeric parratt_numba(q,...) -> (ref, refc)", run_parratt)

print("\n===== SUMMARY =====")
nfail = 0
for label, ok, tb in results:
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        nfail += 1
        print(tb)
print("TOTAL: %d passed, %d failed" % (len(results)-nfail, nfail))
sys.exit(1 if nfail else 0)
