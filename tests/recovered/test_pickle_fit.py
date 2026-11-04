"""
Does the emcee objective (Minimizer._lnprob -> Fit.residual -> Fit QObject)
survive the pickling that multiprocessing.Pool.map performs at cores>1?
"""
import os, sys, pickle
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from importlib import import_module
XLayersPDB = import_module("Functions.XRR.XLayersPDB").XLayersPDB
from Fit_Routines import Fit

x = np.linspace(0.02, 0.55, 60)
fit = Fit(XLayersPDB, x, y=np.ones_like(x), yerr=np.ones_like(x))

for label, obj in [("Fit instance", fit),
                   ("Fit.residual (bound)", fit.residual),
                   ("Fit.func (XLayersPDB)", fit.func)]:
    try:
        pickle.dumps(obj)
        print("pickle OK   :", label)
    except Exception as e:
        print("pickle FAIL :", label, "->", type(e).__name__, str(e)[:120])

# Does dill help? (Note: stdlib multiprocessing.Pool does NOT use dill just because
# it's importable -- so even a dill 'OK' does not mean the Pool path works.)
try:
    import dill
    try:
        dill.dumps(fit.residual)
        print("dill  OK    : Fit.residual")
    except Exception as e:
        print("dill  FAIL  : Fit.residual ->", type(e).__name__, str(e)[:120])
except ImportError:
    print("dill not installed")

# Definitive: run a spawn Pool.map on the bound residual, exactly like emcee's
# compute_log_prob does with pool.map(self._lnprob, ...).
def _call(o):
    return "worker-ok"

if __name__ == "__main__":
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(2) as pool:
            out = pool.map(_call, [fit.residual, fit.residual])
        print("Pool.map(shipping Fit.residual): OK ->", out)
    except Exception as e:
        print("Pool.map(shipping Fit.residual): FAIL ->", type(e).__name__, str(e)[:200])
