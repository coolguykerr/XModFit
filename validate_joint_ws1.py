"""
validate_joint_ws1.py -- WS1 (joint-path correctness) regression checks.

One function per task section; later tasks append T2..T5.

    T1  one PDB model-name list (A1.1): Fit_Routines and xmodfit hold exactly
        lmfit.minimizer.PDB_FUNCNAMES

    QT_QPA_PLATFORM=offscreen python validate_joint_ws1.py
"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Functions'))

# the bundled xraydb/ directory carries no source; stub it before importing xmodfit
try:
    from xraydb import XrayDB as _R  # noqa
except Exception:
    _xr = types.ModuleType("xraydb")
    class _XrayDB:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB; _xr.f0 = lambda *a, **k: 0; _xr.__path__ = []
    sys.modules["xraydb"] = _xr

from PyQt5.QtWidgets import QApplication

from tests.harness import check, finish


def test_t1():
    import lmfit.minimizer as mn
    import Fit_Routines
    import xmodfit
    ref = getattr(mn, 'PDB_FUNCNAMES', None)
    check("T1 lmfit.minimizer.PDB_FUNCNAMES exists", ref is not None)
    check("T1 Fit_Routines.PDB_FUNCNAMES == minimizer set",
          ref is not None and getattr(Fit_Routines, 'PDB_FUNCNAMES', None) == ref)
    check("T1 xmodfit _PDB_MCMC_FUNCS == minimizer set",
          ref is not None and xmodfit.XModFit._PDB_MCMC_FUNCS == ref)
    check("T1 XLayersPDB_XPR in xmodfit _PDB_MCMC_FUNCS",
          'XLayersPDB_XPR' in xmodfit.XModFit._PDB_MCMC_FUNCS)
    check("T1 XLayersPDB_box in Fit_Routines set",
          'XLayersPDB_box' in getattr(Fit_Routines, 'PDB_FUNCNAMES', ()))
    check("T1 legacy alias _PDB_FUNCNAMES is the same object",
          getattr(mn, '_PDB_FUNCNAMES', None) is ref)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    for t in (test_t1,):
        try:
            t()
        except Exception as ex:
            import traceback; traceback.print_exc()
            check("%s runs" % t.__name__, False, repr(ex))
    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
