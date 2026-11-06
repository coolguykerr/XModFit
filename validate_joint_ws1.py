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


class _Line:
    def __init__(self, x=0, a=1.0, b=0.0, mpar={}):
        from lmfit import Parameters
        self.x = x
        self.a, self.b = a, b
        self.__mpar__ = mpar
        self.choices = {}
        self.filepaths = {}
        self.__fit__ = False
        self.output_params = {'scaler_parameters': {}}
        self.params = Parameters()
        self.params.add('a', value=a, vary=0, min=-1e9, max=1e9)
        self.params.add('b', value=b, vary=0, min=-1e9, max=1e9)

    def y(self):
        import numpy as np
        return self.a * np.asarray(self.x, float) + self.b


def test_t2():
    import numpy as np
    import joint_covariance as jcov
    from Joint_Fit_Routines import JointFit
    n = 40
    x = np.linspace(0.1, 2.0, n)
    y = 3.0 * x + 0.5
    e = 0.05 * np.ones(n)
    base = dict(x=x, y_corr=y, sigma_meas=e,
                k_index=(np.arange(n) >= n // 2).astype(int),
                SF_LIST=np.array([1.0, 10.0]),
                Sigma_logc=np.array([[0.01, 0.002], [0.002, 0.02]]))
    v = 0.3 * np.exp(-x / 0.5)
    d = os.path.join(ROOT, 'tests', '_logs')
    os.makedirs(d, exist_ok=True)
    pa, pb, pc = (os.path.join(d, 'ws1_t2_%s.npz' % s) for s in 'abc')
    np.savez(pa, **base)
    np.savez(pb, v=v, var_fwhm=4e-4, **base)
    np.savez(pc, v=v, **base)

    def build(path):
        cov = jcov.load_dataset_cov(path, x=x)
        jf = JointFit()
        jf.add_dataset('A', funcClass=_Line, x=x, y=y, yerr=e, cov=dict(cov))
        jf.add_dataset('B', funcClass=_Line, x=x, y=y, yerr=e, cov=dict(cov))
        jf.build_master_params()
        jf._set_ranges(x[0], x[-1])
        jf.residual(jf.master_params, 'Linear')
        return cov, jf

    cov, jf = build(pa)
    check("T2a no footprint keys -> no v / var_fwhm in cov",
          'v' not in cov and 'var_fwhm' not in cov)
    check("T2a _use_footprint is False", jf._use_footprint is False)

    cov, jf = build(pb)
    check("T2b v and var_fwhm passed through",
          'v' in cov and len(cov['v']) == n and cov['var_fwhm'] == 4e-4)
    check("T2b _use_footprint is True", jf._use_footprint is True)
    cat = lambda a: np.concatenate([a, a])
    C = jcov.build_full_covariance(cat(y), cat(e), cat(base['k_index']),
                                   base['Sigma_logc'], v=cat(v), var_fwhm=4e-4)
    LL = jf._L @ jf._L.T
    rel = np.max(np.abs(LL - C) / np.maximum(np.abs(C), 1e-300))
    check("T2b L L^T == build_full_covariance(v, var_fwhm) to 1e-12",
          LL.shape == C.shape and rel < 1e-12, rel)

    try:
        jcov.load_dataset_cov(pc, x=x)
        check("T2c only v -> ValueError", False, 'no exception')
    except ValueError as ex:
        check("T2c only v -> ValueError naming the file",
              pc in str(ex) and "'v' and 'var_fwhm'" in str(ex), str(ex))


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    for t in (test_t1, test_t2):
        try:
            t()
        except Exception as ex:
            import traceback; traceback.print_exc()
            check("%s runs" % t.__name__, False, repr(ex))
    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
