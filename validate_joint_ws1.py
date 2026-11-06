"""
validate_joint_ws1.py -- WS1 (joint-path correctness) regression checks.

One function per task section; later tasks append T2..T5.

    T1  one PDB model-name list (A1.1): Fit_Routines and xmodfit hold exactly
        lmfit.minimizer.PDB_FUNCNAMES
    T2  .npz sidecars carry the shared-beam footprint term (B2a)
    T3  share-group mismatches (01 6.8): differing start values warn once per
        group; differing min/max/vary raise ShareMismatchError unless
        on_mismatch='first'; JointFit.warnings does not grow across rebuilds
        or re-fits and is printed in the report

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


def test_t3():
    import numpy as np
    from validate_joint_gls import Line, X, make_data, E
    from Joint_Fit_Routines import JointFit, ShareMismatchError
    ys = make_data(3)
    err = E * np.ones_like(X)
    names = ('DS_one', 'DS_two', 'DS_three')

    def build(starts=(3.0, 3.0, 3.0), max2=np.inf, vary2=True, **kw):
        jf = JointFit()
        for i, (nm, y) in enumerate(zip(names, ys)):
            jf.add_dataset(nm, funcClass=Line, x=X, y=y, yerr=err)
            pa = jf.objectives[i]['func'].params['a']
            pa.set(value=starts[i], vary=(vary2 if i == 1 else True),
                   max=(max2 if i == 1 else np.inf))
            jf.objectives[i]['func'].params['b'].set(vary=True)
        grp = jf.share_within('a', list(names))
        jf.build_master_params(groups=[grp], **kw)
        return jf, grp

    # (a) same bounds, different values -> builds, exactly one warning
    jf, grp = build(starts=(2.5, 3.0, 3.5))
    w = list(jf.warnings)
    check("T3a value mismatch builds with exactly 1 warning", len(w) == 1, w)
    check("T3a warning names all three datasets",
          len(w) == 1 and all(nm in w[0] for nm in names), w)
    check("T3a warning text follows the brief",
          len(w) == 1 and w[0].startswith('shared ')
          and 'endpoints start at different values' in w[0]
          and w[0].endswith('using DS_one=2.5'), w)
    mk = list(jf.param_map)[0]
    check("T3a master takes the first endpoint's value",
          jf.master_params[mk].value == 2.5, jf.master_params[mk].value)

    # (e) a second build does not grow the list
    jf.build_master_params(groups=[grp])
    check("T3e rebuilding does not grow warnings", len(jf.warnings) == 1,
          jf.warnings)

    # (f) after perform_fit the report carries the warnings block; re-fit doesn't grow
    rep, msg = jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T3f report contains 'Joint-fit warnings:'",
          'Joint-fit warnings:' in rep and names[2] in rep, rep[-400:])
    rep2, _ = jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T3f re-fit keeps the build warning and does not grow",
          len(jf.warnings) == 1 and rep2.count('Joint-fit warnings:') == 1,
          jf.warnings)
    # run-time entries (perform_fit's own, via _add_run_warning) are replaced
    jf._add_run_warning('stale run-time note')
    jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T3f perform_fit drops its previous run-time warnings only",
          jf.warnings == [w for w in jf.warnings if 'stale' not in w]
          and len(jf.warnings) == 1, jf.warnings)
    jf.clear()
    check("T3 clear() resets warnings", jf.warnings == [], jf.warnings)

    # (b) dataset 2 has max=10, the others inf -> ShareMismatchError
    try:
        build(max2=10.0)
        check("T3b bound mismatch raises ShareMismatchError", False, 'no exception')
    except ShareMismatchError as ex:
        s = str(ex)
        check("T3b ShareMismatchError names 'a' and the dataset",
              "'a'" in s and 'DS_two' in s and isinstance(ex, ValueError), s)

    # (c) on_mismatch='first' -> builds, master max == inf, one warning
    try:
        jf, _ = build(max2=10.0, on_mismatch='first')
        mk = list(jf.param_map)[0]
        check("T3c on_mismatch='first' builds with master max == inf",
              jf.master_params[mk].max == np.inf, jf.master_params[mk].max)
        check("T3c on_mismatch='first' leaves exactly 1 warning",
              len(jf.warnings) == 1, jf.warnings)
    except Exception as ex:
        check("T3c on_mismatch='first' builds", False, repr(ex))

    # (d) vary differs -> ShareMismatchError
    try:
        build(vary2=False)
        check("T3d vary mismatch raises ShareMismatchError", False, 'no exception')
    except ShareMismatchError as ex:
        check("T3d vary mismatch raises ShareMismatchError", True, str(ex))

    # the legacy shared= path obeys the same rule
    try:
        jf = JointFit()
        for i, (nm, y) in enumerate(zip(names, ys)):
            jf.add_dataset(nm, funcClass=Line, x=X, y=y, yerr=err)
        jf.objectives[1]['func'].params['a'].set(min=0.0)
        jf.build_master_params(shared=['a'])
        check("T3 shared= path raises on a bound mismatch", False, 'no exception')
    except ShareMismatchError as ex:
        check("T3 shared= path raises on a bound mismatch", True, str(ex))

    # JointFitSpec.compile passes on_mismatch through (default 'raise')
    from joint_fit_spec import JointFitSpec
    data = {nm: {'c': {'x': X, 'y': y, 'yerr': err}} for nm, y in zip(names, ys)}

    def spec():
        sp = JointFitSpec(default_model_class=Line)
        for i, nm in enumerate(names):
            m = Line(x=X)
            m.params['a'].set(max=(10.0 if i == 1 else np.inf))
            sp.add_dataset(nm, model=m)
        sp.set_shared('a')
        return sp
    try:
        spec().compile(data)
        check("T3 compile() raises ShareMismatchError by default", False,
              'no exception')
    except ShareMismatchError as ex:
        check("T3 compile() raises ShareMismatchError by default", True, str(ex))
    try:
        jf = spec().compile(data, on_mismatch='first')
        check("T3 compile(on_mismatch='first') builds with 1 warning",
              len(jf.warnings) == 1, jf.warnings)
    except Exception as ex:
        check("T3 compile(on_mismatch='first') builds", False, repr(ex))

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    for t in (test_t1, test_t2, test_t3):
        try:
            t()
        except Exception as ex:
            import traceback; traceback.print_exc()
            check("%s runs" % t.__name__, False, repr(ex))
    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
