#!/usr/bin/env python3
"""
validate_gixos.py -- headless regression for Functions/GIXOS/GIXOS.py, the upstream
GIXOS model (chemmatcars/XModFit main, 681f84c) brought in unmodified.

    QT_QPA_PLATFORM=offscreen python validate_gixos.py

Checks that the model
  1. is listed by the GUI (category GIXOS, function GIXOS) and builds a Fit on select,
  2. evaluates finite on its x range, zero only below the horizon (beta <= 0),
  3. pickles (spawn-Pool workers / parallel MCMC need it),
  4. recovers its parameters in a noise-free single-dataset Fit,
  5. ... with a per-column fit mask applied (constant residual length),
  6. works under JointFit with shared film parameters and a mask,
  7. works under JointFit with a covariance attached (GLS path).
The data are synthetic and noise-free, so recovery tolerances are tight.
"""
import functools
import os
import pickle
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
for p in (ROOT, os.path.join(ROOT, 'Functions'), os.path.join(ROOT, 'Functions', 'GIXOS')):
    if p not in sys.path:
        sys.path.insert(0, p)

# xraydb is not installed in cmc-py39 and the local xraydb/ has no source: stub it.
try:
    from xraydb import XrayDB as _R  # noqa
except Exception:
    _xr = types.ModuleType("xraydb")

    class _XrayDB:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB
    _xr.f0 = lambda *a, **k: 0
    _xr.__path__ = []
    sys.modules["xraydb"] = _xr

import numpy as np                                               # noqa: E402
from PyQt5.QtWidgets import QApplication                         # noqa: E402

from tests.harness import check, finish                          # noqa: E402
from GIXOS import GIXOS                                          # noqa: E402

# top phase (air) / film / water subphase; fix_sig ties sig_002 to sig_001
MPAR = {'Model': {'Layers': ['top', 'film', 'bottom'], 'd': [0.0, 15.0, 1.0],
                  'rho': [0.0, 0.45, 0.334], 'mu': [0.0, 0.0, 0.0],
                  'sig': [0.0, 3.0, 3.0]}}
TRUE = {'yscale': 1.0, '__Model_d_001': 15.0, '__Model_rho_001': 0.45,
        '__Model_sig_001': 3.0}
START = {'yscale': 1.3, '__Model_d_001': 13.0, '__Model_rho_001': 0.42,
         '__Model_sig_001': 3.4}
X = np.linspace(0.02, 0.8, 160)          # above the horizon (k0 sin(alpha) ~ 0.009)
Film = functools.partial(GIXOS, mpar=MPAR)


def synth(x, tension=50.0, yscale=1.0):
    f = GIXOS(x=x, mpar=MPAR, tension=tension, yscale=yscale)
    f.__fit__ = False
    f.init_params()
    return np.asarray(f.y(), float)


def rel(a, b):
    return abs(a - b) / abs(b)


# ------------------------------------------------------- 1. GUI listing -----
def test_gui():
    print('\n[1] GUI function list')
    import xmodfit
    # functionChanged swallows any exception into a modal QMessageBox, which
    # segfaults under the offscreen platform. Record the warnings instead.
    warned = []
    xmodfit.QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a[1:3]))
    w = xmodfit.XModFit()
    cats = [w.categoryListWidget.item(i).text() for i in range(w.categoryListWidget.count())]
    check('GIXOS category listed', 'GIXOS' in cats, cats)
    if 'GIXOS' not in cats:
        return
    w.categoryListWidget.setCurrentRow(cats.index('GIXOS'))
    funcs = [w.funcListWidget.item(i).text() for i in range(w.funcListWidget.count())]
    check('GIXOS category holds exactly the GIXOS model', funcs == ['GIXOS'], funcs)
    if 'GIXOS' not in funcs:
        return
    w.funcListWidget.setCurrentRow(funcs.index('GIXOS'))
    # The upstream file has no self.filepaths, which update_fixed_parameters
    # requires (KeyError -> "syntax error" dialog). XModJointFit adds it.
    check('selecting it raises no Function Error dialog', not warned,
          [m[1].strip().splitlines()[-1] for m in warned])
    f = w.fit.func
    check('selecting it builds a GIXOS Fit', type(f).__name__ == 'GIXOS', type(f).__name__)
    f.x = X
    f.__fit__ = False
    check('GUI-built model evaluates finite', np.isfinite(np.asarray(f.y(), float)).all())


# ----------------------------------------------------- 2. evaluation --------
def test_eval():
    print('\n[2] evaluation on the x range')
    x = np.linspace(0.001, 1.0, 200)
    f = GIXOS(x=x)
    f.__fit__ = False
    y = np.asarray(f.y(), float)
    lam = 6.62607004e-34 * 2.99792458e8 * 1e10 / f.E / 1e3 / 1.60217662e-19
    k0 = 2 * np.pi / lam
    below = x / k0 <= np.sin(np.radians(f.alpha))
    check('finite everywhere', np.isfinite(y).all())
    check('zero exactly below the horizon', np.all(y[below] == 0) and np.all(y[~below] > 0),
          '%d zeros, %d below horizon' % (int((y == 0).sum()), int(below.sum())))
    check('reporting path fills the EDP outputs', 'Model_EDP' in f.output_params,
          list(f.output_params))
    yf = synth(X)
    yt = synth(X, tension=25.0)
    check('tension changes the curve', np.max(np.abs(yt / yf - 1)) > 1e-3)


# --------------------------------------------------------- 3. pickle --------
def test_pickle():
    print('\n[3] pickle round trip')
    f = GIXOS(x=X, mpar=MPAR)
    f.__fit__ = False
    y = np.asarray(f.y(), float)
    g = pickle.loads(pickle.dumps(f))
    check('unpickled model gives the identical curve', np.array_equal(np.asarray(g.y(), float), y))


# ------------------------------------------------ 4/5. single Fit ----------
def single_fit(mask=None):
    from Fit_Routines import Fit
    y = synth(X)
    fit = Fit(Film, X, y=y, yerr=0.05 * y, mask=mask)
    fit.func.__fit__ = False
    fit.func.y()                         # installs the fix_sig expr, as update_plot does
    for k, v in START.items():
        fit.fit_params[k].value = v
        fit.fit_params[k].vary = 1
        fit.params[k] = v
    lens = []
    fit.functionCalled.connect(lambda p, n, r, s: lens.append(len(r)))
    fit.perform_fit(X[0], X[-1], fit_scale='Log', fit_method='leastsq', maxiter=2000)
    return fit, lens


def test_single():
    print('\n[4] single-dataset Fit recovers its parameters (Log scale)')
    fit, _ = single_fit()
    rp = fit.result.params
    for k, v in TRUE.items():
        check('%s recovered' % k, rel(rp[k].value, v) < 1e-3, '%.5f vs %.5f' % (rp[k].value, v))
    check('fix_sig tie held (sig_002 == sig_001)',
          abs(rp['__Model_sig_002'].value - rp['__Model_sig_001'].value) < 1e-12)

    print('\n[5] ... with a per-column fit mask')
    m = np.ones(len(X), bool)
    m[40:52] = False
    fit, lens = single_fit(mask=m)
    check('residual length constant and shorter by the mask',
          len(set(lens)) == 1 and lens[0] == len(X) - 12, sorted(set(lens)))
    rp = fit.result.params
    for k, v in TRUE.items():
        check('masked: %s recovered' % k, rel(rp[k].value, v) < 1e-3,
              '%.5f vs %.5f' % (rp[k].value, v))


# ------------------------------------------------------ 6/7. JointFit -------
def build_joint(use_cov):
    from Joint_Fit_Routines import JointFit
    jf = JointFit()
    tensions, yscales = (50.0, 25.0), (1.0, 2.0)
    for i, (t, s) in enumerate(zip(tensions, yscales)):
        y = synth(X, tension=t, yscale=s)
        e = 0.05 * y
        cov = None
        if use_cov:
            cov = dict(y_corr=y, sigma_meas=e, k_index=np.zeros(len(X), int),
                       Sigma_logc=np.array([[0.01]]))
        mask = None
        if i == 0:
            mask = np.ones(len(X), bool)
            mask[40:52] = False
        func = GIXOS(x=X, mpar=MPAR, tension=t)
        jf.add_dataset('T%d' % int(t), func=func, x=X, y=y, yerr=e, cov=cov, mask=mask)
    groups = [jf.share_all('__Model_d_001'), jf.share_all('__Model_rho_001'),
              {'members': [(i, '__Model_sig_%03d' % j) for i in (0, 1) for j in (1, 2)],
               'name': 'sig_film'}]
    jf.build_master_params(groups=groups)
    mk = {}
    for mkey, targets in jf.param_map.items():
        for (idx, lk) in targets:
            mk[(idx, lk)] = mkey
    start = {(0, 'yscale'): 1.3, (1, 'yscale'): 2.4, (0, '__Model_d_001'): 13.0,
             (0, '__Model_rho_001'): 0.42, (0, '__Model_sig_001'): 3.4}
    for (idx, lk), v in start.items():
        jf.master_params[mk[(idx, lk)]].set(value=v, vary=True)
    want = {mk[(0, 'yscale')]: 1.0, mk[(1, 'yscale')]: 2.0,
            mk[(0, '__Model_d_001')]: 15.0, mk[(0, '__Model_rho_001')]: 0.45,
            'sig_film': 3.0}
    return jf, want


def run_joint(tag, use_cov):
    jf, want = build_joint(use_cov)
    lens = []
    jf.functionCalled.connect(lambda p, i, r, s: lens.append(len(r)))
    rep, msg = jf.perform_fit(X[0], X[-1], fit_scale='Log', fit_method='leastsq',
                              maxiter=4000, covariance_mode='data')
    check('%s: residual length constant' % tag, len(set(lens)) == 1, sorted(set(lens)))
    rp = jf.result.params
    for k, v in want.items():
        check('%s: %s recovered' % (tag, k), rel(rp[k].value, v) < 1e-3,
              '%.5f vs %.5f' % (rp[k].value, v))
    return jf, lens


def test_joint():
    print('\n[6] JointFit: two tensions, shared film, mask on dataset 0')
    jf, lens = run_joint('joint', use_cov=False)
    check('joint: residual is 2N minus the 12 masked points',
          bool(lens) and lens[0] == 2 * len(X) - 12, lens[:1])
    for o in jf.objectives:
        g = pickle.loads(pickle.dumps(o['func']))
        check('joint: %s model pickles after the fit' % o['name'],
              np.array_equal(np.asarray(g.y(), float), np.asarray(o['func'].y(), float)))

    print('\n[7] JointFit with a covariance attached (GLS)')
    jc, _ = run_joint('joint+cov', use_cov=True)
    check('joint+cov: covariance whitening was active', bool(jc._use_cov))


def main():
    app = QApplication.instance() or QApplication(sys.argv)     # noqa: F841
    test_eval()
    test_pickle()
    test_single()
    test_joint()
    test_gui()
    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
