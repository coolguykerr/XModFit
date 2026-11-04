"""
validate_fit_mask.py -- engine-level regression for Fit's per-point mask.

The mask composes with imin/imax rather than replacing it, and is FROZEN once per
perform_fit so len(residual) cannot change across iterations (lmfit's leastsq
requires that). Also covers the _window_indices unification that replaced the
inconsistent 0.99*xmin factor.

    QT_QPA_PLATFORM=offscreen python validate_fit_mask.py
"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Functions'))

import numpy as np
from lmfit import Parameters
from PyQt5.QtWidgets import QApplication

from tests.harness import check, finish




class Line:
    """Minimal model: y = a*x + b. Deliberately pointwise in x, so a gappy or
    pre-sliced x is equally valid."""
    def __init__(self, x=0, a=1.0, b=0.0, mpar={}):
        """y = a*x + b"""
        self.x = x
        self.a, self.b = a, b
        self.__mpar__ = mpar
        self.choices = {}
        self.filepaths = {}
        self.__fit__ = False
        self.output_params = {'scaler_parameters': {}}
        self.init_params()

    def init_params(self):
        self.params = Parameters()
        self.params.add('a', value=self.a, vary=0, min=-np.inf, max=np.inf)
        self.params.add('b', value=self.b, vary=0, min=-np.inf, max=np.inf)

    def y(self):
        if isinstance(self.x, dict):
            return {k: self.a * np.asarray(v, float) + self.b for k, v in self.x.items()}
        return self.a * np.asarray(self.x, float) + self.b


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    from Fit_Routines import Fit, _window_indices

    # ---------------------------------------------------- _window_indices ----
    x = np.linspace(0.0, 1.0, 101)
    check("_window_indices exact on positive bounds",
          _window_indices(x, 0.2, 0.8) == (20, 80), _window_indices(x, 0.2, 0.8))
    xn = np.linspace(-1.0, 1.0, 201)
    i0, i1 = _window_indices(xn, -0.5, 0.5)
    check("_window_indices exact on NEGATIVE bounds (the 0.99 bug)",
          (i0, i1) == (50, 150) and xn[i0] >= -0.5 and xn[i1] <= 0.5,
          '(%d,%d) -> x=[%.3f,%.3f]' % (i0, i1, xn[i0], xn[i1]))
    check("_window_indices clamps an out-of-range window",
          _window_indices(x, 5.0, 6.0) == (0, 100), _window_indices(x, 5.0, 6.0))
    check("_window_indices accepts None as unbounded",
          _window_indices(x, None, None) == (0, 100))

    # ------------------------------------------------- mask=None is neutral --
    xd = np.linspace(0.1, 2.0, 60)
    yd = 3.0 * xd + 0.5 + 0.01 * np.sin(40 * xd)
    ed = 0.05 * np.ones_like(xd)
    scales = ('Linear', 'Log', 'Log w/o error', 'Linear w/o error')
    for sc in scales:
        f0 = Fit(Line, xd, y=yd, yerr=ed)
        f0.func.a, f0.func.b = 3.0, 0.5
        f0.imin, f0.imax = 0, len(xd) - 1
        r0 = f0.residual(f0.fit_params, sc)
        f1 = Fit(Line, xd, y=yd, yerr=ed, mask=np.ones(len(xd), bool))
        f1.func.a, f1.func.b = 3.0, 0.5
        f1.imin, f1.imax = 0, len(xd) - 1
        r1 = f1.residual(f1.fit_params, sc)
        check("all-True mask is byte-identical to no mask (%s)" % sc,
              np.array_equal(r0, r1), 'max|d|=%.2e' % np.max(np.abs(r0 - r1)))

    # ------------------------------------ a mask drops exactly k points ------
    m = np.ones(len(xd), bool)
    m[10:17] = False                       # 7 points, non-contiguous vs the window
    fm = Fit(Line, xd, y=yd, yerr=ed, mask=m)
    fm.func.a, fm.func.b = 3.0, 0.5
    fm.imin, fm.imax = 0, len(xd) - 1
    rm = fm.residual(fm.fit_params, 'Linear')
    check("masked residual is shorter by exactly k",
          len(rm) == len(xd) - 7, '%d vs %d-7' % (len(rm), len(xd)))
    check("active_index matches the mask",
          np.array_equal(fm.active_index(), np.where(m)[0]))

    # ---------------------------- excluded NaN / 0 / negative cannot poison --
    ybad = yd.copy()
    ybad[10:17] = [np.nan, 0.0, -5.0, np.nan, 0.0, -1.0, np.nan]
    for sc in ('Linear', 'Log'):
        fb = Fit(Line, xd, y=ybad, yerr=ed, mask=m)
        fb.func.a, fb.func.b = 3.0, 0.5
        for k in ('a', 'b'):
            fb.fit_params[k].vary = 1
        try:
            rep, msg = fb.perform_fit(xd[0], xd[-1], fit_scale=sc,
                                      fit_method='leastsq', maxiter=200)
            ok = np.all(np.isfinite(fb.residual(fb.result.params, sc)))
        except Exception as ex:
            ok = False
            msg = repr(ex)
        check("fit completes with NaN/0/negative EXCLUDED (%s, nan_policy=raise)" % sc,
              ok, msg)

    # ------------------------------------------ the freeze invariant ---------
    lens = []
    ff = Fit(Line, xd, y=yd, yerr=ed, mask=m)
    ff.func.a, ff.func.b = 1.0, 0.0
    for k in ('a', 'b'):
        ff.fit_params[k].vary = 1
    ff.functionCalled.connect(lambda p, n, r, s: lens.append(len(r)))
    ff.perform_fit(xd[0], xd[-1], fit_scale='Linear', fit_method='leastsq', maxiter=200)
    check("len(residual) constant across every iteration",
          len(set(lens)) == 1 and lens[0] == len(xd) - 7,
          'lens=%s (%d iterations)' % (sorted(set(lens)), len(lens)))
    check("recovers the true parameters with a mask applied",
          abs(ff.result.params['a'].value - 3.0) < 1e-3
          and abs(ff.result.params['b'].value - 0.5) < 1e-3,
          'a=%.5f b=%.5f' % (ff.result.params['a'].value, ff.result.params['b'].value))
    check("n_active recorded", ff.n_active == len(xd) - 7, ff.n_active)
    check("redchi uses the ACTIVE count",
          abs(ff.result.redchi - ff.result.chisqr / (ff.n_active - 2)) < 1e-9,
          '%.6g vs %.6g' % (ff.result.redchi, ff.result.chisqr / (ff.n_active - 2)))

    # ------------------------------------------ empty active set is named ----
    fe = Fit(Line, xd, y=yd, yerr=ed, mask=np.zeros(len(xd), bool))
    try:
        fe.perform_fit(xd[0], xd[-1], fit_scale='Linear', fit_method='leastsq')
        check("all-False mask raises a named ValueError", False, 'no exception')
    except ValueError as ex:
        check("all-False mask raises a named ValueError", '0 active' in str(ex), str(ex)[:70])
    except Exception as ex:
        check("all-False mask raises a named ValueError", False, repr(ex))

    # -------------------- both call paths agree over the same active points --
    i0, i1 = 5, 50
    fa = Fit(Line, xd, y=yd, yerr=ed, mask=m)          # full arrays + perform_fit
    fa.func.a, fa.func.b = 3.0, 0.5
    fa.imin, fa.imax = i0, i1
    fa._freeze_imask()
    ra = fa.residual(fa.fit_params, 'Linear')
    fb2 = Fit(Line, xd[i0:i1 + 1], y=yd[i0:i1 + 1], yerr=ed[i0:i1 + 1],
              mask=m[i0:i1 + 1])                       # pre-sliced, like update_plot
    fb2.func.a, fb2.func.b = 3.0, 0.5
    rb = fb2.residual(fb2.fit_params, 'Linear')
    check("full-array and pre-sliced paths give the same residual",
          len(ra) == len(rb) and np.allclose(ra, rb),
          '%d vs %d, max|d|=%.2e' % (len(ra), len(rb),
                                     np.max(np.abs(ra - rb)) if len(ra) == len(rb) else -1))

    # ------------------------------------------------------ dict-x branch ----
    xk = {'A': xd, 'B': xd}
    yk = {'A': yd, 'B': 2.0 * yd}
    ek = {'A': ed, 'B': ed}
    fd = Fit(Line, xk, y=yk, yerr=ek, mask={'A': m, 'B': None})
    fd.func.a, fd.func.b = 3.0, 0.5
    for k in ('A', 'B'):
        fd.imin[k], fd.imax[k] = 0, len(xd) - 1
    fd._freeze_imask()
    rd = fd.residual(fd.fit_params, 'Linear')
    check("dict-x: only the masked key loses points",
          len(rd) == (len(xd) - 7) + len(xd), len(rd))
    check("dict-x: set_mask rejects a wrong-length mask",
          _raises(lambda: fd.set_mask({'A': np.ones(3, bool), 'B': None})))
    check("scalar: set_mask rejects a wrong-length mask",
          _raises(lambda: fm.set_mask(np.ones(3, bool))))
    check("set_x(mask=None) clears a previously installed mask",
          (fm.set_x(xd, y=yd, yerr=ed) or fm.mask) is None)

    print()
    return finish()


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == '__main__':
    sys.exit(main())
