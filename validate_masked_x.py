"""
validate_masked_x.py -- Stage-4 regression: the __masked_x__ opt-in and GID2D's
window='engine' mode.

The point of window='engine' is an HONEST dof: with window='model' the excluded
rows sit in the residual vector contributing exactly zero, so lmfit's reduced
chi^2 is optimistic by roughly n_all/n_active. This checks that the two routes
agree on the parameters and disagree on the dof in exactly that way.

    QT_QPA_PLATFORM=offscreen python validate_masked_x.py
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
for p in (ROOT, os.path.join(ROOT, 'Functions'), os.path.join(ROOT, 'Functions', 'GISAXS')):
    sys.path.insert(0, p)

import numpy as np
from PyQt5.QtWidgets import QApplication

from tests.harness import check, finish




def main():
    app = QApplication.instance() or QApplication(sys.argv)
    import gid2d_core as GC
    from Fit_Routines import Fit
    from GID2D import GID2D

    MAP = os.path.abspath(os.path.join('Data', 'dppc_q2d_gid_indexed.txt'))
    if not os.path.exists(MAP):
        check("packed map present", False, MAP)
        return 1
    mp = GC.load_map(MAP)
    n = mp['n']
    WIN = dict(qxymin=1.38, qxymax=1.56, qzmin=0.02, qzmax=0.95)
    CFG = dict(E=15.126, alpha_i=0.10, qc=0.0217, a0=4.95, t=30.5, psi=0.0,
               eps=0.041, L_dom=400.0, L_ch=35.0, yscale=50.0, bkg0=40.0,
               fisher=False, **WIN)
    VARY = ('a0', 'L_dom', 'L_ch', 'yscale', 'bkg0')

    # the engine-side mask equivalent to the model-side q window
    active = ((mp['qxy'] >= WIN['qxymin']) & (mp['qxy'] <= WIN['qxymax']) &
              (mp['qz'] >= WIN['qzmin']) & (mp['qz'] <= WIN['qzmax']) &
              (mp['Ierr'] > 0))
    n_act = int(active.sum())
    print('   %d of %d rows active (%.1f%%)' % (n_act, n, 100.0 * n_act / n))

    def run(window, mask):
        fit = Fit(GID2D, mp['idx'].astype(float), y=mp['I'], yerr=mp['Ierr'],
                  mask=mask)
        f = fit.func
        f.mapfile, f.mode, f.window = MAP, 'map2d', window
        f.filepaths['mapfile'] = MAP
        for k, v in CFG.items():
            setattr(f, k, v)
            if k in f.params:
                f.params[k].value = v
        fit.sync_param()
        for k in VARY:
            f.params[k].vary = 1
        lens = []
        fit.functionCalled.connect(lambda p, i, r, s: lens.append(len(r)))
        rep, msg = fit.perform_fit(0.0, float(n - 1), fit_scale='Log',
                                   fit_method='leastsq', maxiter=400)
        return fit, lens, msg

    fm, lens_m, msg_m = run('model', None)
    fe, lens_e, msg_e = run('engine', active)

    check("window='model' residual keeps every row",
          lens_m[0] == n, '%d vs %d' % (lens_m[0], n))
    check("window='engine' residual keeps only the active rows",
          lens_e[0] == n_act, '%d vs %d' % (lens_e[0], n_act))
    check("engine residual length constant across iterations",
          len(set(lens_e)) == 1, sorted(set(lens_e)))

    check("__masked_x__ gave y() only the active rows",
          len(np.atleast_1d(fe.func.x)) == n_act,
          '%d vs %d' % (len(np.atleast_1d(fe.func.x)), n_act))
    check("n_active recorded on the Fit", fe.n_active == n_act, fe.n_active)

    pm = {k: fm.result.params[k].value for k in VARY}
    pe = {k: fe.result.params[k].value for k in VARY}
    rel = {k: abs(pe[k] - pm[k]) / max(abs(pm[k]), 1e-12) for k in VARY}
    worst = max(rel.values())
    print('   parameters:  ' + '  '.join('%s %.5g/%.5g' % (k, pm[k], pe[k]) for k in VARY))
    check("both routes find the same parameters (<1% apart)", worst < 1e-2,
          'worst rel diff %.2e on %s' % (worst, max(rel, key=rel.get)))

    # dof: the model route is optimistic by ~n/n_act
    ratio = fm.result.redchi and (fe.result.redchi / fm.result.redchi)
    expect = float(n - len(VARY)) / float(n_act - len(VARY))
    check("engine redchi is larger by ~n/n_active (the honest dof)",
          abs(ratio / expect - 1.0) < 0.05,
          'ratio %.4f vs expected %.4f' % (ratio, expect))
    print('   redchi: model %.6g   engine %.6g   ndata %d/%d'
          % (fm.result.redchi, fe.result.redchi, fm.result.ndata, fe.result.ndata))
    check("lmfit ndata equals the active count under 'engine'",
          fe.result.ndata == n_act, fe.result.ndata)

    # chi2 itself must agree: the excluded rows contributed exactly zero
    check("chisqr agrees between the two routes",
          abs(fe.result.chisqr / fm.result.chisqr - 1.0) < 1e-2,
          '%.6g vs %.6g' % (fe.result.chisqr, fm.result.chisqr))

    # a model WITHOUT __masked_x__ must still get a contiguous x
    class Plain(GID2D):
        __masked_x__ = False
    fp = Fit(Plain, mp['idx'].astype(float), y=mp['I'], yerr=mp['Ierr'], mask=active)
    g = fp.func
    g.mapfile, g.mode, g.window = MAP, 'map2d', 'engine'
    g.filepaths['mapfile'] = MAP
    fp.imin, fp.imax = 0, n - 1
    fp._freeze_imask()
    fp.evaluate()
    check("a model without __masked_x__ receives the FULL contiguous window",
          len(np.atleast_1d(g.x)) == n, '%d vs %d' % (len(np.atleast_1d(g.x)), n))
    r = fp.residual(fp.fit_params, 'Log')
    check("...and its residual is still correctly masked",
          len(r) == n_act, '%d vs %d' % (len(r), n_act))

    # the 1-D reduction modes must be untouched: they use qzmin/qxymin as
    # quadrature/band bounds, not as a fit window
    fi = GID2D(x=np.linspace(1.30, 1.65, 60), mode='inplane', qzmin=0.0, qzmax=1.4,
               a0=4.8743, t=18.85, eps=-0.0321, L_dom=200.0, L_ch=19.0)
    fi.__fit__ = True
    yi = fi.y()
    check("mode='inplane' still uses qzmin/qzmax as quadrature limits",
          len(yi) == 60 and np.all(np.isfinite(yi)) and yi.max() > 0,
          'max %.4g' % yi.max())

    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
