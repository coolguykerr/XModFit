"""
validate_xlayerspdb_joint.py -- does the pre-existing workflow still work?

Written to answer a specific question: after adding the 2D-map view and the
per-column fit limits, can XModFit still joint-fit ordinary 1-D reflectivity with
XLayersPDB? The existing joint-fit tests all use XLayers, so XLayersPDB in a joint
fit was NOT covered.

Also measures the one change in this work that can move an existing result: the
window-index unification. Fit.perform_fit's DICT branch and JointFit._set_ranges
used ``0.99*xmin`` while the scalar branch used ``xmin`` exactly. That is now one
helper using the exact bound everywhere, which can include/exclude one point at
the low edge of a joint fit. This measures it on the real data rather than
asserting it is harmless.

    QT_QPA_PLATFORM=offscreen python validate_xlayerspdb_joint.py
"""
import os, sys, types, faulthandler
faulthandler.enable()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
try:
    from xraydb import XrayDB as _R  # noqa
except Exception:
    _xr = types.ModuleType("xraydb")
    class _XrayDB:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB; _xr.f0 = lambda *a, **k: 0; _xr.__path__ = []
    sys.modules["xraydb"] = _xr
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Functions'))

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

DS = [os.path.join('Data', 'i2_tim3_1000_test.txt'),
      os.path.join('Data', 'd6_tim3_1000_test.txt')]
PDB = os.path.join('Data', 'tim3_hmmm_ext_all_bestframe.pdb')
PAR = os.path.join('Data', 'i2_tim3_1000_test_fitparam.par')

_OK = {"ok": True}


def check(name, cond, detail=''):
    print(("PASS" if cond else "FAIL"), "-", name, (' | ' + str(detail)) if detail else '')
    _OK["ok"] = _OK["ok"] and bool(cond)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    import xmodfit
    import PyQt5.QtWidgets as _QtW
    class _MB:
        Ok = Yes = No = 0
        warning = question = information = staticmethod(lambda *a, **k: 0)
    xmodfit.QMessageBox = _MB
    _QtW.QMessageBox = _MB
    class _Prog:
        def __init__(self, *a, **k): pass
        def wasCanceled(self): return False
        def __getattr__(self, n): return lambda *a, **k: None
    xmodfit.QProgressDialog = _Prog
    class _FRD:
        def __init__(self, *a, **k): pass
        def exec_(self): return 0
        def show(self): pass
    xmodfit.FitResultDialog = _FRD

    for p in DS + [PDB, PAR]:
        if not os.path.exists(p):
            check("required file present: %s" % p, False)
            return 1

    # ---------------------------------------------------------------------
    # 1. the 0.99*xmin unification, measured on the REAL data + REAL window
    # ---------------------------------------------------------------------
    from Fit_Routines import _window_indices
    print('\n[1] window-index unification, on the real data and the real .par window')
    xmin, xmax = 0.026, 0.550          # from Data/i2_tim3_1000_test_fitparam.par
    shifted = 0
    for f in DS:
        q = np.loadtxt(f)[:, 0]
        old_i0 = int(np.where(q >= 0.99 * xmin)[0][0])       # previous dict/joint rule
        new_i0 = int(np.where(q >= xmin)[0][0])              # the rule now used everywhere
        i0, i1 = _window_indices(q, xmin, xmax)
        same = (old_i0 == new_i0)
        shifted += 0 if same else 1
        print('    %-34s old imin=%d (q=%.6f)  new imin=%d (q=%.6f)  %s'
              % (os.path.basename(f), old_i0, q[old_i0], new_i0, q[new_i0],
                 'UNCHANGED' if same else 'SHIFTED BY %d POINT(S)' % (new_i0 - old_i0)))
        check("_window_indices agrees with the exact rule (%s)" % os.path.basename(f),
              (i0, i1) == (new_i0, int(np.where(q <= xmax)[0][-1])))
        check("the new low bound never includes a point below xmin (%s)"
              % os.path.basename(f), q[i0] >= xmin, 'q[imin]=%.6f' % q[i0])
    check("no point shift on this data at its own fit window", shifted == 0,
          '%d of %d datasets shifted' % (shifted, len(DS)))

    # ---------------------------------------------------------------------
    # 2. XLayersPDB: load two real datasets, fit each, then JOINT fit
    # ---------------------------------------------------------------------
    print('\n[2] XLayersPDB individual + joint fit on two real datasets')
    w = xmodfit.XModFit()
    w.addData(fnames=[os.path.abspath(p) for p in DS])
    names = [w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
    check("two reflectivity datasets loaded", len(names) == 2, names)

    w.dataListWidget.clearSelection()
    w.dataListWidget.item(0).setSelected(True)
    w.dataFileSelectionChanged()
    w.categoryListWidget.setCurrentItem(
        w.categoryListWidget.findItems('XRR', Qt.MatchExactly)[0])
    w.funcListWidget.setCurrentItem(
        w.funcListWidget.findItems('XLayersPDB', Qt.MatchExactly)[0])
    check("XLayersPDB selected without error",
          w.funcListWidget.currentItem().text() == 'XLayersPDB')
    check("its PDB filepath parameter is wired",
          'fname' in (w.fit.params.get('filepaths') or {})
          and os.path.exists(str(w.fit.func.fname)),
          os.path.basename(str(w.fit.func.fname)))
    check("no column limits are invented for ordinary 1-D data",
          w._column_mask(w.sfnames[-1]) is None)
    check("Fit.mask is None for ordinary 1-D data", w.fit.mask is None)

    # a real evaluation must produce a finite curve
    w.xminmaxLineEdit.setText('%f:%f' % (xmin, xmax))
    w.xminmaxChanged()
    w.fit.func.__fit__ = False
    yv = np.atleast_1d(np.asarray(w.fit.evaluate(), float))
    check("XLayersPDB evaluates to a finite curve",
          len(yv) > 10 and np.all(np.isfinite(yv)) and np.all(yv > 0),
          'n=%d  range %.3e..%.3e' % (len(yv), yv.min(), yv.max()))

    # ---- individual fits on both datasets (also seeds the joint fit) ----
    w.fitIterationLineEdit.setText('12')
    ok_indiv = 0
    for i in range(2):
        w.dataListWidget.clearSelection()
        w.dataListWidget.item(i).setSelected(True)
        w.dataFileSelectionChanged()
        for k in ('qoff', 'yscale'):
            if k in w.fit.fit_params:
                w.fit.fit_params[k].set(vary=1)
        try:
            w.doFit()
            ok_indiv += 1
        except Exception as ex:
            print('    individual fit %d failed: %r' % (i, ex))
    check("individual XLayersPDB fits run on both datasets", ok_indiv == 2, ok_indiv)
    r0 = getattr(w.fit, 'result', None)
    check("individual fit produced a finite chi-square",
          r0 is not None and np.isfinite(r0.chisqr),
          None if r0 is None else '%.6g (ndata=%d)' % (r0.chisqr, r0.ndata))

    # ---- joint fit, sharing a parameter ----
    w.dataListWidget.clearSelection()
    w.dataListWidget.item(0).setSelected(True)
    w.dataListWidget.item(1).setSelected(True)
    w.dataFileSelectionChanged()
    w.jointFitPanel.add_selected_datasets()
    spec = w.jointFitPanel.spec
    check("both datasets registered in the joint spec", len(spec.datasets) == 2,
          len(spec.datasets))
    for e in spec.datasets:
        e.model.params['qoff'].set(vary=1)
        e.model.params['yscale'].set(vary=1)
    spec.set_shared('qoff')
    w.jointFitEnabled = True
    w.fitIterationLineEdit.setText('12')
    lens = []
    try:
        w.run_joint_fit()
        jf = w.jointFit
        ok = jf is not None and len(jf.objectives) == 2
    except Exception as ex:
        import traceback; traceback.print_exc()
        jf, ok = None, False
    check("joint XLayersPDB fit ran with 2 objectives", ok,
          None if jf is None else len(jf.objectives))
    if jf is not None:
        check("no mask installed on either joint objective (1-D data)",
              all(o.get('mask') is None and o.get('imask') is None
                  for o in jf.objectives))
        check("each objective got its own model instance",
              jf.objectives[0]['func'] is not jf.objectives[1]['func'])
        res = getattr(jf, 'result', None)
        check("joint fit produced a finite chi-square",
              res is not None and np.isfinite(res.chisqr),
              None if res is None else '%.6g (ndata=%d, nvary=%d)'
              % (res.chisqr, res.ndata, res.nvarys))
        # qoff shared => ONE master entry for it
        shared_qoff = [k for k in jf.master_params if k.endswith('qoff')]
        check("shared qoff collapsed to a single master parameter",
              len(shared_qoff) == 1, shared_qoff)
        # residual length must equal the sum of the windows (no silent masking)
        want = sum(o['imax'] - o['imin'] + 1 for o in jf.objectives)
        r = jf.residual(jf.master_params, w.fitScaleComboBox.currentText())
        n_prior = len(getattr(jf, '_priors', {}) or {})
        check("joint residual length == sum of windows (nothing masked away)",
              len(r) == want + n_prior, '%d vs %d (+%d priors)' % (len(r), want, n_prior))

    # ---------------------------------------------------------------------
    # 3. an existing .par (written Nov 2024) must still load
    # ---------------------------------------------------------------------
    print('\n[3] backward compatibility of an existing .par')
    w2 = xmodfit.XModFit()
    w2.addData(fnames=[os.path.abspath(p) for p in DS])
    w2.dataListWidget.clearSelection()
    w2.dataListWidget.item(0).setSelected(True)
    w2.dataFileSelectionChanged()
    w2.categoryListWidget.setCurrentItem(
        w2.categoryListWidget.findItems('XRR', Qt.MatchExactly)[0])
    w2.funcListWidget.setCurrentItem(
        w2.funcListWidget.findItems('XLayersPDB', Qt.MatchExactly)[0])
    try:
        w2.loadParameters(fname=os.path.abspath(PAR))
        loaded = True
    except Exception as ex:
        import traceback; traceback.print_exc()
        loaded = False
    check("a Nov-2024 .par with no #Column Limits= line loads", loaded)
    check("its #Fit Range= was applied",
          w2.xminmaxLineEdit.text().startswith('0.026'), w2.xminmaxLineEdit.text())
    check("loading it invents no column limits",
          all(not [c for c in (v or {}) if c != 'x'] for v in w2.limits.values()),
          {k.split('<>')[0]: v for k, v in w2.limits.items()})
    w2.fit.func.__fit__ = False
    y2 = np.atleast_1d(np.asarray(w2.fit.evaluate(), float))
    check("the loaded model evaluates finite", np.all(np.isfinite(y2)) and y2.min() > 0,
          '%.3e..%.3e' % (y2.min(), y2.max()))

    # ---------------------------------------------------------------------
    # 4. the MCMC data fingerprint is unchanged when there is no mask
    # ---------------------------------------------------------------------
    print('\n[4] MCMC fingerprint with no mask must match the old imin:imax hash')
    import hashlib
    f = w2.fit
    f.imin, f.imax = _window_indices(np.asarray(f.x), xmin, xmax)
    f._freeze_imask()
    old = np.asarray(f.y)[f.imin:f.imax + 1]
    new = np.asarray(f.y)[f.active_index()]
    check("active_index() == the plain window when no mask is set",
          np.array_equal(old, new), '%d vs %d points' % (len(old), len(new)))
    h_old = hashlib.sha1(np.ascontiguousarray(old, dtype='float64').tobytes()).hexdigest()[:16]
    fp = w2._mcmc_fingerprint()
    check("fingerprint datahash unchanged for an unmasked fit",
          fp.get('datahash') == h_old, '%s vs %s' % (fp.get('datahash'), h_old))
    check("fingerprint ndata unchanged", fp.get('ndata') == len(old),
          '%s vs %d' % (fp.get('ndata'), len(old)))

    print()
    print("ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
    return 0 if _OK["ok"] else 1


if __name__ == '__main__':
    sys.exit(main())
