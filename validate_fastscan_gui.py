"""validate_fastscan_gui.py -- the fast global search against a REAL XModFit widget.

Companion to validate_fastscan.py, split out for the same reason
validate_jointfit_gui.py was split from the (since retired) validate_jointfit.py: constructing the real
widget brings up a live IPython kernel that hijacks sys.stdout, so it cannot share a
process with tests that report normally.

validate_fastscan.py parts C and D pin the plugin contract and the spawn-child
protocol. This pins the remaining surface: the Qt wiring against a live host -- the
real parameter tables, the real update_plot, the real undoFit. A synthetic ModeSet
stands in for the scan itself, which parts B and D already cover, so this runs in
under a minute instead of twelve.

Run (needs the Julia/XPR backend; skips cleanly without it):
    "C:/Users/kerr/.conda/envs/cmc-py39/python.exe" validate_fastscan_gui.py

TWO TRAPS this harness exists to encode, both of which cost real time to find:

  * addData (xmodfit.py:4214) auto-accepts its modal Data_Dialog when
    len(fnames) > 1 but calls exec_() on it for a SINGLE file, which blocks
    forever with no display. Always pass two or more datasets.
  * XModFit's console dock starts a live IPython kernel whose stdout replacement
    can block on flush ("IOStream.flush timed out"). Everything after the widget is
    constructed reports through a duplicate of the real fd 1, taken up front.
"""
import copy
import os
import sys
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Functions'))

try:
    from xraydb import XrayDB as _R  # noqa: F401
except Exception:
    _xr = types.ModuleType('xraydb')

    class _XrayDB(object):
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB
    _xr.f0 = lambda *a, **k: 0
    _xr.__path__ = []
    sys.modules['xraydb'] = _xr
    print('NOTE: xraydb stubbed for this run.')

import numpy as np
from importlib import import_module
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

FAIL = []

# XModFit's console dock brings up a live IPython kernel, which replaces sys.stdout
# with a stream whose flush can block ("IOStream.flush timed out"). Everything after
# the widget is constructed therefore reports through a duplicate of the real fd 1,
# taken now, before xmodfit is imported.
_FD = os.dup(1)


def say(msg=''):
    os.write(_FD, (msg + chr(10)).encode('utf-8', 'replace'))


def ok(name, cond, detail=''):
    say('  %-58s %s%s' % (name, 'PASS' if cond else 'FAIL',
                          ('  ' + detail) if detail else ''))
    if not cond:
        FAIL.append(name)
    return bool(cond)


def main():
    from Functions import xpr_backend as xb
    if not xb.julia_available():
        say('SKIP: the Julia/XPR backend is unavailable here.')
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    import xmodfit

    # non-blocking stand-ins for the three modal dialogs
    class _MB(object):
        Ok = Yes = No = 1
        Warning = Information = Question = 1

        @staticmethod
        def warning(*a, **k):
            say('      [QMessageBox.warning] %s' % (a[2][:90] if len(a) > 2 else ''))
            return 1

        @staticmethod
        def information(*a, **k):
            return 1

        @staticmethod
        def question(*a, **k):
            return 1

        @staticmethod
        def critical(*a, **k):
            return 1
    xmodfit.QMessageBox = _MB

    class _Prog(object):
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, n):
            return lambda *a, **k: None

        def wasCanceled(self):
            return False
    xmodfit.QProgressDialog = _Prog

    class _FRD(object):
        def __init__(self, *a, **k):
            pass

        def exec_(self):
            return 0

        def __getattr__(self, n):
            return lambda *a, **k: None
    xmodfit.FitResultDialog = _FRD

    say('\n--- F1  bring up the real widget and select the XPR model ---')
    w = xmodfit.XModFit()
    # TWO datasets, and that is not cosmetic: addData (xmodfit.py:4214) auto-accepts
    # the Data_Dialog when len(fnames) > 1 but calls exec_() on it for a single file,
    # which blocks forever with no display. Any headless test must pass >= 2.
    data = os.path.join(ROOT, 'Data', 'i2_tim3_1000_test.txt')
    data2 = os.path.join(ROOT, 'Data', 'd6_tim3_1000_test.txt')
    w.addData(fnames=[data, data2])
    ok('datasets loaded', w.dataListWidget.count() == 2,
       '%d rows' % w.dataListWidget.count())
    w.dataListWidget.clearSelection()
    w.dataListWidget.setCurrentRow(0)
    w.dataListWidget.item(0).setSelected(True)
    cat = w.categoryListWidget.findItems('XRR', Qt.MatchExactly)
    ok('XRR category present', bool(cat))
    w.categoryListWidget.setCurrentItem(cat[0])
    it = w.funcListWidget.findItems('XLayersPDB_XPR', Qt.MatchExactly)
    ok('XLayersPDB_XPR is in the function list', bool(it))
    if not it:
        return 1
    w.funcListWidget.clearSelection()
    w.funcListWidget.setCurrentItem(it[0])
    it[0].setSelected(True)
    app.processEvents()
    f = w.fit.func
    ok('the model is live on the host', type(f).__name__ == 'XLayersPDB_XPR',
       type(f).__name__)
    ok('its PDB resolves', os.path.isfile(str(f.fname)), os.path.basename(str(f.fname)))

    say('\n--- F2  the tool opens through the Tools loader path ---')
    mod = import_module('Tools.GlobalSearch.FastGlobalSearch')
    dlg = getattr(mod, 'FastGlobalSearch')(w)
    ok('constructed with the real XModFit as parent', dlg.host is w)
    why = dlg._not_ready()
    ok('the tool reports READY against the live host', why is None, (why or '')[:70])
    dlg.refresh()
    ok('refresh() shows the four steps', dlg.tabs.isVisibleTo(dlg))
    ok('four step tabs', dlg.tabs.count() == 4,
       ' | '.join(dlg.tabs.tabText(i) for i in range(dlg.tabs.count())))

    say('\n--- F3  it reads the same active data the fit would ---')
    q, y, e, fname, scale, window = dlg._active()
    from fastscan import core as C
    xmin = w.xmin[w.sfnames[-1]]
    xmax = w.xmax[w.sfnames[-1]]
    q2, y2, e2 = C.active_arrays(w.fit, xmin, xmax)
    ok('active arrays agree with Fit_Routines windowing',
       np.array_equal(q, q2) and np.array_equal(y, y2), '%d points' % len(q))
    ok('the fit scale comes from the GUI combo',
       scale == w.fitScaleComboBox.currentText(), scale)

    say('\n--- F4  step 1 route (c): current membrane as a seed ---')
    dlg._use_current_seed()
    ok('one seed installed from the live model', len(dlg._seeds) == 1)
    ok('the seed has the 1+2n layout',
       len(np.asarray(dlg._seeds[0]).ravel()) == 1 + 2 * dlg._nbox(f),
       np.array2string(np.asarray(dlg._seeds[0]), precision=3))
    ok('the seed table is populated', dlg.seedTable.rowCount() == 1)
    ok('the source is reported as the degraded path',
       dlg._seed_source == 'current-params')

    say('\n--- F5  step 2: the search box builds from the live model ---')
    dlg._rebuild_box_table()
    ok('box table has a row per named column', dlg.boxTable.rowCount() > 0,
       '%d rows' % dlg.boxTable.rowCount())
    names = [dlg.boxTable.item(r, 0).text() for r in range(dlg.boxTable.rowCount())]
    ok('theta and phi appear', 'theta' in names and 'phi' in names)
    r_theta = names.index('theta')
    ok('theta box is the fundamental domain',
       dlg.boxTable.item(r_theta, 1).text() == '0' and
       dlg.boxTable.item(r_theta, 2).text() == '180',
       '%s..%s' % (dlg.boxTable.item(r_theta, 1).text(),
                   dlg.boxTable.item(r_theta, 2).text()))
    ok('lmfit theta stays unbounded (rule 5 intact)',
       not np.isfinite(f.params['theta'].min))

    say('\n--- F6  install a mode into the REAL parameter tables ---')
    from fastscan import modes as M
    nbox = dlg._nbox(f)
    lo, hi, free, colnames = C.scan_box(f, nbox)
    before = {k: float(w.fit.fit_params[k].value) for k in w.fit.fit_params}
    # a synthetic mode built from the model's own values, perturbed, so the names
    # line up with the real tables exactly
    params = {}
    for cname in colnames:
        if cname and cname in w.fit.fit_params:
            params[cname] = float(w.fit.fit_params[cname].value)
    params['theta'] = 47.5
    params['phi'] = 213.25
    prov = M.Provenance(n1=8, k=2, n_active=len(q), num_boxes=nbox,
                        data_sha256=M.data_fingerprint(q, y, e),
                        dataset=str(fname), fit_scale=scale, hessian='exact',
                        template_label='t', spacez=float(f.spacez),
                        param_lower=lo, param_upper=hi, free=tuple(free),
                        varying_names=tuple(colnames[c] for c in free))
    mode = M.Mode(u=np.clip((np.array([params.get(c, 0.0) for c in colnames])
                             - lo) / np.where(hi > lo, hi - lo, 1.0), 0, 1),
                  neglogL=123.456, params=params,
                  theta=params['theta'], phi=params['phi'],
                  protins=params.get('protins', 0.0), cov=params.get('cov', 0.1),
                  chi2=246.912, n_active=len(q), hess_class=M.MIN,
                  hess_source='exact', n_starts=5, template_label='t')
    dlg._modeset = M.ModeSet([mode], prov)
    dlg._show_modes()
    ok('the mode table rendered', dlg.modeTable.rowCount() == 1,
       '%d cols' % dlg.modeTable.columnCount())
    ok('the humility banner is populated and says so',
       'BASIN-FINDER' in dlg.banner.text() and 'LOWER BOUND' in dlg.banner.text())
    ok('the threshold summary rendered', 'distinct position' in dlg.threshSummary.text())

    dlg._install(mode)
    app.processEvents()
    ok('theta reached the real sfit table / lmfit params',
       abs(float(w.fit.fit_params['theta'].value) - 47.5) < 1e-3,
       '%.4f' % float(w.fit.fit_params['theta'].value))
    ok('and the model attribute followed (fit.params IS func.__dict__)',
       abs(float(w.fit.params['theta']) - float(w.fit.fit_params['theta'].value)) < 1e-6)
    tbl_vals = {}
    for row in range(w.sfitParamTableWidget.rowCount()):
        k = w.sfitParamTableWidget.item(row, 0).text()
        tbl_vals[k] = w.sfitParamTableWidget.item(row, 1).text()
    ok('the visible table cell shows the new theta',
       abs(float(tbl_vals.get('theta', 'nan')) - 47.5) < 1e-3,
       tbl_vals.get('theta'))
    ok('cellChanged was reconnected (an edit still triggers the handler)',
       w.sfitParamTableWidget.receivers(
           w.sfitParamTableWidget.cellChanged) > 0)
    ok('Fit.modes carries the set, named modes not minima',
       getattr(w.fit, 'modes', None) is dlg._modeset)
    ok('min/max were NOT written from the scan box',
       not np.isfinite(w.fit.fit_params['theta'].min),
       'theta.min=%s' % w.fit.fit_params['theta'].min)

    say('\n--- F7  XModFit undo restores the pre-install state ---')
    ok('oldParams was armed before the write', hasattr(w, 'oldParams'))
    w.undoFit()
    app.processEvents()
    ok('undoFit restored theta',
       abs(float(w.fit.fit_params['theta'].value) - before['theta']) < 1e-3,
       '%.4f -> %.4f' % (47.5, float(w.fit.fit_params['theta'].value)))

    say('\n--- F8  the mode-curve overlay does not raise ---')
    mode.R = np.asarray(w.fit.evaluate(), dtype=float)[:len(q)]
    try:
        dlg._overlay_curves()
        ok('overlay drew without raising', True)
    except Exception as ex:                    # noqa: BLE001
        ok('overlay drew without raising', False, str(ex)[:70])
    try:
        dlg._mode_selected()
        ok('selection detail pane rendered', 'class' in dlg.detail.toPlainText())
    except Exception as ex:                    # noqa: BLE001
        ok('selection detail pane rendered', False, str(ex)[:70])

    say('\n--- F9  save/load the set through the real dialog helpers ---')
    from fastscan import io as FIO
    tmp = os.path.join(os.environ.get('TEMP', '.'), '_partf.xmodes')
    FIO.save(tmp, dlg._modeset)
    dlg._modeset = None
    dlg._modeset = FIO.load(tmp)
    dlg._show_modes()
    ok('a saved set reloads and re-renders', dlg.modeTable.rowCount() == 1)
    ok('and still reports its own power after the round trip',
       dlg._modeset.prov[0].power.ref_median == 49)
    try:
        os.remove(tmp)
    except OSError:
        pass

    dlg.close()
    w.close()
    say('\n%s' % ('ALL GUI CHECKS PASS' if not FAIL
                  else 'FAILED %d: %s' % (len(FAIL), FAIL)))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
