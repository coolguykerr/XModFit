"""
validate_covariance_gui.py -- the single-fit GUI entry point for a covariance
(handoff/17-spec-general-covariance.md 6.2, 6.5, 6.6; plan task 7). Headless.

    G1  attach_covariance_to: a matching sidecar attaches and marks the item (bold +
        tooltip, text unchanged); a y-mismatched dataset is refused with a message
        naming y; a data_file name mismatch warns; remove_covariance_from detaches
    G2  one Fit moved between datasets: B fits without a covariance, A with its own
    G3  update_plot with a covariance: Chi-Sqr == result.chisqr, Residuals == the
        whitened residual; without one, Residuals stay (y - yfit)/yerr
    G4  .xpar persistence: path + mode round trip; no mode line -> data mode with a
        report note; a missing covariance file warns and continues; no covariance ->
        no covariance lines
    G5  confidence intervals: with a covariance the target is chi2_min + chi2.ppf(CL, 1)
        and the profile compares chisqr; without one, today's redchi formula

    QT_QPA_PLATFORM=offscreen python validate_covariance_gui.py
"""
import os
import shutil
import sys
import tempfile
import types

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
    _xr.XrayDB = _XrayDB
    _xr.f0 = lambda *a, **k: 0
    _xr.__path__ = []
    sys.modules["xraydb"] = _xr
sys.path.insert(0, ROOT)

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from tests.harness import check, finish

FIX = os.path.join(ROOT, 'tests', 'fixtures', 'cov_v1')
GOOD = os.path.join(FIX, 'good.cov.json')
REF = os.path.join(FIX, 'good_ref.txt')
NOTE = 'saved before covariance modes existed; reloaded in data mode'

MSGS = []


def _setup(tmp):
    import xmodfit
    import PyQt5.QtWidgets as _QtW

    class _MB:
        Ok = Yes = No = 0

        @staticmethod
        def warning(*a, **k):
            MSGS.append(' '.join(str(s) for s in a[1:3]))
            return 0
        information = question = warning
    xmodfit.QMessageBox = _MB
    _QtW.QMessageBox = _MB

    class _Prog:
        def __init__(self, *a, **k): pass
        def wasCanceled(self): return False
        def __getattr__(self, n): return lambda *a, **k: None
    xmodfit.QProgressDialog = _Prog

    class _FRD:                      # accept every fit (writes .xfit/.xpar next to the data)
        def __init__(self, *a, **k): pass
        def exec_(self): return 1
        def show(self): pass
    xmodfit.FitResultDialog = _FRD

    w = xmodfit.XModFit()
    w.format = '%.17e'               # tables round-trip the fitted values exactly
    d = np.loadtxt(REF)
    paths = []
    for sub, name, yfac in (('a', 'good_ref.txt', 1.0), ('b', 'cov_b_ref.txt', 1.0),
                            ('c', 'cov_c_bad.txt', 1.05)):
        os.makedirs(os.path.join(tmp, sub))
        p = os.path.join(tmp, sub, name)
        np.savetxt(p, np.c_[d[:, 0], yfac * d[:, 1], d[:, 2]], header='x y yerr')
        paths.append(p)
    w.addData(fnames=paths)
    keys = [w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
    w.categoryListWidget.setCurrentItem(
        w.categoryListWidget.findItems('Backgrounds', Qt.MatchExactly)[0])
    w.funcListWidget.setCurrentItem(w.funcListWidget.findItems('PowerLaw', Qt.MatchExactly)[0])
    return w, keys


def _item(w, key):
    return next(w.dataListWidget.item(i) for i in range(w.dataListWidget.count())
                if w.dataListWidget.item(i).text() == key)


def _select(w, *keys):
    w.dataListWidget.clearSelection()
    for k in keys:
        _item(w, k).setSelected(True)


def _fit(w, key):
    _select(w, key)
    for nm in ('A', 'n'):
        w.fit.fit_params[nm].set(vary=1)
    MSGS.clear()
    w.doFit()


def section_g1(w, kA, kB, kC):
    _select(w, kA)
    MSGS.clear()
    ok = w.attach_covariance_to(kA, GOOD)
    it = _item(w, kA)
    check('G1 matching sidecar attaches (True, no message)', ok is True and not MSGS, str(MSGS))
    check('G1 marker: item bold, tooltip names the covariance file, text unchanged',
          it.font().bold() and 'covariance: good.cov.json' in it.toolTip() and it.text() == kA,
          'bold=%s tip=%r' % (it.font().bold(), it.toolTip()))
    check('G1 mode combo enabled for the selected dataset with a covariance, mode model',
          w.covModeComboBox.isEnabled() and w.covModeComboBox.currentText() == 'model'
          and w.covariance_modes.get(kA) == 'model')
    MSGS.clear()
    ok = w.attach_covariance_to(kC, GOOD)
    check('G1 y-mismatched dataset refused with a message naming y and the index',
          ok is False and kC not in w.covariances and any('y mismatch at index' in m for m in MSGS),
          str(MSGS))
    check('G1 refused dataset not marked', not _item(w, kC).font().bold())
    MSGS.clear()
    ok = w.attach_covariance_to(kB, GOOD)
    check('G1 data_file name differs: attaches, with a warning naming both files',
          ok is True and any('good_ref.txt' in m and 'cov_b_ref.txt' in m for m in MSGS)
          and 'good_ref.txt' in w.covStatusLabel.text(), str(MSGS))
    w.remove_covariance_from(kB)
    check('G1 remove_covariance_from detaches and unmarks',
          kB not in w.covariances and kB not in w.covariance_modes
          and not _item(w, kB).font().bold() and 'covariance' not in _item(w, kB).toolTip())
    _select(w, kB)
    check('G1 mode combo disabled for a dataset without a covariance',
          not w.covModeComboBox.isEnabled())


def section_g2(w, kA, kB):
    import Fit_Routines
    calls = []
    orig = Fit_Routines.Fit.set_x

    def spy(self, x, *a, **k):
        calls.append(k.get('cov'))
        return orig(self, x, *a, **k)
    Fit_Routines.Fit.set_x = spy
    try:
        calls.clear()
        _fit(w, kB)
        rep_b = w.fit_report
        cov_b = list(calls)
        calls.clear()
        _fit(w, kA)
        rep_a = w.fit_report
        cov_a = list(calls)
    finally:
        Fit_Routines.Fit.set_x = orig
    check('G2 dataset B (no covariance) fits without one: every set_x had cov=None',
          len(cov_b) > 0 and all(c is None for c in cov_b) and w.dataset_fits[kB].cov is None
          and 'Covariance:' not in rep_b, '%d calls, %s' % (len(cov_b), cov_b))
    check("G2 dataset A reselected: fitted with A's own covariance",
          any(c is w.covariances[kA] for c in cov_a) and 'Covariance:' in rep_a
          and 'good.cov.json' in rep_a, '%d calls' % len(cov_a))
    check("G2 A's covariance still attached after B's fit", kA in w.covariances
          and w.covariances[kA].source.endswith('good.cov.json'))


def section_g3(w, kA, kB):
    import covariance_core as cc
    _select(w, kA)
    w.covModeComboBox.setCurrentText('data')
    check('G3 mode combo writes the per-dataset mode and the Fit attribute',
          w.covariance_modes[kA] == 'data' and w.fit.covariance_mode == 'data')
    _fit(w, kA)
    res = w.fit.result
    w.update_plot()
    sp = w.fit.params['output_params']['scaler_parameters']
    chi = sp['Chi-Sqr']
    d = abs(chi - res.chisqr) / res.chisqr
    check('G3 displayed Chi-Sqr == result.chisqr (1e-10)', d < 1e-10,
          'shown %r fit %r rel %.2e' % (chi, res.chisqr, d))
    check('G3 Red_Chi_Sqr from the same whitened objective',
          abs(sp['Red_Chi_Sqr'] - res.chisqr / res.nfree) < 1e-10 * max(1.0, res.redchi))
    k0 = list(w.data[kA].keys())[0]
    x, y = w.data[kA][k0]['x'], w.data[kA][k0]['y']
    A, n = res.params['A'].value, res.params['n'].value
    f = A * x ** n
    cov = cc.load_covariance(GOOD)
    C, _ = cc.assemble([cc.Block(cov, y, cov.sigma_meas, y)], 'Linear')
    L = np.linalg.cholesky(C)
    rw = np.linalg.solve(L, y - f)
    R = w.fit.params['output_params']['Residuals']
    dev = np.max(np.abs(np.asarray(R['y']) - rw)) if len(R['y']) == len(rw) else np.inf
    check('G3 Residuals y == independent whitened residual L^-1 (y - f) (1e-10)',
          dev < 1e-10 * max(1.0, np.max(np.abs(rw))) and np.array_equal(R['x'], x),
          'dev %.3g' % dev)
    _select(w, kB)
    fb = w.fit
    R = fb.params['output_params']['Residuals']
    yb = w.data[kB][k0]['y']
    check('G3 without a covariance: Residuals == (y - yfit)/yerr exactly',
          np.array_equal(R['y'], (yb - fb.yfit) / w.data[kB][k0]['yerr']))


def section_g4(w, kA, kB, tmp):
    _select(w, kA)
    w.covModeComboBox.setCurrentText('model')
    xp = os.path.join(tmp, 'a_cov.xpar')
    w.saveParameters(fname=xp)
    txt = open(xp).read()
    check('G4 .xpar has the covariance path and mode lines (no Raw line)',
          ('#Covariance=%s\n' % os.path.abspath(GOOD)) in txt and '#Covariance Mode=model\n' in txt
          and '#Covariance Raw=' not in txt, [l for l in txt.splitlines() if 'Covariance' in l])
    xfit = os.path.join(tmp, 'a', 'good_ref.xfit')
    check('G4 the fit-result header (.xfit) carries the covariance lines too',
          os.path.exists(xfit) and '#Covariance=' in open(xfit).read()
          and '#Covariance Mode=' in open(xfit).read())
    w.remove_covariance_from(kA)
    _select(w, kA)
    MSGS.clear()
    w.loadParameters(fname=xp)
    check('G4 reload: covariance reattached to the dataset, mode restored, item marked',
          kA in w.covariances and w.covariances[kA].source == os.path.abspath(GOOD)
          and w.covariance_modes[kA] == 'model' and w.covModeComboBox.currentText() == 'model'
          and w.covModeComboBox.isEnabled() and _item(w, kA).font().bold()
          and not any('Import Error' in m for m in MSGS), str(MSGS))
    # a file from before covariance modes: no mode line
    xp2 = os.path.join(tmp, 'a_cov_old.xpar')
    with open(xp2, 'w') as fh:
        fh.write(''.join(l for l in open(xp).readlines() if not l.startswith('#Covariance Mode=')))
    w.remove_covariance_from(kA)
    _select(w, kA)
    w.loadParameters(fname=xp2)
    check('G4 no mode line: reloaded in data mode', kA in w.covariances
          and w.covariance_modes[kA] == 'data' and w.covModeComboBox.currentText() == 'data')
    _fit(w, kA)
    rep = w.fit_report
    check('G4 no mode line: the next fit runs in data mode and its report says why',
          NOTE in rep and '  mode: data' in rep, rep[-600:])
    _fit(w, kA)
    check('G4 the note is consumed by that fit (not repeated)', NOTE not in w.fit_report)
    # a missing covariance file
    xp3 = os.path.join(tmp, 'a_cov_missing.xpar')
    gone = os.path.join(tmp, 'gone.cov.json')
    with open(xp3, 'w') as fh:
        fh.write(open(xp).read().replace(os.path.abspath(GOOD), gone))
    w.remove_covariance_from(kA)
    _select(w, kA)
    MSGS.clear()
    w.loadParameters(fname=xp3)
    check('G4 missing covariance file: warning, dataset continues without one',
          kA not in w.covariances and any('not found' in m and gone in m for m in MSGS)
          and not any('Import Error' in m for m in MSGS), str(MSGS))
    w.attach_covariance_to(kA, GOOD)
    # no covariance -> no covariance lines
    _select(w, kB)
    xp4 = os.path.join(tmp, 'b.xpar')
    w.saveParameters(fname=xp4)
    check('G4 dataset without a covariance: no covariance lines in the .xpar',
          'Covariance' not in open(xp4).read())


def _ci(w, key, fpar='A', cl=68.3):
    from scipy.stats import chi2
    _fit(w, key)
    w.confInterval_ChiSqrDist()
    # the dialog's spinbox is an integer QSpinBox (default 68); read 68.3 % through it
    w.confIntervalWidget.confIntervalSpinBox.value = lambda: cl
    tw = w.confIntervalWidget.fitParamTableWidget
    row = next(r for r in range(tw.rowCount()) if tw.cellWidget(r, 0).text() == fpar)
    w.calcAll = True                 # no plot window
    MSGS.clear()
    w.calcErrPushButtonClicked(row, fpar)
    prof = w.chidata.get(fpar)
    at_min = None
    if prof is not None:
        i = int(np.argmin(np.abs(prof[:, 0] - w.min_value[fpar])))
        at_min = prof[i, 1]
    w.confIntervalWidget.close()
    return prof, at_min, chi2


def section_g5(w, kA, kB):
    _select(w, kA)
    w.covModeComboBox.setCurrentText('data')
    prof, at_min, chi2 = _ci(w, kA)
    ok = prof is not None
    # chi2.ppf(0.683, 1) = 1.00128, so "+1.0" holds to 2e-3 (the exact check is the next line)
    check('G5 with a covariance: target = chi2_min + chi2.ppf(CL, 1) (68.3 % -> +1.0)',
          ok and abs((w.targetchisqr - w.minchisqr) - 1.0) < 2e-3
          and abs((w.targetchisqr - w.minchisqr) - chi2.ppf(0.683, 1)) < 1e-12,
          'min %r target %r msgs %s' % (getattr(w, 'minchisqr', None),
                                        getattr(w, 'targetchisqr', None), MSGS))
    check('G5 with a covariance: the profile records chisqr (== chi2_min at the minimum) '
          'and finds both limits',
          ok and abs(at_min - w.minchisqr) <= 1e-12 * max(1.0, w.minchisqr)
          and w.left_limit.get('A') is not None and w.right_limit.get('A') is not None,
          'at_min %r min %r' % (at_min, getattr(w, 'minchisqr', None)))
    _select(w, kB)
    prof, at_min, chi2 = _ci(w, kB)
    ok = prof is not None
    nfree = len(w.data[kB][list(w.data[kB].keys())[0]]['x']) - 1     # A held fixed, n free
    want = w.minchisqr * chi2.isf(1.0 - 0.683, nfree) / nfree if ok else None
    check("G5 without a covariance: today's target redchi * chi2.isf(1-CL, nfree)/nfree",
          ok and abs(w.targetchisqr - want) <= 1e-12 * want,
          'target %r want %r' % (getattr(w, 'targetchisqr', None), want))
    check('G5 without a covariance: the profile records redchi (== redchi_min at the minimum)',
          ok and abs(at_min - w.minchisqr) <= 1e-12 * max(1.0, w.minchisqr)
          and abs(w.minchisqr - w.fit.result.redchi) / w.fit.result.redchi < 0.5)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    tmp = tempfile.mkdtemp(prefix='covgui_')
    try:
        w, keys = _setup(tmp)
        kA, kB, kC = keys
        for sec, args in ((section_g1, (w, kA, kB, kC)), (section_g2, (w, kA, kB)),
                          (section_g3, (w, kA, kB)), (section_g4, (w, kA, kB, tmp)),
                          (section_g5, (w, kA, kB))):
            try:
                sec(*args)
            except Exception as e:
                import traceback
                check('%s runs' % sec.__name__, False, repr(e) + ' | '
                      + traceback.format_exc().strip().splitlines()[-3].strip())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return finish()


if __name__ == '__main__':
    sys.exit(main())
