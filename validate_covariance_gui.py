"""
validate_covariance_gui.py -- the single-fit GUI entry point for a covariance
(handoff/17-spec-general-covariance.md 6.2, 6.5, 6.6; plan task 7). Headless.

    G1  attach_covariance_to: a matching sidecar attaches and marks the item (bold +
        tooltip, text unchanged); a y-mismatched dataset is refused with a message
        naming y; a data_file name mismatch warns; remove_covariance_from detaches
    G2  one Fit moved between datasets: B fits without a covariance, A with its own
    G3  update_plot with a covariance: Chi-Sqr == result.chisqr (data and model mode),
        Residuals == the whitened residual, labelled whitened; without one, Residuals
        stay (y - yfit)/yerr and nothing is labelled
    G4  .xpar persistence: path + mode round trip; no mode line -> data mode with a
        report note; a missing covariance or raw file, or a mismatching one, warns,
        continues, and the next fit's report starts DEGRADED FIT; no covariance ->
        no covariance lines
    G6  a legacy covariance attached: the Data plot shows y_corr; removed: raw y
    J4  joint MCMC uses the fitted JointFit's mode, noting a changed panel combo
    G5  confidence intervals: with a covariance the target is chi2_min + chi2.ppf(CL, 1)
        and the profile compares chisqr; without one, today's redchi formula
    J1  JointFitSpec / build_jointfit_from_selection load sidecars through
        covariance_core.load_covariance: a v1 sidecar on one dataset and none on the
        other fits with Task 4 J1's chi2; spec.covariance_mode reaches the fit; an
        unloadable sidecar at compile time fits that dataset without it (spec 6.5)
    J2  joint panel pre-fill: a covariance attached in the GUI shows in the covTable and
        is used by the joint fit; removing it clears the pre-fill; an explicit spec
        cov_path (the *.cov.json picker) wins
    J3  .jpar '#CovMode:' round trip; '#Cov:' without '#CovMode:' reloads in data mode
        with a report note; a legacy bundle-format fit reproduces an independent GLS to
        1e-12 (data mode, legacy dict route and legacy sidecars named in a pre-mode .jpar)

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
    check('G3 with a covariance: the residual plot is labelled whitened and the list says so',
          'whitened' in str(R.get('names', '')) and 'whitened' in str(sp.get('Chi-Sqr objective'))
          and 'whitened' in w.fit_report, '%s %s' % (R.get('names'), sp.get('Chi-Sqr objective')))
    # model mode (final review M1, Ruling R2 changed): the last round's factor is reused
    w.covModeComboBox.setCurrentText('model')
    _fit(w, kA)
    res = w.fit.result
    rep = w.fit_report                    # (cov_info is reset by the GUI's later set_x)
    check('G3 model mode iterated', '  mode: model' in rep and 'model rounds: 0' not in rep
          and 'model rounds:' in rep, ascii(rep[-500:]))
    w.update_plot()
    chi = w.fit.params['output_params']['scaler_parameters']['Chi-Sqr']
    d = abs(chi - res.chisqr) / res.chisqr
    check('G3 model mode: displayed Chi-Sqr == result.chisqr (1e-10)', d < 1e-10,
          'shown %r fit %r rel %.2e' % (chi, res.chisqr, d))
    w.covModeComboBox.setCurrentText('data')
    _select(w, kB)
    fb = w.fit
    R = fb.params['output_params']['Residuals']
    yb = w.data[kB][k0]['y']
    check('G3 without a covariance: Residuals == (y - yfit)/yerr exactly',
          np.array_equal(R['y'], (yb - fb.yfit) / w.data[kB][k0]['yerr']))
    check('G3 without a covariance: no whitened label, no objective entry',
          'names' not in R and 'Chi-Sqr objective' not in fb.params['output_params']['scaler_parameters'])


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
          NOTE in rep and '  mode: data' in rep, ascii(rep[-600:]))
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
    _fit(w, kA)
    rep = w.fit_report
    check('G4 missing covariance file: the next fit report starts DEGRADED FIT naming the file',
          rep.startswith('DEGRADED FIT:') and 'gone.cov.json' in rep.splitlines()[0]
          and 'Fit warnings:' in rep, rep[:300])
    _fit(w, kA)
    check('G4 ... reported once (the fit after it is a plain no-covariance fit)',
          not w.fit_report.startswith('DEGRADED') and 'gone.cov.json' not in w.fit_report)
    # a legacy *_c_factors.txt whose raw data file is missing
    cf = os.path.join(tmp, 'a_c_factors.txt')
    with open(cf, 'w') as fh:
        fh.write('# SF c sigma(log c)\n1 1.0 0.0\n10 0.9 0.02\n'
                 '# Sigma_logc matrix\n0 0\n0 4e-4\n')
    raw_gone = os.path.join(tmp, 'a_raw_gone.txt')
    xp5 = os.path.join(tmp, 'a_cf_rawmissing.xpar')
    with open(xp5, 'w') as fh:
        fh.write(open(xp).read().replace('#Covariance=%s\n' % os.path.abspath(GOOD),
                                         '#Covariance=%s\n#Covariance Raw=%s\n' % (cf, raw_gone)))
    _select(w, kA)
    MSGS.clear()
    try:
        w.loadParameters(fname=xp5)
        err = None
    except Exception as ex:
        err = repr(ex)
    check('G4 legacy c_factors with a missing raw file: warning naming it, continues without',
          err is None and kA not in w.covariances
          and any(raw_gone in m for m in MSGS)
          and not any('Import Error' in m for m in MSGS), '%s %s' % (err, MSGS))
    _fit(w, kA)
    check('G4 missing raw file: the next fit report starts DEGRADED FIT naming it',
          w.fit_report.startswith('DEGRADED FIT:')
          and 'a_raw_gone.txt' in w.fit_report.splitlines()[0], w.fit_report[:300])
    # a covariance that no longer matches the dataset (x/y edited since the save)
    xp6 = os.path.join(tmp, 'a_cov_mismatch.xpar')
    mm = os.path.join(tmp, 'mismatch.cov.json')
    import json
    doc = json.load(open(GOOD))
    doc['y'] = [1.1 * v for v in doc['y']]
    json.dump(doc, open(mm, 'w'))
    with open(xp6, 'w') as fh:
        fh.write(open(xp).read().replace(os.path.abspath(GOOD), mm))
    _select(w, kA)
    w.loadParameters(fname=xp6)
    _fit(w, kA)
    check('G4 mismatching covariance on reload: next fit report starts DEGRADED FIT naming it',
          kA not in w.covariances and w.fit_report.startswith('DEGRADED FIT:')
          and 'mismatch.cov.json' in w.fit_report.splitlines()[0], w.fit_report[:300])
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


def _target_after_update(w, key, cl=68.3):
    """fit -> update_plot (set_x) -> open the CI dialog -> change the CL spinbox
    (valueChanged -> setTargetChiSqr), before any Calculate."""
    _fit(w, key)
    w.update_plot()
    res = w.fit.result
    w.confInterval_ChiSqrDist()
    sb = w.confIntervalWidget.confIntervalSpinBox
    sb.value = lambda: cl
    sb.valueChanged.emit(int(cl))
    out = (w.minchisqr, w.targetchisqr, res)
    w.confIntervalWidget.close()
    return out


def section_g5(w, kA, kB):
    from scipy.stats import chi2
    _select(w, kA)
    w.covModeComboBox.setCurrentText('data')
    mn, tg, res = _target_after_update(w, kA)
    check('G5 with a covariance, spinbox changed after update_plot: min = chisqr, '
          'target - min == chi2.ppf(CL, 1)',
          mn == res.chisqr and abs((tg - mn) - chi2.ppf(0.683, 1)) < 1e-12,
          'min %r chisqr %r target %r' % (mn, res.chisqr, tg))
    _select(w, kB)
    mn, tg, res = _target_after_update(w, kB)
    want = res.redchi * chi2.isf(1.0 - 0.683, res.nfree) / res.nfree
    check("G5 without a covariance, same sequence: today's redchi formula",
          mn == res.redchi and abs(tg - want) <= 1e-12 * want,
          'min %r redchi %r target %r want %r' % (mn, res.redchi, tg, want))
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


# --------------------------------------------------------------- joint (task 8) --
def _write_v1(path, cov):
    """A v1 JSON sidecar of the in-memory DatasetCovariance ``cov`` (full precision)."""
    import json
    terms = []
    for t in cov.terms:
        terms.append({'name': t.name, 'kind': t.kind, 'scope': t.scope,
                      'cov': np.asarray(t.cov).tolist(),
                      'weights': None if t.weights is None else np.asarray(t.weights).tolist(),
                      'labels': t.labels})
    doc = {'format': 'xmodfit-covariance', 'version': 1, 'x': np.asarray(cov.x).tolist(),
           'y': np.asarray(cov.y).tolist(), 'sigma_meas': np.asarray(cov.sigma_meas).tolist(),
           'terms': terms}
    with open(path, 'w') as fh:
        json.dump(doc, fh)
    return path


def _jdata(**curves):
    from validate_joint_gls import X, E
    return {nm: {'c': {'x': X, 'y': y, 'yerr': E * np.ones_like(X)}} for nm, y in curves.items()}


def _jspec_fit(spec, data, sc='Linear', **kw):
    from validate_joint_gls import X
    jf = spec.compile(data)
    rep, _ = jf.perform_fit(X[0], X[-1], fit_scale=sc, fit_method='leastsq', maxiter=2000, **kw)
    return jf, rep


def section_j1(tmp):
    import covariance_core as cc
    from joint_fit_spec import JointFitSpec
    import joint_fit_orchestration as JO
    from validate_covariance import _j_data, _j_fit, _j_line
    from validate_joint_gls import rel
    X, E, ya, yb, cov = _j_data()
    jp = _write_v1(os.path.join(tmp, 'A.cov.json'), cov)
    data = _jdata(A=ya, B=yb)
    ref, _ = _j_fit([('A', ya, cov), ('B', yb, None)], 'Linear')
    spec = JointFitSpec()
    spec.add_dataset('A', model=_j_line(), cov_path=jp)
    spec.add_dataset('B', model=_j_line())
    check('J1 spec default covariance_mode is model', spec.covariance_mode == 'model')
    jf, rep = _jspec_fit(spec, data)
    d = rel(jf.result.chisqr, ref.result.chisqr)
    check('J1 spec: v1 sidecar on A + none on B compiles and fits (Linear); chi2 == Task 4 '
          'J1 joint chi2 (1e-10)', d < 1e-10,
          'spec %.12e T4 %.12e rel %.2e' % (jf.result.chisqr, ref.result.chisqr, d))
    check('J1 spec: A carries the loaded v1 covariance, B none; model mode; report names it',
          isinstance(jf.objectives[0]['cov'], cc.DatasetCovariance)
          and jf.objectives[1]['cov'] is None and jf.cov_info.get('mode') == 'model'
          and 'A.cov.json' in rep and not rep.startswith('DEGRADED'), rep[:200])
    spec.covariance_mode = 'data'
    jf, rep = _jspec_fit(spec, data)
    refd, _ = _j_fit([('A', ya, cov), ('B', yb, None)], 'Linear', mode='data')
    check('J1 spec covariance_mode=data reaches the JointFit (mode + chi2 == data-mode T4 fit)',
          jf.covariance_mode == 'data' and jf.cov_info.get('mode') == 'data'
          and rel(jf.result.chisqr, refd.result.chisqr) < 1e-10, str(jf.cov_info))
    # orchestration path, same loader
    jo = JO.build_jointfit_from_selection(data, ['A', 'B'], None, cov_paths={'A': jp},
                                          funcs={'A': _j_line(), 'B': _j_line()})
    jo.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq', maxiter=2000)
    check('J1 orchestration: v1 sidecar through load_covariance, same chi2 (1e-10)',
          isinstance(jo.objectives[0]['cov'], cc.DatasetCovariance)
          and rel(jo.result.chisqr, ref.result.chisqr) < 1e-10)
    # compile-time load failures follow the fit-time row of spec 6.5
    bad = os.path.join(tmp, 'broken.cov.json')
    with open(bad, 'w') as fh:
        fh.write('{not json')
    gone = os.path.join(tmp, 'gone.cov.json')
    nocov, _ = _j_fit([('A', ya, None), ('B', yb, None)], 'Linear')
    for path, why in ((gone, 'not found'), (bad, 'unrecognised covariance file format')):
        spec = JointFitSpec()
        spec.add_dataset('A', model=_j_line(), cov_path=path)
        spec.add_dataset('B', model=_j_line())
        try:
            jf, rep = _jspec_fit(spec, data)
            err = None
        except Exception as e:
            err = repr(e)
        first = rep.splitlines()[0] if err is None else ''
        check('J1 compile with an unloadable sidecar (%s): fits A without it, first report '
              'line names the dataset, the file and the reason' % why,
              err is None and jf.objectives[0]['cov'] is None
              and first.startswith('DEGRADED FIT: A:')
              and os.path.basename(path) in first and why in first
              and rel(jf.result.chisqr, nocov.result.chisqr) < 1e-10, err or first)
    try:
        jo = JO.build_jointfit_from_selection(data, ['A', 'B'], None, cov_paths={'A': gone},
                                              funcs={'A': _j_line(), 'B': _j_line()})
        rep, _ = jo.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq',
                                maxiter=2000)
        ok = rep.startswith('DEGRADED FIT:') and 'gone.cov.json' in rep.splitlines()[0]
        err = rep.splitlines()[0]
    except Exception as e:
        ok, err = False, repr(e)
    check('J1 orchestration with a missing sidecar: degraded fit, not an exception', ok, err)


def _panel_row(w, key):
    t = w.jointFitPanel.covTable
    for r in range(t.rowCount()):
        if t.item(r, 0).data(Qt.UserRole) == key:
            return t.item(r, 1).text(), t.item(r, 2).text()
    return None


def _joint_run(w):
    for e in w.jointFitPanel.spec.datasets:
        for nm in ('A', 'n'):
            e.model.params[nm].set(vary=1)
    MSGS.clear()
    w.run_joint_fit()
    return w.fit_report


def section_j2(w, kA, kB, tmp):
    import joint_fit_panel as JP
    _select(w, kA, kB)
    panel = w.jointFitPanel
    panel.add_selected_datasets()
    spec = panel.spec
    by = {e.name: e for e in spec.datasets}
    check('J2 GUI-attached covariance pre-fills the joint panel: A row shows good.cov.json, '
          'B row blank', _panel_row(w, kA) == ('good.cov.json', '')
          and _panel_row(w, kB) == ('', '')
          and by[kA].cov_path == os.path.abspath(GOOD) and by[kB].cov_path is None,
          '%s %s' % (_panel_row(w, kA), _panel_row(w, kB)))
    panel.covModeCombo.setCurrentText('data')
    rep = _joint_run(w)
    check("J2 the joint fit uses A's attached covariance; the panel's mode reaches the fit",
          hasattr(w.fit, 'objectives') and w.fit.objectives[0]['cov'] is not None
          and w.fit.objectives[1]['cov'] is None and w.fit.cov_info.get('mode') == 'data'
          and 'good.cov.json' in rep, ascii(rep[-400:]))
    w.remove_covariance_from(kA)
    check('J2 removing the GUI covariance clears the pre-filled (not overridden) entry',
          _panel_row(w, kA) == ('', '') and by[kA].cov_path is None, str(_panel_row(w, kA)))
    w.attach_covariance_to(kA, GOOD)
    check('J2 re-attaching pre-fills again', _panel_row(w, kA) == ('good.cov.json', ''))
    # an explicit override in the spec wins
    other = os.path.join(tmp, 'override.cov.json')
    shutil.copy(GOOD, other)
    seen = {}

    def _dlg(parent, caption='', directory='', filter='', *a, **k):
        seen['filter'] = filter
        return other, ''
    orig = JP.QFileDialog.getOpenFileName
    JP.QFileDialog.getOpenFileName = staticmethod(_dlg)
    try:
        row = next(r for r in range(panel.covTable.rowCount())
                   if panel.covTable.item(r, 0).data(Qt.UserRole) == kA)
        panel._on_cov_double_clicked(row, 1)
    finally:
        JP.QFileDialog.getOpenFileName = orig
    check('J2 covTable picker filter offers *.cov.json', '*.cov.json' in seen.get('filter', ''),
          seen.get('filter'))
    panel._refresh_cov_table()
    check('J2 the panel override wins over the GUI-attached covariance',
          _panel_row(w, kA) == ('override.cov.json', '') and by[kA].cov_path == other,
          str(_panel_row(w, kA)))
    w.remove_covariance_from(kA)
    check('J2 removing the GUI covariance leaves the explicit override',
          _panel_row(w, kA) == ('override.cov.json', ''))
    rep = _joint_run(w)
    check('J2 the joint fit loads the override file', 'override.cov.json' in rep
          and w.fit.objectives[0]['cov'] is not None, ascii(rep[-400:]))
    w.attach_covariance_to(kA, GOOD)


def section_g6(w, kA, kB, tmp):
    """Final review I1: with a legacy (replaces_y) covariance attached, the Data plot shows
    the corrected y the fit uses; removing it restores the raw curve."""
    k0 = list(w.data[kB].keys())[0]
    d = w.data[kB][k0]
    x, y, e = (np.asarray(d[c], float) for c in ('x', 'y', 'yerr'))
    kidx = (x > np.median(x)).astype(int)
    ycorr = np.where(kidx == 1, 1.3 * y, y)
    p = os.path.join(tmp, 'b_corrected_cov.npz')
    np.savez(p, x=x, y_corr=ycorr, sigma_meas=e, k_index=kidx, SF_LIST=np.array([1.0, 10.0]),
             Sigma_logc=np.array([[4e-4, 0.0], [0.0, 4e-4]]))
    name = '%s:%s' % (kB.split('<>')[0], k0)
    _select(w, kB)
    ok = w.attach_covariance_to(kB, p)
    w.update_plot()
    shown = np.asarray(w.plotWidget.data[name].yData, float)
    check('G6 legacy covariance attached: the Data plot shows y_corr (the data fitted)',
          ok and len(shown) == len(ycorr) and np.allclose(shown, ycorr, rtol=1e-12, atol=0),
          'attached %s, max |shown - y_corr| %.3g' % (ok, np.max(np.abs(shown - ycorr))
                                                       if len(shown) == len(ycorr) else np.inf))
    w.remove_covariance_from(kB)
    w.update_plot()
    shown = np.asarray(w.plotWidget.data[name].yData, float)
    check('G6 legacy covariance removed: the raw curve is shown again',
          np.allclose(shown, y, rtol=1e-12, atol=0))


def section_j4(w, kA, kB):
    """Final review M4/T8: joint MCMC uses the mode the fitted JointFit used, and notes a
    panel combo that has changed since."""
    panel = w.jointFitPanel
    fitted = w.fit.cov_info.get('mode') if hasattr(w.fit, 'objectives') else None
    other = 'model' if fitted == 'data' else 'data'
    panel.covModeCombo.setCurrentText(other)
    try:
        mode, note = w._joint_emcee_mode()
    except Exception as e:
        mode, note = None, repr(e)
    check('J4 joint MCMC mode: the fitted mode, with a note naming the panel mode',
          fitted is not None and mode == fitted and note is not None
          and fitted in str(note) and other in str(note), '%r %r %r' % (fitted, mode, note))
    panel.covModeCombo.setCurrentText(fitted or 'model')
    try:
        mode, note = w._joint_emcee_mode()
    except Exception as e:
        mode, note = None, repr(e)
    check('J4 joint MCMC mode: panel == fitted mode -> no note', mode == fitted and note is None,
          '%r %r' % (mode, note))


def _strip(path, out, prefix):
    with open(out, 'w') as fh:
        fh.write(''.join(l for l in open(path).readlines() if not l.startswith(prefix)))
    return out


def section_j3(w, kA, kB, tmp):
    panel = w.jointFitPanel
    jp = os.path.join(tmp, 'joint.jpar')
    for mode in ('data', 'model'):
        panel.covModeCombo.setCurrentText(mode)
        w.saveJointFit(fname=jp)
        txt = open(jp).read()
        panel.covModeCombo.setCurrentText('data' if mode == 'model' else 'model')
        w.loadJointFit(fname=jp)
        spec = panel.spec
        check('J3 .jpar round trip keeps the mode (%s): #CovMode line, spec, combo' % mode,
              ('#CovMode: %s\n' % mode) in txt and spec.covariance_mode == mode
              and panel.covModeCombo.currentText() == mode and not spec.covariance_notes
              and '#Cov: ' in txt,
              '%s %s' % (spec.covariance_mode, [l for l in txt.splitlines() if 'Cov' in l]))
    old = _strip(jp, os.path.join(tmp, 'joint_old.jpar'), '#CovMode:')
    panel.covModeCombo.setCurrentText('model')
    w.loadJointFit(fname=old)
    spec = panel.spec
    check('J3 a .jpar with #Cov: and no #CovMode: reloads in data mode',
          spec.covariance_mode == 'data' and panel.covModeCombo.currentText() == 'data'
          and NOTE in spec.covariance_notes, str(spec.covariance_notes))
    rep = _joint_run(w)
    check('J3 the next joint fit runs in data mode and its report says why',
          w.fit.cov_info.get('mode') == 'data' and NOTE in rep, ascii(rep[-500:]))
    rep = _joint_run(w)
    check('J3 the note is consumed by that fit (not repeated)', NOTE not in rep
          and w.fit.cov_info.get('mode') == 'data')
    # no covariance anywhere: no #CovMode line, and the reload stays in model mode
    w.remove_covariance_from(kA)
    for e in panel.spec.datasets:
        e.cov_path = e.raw_path = None
    panel._refresh_cov_table()
    jn = os.path.join(tmp, 'joint_nocov.jpar')
    w.saveJointFit(fname=jn)
    txt = open(jn).read()
    panel.covModeCombo.setCurrentText('data')
    w.loadJointFit(fname=jn)
    check('J3 without a covariance: no #Cov/#CovMode lines; reload in model mode, no note',
          '#Cov' not in txt and panel.spec.covariance_mode == 'model'
          and not panel.spec.covariance_notes, [l for l in txt.splitlines() if 'Cov' in l])
    w.attach_covariance_to(kA, GOOD)


def section_j3_legacy(tmp):
    """J3: a legacy bundle-format joint fit, as a .jpar from before covariance modes,
    reproduces the pre-change engine's numbers to 1e-12. The pre-change engine is gone,
    so the reference is an independent GLS solve on joint_covariance.build_full_covariance
    (as validate_joint_gls G3 does)."""
    import joint_covariance as jcov
    import joint_fit_io as jio
    from Joint_Fit_Routines import JointFit
    from joint_fit_spec import JointFitSpec
    from validate_covariance import _legacy_fixture
    from validate_joint_gls import X, Line, rel
    fx = _legacy_fixture(tmp, footprint=True)
    metas, _ = jcov.load_bundle_npz(fx['bundle'])
    names = list(metas)
    # reference: GLS for y_i = a x + b_i with the bundle's full C
    n, nds = len(X), len(names)
    ys = [metas[nm]['y_corr'] for nm in names]
    C = jcov.build_full_covariance(
        np.concatenate(ys), np.concatenate([metas[nm]['sigma_meas'] for nm in names]),
        np.concatenate([metas[nm]['k_index'] for nm in names]), metas[names[0]]['Sigma_logc'],
        v=np.concatenate([metas[nm]['v'] for nm in names]), var_fwhm=metas[names[0]]['var_fwhm'])
    Xd = np.zeros((nds * n, 1 + nds))
    Xd[:, 0] = np.tile(X, nds)
    for i in range(nds):
        Xd[i * n:(i + 1) * n, 1 + i] = 1.0
    d = np.concatenate(ys)
    CiX = np.linalg.solve(C, Xd)
    beta = np.linalg.solve(Xd.T @ CiX, CiX.T @ d)
    rr = d - Xd @ beta
    chi2 = float(rr @ np.linalg.solve(C, rr))
    want = dict(zip(['a'] + ['d%d_b' % i for i in range(nds)], beta))

    def _line():
        f = Line(x=X, a=2.5, b=0.3)
        for nm in ('a', 'b'):
            f.params[nm].vary = True
        return f

    def _cmp(jf, label):
        res = jf.result
        dp = max(rel(res.params[k].value, v) for k, v in want.items())
        dc = rel(res.chisqr, chi2)
        check('J3 legacy bundle (%s): params and chi2 == independent GLS to 1e-12' % label,
              dp < 1e-12 and dc < 1e-12, 'params rel %.2e chi2 rel %.2e' % (dp, dc))
    # the model is linear, so a large finite-difference step is exact up to round-off
    # (the default sqrt(eps) step limits the Jacobian, hence the optimum, to ~1e-8)
    tight = dict(xtol=1e-15, ftol=1e-15, epsfcn=1e-2)
    # joint_covariance's legacy dict route, data mode
    jf = JointFit()
    for nm in names:
        jf.add_dataset(nm, func=_line(), x=X, cov=dict(metas[nm]))
    jf.build_master_params(groups=[jf.share_all('a')])
    jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq', maxiter=2000,
                   covariance_mode='data', **tight)
    _cmp(jf, 'legacy dict route, data mode')
    # the same datasets as per-dataset legacy npz sidecars named in a .jpar written
    # before covariance modes (no #CovMode line)
    blocks = []
    for nm in names:
        m = metas[nm]
        p = os.path.join(tmp, '%s_corrected_cov.npz' % nm)
        np.savez(p, x=m['x'], y_corr=m['y_corr'], sigma_meas=m['sigma_meas'],
                 k_index=m['k_index'], SF_LIST=m['SF_LIST'], Sigma_logc=m['Sigma_logc'],
                 v=m['v'], var_fwhm=m['var_fwhm'])
        blocks.append({'path': nm, 'cov': p})
    txt = jio.format_jpar({'fit_scale': 'Linear'}, blocks, [])
    txt = ''.join(l + '\n' for l in txt.splitlines() if not l.startswith('#CovMode:'))
    parsed = jio.parse_jpar(txt)
    mode, note = jio.jpar_cov_mode(parsed)
    check('J3 jpar_cov_mode: #Cov: without #CovMode: -> data mode with the note',
          mode == 'data' and note == NOTE, '%r %r' % (mode, note))
    spec = JointFitSpec()
    for b in parsed['blocks']:
        spec.add_dataset(b['path'], model=_line(), cov_path=b['cov'])
    spec.set_shared('a')
    spec.covariance_mode = mode
    spec.covariance_notes = [note]
    data = {nm: {'c': {'x': X, 'y': np.zeros_like(X), 'yerr': np.ones_like(X)}} for nm in names}
    jf = spec.compile(data)
    rep, _ = jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq',
                            maxiter=2000, **tight)
    _cmp(jf, 'legacy npz sidecars from a pre-mode .jpar')
    check('J3 legacy .jpar fit: data mode, the note in the report, not degraded',
          jf.cov_info.get('mode') == 'data' and NOTE in rep and not rep.startswith('DEGRADED'),
          rep[:200])
    ok = all(jio.jpar_cov_mode(jio.parse_jpar(jio.format_jpar(
        {'cov_mode': m}, [{'path': 'p', 'cov': 'c.npz'}], [])))[0] == m for m in ('model', 'data'))
    m2, n2 = jio.jpar_cov_mode(jio.parse_jpar(jio.format_jpar({'cov_mode': 'data'},
                                                              [{'path': 'p'}], [])))
    check('J3 format/parse: mode round trips; no #Cov: -> model, no note', ok
          and (m2, n2) == ('model', None), '%r %r' % (m2, n2))


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    tmp = tempfile.mkdtemp(prefix='covgui_')
    try:
        w, keys = _setup(tmp)
        kA, kB, kC = keys
        for sec, args in ((section_g1, (w, kA, kB, kC)), (section_g2, (w, kA, kB)),
                          (section_g3, (w, kA, kB)), (section_g4, (w, kA, kB, tmp)),
                          (section_g5, (w, kA, kB)), (section_g6, (w, kA, kB, tmp)),
                          (section_j1, (tmp,)),
                          (section_j2, (w, kA, kB, tmp)), (section_j4, (w, kA, kB)),
                          (section_j3, (w, kA, kB, tmp)),
                          (section_j3_legacy, (tmp,))):
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
