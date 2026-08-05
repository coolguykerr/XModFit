"""
validate_jointfit_gui.py -- headless regression for Phase-2 joint-fit GUI features:
seeding from individual fits, per-dataset joint view + plotting, .jpar save/load
(with a warn+skip for a missing dataset), .par-into-joint import, and Data-list
labeling. Runs offscreen in an env with the GUI deps (see cmc-py39). xraydb is
stubbed and modal dialogs neutralized (as in validate_perdataset_fit.py).

    QT_QPA_PLATFORM=offscreen python validate_jointfit_gui.py
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

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

_OK = {"ok": True}
def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    _OK["ok"] = _OK["ok"] and bool(cond)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    import xmodfit
    import PyQt5.QtWidgets as _QtW
    class _MB:
        Ok = Yes = No = 0
        warning = question = information = staticmethod(lambda *a, **k: 0)
    xmodfit.QMessageBox = _MB
    _QtW.QMessageBox = _MB      # also covers joint_fit_panel's local `from ... import QMessageBox`
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

    w = xmodfit.XModFit()

    # two datasets
    d1 = os.path.join(os.environ.get("TEMP", ROOT), "jf_ds1.txt")
    d2 = os.path.join(os.environ.get("TEMP", ROOT), "jf_ds2.txt")
    q = np.linspace(0.02, 0.30, 40)
    for p, amp in ((d1, 1.0), (d2, 2.0)):
        np.savetxt(p, np.c_[q, amp * np.exp(-((q - 0.1) ** 2) / 0.002) + 0.01,
                            np.full_like(q, 0.02)], header="x y yerr")
    w.addData(fnames=[d1, d2])
    names = [w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
    check("two datasets loaded", len(names) == 2)

    w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR", Qt.MatchExactly)[0])
    w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers", Qt.MatchExactly)[0])

    # ---- seeding from individual fits ----
    # select both (creates each dataset's store entry), THEN set distinct qoff values
    # directly on the stored funcs -- done LAST so no update_plot round-trip (which
    # reads the param tables) clobbers them before add reads the store.
    w.dataListWidget.clearSelection()
    w.dataListWidget.item(0).setSelected(True)
    w.dataListWidget.item(1).setSelected(True)
    for nm, val in ((names[0], 0.11), (names[1], 0.22)):
        f = w.dataset_fits[nm]
        f.func.params['qoff'].set(value=val, vary=1)
        f.func.__dict__['qoff'] = val
        f._is_fitted = True      # mark as "fitted" for labeling / comparison
    w.jointFitPanel.add_selected_datasets()
    spec = w.jointFitPanel.spec
    by = {e.name: e for e in spec.datasets}
    check("both datasets added to joint fit", len(spec.datasets) == 2)
    check("ds0 seeded from its individual fit (qoff=0.11)",
          abs(by[names[0]].model.params['qoff'].value - 0.11) < 1e-9)
    check("ds1 seeded from its individual fit (qoff=0.22)",
          abs(by[names[1]].model.params['qoff'].value - 0.22) < 1e-9)

    # ---- run the joint fit ----
    for e in spec.datasets:
        e.model.params['qoff'].set(vary=1)
    spec.set_shared('qoff')   # share qoff -> joint has fewer free params than the
                              # individuals, so the nested F-test has a real DOF difference
    w.fitIterationLineEdit.setText("20")
    w.jointFitEnabled = True
    w.run_joint_fit()
    jf = w.jointFit
    check("jointFit persisted with 2 objectives", jf is not None and len(jf.objectives) == 2)
    check("per-dataset objective models are independent",
          jf.objectives[0]["func"] is not jf.objectives[1]["func"])

    # ---- per-dataset joint VIEW + plot ----
    w.jointFitPanel._on_select_dataset(names[0])
    check("selecting a joint dataset activates the JointFit", w.fit is w.jointFit)
    check("joint fit curve drawn in main plot",
          any('[joint:' in k for k in getattr(w.plotWidget, 'data', {}).keys()))
    check("param tables show the current joint objective",
          w.fit.func is jf.objectives[jf._current]["func"])

    # ---- list-driven population + multi-select overlay ----
    jp = w.jointFitPanel
    def _sfit_val(key):
        for r in range(w.sfitParamTableWidget.rowCount()):
            if w.sfitParamTableWidget.item(r, 0).text() == key:
                return w.sfitParamTableWidget.item(r, 1).text()
        return None
    jp.datasetList.clearSelection()
    jp.datasetList.item(0).setSelected(True)
    check("joint list selection populates the Parameters tables", _sfit_val('qoff') is not None)
    jp.datasetList.item(1).setSelected(True)   # both selected -> overlay
    njoint = sum(1 for k in getattr(w.plotWidget, 'data', {}).keys() if '[joint:' in k)
    check("multi-select overlays both joint curves", njoint >= 2)

    # ---- Model Comparison (nested / non-nested / missing) ----
    # chi-square is computed BY EVALUATION now (no lmfit .result needed) -- a loaded
    # .par counts as fit via _is_fitted.
    for nm in names:
        w.dataset_fits[nm]._is_fitted = True
    w.computeModelComparison()                         # nested (XLayers, fname None on all)
    txt = w.compareTextEdit.toPlainText()
    check("model comparison (nested) shows the F-test", 'F =' in txt)
    check("model comparison shows AIC and BIC", 'AIC' in txt and 'BIC' in txt)
    w.dataset_fits[names[0]].func.fname = 'structA.pdb'   # force non-nested
    w.dataset_fits[names[1]].func.fname = 'structB.pdb'
    w.computeModelComparison()
    txt2 = w.compareTextEdit.toPlainText()
    check("model comparison detects non-nested (different PDBs)", 'NON-NESTED' in txt2)
    check("non-nested omits the F-test", 'F =' not in txt2)
    w.dataset_fits[names[0]]._is_fitted = False
    w.computeModelComparison()
    check("comparison aborts cleanly when an individual fit is missing",
          'INDIVIDUAL fit' in w.compareTextEdit.toPlainText())
    w.dataset_fits[names[0]]._is_fitted = True
    # incomplete .jpar -> specific 'redo' error
    jf._is_fitted = False; jf._jpar_incomplete = True
    w.computeModelComparison()
    check("incomplete-.jpar joint -> 'redo' comparison error",
          're-run' in w.compareTextEdit.toPlainText().lower() or 'incomplete' in w.compareTextEdit.toPlainText().lower())
    jf._is_fitted = True; jf._jpar_incomplete = False
    for nm in names:
        w.dataset_fits[nm].func.__dict__.pop('fname', None)

    # ---- fixed params forced-shared (incl the PDB) across the joint fit ----
    jf.objectives[0]['func'].__dict__['E'] = 10.0
    jf.objectives[1]['func'].__dict__['E'] = 8.0
    jf.objectives[0]['func'].fname = 'structA.pdb'
    jf.objectives[1]['func'].fname = 'structB.pdb'
    w.fit = jf
    w._sync_joint_fixed_params(jf.objectives[0]['func'])
    check("fixed scalar (E) shared across joint objectives", jf.objectives[1]['func'].E == 10.0)
    check("PDB fname shared across joint objectives", jf.objectives[1]['func'].fname == 'structA.pdb')
    jf.set_current(0)
    jf.objectives[0]['func'].__dict__['E'] = 5.0
    w._fan_fixed_to_objectives('E')
    check("editing a fixed param fans to all objectives", jf.objectives[1]['func'].E == 5.0)
    w.fit = jf

    # ---- load .par into joint (apply one file's values to all datasets) ----
    par_path = os.path.join(os.environ.get("TEMP", ROOT), "into_joint.par")
    with open(par_path, "w") as fh:
        fh.write("#File\n#Category: XRR\n#Function: XLayers\n#Xrange=np.linspace(0.0,1.0,100)\n"
                 "#Fit Range=0.02:0.3\n#Fit Method=leastsq\n#Fit Scale=Linear\n#Fit Iterations=1000\n"
                 "#Fixed Parameters:\n#param\tvalue\n"
                 "#Single fitting parameters:\n#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n"
                 "qoff\t0.077\t1\t-inf\tinf\tNone\t0.1\n"
                 "#Multiple fitting parameters:\n#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n")
    w.loadParIntoJoint(fname=par_path)
    check("load .par into joint applied to ds0",
          abs(by[names[0]].model.params['qoff'].value - 0.077) < 1e-9)
    check("load .par into joint applied to ds1",
          abs(by[names[1]].model.params['qoff'].value - 0.077) < 1e-9)

    # ---- data-list labeling ----
    w._refresh_data_labels()
    it0 = next(w.dataListWidget.item(i) for i in range(w.dataListWidget.count())
               if w.dataListWidget.item(i).text() == names[0])
    check("labeled dataset is bold", it0.font().bold())
    check("labeled dataset tooltip mentions individual fit",
          "individual fit" in it0.toolTip())
    check("data-list item text unchanged (still the dataset key)", it0.text() == names[0])

    # ---- save .jpar ----
    jpar = os.path.join(os.environ.get("TEMP", ROOT), "test.jpar")
    saved_qoff0 = by[names[0]].model.params['qoff'].value
    w.saveJointFit(fname=jpar)
    check(".jpar file written", os.path.exists(jpar))

    # ---- remove ds1 from Data, then load .jpar -> ds1 skipped, ds0 restored ----
    w.dataListWidget.clearSelection()
    for i in range(w.dataListWidget.count()):
        it = w.dataListWidget.item(i)
        it.setSelected(it.text() == names[1])
    w.removeData()
    check("ds1 removed from Data list", names[1] not in [w.dataListWidget.item(i).text()
                                                          for i in range(w.dataListWidget.count())])
    try:
        os.remove(d2)   # make ds1's file un-findable so the load-missing path warns+skips
    except OSError:
        pass
    w.loadJointFit(fname=jpar)   # prompt stubbed Yes -> tries disk -> not found -> skip
    spec2 = w.jointFitPanel.spec
    check("loadJointFit skipped the not-found dataset (only ds0 associated)",
          spec2.dataset_names == [names[0]])
    check("loadJointFit restored ds0's saved params",
          abs(spec2.datasets[0].model.params['qoff'].value - saved_qoff0) < 1e-6)

    # ==== Phase-4 checks ====
    from joint_fit_panel import display_name as _dn
    check("display_name shows 'num: basename'", _dn('3<>C:/x/foo.pdb') == '3: foo.pdb')
    it = w.dataListWidget.item(0)
    check("data-list item.text() is still the full key (delegate only paints)", '<>' in it.text())
    ck = it.text()
    c1, c2 = w._curve_color_for(ck, 'a'), w._curve_color_for(ck, 'a')
    check("_curve_color_for is stable + hex", c1 == c2 and str(c1).startswith('#'))
    _k0 = list(w.data[ck].keys())[0]
    check("data curve color matches _curve_color_for",
          w.data[ck][_k0].get('color') == w._curve_color_for(ck, _k0))
    jp2 = w.jointFitPanel
    if jp2.datasetList.count() > 0:
        w.dataListWidget.setCurrentRow(0)
        check("selecting individual clears the joint selection",
              len(jp2.datasetList.selectedItems()) == 0)
        jp2.datasetList.item(0).setSelected(True)
        check("selecting joint clears the individual selection",
              len(w.dataListWidget.selectedItems()) == 0)
    # loading a .par marks the (individual) dataset already-fit
    w.dataListWidget.setCurrentRow(0)
    dk = w.dataListWidget.item(0).text()
    if dk in w.dataset_fits:
        w.dataset_fits[dk]._is_fitted = False
    w.loadParameters(fname=par_path)
    check("loading a .par sets individual _is_fitted",
          getattr(w.dataset_fits.get(w.dataListWidget.item(0).text()), '_is_fitted', False))
    # after a joint fit, the Fit button does an INDIVIDUAL fit (dispatch fix) and marks it
    _called = {'joint': False}
    _orig = w.run_joint_fit
    w.run_joint_fit = lambda *a, **k: _called.__setitem__('joint', True)
    w.dataListWidget.setCurrentRow(0)
    w.dataListWidget.item(0).setSelected(True)
    dk = w.dataListWidget.item(0).text()
    w.dataset_fits[dk].fit_params['qoff'].set(vary=1)
    w.dataset_fits[dk]._is_fitted = False
    try:
        w.doFit()
    except Exception:
        import traceback; traceback.print_exc()
    w.run_joint_fit = _orig
    check("Fit button does an individual fit when no joint dataset is selected", not _called['joint'])
    check("individual fit after a joint fit sets _is_fitted",
          getattr(w.dataset_fits.get(dk), '_is_fitted', False))

    # ==== Phase-5 checks ====
    # overlay: collapsible body + no separate dock
    check("joint panel has a collapsible body", hasattr(w.jointFitPanel, 'body'))
    check("no separate joint-fit dock (overlay under Data list)", not hasattr(w, 'jointFitDock'))
    w.jointFitPanel.enableCheck.setChecked(False)
    _off = w.jointFitPanel.body.isVisibleTo(w.jointFitPanel)
    w.jointFitPanel.enableCheck.setChecked(True)
    _on = w.jointFitPanel.body.isVisibleTo(w.jointFitPanel)
    check("enable checkbox expands/collapses the body", (not _off) and _on)

    # CI button reflects per-dataset fit status
    dk0 = w.dataListWidget.item(0).text()
    w.dataListWidget.setCurrentRow(0)
    w.dataset_fits[dk0]._is_fitted = True
    w._update_ci_button_state()
    check("CI button enabled for a fitted dataset", w.calcConfInterButton.isEnabled())
    w.fit = w.jointFit
    w._update_ci_button_state()
    check("CI button disabled for a joint active view", not w.calcConfInterButton.isEnabled())

    # CI baseline: par-loaded (result None) + a varying param -> runs a fit; none -> flag
    f0 = w.dataset_fits[dk0]
    w.fit = f0                       # ensure the individual fit is the active one
    w.sfnames = [dk0]
    f0.result = None
    f0.fit_params['qoff'].set(vary=1)
    ok, _msg = w._ensure_ci_baseline()
    check("CI baseline runs a fit when result is missing",
          ok and getattr(getattr(f0, 'result', None), 'redchi', None) is not None)
    for _p in f0.fit_params.values():
        _p.set(vary=0)
    f0.result = None
    ok2, msg2 = w._ensure_ci_baseline()
    check("CI baseline flags when there are no free parameters",
          (not ok2) and 'fittable' in msg2.lower())

    # fitCallback is JointFit-safe (drives the joint progress screen)
    import numpy as _np
    w.fit = w.jointFit
    w.fit_method = 'leastsq'
    w.showFitInfoDlg()
    w.fitData = []; w.tchisqr = 1e30
    try:
        w.fitCallback({}, 1, _np.zeros(20), 'Linear'); _jc = True
    except Exception:
        import traceback; traceback.print_exc(); _jc = False
    try:
        w.closeFitInfoDlg()
    except Exception:
        pass
    check("fitCallback is JointFit-safe (progress works for joint fits)", _jc and len(w.fitData) == 1)

    print("\nRESULT:", "ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
    return 0 if _OK["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
