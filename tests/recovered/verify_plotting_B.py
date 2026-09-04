"""Headless checks for category-B plotting items:
  B1 data-line toggle (PlotWidget): DATA curves are markers-only by default, gain a
     connecting line when 'Data line' is on; FIT curves always have a line.
  B3 EDP co-plot: a joint fit with >1 dataset overlaid co-plots the SAME output-param
     profile from every dataset with that dataset's data color.
(B2 is covered by validate_jointfit_gui.py.)

    QT_QPA_PLATFORM=offscreen python verify_plotting_B.py
"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(ROOT); sys.path.insert(0, ROOT)
try:
    from xraydb import XrayDB as _R  # noqa
except Exception:
    _xr = types.ModuleType("xraydb")
    class _XrayDB:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB; _xr.f0 = lambda *a, **k: 0; _xr.__path__ = []
    sys.modules["xraydb"] = _xr

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

_OK = {"ok": True}
def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name); _OK["ok"] = _OK["ok"] and bool(cond)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    import xmodfit, PyQt5.QtWidgets as _QtW
    class _MB:
        Ok = Yes = No = 0
        warning = question = information = staticmethod(lambda *a, **k: 0)
    xmodfit.QMessageBox = _MB; _QtW.QMessageBox = _MB
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

    # ---------------- B1: data-line toggle (pure PlotWidget) ----------------
    pw = w.plotWidget
    x = np.linspace(0, 1, 20); y = np.sin(x)
    check("Data line default OFF", not pw.dataLineCheckBox.isChecked())
    pw.add_data(x, y, name='dcurve', fit=False, color='#1f77b4')
    check("data curve markers-only by default (pen is None)",
          pw.data['dcurve'].opts['pen'] is None)
    pw.add_data(x, y, name='fcurve', fit=True, color='#ff7f0e')
    check("fit curve always has a line (pen not None)",
          pw.data['fcurve'].opts['pen'] is not None)
    pw.dataLineCheckBox.setChecked(True)     # -> dataLineChanged -> re-plot
    check("data curve gains a line when toggle ON",
          pw.data['dcurve'].opts['pen'] is not None)
    pw.dataLineCheckBox.setChecked(False)
    check("data curve line removed again when toggle OFF",
          pw.data['dcurve'].opts['pen'] is None)
    # dashed linestyle plumbing (used by B2)
    pw.add_data(x, y, name='dashcurve', fit=True, color='#2ca02c', linestyle=Qt.DashLine)
    check("linestyle recorded + applied",
          pw.linestyle.get('dashcurve') == Qt.DashLine
          and pw.data['dashcurve'].opts['pen'].style() == Qt.DashLine)

    # ---------------- B3: multi-dataset EDP co-plot ----------------
    d1 = os.path.join(os.environ.get("TEMP", ROOT), "bpl_ds1.txt")
    d2 = os.path.join(os.environ.get("TEMP", ROOT), "bpl_ds2.txt")
    q = np.linspace(0.02, 0.30, 40)
    for p, amp in ((d1, 1.0), (d2, 2.0)):
        np.savetxt(p, np.c_[q, amp*np.exp(-((q-0.1)**2)/0.002)+0.01, np.full_like(q, 0.02)],
                   header="x y yerr")
    w.addData(fnames=[d1, d2])
    names = [w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
    w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR", Qt.MatchExactly)[0])
    w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers", Qt.MatchExactly)[0])

    # Regression: adding a dataset must NOT reset the generated-param (EDP) selection,
    # even BEFORE any fit (fchanged==True was the bug -- it skipped the restore).
    w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True)
    _edp0 = next((w.genParamListWidget.item(i) for i in range(w.genParamListWidget.count())
                  if '_EDP :' in w.genParamListWidget.item(i).text()
                  and w.genParamListWidget.item(i).text().split(':',1)[1].strip().startswith('[')), None)
    if _edp0 is not None:
        w.genParamListWidget.clearSelection(); _edp0.setSelected(True)
        _edp_txt = _edp0.text()
        w.dataListWidget.item(1).setSelected(True)   # add ds1 (no fit yet)
        check("adding a dataset preserves the EDP selection (pre-fit)",
              _edp_txt in [it.text() for it in w.genParamListWidget.selectedItems()])

    w.dataListWidget.clearSelection()
    w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
    w.jointFitPanel.add_selected_datasets()
    w.fitIterationLineEdit.setText("5"); w.jointFitEnabled = True
    w.run_joint_fit()

    jp = w.jointFitPanel
    jp.datasetList.clearSelection()
    jp.datasetList.item(0).setSelected(True)
    jp.datasetList.item(1).setSelected(True)        # overlay BOTH -> _joint_overlay_names has 2
    check("two datasets overlaid in joint view", len(getattr(w, '_joint_overlay_names', [])) == 2)

    # find an EDP profile item in the generated-param list and select it
    edp_item = None
    for i in range(w.genParamListWidget.count()):
        t = w.genParamListWidget.item(i).text()
        if '_EDP :' in t and t.strip().split(':', 1)[1].strip().startswith('['):
            edp_item = w.genParamListWidget.item(i); break
    check("an EDP profile is listed for the joint fit", edp_item is not None)
    if edp_item is not None:
        w.genParamListWidget.clearSelection()
        edp_item.setSelected(True)                  # -> plot_extra_param
        ep = w.extra_param_1DplotWidget
        keys = list(ep.data.keys())
        fnums = [n.split('<>')[0] for n in names]
        per_ds = [k for k in keys if any(k.endswith('[%s]' % fn) for fn in fnums)]
        check("EDP co-plotted once per overlaid dataset", len(per_ds) == 2)
        # per-dataset colors match the main data/fit color scheme
        cols = []
        for n in names:
            fk = list(w.data[n].keys())[0]
            cols.append(w._curve_color_for(n, fk))
        # the two co-plotted EDP curves should carry the two distinct dataset colors
        drawn_pens = [ep.data[k].opts['pen'].color().name() for k in per_ds]
        check("co-plotted EDP curves use per-dataset colors",
              len(set(drawn_pens)) == 2 and all(c.lower() in [x.lower() for x in cols] for c in drawn_pens))

    # -------- bottom panel: individual+joint EDP overlay (Ctrl combined view) --------
    w._overlay_modifier_active = lambda: True     # simulate Ctrl -> overlay
    dk = names[0]
    w.dataset_fits[dk]._is_fitted = True          # ensure an individual profile exists
    jp.datasetList.clearSelection(); w.dataListWidget.clearSelection()
    for i in range(jp.datasetList.count()):       # select dk in the joint list
        if jp.datasetList.item(i).text() == dk:
            jp.datasetList.item(i).setSelected(True)
    for i in range(w.dataListWidget.count()):     # ...and in the Data list (Ctrl overlay)
        if w.dataListWidget.item(i).text() == dk:
            w.dataListWidget.item(i).setSelected(True)
    edp2 = None
    for i in range(w.genParamListWidget.count()):
        t = w.genParamListWidget.item(i).text()
        if '_EDP :' in t and t.strip().split(':', 1)[1].strip().startswith('['):
            edp2 = w.genParamListWidget.item(i); break
    check("EDP profile listed in the combined view", edp2 is not None)
    if edp2 is not None:
        w.genParamListWidget.clearSelection(); edp2.setSelected(True)
        ep = w.extra_param_1DplotWidget
        fnum = dk.split('<>')[0]
        jname = next((k for k in ep.data if k.endswith('[joint:%s]' % fnum)), None)
        iname = next((k for k in ep.data if k.endswith('[indiv:%s]' % fnum)), None)
        check("bottom panel co-plots BOTH individual + joint EDP for the same dataset",
              jname is not None and iname is not None)
        if iname is not None:
            check("bottom-panel individual EDP is dashed",
                  ep.linestyle.get(iname) == Qt.DashLine)
        if iname is not None and jname is not None:
            check("individual & joint EDP share the dataset color",
                  ep.data[iname].opts['symbolPen'].color().name()
                  == ep.data[jname].opts['symbolPen'].color().name())

    print("\nRESULT:", "ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
    return 0 if _OK["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
