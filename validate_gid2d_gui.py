"""
validate_gid2d_gui.py -- headless regression for the 2D-map plot tab.

Covers the Part-A wiring: the 'image' output_params convention, the metadata filter
on all three genParamListWidget population sites, routing images to Image2DWidget
instead of the 1-D PlotWidget, the 2D tab's show/raise rules, exact q tick placement
on a non-uniform axis, and saveGenParameters for an image entry.

    QT_QPA_PLATFORM=offscreen python validate_gid2d_gui.py

NOTE on assertions: the pre-fix failure mode for a 2-D array reaching a 1-D plot IS
a QMessageBox (PlotWidget.py:270), and this harness stubs QMessageBox out. So do not
assert "no dialog appeared" -- assert widget state directly, and count calls on the
stub.
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

MAP = os.path.join('Data', 'dppc_q2d_gid_indexed.txt')
_OK = {"ok": True}


def check(name, cond, detail=''):
    print(("PASS" if cond else "FAIL"), "-", name, (' | ' + str(detail)) if detail else '')
    _OK["ok"] = _OK["ok"] and bool(cond)


def select(w, cat, fun):
    w.categoryListWidget.setCurrentItem(
        w.categoryListWidget.findItems(cat, Qt.MatchExactly)[0])
    w.funcListWidget.setCurrentItem(
        w.funcListWidget.findItems(fun, Qt.MatchExactly)[0])


def list_items(w):
    return [w.genParamListWidget.item(i).text()
            for i in range(w.genParamListWidget.count())]


def select_keys(w, keys):
    w.genParamListWidget.clearSelection()
    n = 0
    for i in range(w.genParamListWidget.count()):
        it = w.genParamListWidget.item(i)
        if it.text().split(':')[0].strip() in keys:
            it.setSelected(True)
            n += 1
    return n


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    import xmodfit
    import PyQt5.QtWidgets as _QtW
    calls = {"mb": 0}
    class _MB:
        Ok = Yes = No = 0
        @staticmethod
        def warning(*a, **k):
            calls["mb"] += 1; return 0
        @staticmethod
        def question(*a, **k):
            calls["mb"] += 1; return 0
        @staticmethod
        def information(*a, **k):
            calls["mb"] += 1; return 0
    xmodfit.QMessageBox = _MB
    _QtW.QMessageBox = _MB
    import PlotWidget as _PW
    _PW.QMessageBox = _MB
    class _Prog:
        def __init__(self, *a, **k): pass
        def wasCanceled(self): return False
        def __getattr__(self, n): return lambda *a, **k: None
    xmodfit.QProgressDialog = _Prog

    if not os.path.exists(MAP):
        check("packed map present", False, MAP)
        return 1

    w = xmodfit.XModFit()
    check("2D tab hidden before any function is chosen",
          not w.plotTabWidget.isTabVisible(w._plotTab2D_index)
          if hasattr(w.plotTabWidget, 'isTabVisible') else True)

    select(w, 'GISAXS', 'GID2D')
    import gid2d_core as GC
    mp = GC.load_map(os.path.abspath(MAP))
    w.xLineEdit.setText('np.arange(0.,%d.)' % mp['n'])
    w.xChanged()
    calls["mb"] = 0

    items = list_items(w)
    check("image entry listed with metadata filtered (xChanged path)",
          "map2d_data : ['x', 'y', 'z']" in items,
          [t for t in items if t.startswith('map2d')])

    # ---- select the three map entries only
    n = select_keys(w, {'map2d_data', 'map2d_model', 'map2d_residual'})
    check("three map entries selected", n == 3, n)
    check("imageWidget has maps", w.imageWidget.has_maps())
    check("three panels drawn", w.imageWidget.panel_count() == 3,
          w.imageWidget.panel_count())
    check("two colorbars (shared data+model, own residual)",
          len(w.imageWidget._bars) == 2, len(w.imageWidget._bars))
    check("2D tab visible", w.plotTabWidget.isTabVisible(w._plotTab2D_index))
    check("2D tab raised (no 1-D curve in selection)",
          w.plotTabWidget.currentIndex() == w._plotTab2D_index,
          w.plotTabWidget.currentIndex())
    stray = [k for k in w.extra_param_1DplotWidget.data if k.startswith('map2d')]
    check("no map key reached the 1-D plot widget", not stray, stray)
    check("no QMessageBox raised while plotting images", calls["mb"] == 0, calls["mb"])

    # ---- tick placement must be EXACT, not snapped to a measured column
    e = w.fit.params['output_params']['map2d_data']
    for axis, uq in (('bottom', np.asarray(e['x'])), ('left', np.asarray(e['y']))):
        ticks = w.imageWidget.tick_spec(axis)
        idx = np.arange(len(uq), dtype=float)
        err = max(abs(np.interp(p, idx, uq) - float(l)) for p, l in ticks) if ticks else 1
        check("%s axis: %d major q ticks, exact placement" % (axis, len(ticks)),
              len(ticks) >= 4 and err < 1e-9, 'worst err %.2e' % err)

    # ---- crosshair readout
    txt = w.imageWidget._readout(24, 203)
    check("readout carries both q values and all three roles",
          ('%.5f' % float(e['x'][24])) in txt and ('%.5f' % float(e['y'][203])) in txt
          and 'I_data' in txt and 'I_model' in txt and '(d-f)/s' in txt, txt)

    # ---- adding a 1-D curve must populate both panels and NOT force-switch
    w.plotTabWidget.setCurrentIndex(0)          # counts as a user choice
    select_keys(w, {'map2d_data', 'map2d_model', 'map2d_residual', 'inplane_data'})
    check("1-D curve plotted alongside the maps",
          any(k.startswith('inplane_data') for k in w.extra_param_1DplotWidget.data),
          list(w.extra_param_1DplotWidget.data.keys()))
    check("maps still drawn", w.imageWidget.has_maps())
    check("tab NOT force-switched after a manual tab choice",
          w.plotTabWidget.currentIndex() == 0, w.plotTabWidget.currentIndex())

    # ---- log toggle drops non-positive model pixels
    lin = int(np.isfinite(w.imageWidget._z_for_display(
        w.fit.params['output_params']['map2d_model'])).sum())
    w.imageWidget.set_log(True)
    neg = dict(w.fit.params['output_params']['map2d_model'])
    neg['z'] = np.asarray(neg['z']) - 1e4
    lgneg = int(np.isfinite(w.imageWidget._z_for_display(neg)).sum())
    check("log10 drops non-positive pixels", lgneg < lin, '%d -> %d' % (lin, lgneg))
    w.imageWidget.set_log(False)

    # ---- zoom survives a recalculation
    p0 = w.imageWidget._panels[0]['plot']
    p0.setXRange(10, 30, padding=0)
    before = p0.viewRange()
    w.fit.params['yscale'] = 101.0
    w.fit.fit_params['yscale'].value = 101.0
    w.update_plot()
    after = w.imageWidget._panels[0]['plot'].viewRange()
    check("zoom preserved across update_plot",
          np.allclose(before[0], after[0]) and np.allclose(before[1], after[1]),
          '%s -> %s' % (before[0], after[0]))

    # ---- saveGenParameters on an image entry
    tmp = os.path.join(os.environ.get('TEMP', ROOT), 'gid2d_op')
    select_keys(w, {'map2d_data'})
    try:
        w.saveGenParameters(bfname=tmp)
        f = tmp + '_map2d_data.txt'
        arr = np.loadtxt(f) if os.path.exists(f) else None
        nx, ny = np.asarray(e['z']).shape
        check("saveGenParameters wrote the image as (nx*ny, 3)",
              arr is not None and arr.shape == (nx * ny, 3),
              None if arr is None else arr.shape)
    except Exception as ex:
        check("saveGenParameters on an image entry does not raise", False, repr(ex))

    # ---- switching to a non-2D model hides the tab
    calls["mb"] = 0
    select(w, 'XRR', 'XLayers')
    check("2D tab hidden for a 1-D model",
          not w.plotTabWidget.isTabVisible(w._plotTab2D_index))
    check("current tab is the 1-D plot", w.plotTabWidget.currentIndex() == 0,
          w.plotTabWidget.currentIndex())
    check("imageWidget cleared", not w.imageWidget.has_maps())

    # ---- the path a USER takes: load the map as a dataset, then pick GID2D.
    # Two bugs lived here. (a) The 2D tab is driven by the genParamListWidget
    # selection, which is empty on a fresh function pick, so no tab appeared and
    # nothing on screen said to go select 'map2d_data'. (b) update_plot calls
    # evaluate() BEFORE set_x(), so GID2D's self-arming __fit__ flag kept the
    # outputs from that first evaluation -- built on the x line-edit expression,
    # which as row indices collapses to ~2 pixels of the map (coverage 5e-5).
    w2 = xmodfit.XModFit()
    w2.addData(fnames=[os.path.abspath(MAP), os.path.abspath(MAP)])   # >=2: no modal
    w2.dataListWidget.clearSelection()
    w2.dataListWidget.item(0).setSelected(True)
    w2.dataFileSelectionChanged()
    select(w2, 'GISAXS', 'GID2D')
    sel = [i.text().split(':')[0].strip() for i in w2.genParamListWidget.selectedItems()]
    check("data-loaded path: map entries auto-selected",
          set(['map2d_data', 'map2d_model', 'map2d_residual']) <= set(sel), sel)
    check("data-loaded path: 2D tab visible without any user action",
          w2.plotTabWidget.isTabVisible(w2._plotTab2D_index))
    check("data-loaded path: imageWidget populated",
          w2.imageWidget.has_maps() and w2.imageWidget.panel_count() == 3,
          w2.imageWidget.panel_count())
    s2 = w2.fit.params['output_params']['scaler_parameters']
    check("data-loaded path: coverage is the whole grid, not the default x",
          s2['map_coverage_frac'] > 0.9, s2['map_coverage_frac'])
    check("data-loaded path: no coverage warning",
          not s2.get('WARNING_map_coverage'), s2.get('WARNING_map_coverage'))
    z = np.asarray(w2.fit.params['output_params']['map2d_data']['z'])
    check("data-loaded path: image is actually filled",
          int(np.isfinite(z).sum()) > 30000, int(np.isfinite(z).sum()))

    # The panel must open in LOG, because the entries declare zscale='log'.
    # Linear on data spanning 1e-3..7e2 with a median at 22% of that range puts
    # the whole map in CET-L3's near-black lower third -- it reads as a blank
    # black panel even though the levels are correct.
    iw2 = w2.imageWidget
    check("panel opens in log because the entries declare zscale='log'",
          iw2.logCheckBox.isChecked())
    dz = iw2._z_for_display(w2.fit.params['output_params']['map2d_data'])
    dfin = dz[np.isfinite(dz)]
    lo, hi = iw2._panels[0]['img'].levels
    frac = (np.median(dfin) - lo) / (hi - lo)
    check("the median pixel lands in the readable part of the colormap",
          0.35 < frac < 0.95, 'median at %.0f%% of the level range' % (100 * frac))
    # ...and a hand toggle must win from then on
    iw2.logCheckBox.setChecked(False)
    check("a hand toggle of log10 is respected on the next rebuild",
          iw2._user_log and not iw2.logCheckBox.isChecked())
    w2.update_plot()
    check("...and survives it", not iw2.logCheckBox.isChecked())
    iw2._user_log = False
    iw2.logCheckBox.setChecked(True)
    iw2._user_log = False

    # default yscale/bkg0 must put the model on the same ORDER as the data, or
    # the model panel opens black too
    zm = np.asarray(w2.fit.params['output_params']['map2d_model']['z'])
    mfin = zm[np.isfinite(zm)]
    dfin_lin = z[np.isfinite(z)]
    ratio = np.median(mfin) / np.median(dfin_lin)
    check("default parameters put the model within ~30x of the data",
          0.03 < ratio < 30.0, 'median model/data = %.3g' % ratio)
    # a manual deselect must be respected, not re-selected on the next update
    w2.genParamListWidget.clearSelection()
    w2.gen_param_items = []
    w2.update_plot()
    check("a deliberate deselect STICKS across update_plot (auto-select is "
          "once per function pick)",
          len(w2.genParamListWidget.selectedItems()) == 0
          and not w2.plotTabWidget.isTabVisible(w2._plotTab2D_index),
          'selected %d, tab visible %s'
          % (len(w2.genParamListWidget.selectedItems()),
             w2.plotTabWidget.isTabVisible(w2._plotTab2D_index)))
    # ...and re-picking the function offers them again
    select(w2, 'XRR', 'XLayers')
    select(w2, 'GISAXS', 'GID2D')
    check("re-picking the function offers the maps again",
          len(w2.genParamListWidget.selectedItems()) == 3
          and w2.plotTabWidget.isTabVisible(w2._plotTab2D_index),
          len(w2.genParamListWidget.selectedItems()))

    # ---- the WRONG file loaded as the dataset must be diagnosed LOUDLY.
    # Loading the raw (q_xy, q_z, I, Ierr) map makes Data_Dialog pick q_xy as x;
    # row_lookup then rounds all 41797 q values into rows 0-2, so 2 pixels of a
    # 39739-cell grid are filled and the panel looks blank. The fit is also
    # regressing the model against q_z, so this is an error, not a cosmetic note.
    raw = os.path.join(os.environ.get('TEMP', ROOT), 'gid2d_raw_map.txt')
    mp2 = GC.load_map(os.path.abspath(MAP))
    np.savetxt(raw, np.c_[mp2['qxy'], mp2['qz'], mp2['I'], mp2['Ierr']], delimiter='\t')
    w3 = xmodfit.XModFit()
    w3.addData(fnames=[raw, raw])
    w3.dataListWidget.clearSelection()
    w3.dataListWidget.item(0).setSelected(True)
    w3.dataFileSelectionChanged()
    select(w3, 'GISAXS', 'GID2D')
    s3 = w3.fit.params['output_params']['scaler_parameters']
    msg = str(s3.get('WARNING_map_coverage', ''))
    check("raw-file mistake: coverage collapses as expected",
          s3['map_coverage_frac'] < 0.01, s3['map_coverage_frac'])
    check("raw-file mistake: the message names the ACTUAL cause",
          'NOT A ROW INDEX' in msg and 'RAW' in msg and 'pack_map' in msg,
          msg[:80])
    check("raw-file mistake: the message says the fit is meaningless",
          'meaningless' in msg)
    check("raw-file mistake: a red banner is shown ON the 2D panel",
          not w3.imageWidget.warnLabel.isHidden()
          and 'NOT A ROW INDEX' in w3.imageWidget.warnLabel.text(),
          w3.imageWidget.warnLabel.text()[:60])
    # ...and the banner clears once the right file is loaded
    check("banner is clear on the correctly-loaded dataset",
          w2.imageWidget.warnLabel.isHidden(), w2.imageWidget.warnLabel.text()[:60])

    # ---- the Tools packer must take a raw map to a fittable state in one click
    import importlib, glob
    tools = [os.path.splitext(os.path.basename(a))[0]
             for a in glob.glob(os.path.join('Tools', 'GrazingIncidence', '*.py'))]
    check("packer is discoverable by the Tools menu scan",
          'GID2DMapPacker' in tools, sorted(tools))
    tmod = importlib.import_module('Tools.GrazingIncidence.GID2DMapPacker')
    tcls = getattr(tmod, 'GID2DMapPacker', None)      # launch_tApp needs name==file
    check("packer class name equals its file name", tcls is not None)
    # addData() with ONE file pops a modal Data_Dialog.exec_() -- correct in the
    # real GUI (the user confirms the columns) but it blocks headless
    import Data_Dialog as _DD
    _real_exec = _DD.Data_Dialog.exec_
    _DD.Data_Dialog.exec_ = lambda self: 1
    try:
        w4 = xmodfit.XModFit()
        dlg = tcls(w4)                                 # exactly how launch_tApp builds it
        out = os.path.join(os.environ.get('TEMP', ROOT), 'packer_val_indexed.txt')
        dlg.srcLineEdit.setText(raw)
        dlg.dstLineEdit.setText(out)
        dlg.loadCheckBox.setChecked(True)
        dlg._pack()
        log = dlg.log.toPlainText()
        check("packer wrote the indexed file and reported the grid",
              os.path.exists(out) and 'grid 49 q_xy x 811 q_z' in log,
              [l for l in log.splitlines() if 'grid' in l])
        k4 = list(w4.data.keys())[-1]
        sub4 = list(w4.data[k4].keys())[0]
        d4 = w4.data[k4][sub4]
        mp4 = GC.load_map(out)
        check("packer's file loads with x=idx, y=I, yerr=Ierr by DEFAULT",
              np.array_equal(d4['x'], mp4['idx']) and np.allclose(d4['y'], mp4['I'])
              and np.allclose(d4['yerr'], mp4['Ierr']))
        check("packer selected GID2D and pointed mapfile at its own output",
              w4.funcListWidget.currentItem().text() == 'GID2D'
              and os.path.abspath(str(w4.fit.func.mapfile)) == os.path.abspath(out))
        s4 = w4.fit.params['output_params']['scaler_parameters']
        check("packer leaves a fully-covered map and no warning",
              s4['map_coverage_frac'] > 0.9 and not s4.get('WARNING_map_coverage'),
              s4['map_coverage_frac'])
        check("packer leaves the 2D tab visible with the banner clear",
              w4.plotTabWidget.isTabVisible(w4._plotTab2D_index)
              and w4.imageWidget.warnLabel.isHidden())
    finally:
        _DD.Data_Dialog.exec_ = _real_exec

    print()
    print("ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
    return 0 if _OK["ok"] else 1


if __name__ == '__main__':
    sys.exit(main())
