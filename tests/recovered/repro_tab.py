"""Reproduce: LOAD the GID data as a dataset, then pick GID2D. Does the 2D tab show?"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(ROOT)
_xr = types.ModuleType("xraydb")
class _XrayDB:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: 0
_xr.XrayDB = _XrayDB; _xr.f0 = lambda *a, **k: 0; _xr.__path__ = []
sys.modules.setdefault("xraydb", _xr)
sys.path.insert(0, ROOT)
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
app = QApplication(sys.argv)
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

MAP = os.path.abspath(os.path.join('Data', 'dppc_q2d_gid_indexed.txt'))
w = xmodfit.XModFit()
# exactly what a user does: load the map file as a dataset (twice, because a
# single file pops a modal dialog that hangs headless)
w.addData(fnames=[MAP, MAP])
print("datasets:", len(w.data))
k0 = list(w.data.keys())[0]
print("sub_keys:", list(w.data[k0].keys()))
print("x range: %.1f..%.1f  n=%d" % (w.data[k0][list(w.data[k0])[0]]['x'].min(),
                                     w.data[k0][list(w.data[k0])[0]]['x'].max(),
                                     len(w.data[k0][list(w.data[k0])[0]]['x'])))
w.dataListWidget.clearSelection()
w.dataListWidget.item(0).setSelected(True)
w.dataFileSelectionChanged()

w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems('GISAXS', Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems('GID2D', Qt.MatchExactly)[0])

print("\n--- after selecting GID2D with data loaded ---")
items = [w.genParamListWidget.item(i).text() for i in range(w.genParamListWidget.count())]
print("genParam list has %d entries; map2d ones:" % len(items))
for t in items:
    if t.startswith('map2d'):
        print("   ", t)
sel = [i.text() for i in w.genParamListWidget.selectedItems()]
print("SELECTED:", sel)
print("2D tab visible:", w.plotTabWidget.isTabVisible(w._plotTab2D_index))
print("current tab:", w.plotTabWidget.currentIndex(),
      "(0=1D, %d=2D)" % w._plotTab2D_index)
print("imageWidget.has_maps():", w.imageWidget.has_maps())
op = w.fit.params['output_params']
print("output_params has map2d keys:", [k for k in op if k.startswith('map2d')])
s = op.get('scaler_parameters', {})
print("coverage:", s.get('map_coverage_frac'), " warning:", repr(s.get('WARNING_map_coverage'))[:70])
print("func.mode =", repr(w.fit.func.mode), " mapfile =", os.path.basename(str(w.fit.func.mapfile)))
