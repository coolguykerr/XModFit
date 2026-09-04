"""Reproduce the user's state: load the RAW 4-column file as the dataset."""
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
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Functions'))
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

RAW = r"T:\SURFACE\NX_school\2026sep\extracted_data\dppc_q2d_gid.txt"
if not os.path.exists(RAW):
    RAW = os.path.abspath(os.path.join('Data', 'dppc_q2d_gid_raw_copy.txt'))
    import gid2d_core as GC
    mp = GC.load_map(os.path.join('Data', 'dppc_q2d_gid_indexed.txt'))
    np.savetxt(RAW, np.c_[mp['qxy'], mp['qz'], mp['I'], mp['Ierr']], delimiter='\t')
print("raw file:", RAW)
d = np.loadtxt(RAW)
print("raw shape:", d.shape)

w = xmodfit.XModFit()
w.addData(fnames=[RAW, RAW])
k0 = list(w.data.keys())[0]
sub = list(w.data[k0].keys())[0]
xx = w.data[k0][sub]['x']
print("dataset sub_key=%r  x[0:3]=%s  x range %.4f..%.4f  n=%d"
      % (sub, xx[:3], xx.min(), xx.max(), len(xx)))
w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True)
w.dataFileSelectionChanged()
print("Xmin:Xmax box =", w.xminmaxLineEdit.text())
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems('GISAXS', Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems('GID2D', Qt.MatchExactly)[0])

s = w.fit.params['output_params']['scaler_parameters']
print("\nn_active_pixels =", s.get('n_active_pixels'))
print("map_coverage_frac =", s.get('map_coverage_frac'))
print("WARNING_map_coverage =", repr(s.get('WARNING_map_coverage'))[:110])
z = np.asarray(w.fit.params['output_params']['map2d_data']['z'])
print("filled pixels in the image: %d of %d" % (int(np.isfinite(z).sum()), z.size))
print("func.mapfile =", os.path.basename(str(w.fit.func.mapfile)))
print("2D tab visible:", w.plotTabWidget.isTabVisible(w._plotTab2D_index))

print("\n--- panel banner ---")
print("visible:", w.imageWidget.warnLabel.isVisible())
print(w.imageWidget.warnLabel.text())
