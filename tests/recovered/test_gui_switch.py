"""Offscreen Part-A test: construct XModFit, load 2 datasets, switch between them,
verify per-dataset fit memory. xraydb is stubbed (not installed in cmc-py39; repo's
local ./xraydb has no source) -- it is only used for X-ray calcs, not construction."""
import os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(ROOT)

# --- stub xraydb before anything imports Chemical_Formula ---
xr = types.ModuleType("xraydb")
class _XrayDB:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return lambda *a, **k: 0
xr.XrayDB = _XrayDB
xr.f0 = lambda *a, **k: 0
xr.__path__ = []
sys.modules["xraydb"] = xr

sys.path.insert(0, ROOT)

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
app = QApplication(sys.argv)
import xmodfit
w = xmodfit.XModFit()
print("constructed OK; dataset_fits present:", hasattr(w, "dataset_fits"))

# write two tiny 3-column data files
d1 = os.path.join(os.environ["TEMP"], "ds1.txt")
d2 = os.path.join(os.environ["TEMP"], "ds2.txt")
q = np.linspace(0.02, 0.30, 40)
for path, amp in ((d1, 1.0), (d2, 2.0)):
    y = amp * np.exp(-((q - 0.1) ** 2) / 0.002) + 0.01
    np.savetxt(path, np.c_[q, y, 0.05 * y], header="x y yerr")

w.addData(fnames=[d1, d2])   # >1 file -> non-interactive accept() path
names = [w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
print("loaded datasets:", len(names))
assert len(names) == 2, names

# pick a cheap non-PDB function: category XRR, function XLayers
cats = [w.categoryListWidget.item(i).text() for i in range(w.categoryListWidget.count())]
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR", Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers", Qt.MatchExactly)[0])
print("function selected:", w.funcListWidget.currentItem().text())

# select dataset 0 -> should create a stored fit for it
w.dataListWidget.setCurrentRow(0)
print("after selecting ds0, in store:", names[0] in w.dataset_fits)
# give ds0 a distinctive param value
fit0 = w.dataset_fits[names[0]]
k = [kk for kk in fit0.fit_params.keys() if kk[:2] != "__"][0]
fit0.fit_params[k].set(value=0.4242)
fit0.func.__dict__[k] = 0.4242

# switch to dataset 1 -> its own (different) stored fit
w.dataListWidget.setCurrentRow(1)
same_obj = w.fit is w.dataset_fits[names[1]]
print("ds1 active fit is its own store entry:", same_obj)
ds1_is_distinct = w.dataset_fits[names[1]] is not w.dataset_fits[names[0]]
print("ds0 and ds1 have distinct Fit objects:", ds1_is_distinct)

# switch back to dataset 0 -> value must be remembered
w.dataListWidget.setCurrentRow(0)
back = w.dataset_fits[names[0]]
print("ds0 fit object persisted (same instance):", back is fit0)
print("ds0 param value remembered:", back.fit_params[k].value)
assert back is fit0 and abs(back.fit_params[k].value - 0.4242) < 1e-9

print("\nPART_A_OK")
