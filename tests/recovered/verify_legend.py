"""Main-plot legend: (1) data + fit ordered by dataset index, individual before joint;
(2) legend labels show the dataset NAME (not raw 'N:Col_1' keys); (3) the detail fit
carries its index (no longer placed randomly)."""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
ROOT=r"C:\Users\kerr\Documents\GitHub\XModJointFit"; os.chdir(ROOT); sys.path.insert(0,ROOT)
try: from xraydb import XrayDB as _R
except Exception:
    _xr=types.ModuleType("xraydb")
    class _X:
        def __init__(s,*a,**k): pass
        def __getattr__(s,n): return lambda *a,**k:0
    _xr.XrayDB=_X; _xr.f0=lambda *a,**k:0; _xr.__path__=[]; sys.modules["xraydb"]=_xr
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
_OK={"ok":True}
def check(n,c): print(("PASS" if c else "FAIL"),"-",n); _OK["ok"]=_OK["ok"] and bool(c)
app=QApplication.instance() or QApplication(sys.argv)
import xmodfit, PyQt5.QtWidgets as _W
class _MB:
    Ok=Yes=No=0
    warning=question=information=staticmethod(lambda *a,**k:0)
xmodfit.QMessageBox=_MB; _W.QMessageBox=_MB
class _P:
    def __init__(s,*a,**k): pass
    def wasCanceled(s): return False
    def __getattr__(s,n): return lambda *a,**k:None
xmodfit.QProgressDialog=_P
class _FRD:
    def __init__(s,*a,**k): pass
    def exec_(s): return 0
    def show(s): pass
xmodfit.FitResultDialog=_FRD
w=xmodfit.XModFit()
d1=os.path.join(os.environ.get("TEMP",ROOT),"lg_alpha.txt"); d2=os.path.join(os.environ.get("TEMP",ROOT),"lg_beta.txt")
q=np.linspace(0.02,0.30,40)
for p,a in ((d1,1.),(d2,2.)):
    np.savetxt(p,np.c_[q,a*np.exp(-((q-0.1)**2)/0.002)+0.01,np.full_like(q,0.02)],header="x y yerr")
w.addData(fnames=[d1,d2])
names=[w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR",Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers",Qt.MatchExactly)[0])
# fit both individually
for i in range(2):
    w.dataListWidget.clearSelection(); w.dataListWidget.item(i).setSelected(True)
    w.dataset_fits[names[i]].fit_params['qoff'].set(vary=1); w.fitIterationLineEdit.setText("5")
    try: w.doFit()
    except Exception: import traceback; traceback.print_exc()
# multi-select both
w.dataListWidget.clearSelection()
w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
plotted=list(w.plotWidget.selDataNames or [])
print("plotted keys:", plotted)
# (1) ordering: every dataset-0 curve before every dataset-1 curve; data before fit
meta=[w._curve_meta(k) for k in plotted]
idxs=[m[0] for m in meta if m[0] is not None]
check("legend ordered by dataset index (non-decreasing)", idxs==sorted(idxs))
# within dataset, data(0) precedes indiv fit(1)
for di in (0,1):
    ranks=[m[1] for m in meta if m[0]==di]
    check("dataset %d: data before fit (ranks non-decreasing)"%di, ranks==sorted(ranks))
# (2) detail fit carries its index (no rank-3 'random' fit among the individual curves)
check("no unindexed individual fit curves (detail has its index)",
      all(m[0] is not None for m in meta))
# (3) legend LABELS show dataset names, not 'N:Col_1'
labels={k: w._legend_label(k) for k in plotted}
print("labels:", labels)
data_keys=[k for k in plotted if w._curve_meta(k)[1]==0]
check("data legend label shows the dataset name (not 'Col_1')",
      all('Col' not in labels[k] for k in data_keys)
      and any('alpha' in labels[k] for k in data_keys)
      and any('beta' in labels[k] for k in data_keys))
fit_keys=[k for k in plotted if w._curve_meta(k)[1]==1]
check("fit legend label is 'name (fit)'", all(labels[k].endswith('(fit)') for k in fit_keys))

# (4) legend contains EXACTLY the currently-plotted curves (no stale/duplicate entries)
leg=w.plotWidget.legendItem
check("legend has one entry per plotted curve (no duplicates/stale)",
      len(leg.items)==len(plotted))
# switching selection updates the legend (no leftovers)
w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True)
plotted1=list(w.plotWidget.selDataNames or [])
check("legend shrinks to the new selection (no leftover entries)",
      len(w.plotWidget.legendItem.items)==len(plotted1) and len(plotted1)<len(plotted))

print("\nRESULT:", "ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
sys.exit(0 if _OK["ok"] else 1)
