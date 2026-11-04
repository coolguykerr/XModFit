"""Request 3: plain-click a dataset WITHIN a multi-selection switches which dataset's
fit fills the param tables (without collapsing the selection); if that dataset has no
fit, the tables are left as-is and a no-fit indicator (banner + table tint) appears.

Tests the host logic _set_detail_dataset / indicator directly (the DataListWidget
mouse override just routes plain within-selection clicks to it).
"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
os.chdir(ROOT); sys.path.insert(0, ROOT)
try:
    from xraydb import XrayDB as _R
except Exception:
    _xr = types.ModuleType("xraydb")
    class _XrayDB:
        def __init__(self,*a,**k): pass
        def __getattr__(self,n): return lambda *a,**k: 0
    _xr.XrayDB=_XrayDB; _xr.f0=lambda *a,**k:0; _xr.__path__=[]
    sys.modules["xraydb"]=_xr
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from tests.harness import check, finish

app=QApplication.instance() or QApplication(sys.argv)
import xmodfit, PyQt5.QtWidgets as _QtW
class _MB:
    Ok=Yes=No=0
    warning=question=information=staticmethod(lambda *a,**k:0)
xmodfit.QMessageBox=_MB; _QtW.QMessageBox=_MB
class _Prog:
    def __init__(self,*a,**k): pass
    def wasCanceled(self): return False
    def __getattr__(self,n): return lambda *a,**k: None
xmodfit.QProgressDialog=_Prog
class _FRD:
    def __init__(self,*a,**k): pass
    def exec_(self): return 0
    def show(self): pass
xmodfit.FitResultDialog=_FRD

w=xmodfit.XModFit()
check("data list is the custom DataListWidget", type(w.dataListWidget).__name__=="DataListWidget")

d1=os.path.join(SCRATCH,"ds_d1.txt")
d2=os.path.join(SCRATCH,"ds_d2.txt")
q=np.linspace(0.02,0.30,40)
for p,amp in ((d1,1.0),(d2,2.0)):
    np.savetxt(p, np.c_[q, amp*np.exp(-((q-0.1)**2)/0.002)+0.01, np.full_like(q,0.02)], header="x y yerr")
w.addData(fnames=[d1,d2])
names=[w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR",Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers",Qt.MatchExactly)[0])

# fit ONLY ds0 individually; leave ds1 unfit
w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True)
w.dataset_fits[names[0]].fit_params['qoff'].set(vary=1)
w.fitIterationLineEdit.setText("5")
try: w.doFit()
except Exception: import traceback; traceback.print_exc()
# make sure ds1 is explicitly NOT marked fitted
w.dataListWidget.clearSelection(); w.dataListWidget.item(1).setSelected(True)
if names[1] in w.dataset_fits:
    w.dataset_fits[names[1]]._is_fitted=False
    w.dataset_fits[names[1]].result=None
check("ds0 fitted, ds1 not",
      (w.dataset_fits[names[0]]._is_fitted or w.dataset_fits[names[0]].result is not None)
      and not (getattr(w.dataset_fits[names[1]],'_is_fitted',False) or getattr(w.dataset_fits[names[1]],'result',None)))

# multi-select ds0(fitted)+ds1(unfit): banner shows on FIRST addition; detail kept on
# the FITTED dataset (params not overwritten by the unfit one).
w.dataListWidget.clearSelection()
w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
check("adding an unfit dataset shows the banner immediately", not w.noFitBanner.isHidden())
check("detail kept on the fitted dataset (self.fit is ds0)", w.fit is w.dataset_fits[names[0]])

# plain-click ds0 (fitted) within the selection -> detail switches, banner hidden
w._set_detail_dataset(names[0])
check("detail switched to fitted ds0 (self.fit is ds0's fit)", w.fit is w.dataset_fits[names[0]])
check("sfnames detail is ds0", w.sfnames[-1]==names[0])
check("no indicator for a fitted dataset", w.noFitBanner.isHidden())
check("selection preserved (both still selected)", len(w.dataListWidget.selectedItems())==2)

# plain-click ds1 (unfit) within the selection -> tables NOT updated + indicator shown
_fit_before=w.fit
w._set_detail_dataset(names[1])
check("param fields NOT updated for unfit ds1 (self.fit unchanged)", w.fit is _fit_before)
check("no-fit banner shown", not w.noFitBanner.isHidden())
check("banner names the unfit dataset", os.path.basename(names[1].split('<>')[-1]) in w.noFitBanner.text())
check("param table tinted", w.fixedParamTableWidget.styleSheet()!='' and w.sfitParamTableWidget.styleSheet()!='')

# a normal selection change clears the indicator
w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True)
check("normal selection clears the banner", w.noFitBanner.isHidden())
check("normal selection clears the tint", w.fixedParamTableWidget.styleSheet()=='')

# --- EDP: multi-select skips the UNFIT dataset; single-select still plots it ---
def _find_edp():
    for i in range(w.genParamListWidget.count()):
        t=w.genParamListWidget.item(i).text()
        if '_EDP :' in t and t.split(':',1)[1].strip().startswith('['):
            return w.genParamListWidget.item(i)
    return None
f0=names[0].split('<>')[0]; f1=names[1].split('<>')[0]
w.dataListWidget.clearSelection()
w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)  # ds0 fit, ds1 unfit
e=_find_edp()
if e is not None:
    w.genParamListWidget.clearSelection(); e.setSelected(True)
    plotted=set(w.extra_param_1DplotWidget.selDataNames or [])
    check("multi-select EDP: fitted ds0 IS plotted", any(k.endswith('[%s]'%f0) for k in plotted))
    check("multi-select EDP: unfit ds1 NOT plotted", not any(k.endswith('[%s]'%f1) for k in plotted))
# single-select the UNFIT ds1 -> its EDP DOES plot (initial values for setup)
w.dataListWidget.clearSelection(); w.dataListWidget.item(1).setSelected(True)
e=_find_edp()
if e is not None:
    w.genParamListWidget.clearSelection(); e.setSelected(True)
    check("single-select unfit dataset still plots its EDP",
          len(set(w.extra_param_1DplotWidget.selDataNames or []))>=1)

sys.exit(finish())
