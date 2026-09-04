"""Sequential multi-fit: (1) the multi-selection PERSISTS after fitting; (2) each
dataset is fit from its OWN current params (distinct starting qoff -> distinct fitted
results); (3) both are marked fitted."""
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
    def exec_(s): return 0        # accept fit result dialog (single-fit path only)
    def show(s): pass
xmodfit.FitResultDialog=_FRD
w=xmodfit.XModFit()
d1=os.path.join(os.environ.get("TEMP",ROOT),"mf_d1.txt"); d2=os.path.join(os.environ.get("TEMP",ROOT),"mf_d2.txt")
q=np.linspace(0.02,0.30,40)
for p,a in ((d1,1.),(d2,2.)):
    np.savetxt(p,np.c_[q,a*np.exp(-((q-0.1)**2)/0.002)+0.01,np.full_like(q,0.02)],header="x y yerr")
w.addData(fnames=[d1,d2])
names=[w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR",Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers",Qt.MatchExactly)[0])
# give each dataset its OWN model with a distinct starting qoff + vary
for i,val in ((0,0.05),(1,0.15)):
    w.dataListWidget.clearSelection(); w.dataListWidget.item(i).setSelected(True)
    dk=names[i]
    f=w.dataset_fits[dk]
    f.fit_params['qoff'].set(value=val, vary=1)
    f.func.__dict__['qoff']=val
start0=w.dataset_fits[names[0]].fit_params['qoff'].value
start1=w.dataset_fits[names[1]].fit_params['qoff'].value
check("distinct starting qoff per dataset", abs(start0-start1)>1e-6)
# multi-select both and press Fit (least-squares)
w.dataListWidget.clearSelection()
w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
w.fitMethodComboBox.setCurrentText([k for k in w.fitMethods if w.fitMethods[k]=='leastsq'][0]) if hasattr(w,'fitMethods') else None
w.fitIterationLineEdit.setText("50")
try:
    w.doFit()
except Exception:
    import traceback; traceback.print_exc()
sel_after=[it.text() for it in w.dataListWidget.selectedItems()]
check("multi-selection PERSISTS after fitting", set(sel_after)==set(names))
check("both datasets marked fitted",
      (w.dataset_fits[names[0]]._is_fitted or w.dataset_fits[names[0]].result is not None)
      and (w.dataset_fits[names[1]]._is_fitted or w.dataset_fits[names[1]].result is not None))
# each fit started from its own params -> results are independent (not identical)
r0=w.dataset_fits[names[0]].fit_params['qoff'].value if w.dataset_fits[names[0]].result is None else w.dataset_fits[names[0]].result.params['qoff'].value
r1=w.dataset_fits[names[1]].fit_params['qoff'].value if w.dataset_fits[names[1]].result is None else w.dataset_fits[names[1]].result.params['qoff'].value
print("fitted qoff: ds0=%.5f ds1=%.5f (started %.3f / %.3f)"%(r0,r1,start0,start1))
check("each dataset fit independently (results differ)", abs(r0-r1)>1e-9)
print("\nRESULT:", "ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
sys.exit(0 if _OK["ok"] else 1)
