"""No-fit banner/tint lifecycle across the individual<->joint transition:
  * individual multi-select with an unfit dataset -> banner+tint;
  * selecting a JOINT dataset (pure joint view) clears banner+tint (both directions);
  * order-independent: joint-first then individual-multi still shows the banner.
"""
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
class _F:
    def __init__(s,*a,**k): pass
    def exec_(s): return 0
    def show(s): pass
xmodfit.FitResultDialog=_F
w=xmodfit.XModFit()
d1=os.path.join(os.environ.get("TEMP",ROOT),"bl_d1.txt"); d2=os.path.join(os.environ.get("TEMP",ROOT),"bl_d2.txt")
q=np.linspace(0.02,0.30,40)
for p,a in ((d1,1.),(d2,2.)):
    np.savetxt(p,np.c_[q,a*np.exp(-((q-0.1)**2)/0.002)+0.01,np.full_like(q,0.02)],header="x y yerr")
w.addData(fnames=[d1,d2])
names=[w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR",Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers",Qt.MatchExactly)[0])
w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True)
w.dataset_fits[names[0]].fit_params['qoff'].set(vary=1); w.fitIterationLineEdit.setText("5")
try: w.doFit()
except Exception: import traceback; traceback.print_exc()
w.dataListWidget.clearSelection(); w.dataListWidget.item(1).setSelected(True)
w.dataset_fits[names[1]]._is_fitted=False; w.dataset_fits[names[1]].result=None
w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
w.jointFitPanel.add_selected_datasets(); w.jointFitEnabled=True; w.fitIterationLineEdit.setText("5")
w.run_joint_fit()
jp=w.jointFitPanel
w._overlay_modifier_active=lambda: False
def shown(): return (not w.noFitBanner.isHidden())
def tinted(): return w.fixedParamTableWidget.styleSheet()!='' and w.sfitParamTableWidget.styleSheet()!=''

# Scenario 1: individual multi (unfit) -> banner; then joint clears it
w.dataListWidget.clearSelection()
w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
check("S1 individual multi (unfit): banner shown", shown() and tinted())
jp.datasetList.clearSelection(); jp.datasetList.item(0).setSelected(True)
check("S1 selecting a joint dataset clears banner", not shown())
check("S1 selecting a joint dataset clears tint", not tinted())

# Scenario 2 (converse): joint first -> then individual multi (unfit) shows banner
jp.datasetList.clearSelection(); jp.datasetList.item(0).setSelected(True)
check("S2 joint view: no banner", not shown() and not tinted())
w.dataListWidget.clearSelection()
w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
check("S2 individual multi after joint: banner shown", shown() and tinted())

print("\nRESULT:", "ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
sys.exit(0 if _OK["ok"] else 1)
