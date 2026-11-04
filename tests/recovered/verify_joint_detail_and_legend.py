"""(1) Plain click on an already-selected joint dataset within a multi-selection keeps
the selection and just switches the detail (params). Tested via the panel's
_on_joint_detail_click (the JointDatasetList mouse override routes to it).
(2) Legend order: by dataset index, and data < individual fit < joint fit."""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0,ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
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
from tests.harness import check, finish
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

# ---- (2) unit-test the legend-order helper directly ----
w=xmodfit.XModFit()
names_in=['1 [joint:1]','0:x','1:x','0 [joint:0]','0 [indiv:0]','1 [indiv:1]']
ordered=w._legend_order(names_in)
expected=['0:x','0 [indiv:0]','0 [joint:0]','1:x','1 [indiv:1]','1 [joint:1]']
check("legend order: by index, data<indiv<joint", ordered==expected)
check("bottom-panel [N] suffix ordered by index",
      w._legend_order(['EDP [1]','EDP [0]'])==['EDP [0]','EDP [1]'])

# ---- (1) joint within-selection detail click ----
d1=os.path.join(SCRATCH,"jd_d1.txt"); d2=os.path.join(SCRATCH,"jd_d2.txt")
q=np.linspace(0.02,0.30,40)
for p,a in ((d1,1.),(d2,2.)):
    np.savetxt(p,np.c_[q,a*np.exp(-((q-0.1)**2)/0.002)+0.01,np.full_like(q,0.02)],header="x y yerr")
w.addData(fnames=[d1,d2])
names=[w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR",Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers",Qt.MatchExactly)[0])
w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True); w.dataListWidget.item(1).setSelected(True)
w.jointFitPanel.add_selected_datasets(); w.jointFitEnabled=True; w.fitIterationLineEdit.setText("5")
w.run_joint_fit()
jp=w.jointFitPanel
w._overlay_modifier_active=lambda: False
# multi-select both joint datasets
jp.datasetList.clearSelection()
jp.datasetList.item(0).setSelected(True); jp.datasetList.item(1).setSelected(True)
check("joint multi-select: both selected", len(jp.datasetList.selectedItems())==2)
jf=w.jointFit
det_before=jf._current
# plain click within selection on the OTHER dataset -> detail switches, selection kept
other = 0 if jf._current==1 else 1
jp._on_joint_detail_click(jp.datasetList.item(other).text())
check("within-selection detail click keeps the multi-selection",
      len(jp.datasetList.selectedItems())==2)
check("within-selection detail click switches the detail (JointFit._current)",
      jf._current==other)
check("param tables reflect the clicked joint dataset",
      w.fit is jf and jf.objectives[jf._current]['func'] is jf.func)
# legend order in the main plot after the joint overlay
plotted=list(getattr(w.plotWidget,'selDataNames',[]) or [])
order_ok = plotted==w._legend_order(plotted)
check("main-plot legend is in _legend_order", order_ok)

sys.exit(finish())
