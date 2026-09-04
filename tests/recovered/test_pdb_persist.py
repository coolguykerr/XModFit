"""Reproduce + verify the per-dataset PDB-file persistence fix (offscreen)."""
import os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(ROOT)
_xr = types.ModuleType("xraydb")
class _XrayDB:
    def __init__(self,*a,**k): pass
    def __getattr__(self,n): return lambda *a,**k: 0
_xr.XrayDB=_XrayDB; _xr.f0=lambda *a,**k:0; _xr.__path__=[]
sys.modules["xraydb"]=_xr
sys.path.insert(0, ROOT)

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
app = QApplication(sys.argv)
import xmodfit
class _MB:
    Ok=Yes=No=0
    warning=question=information=staticmethod(lambda *a,**k:0)
xmodfit.QMessageBox=_MB

w = xmodfit.XModFit()

# two datasets
d1=os.path.join(os.environ["TEMP"],"pp1.txt"); d2=os.path.join(os.environ["TEMP"],"pp2.txt")
q=np.linspace(0.02,0.30,30)
for p,a in ((d1,1.0),(d2,2.0)):
    np.savetxt(p, np.c_[q, a*np.exp(-((q-0.1)**2)/0.002)+0.01, np.full_like(q,0.01)], header="x y yerr")
w.addData(fnames=[d1,d2])
names=[w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]

# two distinct, existing PDBs
PDB1=os.path.join("Data","Pr4.pdb"); PDB2=os.path.join("Data","Nd9.pdb")
assert os.path.exists(PDB1) and os.path.exists(PDB2)

# assign XLayersPDB
w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR",Qt.MatchExactly)[0])
w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayersPDB",Qt.MatchExactly)[0])

def set_pdb(path):
    """simulate the user picking a PDB via the fixed-param file button"""
    w.fit.func.fname = path
    w.update_fixed_parameters()   # rebuild button from live fname (the fix)
    w.update_plot()               # round-trip button -> fname

ok=True
def check(n,c):
    global ok; print(("PASS" if c else "FAIL"),"-",n); ok=ok and c

# ds0 -> PDB1
w.dataListWidget.setCurrentRow(0)
set_pdb(PDB1)
check("ds0 fname set to PDB1", os.path.basename(w.fit.func.fname)=="Pr4.pdb")
# ds1 -> PDB2
w.dataListWidget.setCurrentRow(1)
set_pdb(PDB2)
check("ds1 fname set to PDB2", os.path.basename(w.fit.func.fname)=="Nd9.pdb")
# switch back to ds0 -> must STILL be PDB1 (the bug reverted it to default)
w.dataListWidget.setCurrentRow(0)
check("ds0 remembers PDB1 after switch", os.path.basename(w.fit.func.fname)=="Pr4.pdb")
check("ds0 store holds PDB1", os.path.basename(w.dataset_fits[names[0]].func.fname)=="Pr4.pdb")
# and ds1 still PDB2
w.dataListWidget.setCurrentRow(1)
check("ds1 remembers PDB2 after switch", os.path.basename(w.fit.func.fname)=="Nd9.pdb")

print("\nRESULT:", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
