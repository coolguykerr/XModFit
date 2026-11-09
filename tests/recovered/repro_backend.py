import os, sys, types as _t
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
_xr = _t.ModuleType('xraydb')
class X:
    def __getattr__(s, n): return lambda *a, **k: None
_xr.XrayDB = X; sys.modules['xraydb'] = _xr

import numpy as np
from tests.harness import check, finish
from PyQt5.QtWidgets import QApplication, QFileDialog
import xmodfit
from Joint_Fit_Routines import JointFit
from lmfit import Parameters as LP

app = QApplication([])
w = xmodfit.XModFit()

# --- build a real 2-dataset joint least-squares fit ---
class Trivial:
    def __init__(self):
        self.params = LP(); self.params.add('scale', value=1.0, vary=True)
        self.x = np.linspace(0.1, 1.0, 20); self.output_params = {}
    def init_params(self): pass
    def y(self): return self.params['scale'].value * np.ones_like(self.x)

jf = JointFit()
x = np.linspace(0.1, 1.0, 20)
for nm in ('A', 'B'):
    m = Trivial(); m.x = x
    jf.add_dataset(nm, func=m, x=x, y=2.0*np.ones_like(x), yerr=0.1*np.ones_like(x))
jf.build_master_params()
jf.perform_fit(0.1, 1.0, fit_method='leastsq')
jf._is_fitted = True; jf._jpar_incomplete = False
w.jointFit = jf
w.jointFitEnabled = True
w.jointFitPanel.spec.add_dataset('A', model=jf.objectives[0]['func'])
w.jointFitPanel.spec.add_dataset('B', model=jf.objectives[1]['func'])

# --- open the joint MCMC window ---
w.confInterval_joint_emcee()
print('joint window open, is_joint:', w._mcmc_is_joint)

# --- inspect funcListWidget current item (the suspect) ---
ci = w.funcListWidget.currentItem()
print('funcListWidget.currentItem():', None if ci is None else ci.text())

# --- mock the save dialog and run startMCMCBackendFile (what "Start MCMC file" does) ---
tmp = os.path.join(SCRATCH, 'joint_test_chain.h5')
if os.path.exists(tmp):
    os.remove(tmp)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (tmp, 'MCMC Files (*.h5)'))
_raised = None
try:
    w.startMCMCBackendFile()
    print('startMCMCBackendFile returned normally')
except Exception as e:
    _raised = e
    import traceback; traceback.print_exc()
    print('startMCMCBackendFile RAISED:', type(e).__name__, e)

print('self.backend after Start MCMC file:', getattr(w, 'backend', 'MISSING'))
print('backend is None?', getattr(w, 'backend', None) is None)

check('joint MCMC window opened as joint', w._mcmc_is_joint is True)
check('startMCMCBackendFile returned normally (joint fit, no current function item)',
      _raised is None, repr(_raised))
check('Start MCMC file attached an HDF backend', getattr(w, 'backend', None) is not None,
      repr(getattr(w, 'backend', None)))
check('the backend writes the chosen file', getattr(getattr(w, 'backend', None),
                                                   'filename', None) == tmp)
sys.exit(finish())
