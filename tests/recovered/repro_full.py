import os, sys, types as _t
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
_xr = _t.ModuleType('xraydb')
class X:
    def __getattr__(s, n): return lambda *a, **k: None
_xr.XrayDB = X; sys.modules['xraydb'] = _xr

import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import xmodfit
from Joint_Fit_Routines import JointFit
from lmfit import Parameters as LP

app = QApplication([])
_warns = []
QMessageBox.warning = staticmethod(lambda *a, **k: _warns.append(a[2] if len(a) > 2 else ''))
w = xmodfit.XModFit()

class Trivial:
    def __init__(self):
        self.params = LP(); self.params.add('scale', value=1.0, vary=True, min=0.0, max=5.0)
        self.x = np.linspace(0.1, 1.0, 20); self.output_params = {}
    def init_params(self): pass
    def y(self): return self.params['scale'].value * np.ones_like(self.x)

jf = JointFit()
x = np.linspace(0.1, 1.0, 20)
rng = np.random.default_rng(0)
for nm in ('A', 'B'):
    m = Trivial(); m.x = x
    jf.add_dataset(nm, func=m, x=x, y=2.0 * np.ones_like(x) + 0.01 * rng.normal(size=20),
                   yerr=0.1 * np.ones_like(x))
jf.build_master_params()
jf.perform_fit(0.1, 1.0, fit_method='leastsq')
jf._is_fitted = True; jf._jpar_incomplete = False
w.jointFit = jf
w.jointFitEnabled = True
w.jointFitPanel.spec.add_dataset('A', model=jf.objectives[0]['func'])
w.jointFitPanel.spec.add_dataset('B', model=jf.objectives[1]['func'])
# per-dataset ranges: give A and B different windows
w.jointFitPanel.spec.datasets[0].fit_xmin = 0.15; w.jointFitPanel.spec.datasets[0].fit_xmax = 0.9
w.xminmaxLineEdit.setText('0.1:1.0')

# open joint MCMC window
w.confInterval_joint_emcee()
cw = w.emceeConfIntervalWidget
print('backend before Start MCMC file:', getattr(w, 'backend', None))

# "Start MCMC file"
tmp = os.path.join(SCRATCH, 'joint_full_chain.h5')
if os.path.exists(tmp):
    os.remove(tmp)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (tmp, 'MCMC Files (*.h5)'))
w.startMCMCBackendFile()
print('backend after Start MCMC file:', type(getattr(w, 'backend', None)).__name__)
assert w.backend is not None

# set a short run
cw.MCMCWalkerLineEdit.setText('6')
cw.MCMCStepsLineEdit.setText('40')
cw.MCMCBurnLineEdit.setText('0')
cw.MCMCThinLineEdit.setText('1')
cw.ParallelCoresLineEdit.setText('1')

# "Start Sampling"
w.start_emcee_sampling()
print('warnings during sampling:', _warns)
assert not _warns, 'Start Sampling warned: %s' % _warns
print('result chain shape:', getattr(w.fit.result, 'chain', np.array([])).shape)

# now compute a subset corner plot + summary (the joint viz)
w._set_joint_subset_checks('all')
w._plot_joint_subset()
print('plotted:', w._corner_names_cache)
print('CI table rows:', cw.jointCITable.rowCount())
assert cw.jointCITable.rowCount() == len(w._corner_names_cache)
# per-dataset range actually applied to objective A
print('obj A imin/imax:', jf.objectives[0]['imin'], jf.objectives[0]['imax'],
      'x=', round(x[jf.objectives[0]['imin']], 3), round(x[jf.objectives[0]['imax']], 3))
assert jf.objectives[0]['imin'] >= 2 and jf.objectives[0]['imax'] <= 17   # window narrowed
assert (jf.objectives[1]['imin'], jf.objectives[1]['imax']) == (0, 19)     # B: scalar range
print('END-TO-END JOINT MCMC OK')
