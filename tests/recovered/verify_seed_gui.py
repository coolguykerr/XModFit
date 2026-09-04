import os, sys, types as _t
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r'C:\Users\kerr\Documents\GitHub\XModJointFit')
_xr = _t.ModuleType('xraydb')
class X:
    def __getattr__(s, n): return lambda *a, **k: None
_xr.XrayDB = X; sys.modules['xraydb'] = _xr
import numpy as np
from lmfit import Parameters
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import xmodfit
from Joint_Fit_Routines import JointFit
PI = np.pi

def shape(t, p, x):
    return 1 + 0.3*np.cos(t*PI/180)*np.cos(2*PI*x) + 0.3*np.sin(t*PI/180)*np.cos(p*PI/180)*np.sin(2*PI*x)
class SynPDB:
    def __init__(s):
        s.params = Parameters()
        s.params.add('theta', value=50., min=-np.inf, max=np.inf, vary=True)
        s.params.add('phi', value=130., min=-np.inf, max=np.inf, vary=True)
        s.params.add('scale', value=1., min=-np.inf, max=np.inf, vary=True)
        s.x = np.linspace(0.02, 0.5, 40); s.output_params = {}
    def init_params(s): pass
    def y(s): return s.params['scale'].value*shape(s.params['theta'].value, s.params['phi'].value, s.x)
SynPDB.__name__ = 'XLayersPDB'

def grid_row(widget, sb):
    g = widget.gridLayout
    r, c, rs, cs = g.getItemPosition(g.indexOf(sb))
    return r, c

# --- individual XLayersPDB window: temp enabled, seed-mode disabled ---
w1 = xmodfit.XModFit()
class R: pass
class F:
    def __init__(s): s.result = R(); s.result.nvarys = 3
w1.fit = F(); w1.errorAvailable = False; w1.sfnames = ['0<>C:/d/x.rrf']
w1.funcListWidget.addItem('XLayersPDB'); w1.funcListWidget.setCurrentRow(0)
w1.confInterval_emcee(joint=False)
cw1 = w1.emceeConfIntervalWidget
tr, tc = grid_row(cw1, cw1.seedTempSpinBox)
print('individual: seedTemp at grid(row=%d,col=%d) enabled=%s | seedMode enabled=%s'
      % (tr, tc, cw1.seedTempSpinBox.isEnabled(), cw1.seedModeCombo.isEnabled()))
assert tr == 4, 'temp not at row 4'
assert cw1.seedTempSpinBox.isEnabled() and not cw1.seedModeCombo.isEnabled()

# --- joint both-shared window: temp enabled, seed-mode enabled ---
w2 = xmodfit.XModFit()
jf = JointFit(); x = np.linspace(0.02, 0.5, 40); rng = np.random.RandomState(0)
for i, s in enumerate((1.0, 1.4)):
    m = SynPDB(); m.x = x
    jf.add_dataset('D%d' % i, func=m, x=x, y=s*shape(50, 130, x)+0.01*rng.randn(40), yerr=0.01*np.ones(40))
jf.build_master_params(groups=[{'members': [(0,'theta'),(1,'theta')], 'bare': True, 'name': None},
                               {'members': [(0,'phi'),  (1,'phi')],   'bare': True, 'name': None}])
jf.perform_fit(0.02, 0.5, fit_method='leastsq')
jf._is_fitted = True; jf._jpar_incomplete = False
w2.jointFit = jf; w2.jointFitEnabled = True
w2.confInterval_joint_emcee()
cw2 = w2.emceeConfIntervalWidget
tr2, tc2 = grid_row(cw2, cw2.seedTempSpinBox)
sr2, sc2 = grid_row(cw2, cw2.seedModeCombo)
print('joint both-shared: seedTemp grid(row=%d) enabled=%s | seedMode grid(row=%d) enabled=%s items=%s'
      % (tr2, cw2.seedTempSpinBox.isEnabled(), sr2, cw2.seedModeCombo.isEnabled(),
         [cw2.seedModeCombo.itemText(i) for i in range(cw2.seedModeCombo.count())]))
assert tr2 == 4 and sr2 == 5
assert cw2.seedTempSpinBox.isEnabled() and cw2.seedModeCombo.isEnabled()
print('SEED GUI CONTROLS OK')
