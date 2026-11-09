import os, sys, types as _t
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
_xr = _t.ModuleType('xraydb')
class X:
    def __getattr__(s, n): return lambda *a, **k: None
_xr.XrayDB = X; sys.modules['xraydb'] = _xr

import numpy as np
from PyQt5.QtWidgets import QApplication, QListWidgetItem
from lmfit import Parameters, Minimizer
import xmodfit
from Joint_Fit_Routines import JointFit

app = QApplication([])
w = xmodfit.XModFit()

# ---- fake an INDIVIDUAL fit + result (what an individual MCMC window needs) ----
class FakeResult:
    pass
def resid(p):
    return np.array([p['a'].value - 1.0, p['b'].value - 2.0])
pars = Parameters(); pars.add('a', value=0.5); pars.add('b', value=1.0)
mini = Minimizer(resid, pars)
indiv_res = mini.minimize(method='leastsq')

class FakeIndivFit:
    def __init__(self, res): self.result = res
w.fit = FakeIndivFit(indiv_res)
w.errorAvailable = False
w.sfnames = ['0<>C:/data/indiv.rrf']
# give funcListWidget a current item (the MCMC window reads its text). Signals are
# blocked: with no category selected, functionChanged would raise AttributeError
# (curr_category unset) inside the slot and replace the faked w.fit.
w.funcListWidget.blockSignals(True)
w.funcListWidget.addItem('XLayers'); w.funcListWidget.setCurrentRow(0)
w.funcListWidget.blockSignals(False)

# ---- open the INDIVIDUAL MCMC window ----
w.confInterval_emcee(joint=False)
iw = w.emceeConfIntervalWidget
print('individual window visible:', iw.isVisible(), 'is_joint:', w._mcmc_is_joint)
n_indiv_tabs = iw.tabWidget.count()
print('individual tabs:', n_indiv_tabs)
assert w._mcmc_is_joint is False

# ---- now build a REAL 2-dataset joint fit (least-squares) ----
# minimal trivial model class with the JointFit contract (params, x, y())
from lmfit import Parameters as LP
class Trivial:
    def __init__(self):
        self.params = LP(); self.params.add('scale', value=1.0, vary=True)
        self.x = np.linspace(0.1, 1.0, 20)
        self.output_params = {}
    def init_params(self):
        pass
    def y(self):
        return self.params['scale'].value * np.ones_like(self.x)

jf = JointFit()
x = np.linspace(0.1, 1.0, 20)
for nm in ('A', 'B'):
    m = Trivial(); m.x = x
    jf.add_dataset(nm, func=m, x=x, y=2.0*np.ones_like(x), yerr=0.1*np.ones_like(x))
jf.build_master_params()
rep, msg = jf.perform_fit(0.1, 1.0, fit_method='leastsq')
jf._is_fitted = True
jf._jpar_incomplete = False
w.jointFit = jf
# put 2 datasets into the panel spec so the title/ranges code has something
w.jointFitEnabled = True
w.jointFitPanel.spec.add_dataset('A', model=jf.objectives[0]['func'])
w.jointFitPanel.spec.add_dataset('B', model=jf.objectives[1]['func'])

# ---- click the joint MCMC button path (individual window STILL open) ----
w.confInterval_joint_emcee()
jw = w.emceeConfIntervalWidget
print('after joint click: is_joint:', w._mcmc_is_joint)
print('individual window still visible:', iw.isVisible())
print('new (joint) window is different object:', jw is not iw)
print('joint window visible:', jw.isVisible())
print('joint tabs:', jw.tabWidget.count(), '(has augment:', hasattr(jw, 'jointSubsetList'), ')')
assert w._mcmc_is_joint is True, 'flag not switched to joint'
assert jw is not iw, 'joint window did not replace individual'
assert hasattr(jw, 'jointSubsetList'), 'joint tabs not augmented'
assert not iw.isVisible(), 'old individual window still visible'
print('SWITCH OK')
