import os, sys, types as _t
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r'C:\Users\kerr\Documents\GitHub\XModJointFit')
_xr = _t.ModuleType('xraydb')
class X:
    def __getattr__(s, n): return lambda *a, **k: None
_xr.XrayDB = X; sys.modules['xraydb'] = _xr

import numpy as np
from collections import OrderedDict
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt
from PyQt5 import uic
import xmodfit

app = QApplication([])
w = xmodfit.XModFit()

# --- open the joint MCMC window UI shell manually (no real fit needed) ---
w._mcmc_is_joint = True
w.emceeConfIntervalWidget = QWidget()
uic.loadUi('./UI_Forms/EMCEE_ConfInterval_Widget.ui', w.emceeConfIntervalWidget)
w._add_corner_popout_button()
n_tabs_before = w.emceeConfIntervalWidget.tabWidget.count()
w._augment_joint_mcmc_tabs()
n_tabs_after = w.emceeConfIntervalWidget.tabWidget.count()
print('tabs before/after augment:', n_tabs_before, n_tabs_after)
assert n_tabs_after == n_tabs_before + 2
assert hasattr(w.emceeConfIntervalWidget, 'jointSubsetList')
assert hasattr(w.emceeConfIntervalWidget, 'jointCITable')
assert hasattr(w.emceeConfIntervalWidget, 'jointCorrMPLWidget')

# --- synthesize a chain (joint-style master keys) ---
from lmfit import Parameters
nsteps, nwalk = 300, 8
keys = ['theta', 'phi', 'd0_qoff', 'd1_qoff', '__lnsigma']
w.chain_shape = (nsteps, nwalk)
rng = np.random.default_rng(1)
chain = rng.normal(size=(nsteps, nwalk, len(keys)))
w.param_chain = OrderedDict()
for i, k in enumerate(keys):
    w.param_chain[k] = OrderedDict((j, chain[:, j, i]) for j in range(nwalk))
fp = Parameters()
for k in keys:
    if k != '__lnsigma':
        fp.add(k, value=0.0)
w.final_fit_params = fp

# populate the subset list (default structural check) and confirm defaults
w._populate_joint_subset_list()
lst = w.emceeConfIntervalWidget.jointSubsetList
checked = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]
print('subset items:', [lst.item(i).text() for i in range(lst.count())])
print('default checked:', checked)
assert 'theta' in checked and 'phi' in checked
assert 'd0_qoff' not in checked and 'd1_qoff' not in checked   # per-dataset -> unchecked
assert '__lnsigma' not in [lst.item(i).text() for i in range(lst.count())]

# --- drive the subset corner plot + summary through the real widgets ---
w._plot_joint_subset()   # uses the checked subset (theta, phi)
names = w._corner_names_cache
print('plotted subset:', names)
assert names == ['theta', 'phi']
# CI table populated
t = w.emceeConfIntervalWidget.jointCITable
print('CI table rows:', t.rowCount())
assert t.rowCount() == 2
assert t.item(0, 0).text() in ('theta', 'phi')
# correlation heatmap drawn (has an image + colorbar axes)
axn = len(w.emceeConfIntervalWidget.jointCorrMPLWidget.fig.get_axes())
print('heatmap axes:', axn)
assert axn >= 1

# --- plot ALL params, confirm calcMCMCerrorbars stays consistent (4 real params) ---
w._set_joint_subset_checks('all')
w._plot_joint_subset()
print('all plotted:', w._corner_names_cache)
assert w._corner_names_cache == ['theta', 'phi', 'd0_qoff', 'd1_qoff']
assert w.emceeConfIntervalWidget.jointCITable.rowCount() == 4

# --- pop-out renders without error ---
w.popOutCornerPlot()
assert getattr(w, '_cornerPopoutDlgs', None) and len(w._cornerPopoutDlgs) == 1
print('popout dialogs:', len(w._cornerPopoutDlgs))

print('JOINT WINDOW INTEGRATION OK')
