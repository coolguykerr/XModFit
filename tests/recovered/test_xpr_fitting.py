"""Stage-1 part 2: drive XLayersPDB_XPR through XModFit's own Fit_Routines --
leastsq, then emcee -- exactly as the GUI does.
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XPRMAIN = os.path.join(os.path.dirname(REPO), 'XPR-main')
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'Functions'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtWidgets import QApplication          # noqa: E402
_app = QApplication.instance() or QApplication([])

from Fit_Routines import Fit                       # noqa: E402
from Functions import pdb_geometry as pg                  # noqa: E402
from Functions.XRR.XLayersPDB_XPR import XLayersPDB_XPR   # noqa: E402

from tests.harness import check as _hcheck, finish   # noqa: E402


def rrf(fname):
    v = np.genfromtxt(fname, unpack=True)
    a = np.transpose(np.array(v))
    a = a[np.where(a[:, 0] >= 0.026), :][0, :, :]
    return a[:, 0], a[:, 1], a[:, 2]


x, y, e = rrf(os.path.join(XPRMAIN, 'test_data', '2box.rrf'))
best = dict(qoff=-0.0005197557108984115, d1=12.870861541436838, d2=9.400266410116624,
            r1=0.21726162435306018, r2=0.4564847693916873)

print('=== leastsq through Fit_Routines ===')
fit = Fit(XLayersPDB_XPR, x, y=y, yerr=e)
f = fit.func
f.fname = None
f.use_protein = False
f.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
f.__mpar__['Model']['d'] = [0.0, 12.0, 10.0, 1.0]
f.__mpar__['Model']['rho'] = [0.0, 0.20, 0.42, 0.334]
f.init_params()
fit.fit_params = f.params
fit.params = f.__dict__

for k in ('theta', 'phi', 'protins', 'cov'):
    fit.fit_params[k].vary = 0
fit.fit_params['qoff'].vary = 1
for k in ('__Model_d_001', '__Model_d_002', '__Model_rho_001', '__Model_rho_002'):
    fit.fit_params[k].vary = 1

start = np.sum(((y - fit.evaluate()) / e) ** 2)
t0 = time.time()
report, msg = fit.perform_fit(x.min(), x.max(), fit_scale='Linear',
                              fit_method='leastsq', maxiter=1000)
dt = time.time() - t0
chi2 = np.sum(fit.result.residual ** 2)
print('leastsq: %d nfev in %.1f s, chi2 %.1f -> %.1f  (%s)'
      % (fit.result.nfev, dt, start, chi2, msg))
for k in ('qoff', '__Model_d_001', '__Model_d_002', '__Model_rho_001', '__Model_rho_002'):
    print('   %-18s %12.6f' % (k, fit.result.params[k].value))

ref_chi2 = 2 * 1212.2127
print('reference chi2 at the published best fit: %.1f' % ref_chi2)
_hcheck('leastsq reached the published optimum', chi2 < start and chi2 < ref_chi2 * 1.02,
        'chi2 %.1f start %.1f ref %.1f' % (chi2, start, ref_chi2))
for k, ref in (('qoff', best['qoff']), ('__Model_d_001', best['d1']),
               ('__Model_d_002', best['d2']), ('__Model_rho_001', best['r1']),
               ('__Model_rho_002', best['r2'])):
    got = fit.result.params[k].value
    _hcheck('leastsq %s within 2%% of published' % k,
            abs(got - ref) <= 0.02 * max(abs(ref), 1e-3), '%g vs published %g' % (got, ref))

print('\n=== emcee through Fit_Routines (serial) ===')
t0 = time.time()
report2, _ = fit.perform_fit(x.min(), x.max(), fit_scale='Linear', fit_method='emcee',
                             emcee_walkers=20, emcee_steps=40, emcee_burn=0,
                             emcee_cores=1, emcee_thin=1)
print('emcee: %d walkers x 40 steps in %.1f s' % (20, time.time() - t0))
ch = fit.result.flatchain
print('flatchain shape:', ch.shape, 'columns:', list(ch.columns))
print(ch.describe().loc[['mean', 'std']].to_string())
_hcheck('emcee chain shape', ch.shape[0] == 20 * 40, '%s' % (ch.shape,))
_hcheck('emcee lnprob finite', np.isfinite(fit.result.lnprob).all())

print('\n=== emcee with theta/phi varying (orientation seeding path) ===')
fp = os.path.join(XPRMAIN, 'test_data', 'protein_2box.rrf')
xp, yp, ep = rrf(fp)
fit2 = Fit(XLayersPDB_XPR, xp, y=yp, yerr=ep)
g = fit2.func
g.fname = os.path.join(XPRMAIN, 'test_data', 'protein.pdb')
g.use_protein = True
g.vol_calc = 'approx'
g.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
g.__mpar__['Model']['d'] = [0.0, 11.66753477743085, 9.643473851409357, 1.0]
g.__mpar__['Model']['rho'] = [0.0, 0.21702095531305382, 0.4470646192673328, 0.334]
g.init_params()
fit2.fit_params = g.params
fit2.params = g.__dict__
fit2.fit_params['qoff'].value = 0.0002054238403330481
fit2.fit_params['theta'].value = 32.92088491028436
fit2.fit_params['phi'].value = 112.59475254401123
# centroid-referenced protins; upstream's demo pose (handoff/05-xpr-julia.md)
fit2.fit_params['protins'].value = -13.717
fit2.fit_params['cov'].value = 2.20516005726808
for k in ('theta', 'phi', 'protins', 'cov'):
    fit2.fit_params[k].vary = 1

t0 = time.time()
r3, m3 = fit2.perform_fit(xp.min(), xp.max(), fit_scale='Linear',
                          fit_method='leastsq', maxiter=200)
chi2p = np.sum(fit2.result.residual ** 2)
print('protein leastsq: %d nfev in %.1f s, chi2 = %.1f (published %.1f)'
      % (fit2.result.nfev, time.time() - t0, chi2p, 2 * 464.6397))
_hcheck('protein leastsq chi2 finite', np.isfinite(chi2p))

t0 = time.time()
fit2.perform_fit(xp.min(), xp.max(), fit_scale='Linear', fit_method='emcee',
                 emcee_walkers=16, emcee_steps=20, emcee_burn=0, emcee_cores=1,
                 emcee_thin=1)
print('protein emcee (with theta/phi grid seeding): %.1f s' % (time.time() - t0))
ch2 = fit2.result.flatchain
th = ch2['theta'].values
ph = ch2['phi'].values
print('theta range %.1f..%.1f   phi range %.1f..%.1f' %
      (th.min(), th.max(), ph.min(), ph.max()))
_hcheck('theta stays in [0,180] (angle wrapping applied)',
        th.min() >= -1e-9 and th.max() <= 180 + 1e-9)
_hcheck('phi stays in [0,360] (angle wrapping applied)',
        ph.min() >= -1e-9 and ph.max() <= 360 + 1e-9)

sys.exit(finish())
