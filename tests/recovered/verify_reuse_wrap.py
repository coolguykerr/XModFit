import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
import numpy as np
from lmfit import Parameters, Minimizer
from lmfit.customhdf import CustomHDFBackend

def make_params():
    p = Parameters()
    p.add('theta', value=40.0, min=-np.inf, max=np.inf, vary=True)
    p.add('phi',   value=70.0, min=-np.inf, max=np.inf, vary=True)
    p.add('a',     value=1.0,  min=-np.inf, max=np.inf, vary=True)
    return p

def resid(params, scale):
    th = params['theta'].value; ph = params['phi'].value; a = params['a'].value
    dphi = ((ph - 70.0 + 180.0) % 360.0) - 180.0
    return np.array([a - 1.0, 0.1 * (th - 40.0), 0.1 * dphi, 0.05 * (a - 1.0)])

tmp = os.path.join(SCRATCH, 'reuse_wrap.h5')
if os.path.exists(tmp): os.remove(tmp)

NW = 8
# ---- fresh run (seeds + samples; angleindex applied on the fresh path) ----
be = CustomHDFBackend(tmp)
m1 = Minimizer(resid, make_params(), fcn_args=('Linear',), nan_policy='propagate')
r1 = m1.emcee(params=make_params(), steps=20, nwalkers=NW, reuse_sampler=False,
              backend=be, funcname='XLayersPDB', workers=1, progress=False)
print('fresh angleindex:', list(np.asarray(m1.sampler.angleindex).astype(int)))
assert 2 in list(np.asarray(m1.sampler.angleindex).astype(int)), 'fresh missing phi wrap'

# ---- RESUME run: fresh Minimizer (as XModFit does), reuse the same backend ----
be2 = CustomHDFBackend(tmp)
m2 = Minimizer(resid, make_params(), fcn_args=('Linear',), nan_policy='propagate')
r2 = m2.emcee(params=make_params(), steps=20, nwalkers=NW, reuse_sampler=True,
              backend=be2, funcname='XLayersPDB', workers=1, progress=False)
ai = list(np.asarray(m2.sampler.angleindex).astype(int))
print('resume angleindex:', ai)
# THE FIX: the resumed sampler must carry the angle-wrap index (2 = phi, 1 = theta)
assert 2 in ai and 1 in ai, 'RESUME sampler has no angleindex -> phi would not wrap'

# phi in the appended chain must be wrapped into [0, 360)
chain = be2.get_chain()
var_names = list(r2.var_names)
phi_col = var_names.index('phi')
phi = chain[:, :, phi_col]
print('phi range over full chain: [%.2f, %.2f]' % (np.nanmin(phi), np.nanmax(phi)))
assert np.nanmin(phi) >= 0.0 and np.nanmax(phi) < 360.0, 'phi not wrapped to [0,360)'

print('REUSE WRAP OK')
