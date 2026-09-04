import os, sys
sys.path.insert(0, r'C:\Users\kerr\Documents\GitHub\XModJointFit')
import numpy as np
from lmfit import Parameters
from lmfit.customhdf import CustomHDFBackend
from Joint_Fit_Routines import JointFit
PI = np.pi

def shape(theta, phi, x):
    tr = theta*PI/180; pr = phi*PI/180
    return 1.0 + 0.3*np.cos(tr)*np.cos(2*PI*x) + 0.3*np.sin(tr)*np.cos(pr)*np.sin(2*PI*x)

class SynPDB:
    def __init__(self):
        self.params = Parameters()
        self.params.add('theta', value=90.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('phi',   value=180.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('scale', value=1.0, min=-np.inf, max=np.inf, vary=True)
        self.x = np.linspace(0.02, 0.5, 50); self.output_params = {}
    def init_params(self): pass
    def y(self):
        p = self.params
        return p['scale'].value * shape(p['theta'].value, p['phi'].value, self.x)
SynPDB.__name__ = 'XLayersPDB'

def build():
    jf = JointFit(); rng = np.random.RandomState(0); x = np.linspace(0.02, 0.5, 50)
    for i, s in enumerate((1.0, 1.5)):
        m = SynPDB(); m.x = x
        y = s*shape(50, 130, x) + 0.01*rng.randn(len(x))     # shared true orientation
        jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01*np.ones_like(x))
    groups = [{'members': [(0,'theta'),(1,'theta')], 'bare': True, 'name': None},
              {'members': [(0,'phi'),  (1,'phi')],   'bare': True, 'name': None}]
    jf.build_master_params(groups=groups)
    jf.perform_fit(0.02, 0.5, fit_method='leastsq')
    return jf

def run(mode):
    jf = build()
    tmp = os.path.join(os.environ.get('TEMP', '.'), 'seedmode_%s.h5' % mode)
    if os.path.exists(tmp): os.remove(tmp)
    jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=12, emcee_steps=20,
                   emcee_cores=1, backend=CustomHDFBackend(tmp), seed_temperature=1.0,
                   seed_mode=mode)
    has_seedpos = getattr(jf.fitter, '_seed_pos', None) is not None
    samp = jf.fitter.sampler
    be = CustomHDFBackend(tmp); chain = be.get_chain(); vns = list(jf.result.var_names)
    th = chain[:, :, vns.index('theta')]; ph = chain[:, :, vns.index('phi')]
    print('mode=%-11s seed_pos_precomputed=%s sampler=%s theta[%.0f,%.0f] phi[%.0f,%.0f] seedmed(th=%.0f,ph=%.0f)'
          % (mode, has_seedpos, type(samp).__name__, np.nanmin(th), np.nanmax(th),
             np.nanmin(ph), np.nanmax(ph), np.median(th[0]), np.median(ph[0])))
    assert type(samp).__name__ == 'CustomSampler'
    assert np.nanmin(th) >= 0 and np.nanmax(th) <= 180 and np.nanmin(ph) >= 0 and np.nanmax(ph) < 360
    return has_seedpos

# individual mode -> precomputed _seed_pos; joint mode -> in-emcee scan (no _seed_pos)
assert run('individual') is True, 'individual should precompute _seed_pos'
assert run('joint') is False, 'joint should use the in-emcee scan (no _seed_pos)'
print('SEED MODE (joint vs individual) OK')
