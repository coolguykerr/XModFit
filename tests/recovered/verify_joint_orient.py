import os, sys
sys.path.insert(0, r'C:\Users\kerr\Documents\GitHub\XModJointFit')
import numpy as np
from lmfit import Parameters
from lmfit.customhdf import CustomHDFBackend
from Joint_Fit_Routines import JointFit

PI = np.pi

class SynPDB:
    """Synthetic 'XLayersPDB'-like model: y depends on theta/phi (periodic) + a scale
    nuisance, with a minimum near (theta=40, phi=70)."""
    def __init__(self):
        self.params = Parameters()
        self.params.add('theta', value=40.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('phi',   value=70.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('scale', value=1.0,  min=-np.inf, max=np.inf, vary=True)
        self.x = np.linspace(0.02, 0.5, 60)
        self.output_params = {}
    def init_params(self):
        pass
    def y(self):
        th = self.params['theta'].value; ph = self.params['phi'].value; s = self.params['scale'].value
        shape = 1.0 + 0.3 * np.cos(2*PI*self.x) * np.cos((th - 40.0)*PI/180.0) \
                    + 0.3 * np.sin(2*PI*self.x) * np.cos((ph - 70.0)*PI/180.0)
        return s * shape
# make the class look like a PDB model to the has_angles/_pdb detection
SynPDB.__name__ = 'XLayersPDB'

def truth(x, s):
    shape = 1.0 + 0.3*np.cos(2*PI*x)*np.cos(0.0) + 0.3*np.sin(2*PI*x)*np.cos(0.0)
    return s * shape

jf = JointFit()
rng = np.random.RandomState(0)
x = np.linspace(0.02, 0.5, 60)
for i, s in enumerate((1.0, 1.7)):
    m = SynPDB(); m.x = x
    y = truth(x, s) + 0.01*rng.randn(len(x))
    jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01*np.ones_like(x))
# share theta & phi (bare -> master keys stay 'theta'/'phi')
groups = [{'members': [(0, 'theta'), (1, 'theta')], 'bare': True, 'name': None},
          {'members': [(0, 'phi'),   (1, 'phi')],   'bare': True, 'name': None}]
jf.build_master_params(groups=groups)
print('master var_names:', [k for k in jf.master_params if jf.master_params[k].vary])

# baseline least-squares to populate jf.result
jf.perform_fit(0.02, 0.5, fit_method='leastsq')
assert 'theta' in jf.result.params and 'phi' in jf.result.params

# ---- joint emcee with orientation seeding (shared theta/phi) ----
tmp = os.path.join(os.environ.get('TEMP', '.'), 'joint_orient.h5')
if os.path.exists(tmp): os.remove(tmp)
be = CustomHDFBackend(tmp)
NW = 12
rep, msg = jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=NW,
                          emcee_steps=30, emcee_cores=1, backend=be, seed_temperature=1.0)
samp = jf.fitter.sampler
print('sampler class:', type(samp).__name__)
print('sampler angleindex:', list(np.asarray(getattr(samp, 'angleindex', [])).astype(int)))
# orientation seeding path -> CustomSampler with theta=1, phi=2 in angleindex
assert type(samp).__name__ == 'CustomSampler', 'joint orient-seed did not use CustomSampler'
ai = list(np.asarray(samp.angleindex).astype(int))
assert 1 in ai and 2 in ai, 'angleindex missing theta/phi wrap for joint'

chain = be.get_chain()
vns = list(jf.result.var_names)
ti = vns.index('theta'); pi = vns.index('phi')
th = chain[:, :, ti]; ph = chain[:, :, pi]
print('theta range: [%.1f, %.1f]  phi range: [%.1f, %.1f]'
      % (np.nanmin(th), np.nanmax(th), np.nanmin(ph), np.nanmax(ph)))
assert np.nanmin(ph) >= 0.0 and np.nanmax(ph) < 360.0, 'phi not wrapped in joint'
assert np.nanmin(th) >= 0.0 and np.nanmax(th) <= 180.0, 'theta not folded in joint'
print('chain shape:', chain.shape)
print('JOINT ORIENT SEED OK')
