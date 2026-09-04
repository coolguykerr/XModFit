import os, sys
sys.path.insert(0, r'C:\Users\kerr\Documents\GitHub\XModJointFit')
import numpy as np
from lmfit import Parameters
from lmfit.customhdf import CustomHDFBackend
from Joint_Fit_Routines import JointFit
PI = np.pi

def shape(theta, phi, x):
    tr = theta * PI / 180.0; pr = phi * PI / 180.0
    return (1.0 + 0.3 * np.cos(tr) * np.cos(2 * PI * x)
            + 0.3 * np.sin(tr) * np.cos(pr) * np.sin(2 * PI * x)
            + 0.3 * np.sin(tr) * np.sin(pr) * np.sin(4 * PI * x))   # cos+sin(phi) -> unique phi

class SynPDB:
    def __init__(self):
        self.params = Parameters()
        self.params.add('theta', value=90.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('phi',   value=180.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('scale', value=1.0,  min=-np.inf, max=np.inf, vary=True)
        self.x = np.linspace(0.02, 0.5, 60); self.output_params = {}
    def init_params(self): pass
    def y(self):
        return self.params['scale'].value * shape(self.params['theta'].value,
                                                  self.params['phi'].value, self.x)
SynPDB.__name__ = 'XLayersPDB'

# SHARED theta (both true 60), PER-DATASET phi (70 vs 220)
truth = [(60.0, 70.0, 1.0), (60.0, 220.0, 1.6)]
jf = JointFit(); rng = np.random.RandomState(0); x = np.linspace(0.02, 0.5, 60)
for i, (th, ph, s) in enumerate(truth):
    m = SynPDB(); m.x = x
    y = s * shape(th, ph, x) + 0.01 * rng.randn(len(x))
    jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01 * np.ones_like(x))
groups = [{'members': [(0, 'theta'), (1, 'theta')], 'bare': True, 'name': None}]  # share theta only
jf.build_master_params(groups=groups)
lay = jf.orientation_layout()
print('both_shared:', lay['both_shared'], 'neither_shared:', lay['neither_shared'])
angle_cols = {k: v for k, v in lay['col'].items() if 'theta' in k or 'phi' in k}
print('angle cols:', angle_cols)
assert not lay['both_shared'] and not lay['neither_shared']   # mixed
assert 'theta' in lay['col'] and 'd0_phi' in lay['col'] and 'd1_phi' in lay['col']

jf.perform_fit(0.02, 0.5, fit_method='leastsq')
tmp = os.path.join(os.environ.get('TEMP', '.'), 'joint_mixed.h5')
if os.path.exists(tmp): os.remove(tmp)
be = CustomHDFBackend(tmp)
NW = 16
jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=NW, emcee_steps=30,
               emcee_cores=1, backend=be, seed_temperature=1.0)
samp = jf.fitter.sampler
ai = list(np.asarray(samp.angleindex).astype(int))
print('sampler:', type(samp).__name__, 'angleindex:', ai, 'anglepairs:', samp.anglepairs)
assert type(samp).__name__ == 'CustomSampler'
assert ai.count(1) == 1 and ai.count(2) == 2, 'expected 1 shared theta + 2 phi'
# anglepairs: both datasets pair their phi with the SAME shared theta col
tcol = lay['col']['theta']
assert all(tp[0] == tcol for tp in samp.anglepairs), samp.anglepairs

chain = be.get_chain(); vns = list(jf.result.var_names)
def cv(name): return chain[:, :, vns.index(name)]
for name in ('theta', 'd0_phi', 'd1_phi'):
    v = cv(name); print('%-8s range [%.1f, %.1f]' % (name, np.nanmin(v), np.nanmax(v)))
# wrapping
assert np.nanmin(cv('theta')) >= 0 and np.nanmax(cv('theta')) <= 180
for name in ('d0_phi', 'd1_phi'):
    v = cv(name); assert np.nanmin(v) >= 0 and np.nanmax(v) < 360, name
# seeding: shared theta ~60, phi_0 ~70, phi_1 ~220 (unique-phi model)
mt = np.median(cv('theta')[0]); m0 = np.median(cv('d0_phi')[0]); m1 = np.median(cv('d1_phi')[0])
print('seed medians: theta=%.1f (60)  d0_phi=%.1f (70)  d1_phi=%.1f (220)' % (mt, m0, m1))
assert abs(mt - 60) < 25, mt
assert abs(m0 - 70) < 35 and abs(m1 - 220) < 35, (m0, m1)
print('JOINT MIXED (shared-theta) SEED OK')
