import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
import numpy as np
from lmfit import Parameters
from lmfit.customhdf import CustomHDFBackend
from Joint_Fit_Routines import JointFit
PI = np.pi

def shape(theta, phi, x):
    tr = theta * PI / 180.0; pr = phi * PI / 180.0
    return 1.0 + 0.3 * np.cos(tr) * np.cos(2 * PI * x) + 0.3 * np.sin(tr) * np.cos(pr) * np.sin(2 * PI * x)

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

# two datasets, DIFFERENT true orientations, per-dataset angles (no sharing)
truth = [(40.0, 70.0, 1.0), (120.0, 200.0, 1.6)]
jf = JointFit(); rng = np.random.RandomState(0); x = np.linspace(0.02, 0.5, 60)
for i, (th, ph, s) in enumerate(truth):
    m = SynPDB(); m.x = x
    y = s * shape(th, ph, x) + 0.01 * rng.randn(len(x))
    jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01 * np.ones_like(x))
jf.build_master_params(groups=None)     # everything per-dataset -> d0_*, d1_*
lay = jf.orientation_layout()
print('both_shared:', lay['both_shared'], 'neither_shared:', lay['neither_shared'])
print('columns:', {k: v for k, v in lay['col'].items() if 'theta' in k or 'phi' in k})
assert lay['neither_shared'] and not lay['both_shared']

jf.perform_fit(0.02, 0.5, fit_method='leastsq')

tmp = os.path.join(SCRATCH, 'joint_neither.h5')
if os.path.exists(tmp): os.remove(tmp)
be = CustomHDFBackend(tmp)
NW = 16
jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=NW, emcee_steps=30,
               emcee_cores=1, backend=be, seed_temperature=1.0)
samp = jf.fitter.sampler
ai = list(np.asarray(samp.angleindex).astype(int))
print('sampler:', type(samp).__name__, 'angleindex:', ai)
assert type(samp).__name__ == 'CustomSampler'
assert ai.count(1) == 2 and ai.count(2) == 2, 'expected 2 theta + 2 phi angle cols'

chain = be.get_chain()
vns = list(jf.result.var_names)
def colvals(name):
    return chain[:, :, vns.index(name)]
for name in ('d0_theta', 'd0_phi', 'd1_theta', 'd1_phi'):
    v = colvals(name)
    print('%-9s range [%.1f, %.1f]' % (name, np.nanmin(v), np.nanmax(v)))
# wrapping: every theta in [0,180], every phi in [0,360)
for name in ('d0_theta', 'd1_theta'):
    v = colvals(name); assert np.nanmin(v) >= 0 and np.nanmax(v) <= 180, name
for name in ('d0_phi', 'd1_phi'):
    v = colvals(name); assert np.nanmin(v) >= 0 and np.nanmax(v) < 360, name
# per-dataset seeding put each dataset's theta near ITS OWN truth (unique cos(theta) min)
med0 = np.median(colvals('d0_theta')[0]); med1 = np.median(colvals('d1_theta')[0])
print('seed median theta: d0=%.1f (true 40)  d1=%.1f (true 120)' % (med0, med1))
assert abs(med0 - 40) < 25 and abs(med1 - 120) < 25, (med0, med1)
print('JOINT NEITHER-SHARED SEED OK')
