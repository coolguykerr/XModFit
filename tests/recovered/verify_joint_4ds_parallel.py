import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
import numpy as np
from lmfit import Parameters
PI = np.pi

def shape(theta, phi, x):
    tr = theta*PI/180; pr = phi*PI/180
    return 1.0 + 0.3*np.cos(tr)*np.cos(2*PI*x) + 0.3*np.sin(tr)*np.cos(pr)*np.sin(2*PI*x)

class SynPDB:
    def __init__(self):
        self.params = Parameters()
        for n, v in (('theta', 90.), ('phi', 180.), ('protins', 8.), ('scale', 1.)):
            self.params.add(n, value=v, min=-np.inf, max=np.inf, vary=True)
        self.x = np.linspace(0.02, 0.5, 50); self.output_params = {}
    def init_params(self): pass
    def y(self):
        p = self.params
        return p['scale'].value * (shape(p['theta'].value, p['phi'].value, self.x)
                                   + 0.02 * p['protins'].value * np.cos(6*PI*self.x))

def build():
    from Joint_Fit_Routines import JointFit
    jf = JointFit(); rng = np.random.RandomState(0); x = np.linspace(0.02, 0.5, 50)
    for i, s in enumerate((1.0, 1.5, 0.8, 1.2)):   # 4 datasets
        m = SynPDB(); m.x = x
        y = s*(shape(55, 110, x) + 0.02*8.5*np.cos(6*PI*x)) + 0.01*rng.randn(len(x))
        jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01*np.ones_like(x))
    groups = [{'members': [(k, 'theta') for k in range(4)], 'bare': True, 'name': None},
              {'members': [(k, 'phi')   for k in range(4)], 'bare': True, 'name': None},
              {'members': [(k, 'protins') for k in range(4)], 'bare': True, 'name': None}]
    jf.build_master_params(groups=groups)
    return jf

def main():
    from lmfit.customhdf import CustomHDFBackend
    jf = build()
    jf.perform_fit(0.02, 0.5, fit_method='leastsq')
    print('both_shared:', jf.orientation_layout()['both_shared'])
    tmp = os.path.join(SCRATCH, 'joint_par_scan.h5')
    if os.path.exists(tmp): os.remove(tmp)
    t0 = time.time()
    jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=16, emcee_steps=20,
                   emcee_cores=4, backend=CustomHDFBackend(tmp), seed_temperature=1.0)
    print('parallel-scan both-shared init+sample wall: %.1fs' % (time.time()-t0))
    samp = jf.fitter.sampler
    print('sampler:', type(samp).__name__, 'pool:', samp.pool)   # pool must be None (serial sampling)
    assert type(samp).__name__ == 'CustomSampler'
    assert samp.pool is not None, 'both-shared 4-ds sampling must now be PARALLEL (pool set)'; assert getattr(jf.fitter, '_ext_objective', None) is not None; assert samp.step_callback is not None
    be = CustomHDFBackend(tmp); chain = be.get_chain(); vns = list(jf.result.var_names)
    th = chain[:, :, vns.index('theta')]; ph = chain[:, :, vns.index('phi')]
    print('theta [%.1f,%.1f] phi [%.1f,%.1f]' % (np.nanmin(th), np.nanmax(th), np.nanmin(ph), np.nanmax(ph)))
    assert np.nanmin(th) >= 0 and np.nanmax(th) <= 180 and np.nanmin(ph) >= 0 and np.nanmax(ph) < 360
    print('JOINT 4-DS PARALLEL SAMPLING (shared protins) OK')

if __name__ == '__main__':
    main()
