"""Edge cases for parallel joint MCMC: per-dataset (neither-shared) parallel path,
abort THROUGH the pool (step_callback raises in the main process), and resume+pool.

Run under `if __name__ == '__main__':` (spawn Pool re-imports this module).
"""
import os, sys
sys.path.insert(0, r'C:\Users\kerr\Documents\GitHub\XModJointFit')
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
        self.params.add('theta', value=90.0, min=0.0, max=180.0, vary=True)
        self.params.add('phi',   value=180.0, min=0.0, max=360.0, vary=True)
        self.params.add('scale', value=1.0,  min=-np.inf, max=np.inf, vary=True)
        self.x = np.linspace(0.02, 0.5, 60); self.output_params = {}
    def init_params(self): pass
    def y(self):
        return self.params['scale'].value * shape(self.params['theta'].value,
                                                  self.params['phi'].value, self.x)


SynPDB.__name__ = 'XLayersPDB'


def build_neither():
    truth = [(40.0, 70.0, 1.0), (120.0, 200.0, 1.6)]
    jf = JointFit(); rng = np.random.RandomState(0); x = np.linspace(0.02, 0.5, 60)
    for i, (th, ph, s) in enumerate(truth):
        m = SynPDB(); m.x = x
        y = s * shape(th, ph, x) + 0.01 * rng.randn(len(x))
        jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01 * np.ones_like(x))
    jf.build_master_params(groups=None)
    jf.perform_fit(0.02, 0.5, fit_method='leastsq')
    return jf


def test_neither_parallel():
    jf = build_neither()
    tmp = os.path.join(os.environ.get('TEMP', '.'), 'jpe_neither.h5')
    if os.path.exists(tmp): os.remove(tmp)
    be = CustomHDFBackend(tmp)
    jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=16, emcee_steps=20,
                   emcee_cores=4, backend=be, seed_temperature=1.0)
    samp = jf.fitter.sampler
    ai = list(np.asarray(samp.angleindex).astype(int))
    assert type(samp).__name__ == 'CustomSampler'
    assert getattr(samp, 'pool', None) is not None, 'neither-shared not parallel'
    assert getattr(jf.fitter, '_ext_objective', None) is not None
    assert samp.step_callback is not None
    assert ai.count(1) == 2 and ai.count(2) == 2
    chain = be.get_chain(); vns = list(jf.result.var_names)
    for name in ('d0_theta', 'd1_theta'):
        v = chain[:, :, vns.index(name)]
        assert np.nanmin(v) >= 0 and np.nanmax(v) <= 180, name
    for name in ('d0_phi', 'd1_phi'):
        v = chain[:, :, vns.index(name)]
        assert np.nanmin(v) >= 0 and np.nanmax(v) < 360, name
    med0 = np.median(chain[0, :, vns.index('d0_theta')])
    med1 = np.median(chain[0, :, vns.index('d1_theta')])
    print('[neither cores=4] pool=True wrapped OK; seed theta d0=%.1f (40) d1=%.1f (120)' % (med0, med1))
    assert abs(med0 - 40) < 25 and abs(med1 - 120) < 25
    print('NEITHER-SHARED PARALLEL OK')


def test_abort_through_pool():
    jf = build_neither()
    # perform_fit resets self.fit_abort, so trigger the abort mid-run: the Minimizer's
    # iter_cb is bound to jf.callback; return truthy after a few steps -> step_callback
    # (main process) raises AbortFitException -> caught -> result.aborted.
    state = {'n': 0}
    def aborting_callback(params, iterations, residual, fit_scale):
        state['n'] += 1
        return True if state['n'] >= 3 else None
    jf.callback = aborting_callback
    tmp = os.path.join(os.environ.get('TEMP', '.'), 'jpe_abort.h5')
    if os.path.exists(tmp): os.remove(tmp)
    be = CustomHDFBackend(tmp)
    jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=24, emcee_steps=50,
                   emcee_cores=4, backend=be, seed_temperature=1.0)
    res = jf.result
    assert getattr(res, 'aborted', False), 'abort flag not honoured through the pool'
    print('[abort cores=4] result.aborted=%s (clean abort via main-process step_callback)' % res.aborted)
    print('ABORT-THROUGH-POOL OK')


def test_resume_pool():
    jf = build_neither()
    tmp = os.path.join(os.environ.get('TEMP', '.'), 'jpe_resume.h5')
    if os.path.exists(tmp): os.remove(tmp)
    be = CustomHDFBackend(tmp)
    jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=24, emcee_steps=15,
                   emcee_cores=4, backend=be, seed_temperature=1.0)
    n1 = be.get_chain().shape[0]
    # resume: fresh backend on same file, reuse_sampler=True, still parallel
    be2 = CustomHDFBackend(tmp)
    jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=24, emcee_steps=10,
                   emcee_cores=4, backend=be2, reuse_sampler=True, seed_temperature=1.0)
    n2 = be2.get_chain().shape[0]
    samp = jf.fitter.sampler
    chain = be2.get_chain(); vns = list(jf.result.var_names)
    ph = chain[:, :, vns.index('d0_phi')]
    print('[resume cores=4] chain grew %d -> %d, pool=%s, phi[%.1f,%.1f]'
          % (n1, n2, getattr(samp, 'pool', None) is not None, np.nanmin(ph), np.nanmax(ph)))
    assert n2 == n1 + 10, 'resume did not append (%d -> %d)' % (n1, n2)
    assert getattr(samp, 'pool', None) is not None, 'resume lost the pool'
    assert np.nanmin(ph) >= 0 and np.nanmax(ph) < 360, 'wrapping broke across resume'
    print('RESUME+POOL OK')


if __name__ == '__main__':
    test_neither_parallel()
    test_abort_through_pool()
    test_resume_pool()
    print('\nALL EDGE-CASE TESTS PASSED')
