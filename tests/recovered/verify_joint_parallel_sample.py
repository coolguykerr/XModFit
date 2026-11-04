"""Validate PARALLEL joint XLayersPDB MCMC sampling (both-shared orientation).

Must run under an `if __name__ == '__main__':` guard: emcee_cores>1 triggers a
spawn Pool that re-imports this script in each worker -- without the guard the
re-import re-runs the fit -> fork bomb. SynPDB is defined at MODULE level so the
spawn workers can import + unpickle the shipped model instances.

Env: C:\\Users\\kerr\\.conda\\envs\\cmc-py39\\python.exe  (QT not needed here).
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
import numpy as np
from lmfit import Parameters
from lmfit.customhdf import CustomHDFBackend
from Joint_Fit_Routines import JointFit

PI = np.pi


class SynPDB:
    """Synthetic 'XLayersPDB'-like model with a minimum near (theta=40, phi=70)."""
    def __init__(self):
        self.params = Parameters()
        self.params.add('theta', value=40.0, min=0.0, max=180.0, vary=True)
        self.params.add('phi',   value=70.0, min=0.0, max=360.0, vary=True)
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


SynPDB.__name__ = 'XLayersPDB'   # look like a PDB model to the orientation detection


def truth(x, s):
    shape = 1.0 + 0.3*np.cos(2*PI*x)*np.cos(0.0) + 0.3*np.sin(2*PI*x)*np.cos(0.0)
    return s * shape


def build_jf():
    jf = JointFit()
    rng = np.random.RandomState(0)
    x = np.linspace(0.02, 0.5, 60)
    for i, s in enumerate((1.0, 1.7)):
        m = SynPDB(); m.x = x
        y = truth(x, s) + 0.01*rng.randn(len(x))
        jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01*np.ones_like(x))
    groups = [{'members': [(0, 'theta'), (1, 'theta')], 'bare': True, 'name': None},
              {'members': [(0, 'phi'),   (1, 'phi')],   'bare': True, 'name': None}]
    jf.build_master_params(groups=groups)
    jf.perform_fit(0.02, 0.5, fit_method='leastsq')   # populate jf.result
    return jf


def test_bitforbit():
    """JointLogProb(theta) == -0.5*sum(JointFit.residual(params)**2) for the no-cov branch."""
    import pdb_joint_logprob as pjl
    jf = build_jf()
    jf._set_ranges(0.02, 0.5)
    ep = jf.result.params.copy()
    var_names = [k for k in ep if ep[k].vary]
    bounds = [(ep[k].min, ep[k].max) for k in var_names]
    models = [o["func"] for o in jf.objectives]
    datas = [(o["x"][o["imin"]:o["imax"]+1], o["y"][o["imin"]:o["imax"]+1],
              o["yerr"][o["imin"]:o["imax"]+1]) for o in jf.objectives]
    lp = pjl.JointLogProb(var_names, bounds, 'Linear', True, jf.param_map, {},
                          ep.copy(), has_lnsigma=False, models=models, datas=datas)
    rng = np.random.RandomState(3)
    worst = 0.0
    for _ in range(8):
        theta = np.array([rng.uniform(max(lo, -5.0), min(hi, 5.0) if hi > 5.0 else hi)
                          if not np.isfinite(lo) or not np.isfinite(hi)
                          else rng.uniform(lo, hi) for (lo, hi) in bounds])
        # reference via JointFit.residual
        p = ep.copy()
        for n, v in zip(var_names, theta):
            p[n].value = v
        p.update_constraints()
        ref = -0.5 * np.sum(np.asarray(jf.residual(p, 'Linear'))**2)
        got = lp(theta)
        worst = max(worst, abs(ref - got))
    print('bit-for-bit worst |JointLogProb - (-0.5*sum resid^2)| = %.3e' % worst)
    assert worst < 1e-8, 'JointLogProb does not match JointFit.residual'
    print('BIT-FOR-BIT OK')


def run_sample(cores, tag, steps=25, nw=12):
    jf = build_jf()
    tmp = os.path.join(SCRATCH, 'jps_%s.h5' % tag)
    if os.path.exists(tmp):
        os.remove(tmp)
    be = CustomHDFBackend(tmp)
    t0 = time.time()
    rep, msg = jf.perform_fit(0.02, 0.5, fit_method='emcee', emcee_walkers=nw,
                              emcee_steps=steps, emcee_cores=cores, backend=be,
                              seed_temperature=1.0)
    dt = time.time() - t0
    samp = jf.fitter.sampler
    pool = getattr(samp, 'pool', None)
    ext = getattr(jf.fitter, '_ext_objective', None)
    chain = be.get_chain()
    vns = list(jf.result.var_names)
    ti = vns.index('theta'); pi = vns.index('phi')
    th = chain[:, :, ti]; ph = chain[:, :, pi]
    nfev = int(np.asarray(jf.fitter.result.nfev).ravel()[-1]) if hasattr(jf.fitter, 'result') else -1
    print('[%s cores=%d] sampler=%s pool=%s ext_obj=%s chain=%s  theta[%.1f,%.1f] phi[%.1f,%.1f] nfev=%s t=%.1fs'
          % (tag, cores, type(samp).__name__, pool is not None, ext is not None,
             chain.shape, np.nanmin(th), np.nanmax(th), np.nanmin(ph), np.nanmax(ph), nfev, dt))
    assert type(samp).__name__ == 'CustomSampler', 'expected CustomSampler for orient-seed'
    assert chain.shape[0] == steps, 'chain not fully populated'
    assert np.nanmin(th) >= 0.0 and np.nanmax(th) <= 180.0, 'theta not folded'
    assert np.nanmin(ph) >= 0.0 and np.nanmax(ph) < 360.0, 'phi not wrapped'
    return samp, pool, ext, nfev


def test_parallel():
    samp, pool, ext, nfev = run_sample(4, 'par')
    assert pool is not None, 'parallel run has no pool'
    assert ext is not None, '_ext_objective not attached for parallel joint'
    assert samp.step_callback is not None, 'step_callback (progress/abort) not wired'
    assert nfev > 0, 'per-step nfev not incremented in main process'
    print('PARALLEL JOINT SAMPLING OK')


def test_serial_regression():
    samp, pool, ext, nfev = run_sample(1, 'ser')
    assert pool is None, 'serial run unexpectedly got a pool'
    assert ext is None, 'serial run unexpectedly attached _ext_objective'
    print('SERIAL REGRESSION OK (unchanged: no pool, no ext objective)')


if __name__ == '__main__':
    test_bitforbit()
    test_serial_regression()
    test_parallel()
    print('\nALL JOINT-PARALLEL-SAMPLING TESTS PASSED')
