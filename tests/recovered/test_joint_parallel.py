"""
Validate parallel no-covariance joint MCMC:
  (1) JointLogProb == -0.5*sum(JointFit.residual**2) on random theta (airtight)
  (2) parallel (workers>1) joint emcee completes with a finite chain
Guarded for Windows spawn.
"""
import os, sys, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
from PyQt5.QtWidgets import QApplication
_app = QApplication.instance() or QApplication([])
from importlib import import_module
XLayersPDB = import_module("Functions.XRR.XLayersPDB").XLayersPDB
from Joint_Fit_Routines import JointFit
from emcee import backends
import pdb_joint_logprob as pjl

RRF = r"C:\Users\kerr\Documents\XLayersProtTest\mTIM3 analysis\i2_tim3_1000.rrf"
raw = np.loadtxt(RRF); m = (raw[:, 0] >= 0.026) & (raw[:, 0] <= 0.550)
XD, YD, ED = raw[m, 0], raw[m, 1], raw[m, 2]
MPAR = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                  'd': [0.0, 11.3182806, 8.75928561, 1.0],
                  'rho': [0.0, 0.23396671, 0.46419943, 0.334],
                  'mu': [0.0, 0.0, 0.0, 0.0], 'sig': [3.4, 3.4, 3.4, 3.4]}}
VARY = ['qoff', 'theta', 'phi', 'protins', 'cov',
        '__Model_d_001', '__Model_rho_001', '__Model_d_002', '__Model_rho_002']


def build_func():
    f = XLayersPDB(x=XD, E=10.0, fname='./Data/tim3_hmmm_ext_all_bestframe.pdb', mpar=MPAR,
                   dz=0.5, fslice=0.5, rrf=True, fix_sig=True, qoff=9.83e-05, yscale=1.0,
                   bkg=0.0, theta=40.88, phi=74.08, protins=6.094141, cov=0.1582754)
    for k, v in [('__Model_d_001', 11.3182806), ('__Model_d_002', 8.75928561),
                 ('__Model_rho_001', 0.23396671), ('__Model_rho_002', 0.46419943)]:
        f.params[k].value = v
    for k in f.params:
        f.params[k].set(vary=1 if k in VARY else 0)
    return f


def make_joint():
    jf = JointFit()
    jf.add_dataset('dsA', func=build_func(), x=XD, y=YD, yerr=ED)
    jf.add_dataset('dsB', func=build_func(), x=XD, y=YD, yerr=ED)
    jf.build_master_params(shared=VARY)
    jf.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=100)
    return jf


def main():
    ncpu = os.cpu_count() or 2
    workers = max(2, min(6, ncpu - 2))

    # (1) objective identity vs the REAL JointFit.residual
    jf = make_joint()
    var_names = [k for k in jf.result.params if jf.result.params[k].vary]
    bounds = [(jf.result.params[k].min, jf.result.params[k].max) for k in var_names]
    models = [o["func"] for o in jf.objectives]
    datas = [(o["x"][o["imin"]:o["imax"] + 1], o["y"][o["imin"]:o["imax"] + 1],
              o["yerr"][o["imin"]:o["imax"] + 1]) for o in jf.objectives]
    lp = pjl.JointLogProb(var_names, bounds, 'Linear', True, jf.param_map, {},
                          jf.result.params.copy(), models=models, datas=datas)
    base = np.array([jf.result.params[k].value for k in var_names])
    rng = np.random.RandomState(0)
    maxdiff = 0.0
    for _ in range(25):
        th = base * (1 + 0.03 * rng.randn(len(base)))
        r = jf.residual(_set(jf.result.params, var_names, th), 'Linear')  # real joint residual
        ref = -0.5 * np.sum(np.asarray(r) ** 2)
        got = lp(th)
        maxdiff = max(maxdiff, abs(ref - got))
    print("(1) JointLogProb vs -0.5*sum(JointFit.residual^2): max|diff| = %.3e" % maxdiff)

    # (2) parallel joint emcee end-to-end
    jf2 = make_joint()
    bp = tempfile.mktemp(suffix='.h5')
    if os.path.exists(bp):
        os.remove(bp)
    rep, msg = jf2.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='emcee',
                               emcee_walkers=20, emcee_steps=120, emcee_burn=20,
                               emcee_cores=workers, reuse_sampler=False, emcee_thin=1,
                               backend=backends.HDFBackend(bp))
    used_ext = getattr(jf2.fitter, '_ext_objective', None) is not None
    chain = jf2.result.chain
    print("(2) PARALLEL joint emcee (workers=%d, ext_objective=%s): chain %s finite=%s"
          % (workers, used_ext, chain.shape, bool(np.all(np.isfinite(chain)))))
    fc = jf2.result.flatchain
    for nm in ['theta', 'phi', 'cov']:
        col = np.asarray(fc[nm])
        print("    %-6s median=%.4f std=%.4f" % (nm, np.median(col), np.std(col)))


def _set(params, names, vals):
    for n, v in zip(names, vals):
        params[n].value = v
    params.update_constraints()
    return params


if __name__ == '__main__':
    main()
