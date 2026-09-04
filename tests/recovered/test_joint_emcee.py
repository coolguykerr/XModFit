"""Validate emcee on a 2-dataset JointFit (engine-level, serial)."""
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


jf = JointFit()
jf.add_dataset('dsA', func=build_func(), x=XD, y=YD, yerr=ED)
jf.add_dataset('dsB', func=build_func(), x=XD, y=YD, yerr=ED)
jf.build_master_params(shared=VARY)   # share all varying params across both datasets
print("master var params:", [k for k in jf.master_params if jf.master_params[k].vary])

# baseline leastsq
rep, msg = jf.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=100)
print("joint leastsq redchi=%.4f  nvarys=%d" % (jf.result.redchi, jf.result.nvarys))

# emcee
bp = tempfile.mktemp(suffix='.h5')
if os.path.exists(bp):
    os.remove(bp)
rep, msg = jf.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='emcee',
                          emcee_walkers=20, emcee_steps=200, emcee_burn=20, emcee_cores=1,
                          reuse_sampler=False, emcee_thin=1, backend=backends.HDFBackend(bp))
chain = jf.result.chain
print("JOINT emcee OK: chain shape", chain.shape, "finite:", bool(np.all(np.isfinite(chain))))
print("var_names:", list(jf.result.var_names))
fc = jf.result.flatchain
print("flatchain keys:", list(fc.keys()))
for nm in ['theta', 'phi', 'cov']:
    if nm in jf.result.var_names:
        col = np.asarray(fc[nm])
        print("  %-6s median=%.4f std=%.4f" % (nm, np.median(col), np.std(col)))
print("acceptance:", np.mean(jf.fitter.sampler.acceptance_fraction))
