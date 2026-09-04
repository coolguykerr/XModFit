"""Check PdbLogProb == -0.5*chi^2 at the two known minima, and that it pickles."""
import os, sys, pickle
import numpy as np
REPO = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(REPO); sys.path.insert(0, REPO)
from importlib import import_module
XLayersPDB = import_module("Functions.XRR.XLayersPDB").XLayersPDB
from pdb_logprob import PdbLogProb

RRF = r"C:\Users\kerr\Documents\XLayersProtTest\mTIM3 analysis\i2_tim3_1000.rrf"
raw = np.loadtxt(RRF)
xall, yall, eall = raw[:, 0], raw[:, 1], raw[:, 2]
m = (xall >= 0.026) & (xall <= 0.550)
xd, yd, ed = xall[m], yall[m], eall[m]

mpar = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                  'd':   [0.0, 11.3182806, 8.75928561, 1.0],
                  'rho': [0.0, 0.23396671, 0.46419943, 0.334],
                  'mu':  [0.0, 0.0, 0.0, 0.0],
                  'sig': [3.4, 3.4, 3.4, 3.4]}}
func = XLayersPDB(x=xd, E=10.0, fname='./Data/tim3_hmmm_ext_all_bestframe.pdb',
                  mpar=mpar, dz=0.5, fslice=0.5, rrf=True, fix_sig=True,
                  qoff=9.830348e-05, yscale=1.0, bkg=0.0,
                  theta=40.88, phi=74.08, protins=6.094141, cov=0.1582754)
func.params['__Model_d_001'].value = 11.3182806
func.params['__Model_d_002'].value = 8.75928561
func.params['__Model_rho_001'].value = 0.23396671
func.params['__Model_rho_002'].value = 0.46419943

var_names = ['qoff', 'theta', 'phi', 'protins', 'cov',
             '__Model_d_001', '__Model_rho_001', '__Model_d_002', '__Model_rho_002']
bounds = np.array([[-np.inf, np.inf]] * len(var_names))
lp = PdbLogProb(var_names, bounds, 'Linear', is_weighted=True,
                model=func, x=xd, y=yd, err=ed)

primary = [9.830348e-05, 40.8812499, 74.0780823, 6.09414125, 0.15827540,
           11.3182806, 0.23396671, 8.75928561, 0.46419943]
secondary = [9.1524e-05, 31.4256901, 148.961643, 13.9613163, 0.14705280,
             11.3394087, 0.23344522, 8.73373630, 0.46175949]

for label, th, chi2 in [("primary", primary, 148.911112), ("secondary", secondary, 157.189756)]:
    v = lp(np.array(th))
    print("%-10s lnprob=%.4f  expected -chi2/2=%.4f  diff=%.2e" %
          (label, v, -0.5 * chi2, abs(v - (-0.5 * chi2))))

# out-of-bounds -> -inf
bb = bounds.copy(); bb[1] = [0, 90]  # theta in [0,90]
lp2 = PdbLogProb(var_names, bb, 'Linear', True, model=func, x=xd, y=yd, err=ed)
oob = list(primary); oob[1] = 95.0
print("out-of-bounds theta=95 ->", lp2(np.array(oob)))

# picklable (model excluded via __getstate__)
blob = pickle.dumps(lp)
print("pickle(PdbLogProb) OK, %d bytes (model excluded)" % len(blob))
