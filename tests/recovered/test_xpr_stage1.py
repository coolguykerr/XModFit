"""Stage-1 verification for XLayersPDB_XPR.

Reference log-likelihoods come from XPR-main/demo.ipynb (stored outputs), which is
the upstream author's own run of the same Julia model through XPRsim.py.
"""
import os
import sys
import time

import numpy as np

REPO = r'C:\Users\kerr\Documents\GitHub\XModJointFit'
XPRMAIN = r'C:\Users\kerr\Documents\GitHub\XPR-main'
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'Functions'))

from Functions import xpr_backend as xb          # noqa: E402
from Functions import pdb_geometry as pg         # noqa: E402
from Functions.XRR.XLayersPDB_XPR import XLayersPDB_XPR   # noqa: E402

FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print('%-40s got %14.4f  want %14.4f  %s' % (name, got, want, 'OK' if ok else 'FAIL'))
    if not ok:
        FAIL.append(name)


def rrf(fname):
    v = np.genfromtxt(fname, unpack=True)
    a = np.transpose(np.array(v))
    a = a[np.where(a[:, 0] >= 0.026), :][0, :, :]
    return a[:, 0], a[:, 1], a[:, 2]


def model_logL(x, y, e, mpar_d, mpar_rho, **kw):
    """Run the XModFit model class and score it the way XPR does."""
    # The published references come from upstream XPRsim, which applies no
    # protvol normalisation, so score against cov_norm='xpr' unless told otherwise.
    kw.setdefault('cov_norm', 'xpr')
    f = XLayersPDB_XPR(x=x, **kw)
    f.__mpar__['Model']['Layers'] = ['top'] + ['b%d' % i for i in range(len(mpar_d) - 2)] + ['bottom']
    f.__mpar__['Model']['d'] = list(mpar_d)
    f.__mpar__['Model']['rho'] = list(mpar_rho)
    f.init_params()
    R = f.y()
    chi2 = np.sum(((y - R) / e) ** 2)
    return f, R, -0.5 * chi2


print('=== 1. backend up ===')
t0 = time.time()
print('julia project:', xb.julia_project_dir())
xb.get_jl()
print('julia session in %.1f s' % (time.time() - t0))

# ---------------------------------------------------------------- 2-box, no protein
print('\n=== 2. 2-box lipid (no protein) ===')
x, y, e = rrf(os.path.join(XPRMAIN, 'test_data', '2box.rrf'))
p2 = [-0.0005197557108984115, 12.870861541436838, 9.400266410116624,
      0.21726162435306018, 0.4564847693916873]
t0 = time.time()
f2, R2, l2 = model_logL(x, y, e,
                        [0.0, p2[1], p2[2], 1.0], [0.0, p2[3], p2[4], 0.334],
                        fname=None, use_protein=False, qoff=p2[0], sig=3.4,
                        yscale=1.0, bkg=0.0)
print('first eval %.1f s (includes Julia JIT)' % (time.time() - t0))
check('2box logL (via XLayersPDB_XPR)', l2, -1212.2127, 1e-3)

t0 = time.time()
for _ in range(10):
    f2.y()
print('warm eval: %.1f ms' % ((time.time() - t0) * 100))

# ---------------------------------------------------------- 3-box, rho_top, fit sig
print('\n=== 3. 3-box lipid (rho_top=0.2574, sigma fitted) ===')
x3, y3, e3 = rrf(os.path.join(XPRMAIN, 'test_data', '3box.rrf'))
p3 = [0.0008998796447718044, 2.511818916572768,
      26.886814178891647, 18.579820992512868, 11.532122447496487,
      0.30077908545466, 0.25596604523845007, 0.31662370213285806]
f3, R3, l3 = model_logL(x3, y3, e3,
                        [0.0, p3[2], p3[3], p3[4], 1.0],
                        [0.2574, p3[5], p3[6], p3[7], 0.334],
                        fname=None, use_protein=False, qoff=p3[0], sig=p3[1])
check('3box logL (via XLayersPDB_XPR)', l3, -600.1768, 1e-3)

# ------------------------------------------------------------------- protein + 2box
print('\n=== 4. protein + 2-box ===')
xp, yp, ep = rrf(os.path.join(XPRMAIN, 'test_data', 'protein_2box.rrf'))
pdb = os.path.join(XPRMAIN, 'test_data', 'protein.pdb')
# protins is CENTROID-referenced since XPR's 2026-09-08 revision; upstream's demo
# now quotes the corresponding pose directly as -13.717. See handoff/05-xpr-julia.md.
pp = [0.0002054238403330481, 32.92088491028436, 112.59475254401123,
      -13.717,
      2.20516005726808,
      11.66753477743085, 9.643473851409357,
      0.21702095531305382, 0.4470646192673328]
t0 = time.time()
fp, Rp, lp = model_logL(xp, yp, ep,
                        [0.0, pp[5], pp[6], 1.0], [0.0, pp[7], pp[8], 0.334],
                        fname=pdb, use_protein=True, vol_calc='approx',
                        qoff=pp[0], theta=pp[1], phi=pp[2], protins=pp[3], cov=pp[4])
print('first protein eval %.1f s' % (time.time() - t0))
check('protein logL (via XLayersPDB_XPR)', lp, -464.6397, 1e-3)
t0 = time.time()
for _ in range(5):
    fp.y()
print('warm protein eval: %.1f ms' % ((time.time() - t0) * 200))

# ------------------------------------------------------------------------- EDP + AD
print('\n=== 5. EDP output + gradient/Hessian ===')
print('output_params keys:', sorted(fp.output_params.keys()))
z, rho = fp.output_params['Model_EDP']['x'], fp.output_params['Model_EDP']['y']
print('EDP: %d points, z %.1f..%.1f, rho %.4f..%.4f' %
      (len(z), z.min(), z.max(), rho.min(), rho.max()))
if not (len(z) > 10 and np.isfinite(rho).all()):
    FAIL.append('EDP sane')

full = fp._full_params(fp.__d__['Model'], fp.__rho__['Model'], 2)
# grad/hess are only meaningful against the real data, so reload with it
xb.load(xp, 2, pdbfile=pdb, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
        E=10.0, rrf=True, max_box_length=30.0, sig_max=5.0,
        d_protein_bounds=(-40.0, 20.0),
        data_R=yp, data_err=ep, force=True)
lg, _ = xb.sim(full)
check('protein logL (backend, real data)', lg, -464.6397, 1e-3)
t0 = time.time()
g = xb.grad(full)
print('grad (%.1f s): %s' % (time.time() - t0, np.array2string(g, precision=4)))
t0 = time.time()
H = xb.hess(full)
print('hess (%.1f s): shape %s, symmetric=%s' %
      (time.time() - t0, H.shape, np.allclose(H, H.T, atol=1e-6)))
if not (np.isfinite(g).all() and np.isfinite(H).all()):
    FAIL.append('grad/hess finite')

print('\n' + ('ALL CHECKS PASSED' if not FAIL else 'FAILURES: %s' % FAIL))
sys.exit(1 if FAIL else 0)
