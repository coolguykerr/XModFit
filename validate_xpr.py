"""Regression check for the XLayersPDB_XPR function (Julia/XPR backend).

Reference log-likelihoods are the stored outputs of ``XPR-main/demo.ipynb``, i.e.
the upstream author's own run of the same Julia model through ``XPRsim.py``. If
these three numbers still match, the XModFit wrapper -- layer-table mapping,
parameter vector order, q handling, Fresnel normalisation -- is faithful to it.

    python validate_xpr.py [path/to/XPR-main/test_data]

Needs juliacall and the vendored julia/XPR project; see julia/README.md.
"""

import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'Functions'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

DATA = (sys.argv[1] if len(sys.argv) > 1
        else os.path.join(os.path.dirname(REPO), 'XPR-main', 'test_data'))

from PyQt5.QtWidgets import QApplication                     # noqa: E402
_app = QApplication.instance() or QApplication([])

from Fit_Routines import Fit                                  # noqa: E402
from Functions import xpr_backend as xb                       # noqa: E402
from Functions import pdb_geometry as pg                      # noqa: E402
from Functions.XRR.XLayersPDB_XPR import XLayersPDB_XPR       # noqa: E402

FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print('%-42s got %13.4f  want %13.4f  %s'
          % (name, got, want, 'OK' if ok else 'FAIL'))
    if not ok:
        FAIL.append(name)


def rrf(name):
    """Read a .rrf and drop q < 0.026, matching XPRsim._import_rrf."""
    a = np.transpose(np.array(np.genfromtxt(os.path.join(DATA, name), unpack=True)))
    a = a[np.where(a[:, 0] >= 0.026), :][0, :, :]
    return a[:, 0], a[:, 1], a[:, 2]


def build(x, d, rho, **kw):
    # The reference numbers below come from upstream XPRsim, which has no
    # normalisation, so default to 'xpr' and opt in to 'protvol' where wanted.
    kw.setdefault('cov_norm', 'xpr')
    f = XLayersPDB_XPR(x=x, **kw)
    f.__mpar__['Model']['Layers'] = (['top'] + ['b%d' % i for i in range(len(d) - 2)]
                                     + ['bottom'])
    f.__mpar__['Model']['d'] = list(d)
    f.__mpar__['Model']['rho'] = list(rho)
    f.init_params()
    return f


def logL(f, y, e):
    return -0.5 * np.sum(((y - f.y()) / e) ** 2)


if not os.path.isdir(DATA):
    sys.exit('No XPR test data at %s -- pass its path as an argument.' % DATA)

print('julia project: %s' % xb.julia_project_dir())
t0 = time.time()
xb.get_jl()
print('julia session up in %.1f s\n' % (time.time() - t0))

# --- forward model against the three published log-likelihoods -------------------
x, y, e = rrf('2box.rrf')
f2 = build(x, [0.0, 12.870861541436838, 9.400266410116624, 1.0],
           [0.0, 0.21726162435306018, 0.4564847693916873, 0.334],
           fname=None, use_protein=False, qoff=-0.0005197557108984115, sig=3.4)
check('2-box lipid', logL(f2, y, e), -1214.2981, 1e-3)

x3, y3, e3 = rrf('3box.rrf')
f3 = build(x3, [0.0, 26.886814178891647, 18.579820992512868, 11.532122447496487, 1.0],
           [0.2574, 0.30077908545466, 0.25596604523845007, 0.31662370213285806, 0.334],
           fname=None, use_protein=False, qoff=0.0008998796447718044,
           sig=2.511818916572768)
check('3-box lipid (rho_top, fitted sigma)', logL(f3, y3, e3), -600.6506, 1e-3)

xp, yp, ep = rrf('protein_2box.rrf')
pdb = os.path.join(DATA, 'protein.pdb')
prot = dict(fname=pdb, use_protein=True, vol_calc='approx',
            qoff=0.0002054238403330481, theta=32.92088491028436,
            phi=112.59475254401123, protins=13.358030329245691, cov=2.20516005726808)
fp = build(xp, [0.0, 11.66753477743085, 9.643473851409357, 1.0],
           [0.0, 0.21702095531305382, 0.4470646192673328, 0.334], **prot)
check('protein + 2-box', logL(fp, yp, ep), -466.4557, 1e-3)

# --- the grid footprint must equal the one XPR computes internally ---------------
# XPR never exposes tempvars.A, but it can be extracted exactly: zero every rho, so
# rho_hard = C*rho_prot, and rho_prot = sum_i Z_i f_ij/(A*dz) shares its numerator
# between the two vol_calc modes. The pointwise EDP ratio is then A_exact/A_xpr.
print('\n--- Python grid footprint vs XPR\'s own union area ---')
n_at, a_xpr, _vs = pg.sphere_sums(pdb)
zero_rho = [0.0, 0.0, 0.0, 0.0]


def _xpr_internal_area(theta, phi):
    out = {}
    for vc in ('exact', 'approx'):
        xb.load(xp, 2, pdbfile=pdb, vol_calc=vc, rho_top=0.0, rho_bottom=0.0,
                rrf=False, max_box_length=30.0, max_sig=6.0, slab_headroom=10.0)
        out[vc] = xb.edp(np.array([0.0, theta, phi, 13.0, 1.0, 3.4, 1.0, 0.0,
                                   12.0, 10.0, 0.0, 0.0]))[1]
    m = (out['exact'] > 1e-6) & (out['approx'] > 1e-6)
    return a_xpr * float(np.median(out['approx'][m] / out['exact'][m]))


for th, ph in [(0.0, 0.0), (32.92, 112.59), (90.0, 45.0), (150.0, 300.0)]:
    a_int = _xpr_internal_area(th, ph)
    a_grid = pg.projected_area(pdb, th, ph, 0.5)
    err = 100 * (a_grid - a_int) / a_int
    print('  theta=%6.2f phi=%6.2f : grid %7.1f A^2  XPR %7.1f A^2  (%+.2f%%)'
          % (th, ph, a_grid, a_int, err))
    if abs(err) > 1.0:
        FAIL.append('projected_area off by %.2f%% at theta=%g phi=%g' % (err, th, ph))

# --- the reported coverage means the same thing in both vol_calc modes ----------
# cov itself is raw and unequal between the modes by design. What must agree is the
# DERIVED covered fraction: 'approx' at cov = frac*cov_scale should reproduce
# 'exact' at cov = frac, since 'exact' already parameterises the fraction directly.
print('\n--- derived coverage: approx at cov*cov_scale == exact at cov? ---')
LAYD = [0.0, 11.66753477743085, 9.643473851409357, 1.0]
LAYR = [0.0, 0.21702095531305382, 0.4470646192673328, 0.334]
worst = 0.0
for (th, ph) in [(32.92, 112.59), (0.0, 0.0), (90.0, 45.0)]:
    for frac in (0.02, 0.10):
        ma = build(xp, LAYD, LAYR, **dict(prot, theta=th, phi=ph,
                                          vol_calc='approx', cov_norm='protvol'))
        scale = ma.cov_scale()
        ma.cov = frac * scale
        ma.params['cov'].value = ma.cov
        Ra = ma.y()
        me = build(xp, LAYD, LAYR,
                   **dict(prot, theta=th, phi=ph, cov=frac, vol_calc='exact',
                          cov_norm='protvol'))
        Re = me.y()
        rms = float(np.sqrt(np.mean((Ra - Re) ** 2) / np.mean(Re ** 2)))
        worst = max(worst, rms)
        back = ma.coverage_fraction()
        print('  theta=%5.1f phi=%5.1f frac=%.2f : cov_scale %6.2f  round-trip %.4f'
              '  rms %.2e' % (th, ph, frac, scale, back, rms))
        if abs(back - frac) > 1e-9:
            FAIL.append('coverage_fraction round-trip failed at theta=%g' % th)
print('  worst %.2e (comparing at the same RAW cov instead gives ~2.8e-02)' % worst)
if worst > 5e-3:
    FAIL.append('derived coverage: approx still differs from exact by %.2e' % worst)

# cov must reach Julia untouched -- that is what keeps ForwardDiff exact.
_m = build(xp, LAYD, LAYR, **dict(prot, cov=0.037, vol_calc='approx',
                                  cov_norm='protvol'))
if _m._full_params(np.array(LAYD), np.array(LAYR), 2)[xb.IDX_COV] != 0.037:
    FAIL.append('cov is being rescaled before it reaches Julia')
else:
    print('  cov passed to Julia unmodified: OK (gradient stays exact)')

# --- EDP output and the autodiff derivatives -------------------------------------
z = fp.output_params['Model_EDP']['x']
rho = fp.output_params['Model_EDP']['y']
print('\nEDP: %d points, z %.1f..%.1f A, rho %.4f..%.4f el/A^3'
      % (len(z), z.min(), z.max(), rho.min(), rho.max()))
if not (len(z) > 10 and np.isfinite(rho).all()):
    FAIL.append('EDP sane')

full = fp._full_params(fp.__d__['Model'], fp.__rho__['Model'], 2)
xb.load(xp, 2, pdbfile=pdb, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
        max_box_length=30.0, max_sig=5.0, slab_headroom=10.0,
        data_R=yp, data_err=ep, force=True)
check('protein logL from the backend', xb.sim(full)[0], -466.4557, 1e-3)
g, H = xb.grad(full), xb.hess(full)
print('grad: %d entries, all finite = %s' % (len(g), bool(np.isfinite(g).all())))
print('hess: %s, symmetric = %s, all finite = %s'
      % (H.shape, bool(np.allclose(H, H.T, atol=1e-6)), bool(np.isfinite(H).all())))
if not (np.isfinite(g).all() and np.isfinite(H).all() and H.shape == (12, 12)):
    FAIL.append('grad/hess')

# --- through XModFit's own fitter: leastsq, then emcee ---------------------------
print('\n--- Fit_Routines: leastsq on 2box.rrf from a perturbed start ---')
fit = Fit(XLayersPDB_XPR, x, y=y, yerr=e)
fit.func.fname = None
fit.func.use_protein = False
fit.func.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
fit.func.__mpar__['Model']['d'] = [0.0, 12.0, 10.0, 1.0]
fit.func.__mpar__['Model']['rho'] = [0.0, 0.20, 0.42, 0.334]
fit.func.init_params()
fit.fit_params, fit.params = fit.func.params, fit.func.__dict__
for k in ('theta', 'phi', 'protins', 'cov'):
    fit.fit_params[k].vary = 0
for k in ('qoff', '__Model_d_001', '__Model_d_002',
          '__Model_rho_001', '__Model_rho_002'):
    fit.fit_params[k].vary = 1
fit.perform_fit(x.min(), x.max(), fit_scale='Linear', fit_method='leastsq',
                maxiter=1000)
check('leastsq chi2 vs published optimum',
      float(np.sum(fit.result.residual ** 2)), 2 * 1214.2981, 1.0)
for k, ref in (('qoff', -0.0005197557108984115), ('__Model_d_001', 12.870861541436838),
               ('__Model_d_002', 9.400266410116624),
               ('__Model_rho_001', 0.21726162435306018),
               ('__Model_rho_002', 0.4564847693916873)):
    got = fit.result.params[k].value
    if abs(got - ref) > 0.02 * max(abs(ref), 1e-3):
        FAIL.append('leastsq %s: %g vs %g' % (k, got, ref))

print('\n--- Fit_Routines: emcee (serial, no backend file) ---')
fit.perform_fit(x.min(), x.max(), fit_scale='Linear', fit_method='emcee',
                emcee_walkers=20, emcee_steps=40, emcee_burn=0, emcee_cores=1,
                emcee_thin=1)
ch = fit.result.flatchain
print('chain %s, posterior mean vs best fit:' % (ch.shape,))
for k in ch.columns:
    print('   %-18s %12.6f  (%.6f)' % (k, ch[k].mean(), fit.result.params[k].value))
if ch.shape != (800, 5) or not np.isfinite(fit.result.lnprob).all():
    FAIL.append('emcee chain')

print('\n--- Fit_Routines: emcee with theta/phi (orientation seeding) ---')
fit2 = Fit(XLayersPDB_XPR, xp, y=yp, yerr=ep)
for k, v in prot.items():
    setattr(fit2.func, k, v)
fit2.func.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
fit2.func.__mpar__['Model']['d'] = [0.0, 11.66753477743085, 9.643473851409357, 1.0]
fit2.func.__mpar__['Model']['rho'] = [0.0, 0.21702095531305382, 0.4470646192673328, 0.334]
fit2.func.init_params()
fit2.fit_params, fit2.params = fit2.func.params, fit2.func.__dict__
for k in ('theta', 'phi', 'protins', 'cov'):
    fit2.fit_params[k].value = prot[k]
    fit2.fit_params[k].vary = 1
fit2.perform_fit(xp.min(), xp.max(), fit_scale='Linear', fit_method='leastsq',
                 maxiter=200)
fit2.perform_fit(xp.min(), xp.max(), fit_scale='Linear', fit_method='emcee',
                 emcee_walkers=16, emcee_steps=20, emcee_burn=0, emcee_cores=1,
                 emcee_thin=1)
th, ph = fit2.result.flatchain['theta'].values, fit2.result.flatchain['phi'].values
print('theta %.1f..%.1f   phi %.1f..%.1f' % (th.min(), th.max(), ph.min(), ph.max()))
if not (-1e-9 <= th.min() and th.max() <= 180 + 1e-9
        and -1e-9 <= ph.min() and ph.max() <= 360 + 1e-9):
    FAIL.append('theta/phi escaped the fundamental domain (angle wrapping)')

# --- angles come back inside their conventional window ----------------------------
# theta/phi are unbounded so the optimiser and the emcee moves can cross the poles
# freely; lmfit then reports its own raw coordinate, which for a periodic parameter
# can be arbitrarily far out (observed: theta=379, phi=-8208 for a fit that was
# actually evaluating 19.3 / 71.8). Fit._canonicalize_result folds it back.
print('\n--- post-fit angle canonicalisation ---')
fa = Fit(XLayersPDB_XPR, xp, y=yp, yerr=ep)
ga = fa.func
ga.fname = pdb; ga.use_protein = True; ga.sig = 3.37
ga.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
ga.__mpar__['Model']['d'] = [0.0, 11.13, 9.07, 1.0]
ga.__mpar__['Model']['rho'] = [0.0, 0.21, 0.44, 0.334]
ga.init_params()
fa.fit_params, fa.params = ga.params, ga.__dict__
for k, v in dict(qoff=0.0, theta=185.0, phi=-10.0, protins=13.137, cov=2.0).items():
    fa.fit_params[k].value = v; ga.__dict__[k] = v
for k in ('theta', 'phi', 'protins', 'cov'):
    fa.fit_params[k].vary = 1
fa.perform_fit(xp.min(), xp.max(), fit_scale='Linear', fit_method='leastsq',
               maxiter=400)
at, ap = fa.result.params['theta'].value, fa.result.params['phi'].value
print('  started at theta=185, phi=-10 -> reported theta=%.4f phi=%.4f' % (at, ap))
if not (0.0 <= at <= 180.0 and 0.0 <= ap <= 360.0):
    FAIL.append('angles left outside their window: theta=%g phi=%g' % (at, ap))
if abs(at - ga.theta) > 1e-9 or abs(ap - ga.phi) > 1e-9:
    FAIL.append('reported angles disagree with the ones the model evaluated')

# --- normalised (unit-cube) load -------------------------------------------------
# Stage 2 reparameterises the model onto [0,1]^n at load time so a bounded global
# search sees a well-scaled problem. The mapping lives inside XPR!, hence inside the
# AD tape: derivatives must come back already chain-ruled.
print('\n--- normalised unit-cube load ---')
NLO = np.array([-1e-3,  0.,   0.,  0.,  0., 1.0, 0.0, -1.0,  0.,  0.,  0.,  0.])
NHI = np.array([ 1e-3,180., 360., 20., 10., 6.0, 2.0,  1.0, 30., 30., 1.0, 2.0])
NP  = np.array([2.054e-4, 32.92, 112.59, 13.137, 2.205, 3.37, 1.0, 0.0,
                11.13, 9.07, 0.21, 0.44])
NU = (NP - NLO) / (NHI - NLO)
nkw = dict(pdbfile=pdb, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
           data_R=yp, data_err=ep, max_box_length=40.0, max_sig=8.0,
           slab_headroom=20.0)
xb.load(xp, 2, force=True, **nkw)
_lp, _Rp = xb.sim(NP); _gp = xb.grad(NP); _Jp = xb.jac(NP, np.arange(12))
xb.load(xp, 2, param_lower=NLO, param_upper=NHI, force=True, **nkw)
_lu, _Ru = xb.sim(NU); _gu = xb.grad(NU); _Ju = xb.jac(NU, np.arange(12))
_SC = NHI - NLO
print('  logL physical %.6f == normalised %.6f' % (_lp, _lu))
if abs(_lp - _lu) > 1e-9 * max(1.0, abs(_lp)) or not np.allclose(_Rp, _Ru, atol=1e-12):
    FAIL.append('normalised load changed the model')
if not np.allclose(_gu, _gp * _SC, rtol=1e-8, atol=1e-12):
    FAIL.append('normalised gradient is not chain-ruled')
if not np.allclose(_Ju, _Jp * _SC[None, :], rtol=1e-8, atol=1e-14):
    FAIL.append('normalised Jacobian is not chain-ruled')
if not np.allclose(xb.to_physical(xb.to_normalized(NP)), NP):
    FAIL.append('to_physical/to_normalized do not round-trip')
print('  grad and Jacobian chain-ruled, round-trip exact: OK')
xb.load(xp, 2, force=True, **nkw)          # leave the backend in physical units

# --- analytic Jacobian ------------------------------------------------------------
# NB the reference values below are at GENERIC parameters on purpose. XPR lays its
# profile on a fixed spacez=0.5 A lattice, so R has a (tiny) slope discontinuity
# wherever a layer boundary or the smearing extent lands exactly on a node --
# sum(d) a multiple of 0.5, or 10*sig likewise. Central differences straddle those
# and disagree with the (correct) one-sided AD derivative by ~1e-2. Round-number
# defaults such as d=[11.0, 9.0] and sig=3.4 sit exactly on them.
print('\n--- analytic Jacobian vs central differences (generic parameters) ---')
JN = ['qoff', 'theta', 'phi', 'protins', 'cov', 'sig', 'yscale', 'bkg',
      '__Model_d_001', '__Model_d_002', '__Model_rho_001', '__Model_rho_002']
fj = build(xp, [0.0, 11.13, 9.07, 1.0], [0.0, 0.21, 0.44, 0.334],
           fname=pdb, use_protein=True, vol_calc='approx', qoff=0.0, theta=30.0,
           phi=110.0, protins=13.137, cov=2.0, sig=3.37)
fj.__fit__ = True
fj.y()
Jac = fj.dydp(JN)


def _get(nm):
    return fj.params[nm].value if nm.startswith('__') else getattr(fj, nm)


def _set(nm, v):
    fj.params[nm].value = v
    if not nm.startswith('__'):
        setattr(fj, nm, v)


worst_j = 0.0
for jc, nm in enumerate(JN):
    v0 = _get(nm)
    h = 1e-5 * max(abs(v0), 1.0)
    _set(nm, v0 + h); rp = np.asarray(fj.y())
    _set(nm, v0 - h); rm = np.asarray(fj.y())
    _set(nm, v0)
    fd = (rp - rm) / (2 * h)
    worst_j = max(worst_j, float(np.max(np.abs(Jac[:, jc] - fd))
                                 / max(np.max(np.abs(fd)), 1e-12)))
print('  %d columns, worst relative error %.2e' % (len(JN), worst_j))
if worst_j > 1e-4:
    FAIL.append('analytic Jacobian disagrees with finite differences: %.2e' % worst_j)
if fj.parameter_columns(['__Model_rho_003']) is not None:
    FAIL.append('parameter_columns should refuse rho of the bulk (load-time constant)')

print('  leastsq with vs without the Jacobian:')
res_j = {}
for use in (False, True):
    fk = Fit(XLayersPDB_XPR, xp, y=yp, yerr=ep)
    gk = fk.func
    gk.fname = pdb; gk.use_protein = True; gk.sig = 3.37
    gk.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
    gk.__mpar__['Model']['d'] = [0.0, 11.13, 9.07, 1.0]
    gk.__mpar__['Model']['rho'] = [0.0, 0.21, 0.44, 0.334]
    gk.init_params()
    fk.fit_params, fk.params = gk.params, gk.__dict__
    for k, v in dict(qoff=0.0, theta=30.0, phi=110.0, protins=13.137, cov=2.0).items():
        fk.fit_params[k].value = v; gk.__dict__[k] = v
    for k in ('qoff', '__Model_d_001', '__Model_d_002',
              '__Model_rho_001', '__Model_rho_002'):
        fk.fit_params[k].vary = 1
    fk.use_jacobian = use
    calls = [0]
    fk.functionCalled.connect(lambda p, i, r, s: calls.__setitem__(0, calls[0] + 1))
    fk.perform_fit(xp.min(), xp.max(), fit_scale='Linear', fit_method='leastsq',
                   maxiter=2000)
    res_j[use] = (calls[0], float(np.sum(fk.result.residual ** 2)))
    print('     use_jacobian=%-5s : %4d residual calls, chi2 %.6f'
          % (use, res_j[use][0], res_j[use][1]))
if abs(res_j[True][1] - res_j[False][1]) > 1e-4 * max(res_j[False][1], 1.0):
    FAIL.append('Jacobian changed the optimum')
if res_j[True][0] >= res_j[False][0]:
    FAIL.append('Jacobian did not reduce the model-call count')

print('\n' + ('ALL CHECKS PASSED' if not FAIL else 'FAILURES:\n  ' + '\n  '.join(FAIL)))
sys.exit(1 if FAIL else 0)
