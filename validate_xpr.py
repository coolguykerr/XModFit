"""Regression check for the XLayersPDB_XPR function (Julia/XPR backend).

Reference log-likelihoods are the stored outputs of ``XPR-main/demo.ipynb``, i.e.
the upstream author's own run of the same Julia model through ``XPRsim.py``. If
these numbers still match, the XModFit wrapper -- layer-table mapping, parameter
vector order, q handling, Fresnel normalisation -- is faithful to it.

Baselined on the **2026-09-10** upstream revision. Recent upstream history:

* 2026-09-08 redefined ``protins`` from the top of the protein to its MASS CENTROID
  and rebuilt the smearing lattice, both for smoothness.
* 2026-09-09 replaced the fixed 0.5 A lattice with a fixed slab COUNT and
  parameter-dependent spacing (``d_b/cells_per_box`` in box b, rigid ``spacez``
  outside). This removed the last kink AND the segfault class.
* 2026-09-10 exposed ``spacez`` and ``min_box_length_sum``, and made an
  out-of-window protein a hard ``error()`` instead of silent clipping.

All four reference log-likelihoods below are upstream's own ``demo.ipynb`` outputs
for the 2026-09-10 code and reproduce to every printed digit. Note the protein cell
now sits at ``protins = -13.717``, the physically-corresponding pose under the
centroid convention, so it can be compared directly.

**The mesh is now part of the model.** ``spacez`` and ``box_ref_length`` set the
discretization and therefore the absolute chi2; ``d_protein_bounds``/``sig_max``
change only the cost. The mesh-convergence and mesh-invariance checks below pin
that split, because getting it wrong is silent.

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

# The cmcars env has an unrelated regular `tests` package in site-packages that
# shadows the repo's namespace package, so load the harness by file path here.
import importlib.util as _ilu                                 # noqa: E402
_hs = _ilu.spec_from_file_location('xmf_harness', os.path.join(REPO, 'tests', 'harness.py'))
_h = _ilu.module_from_spec(_hs)
_hs.loader.exec_module(_h)
check, check_close, skip, finish = _h.check, _h.check_close, _h.skip, _h.finish

FAIL = []   # ad-hoc failures appended below; folded into the harness count at the end


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
    skip('No XPR test data at %s -- pass its path as an argument.' % DATA)

print('julia project: %s' % xb.julia_project_dir())
t0 = time.time()
xb.get_jl()
print('julia session up in %.1f s\n' % (time.time() - t0))

# --- forward model against the three published log-likelihoods -------------------
x, y, e = rrf('2box.rrf')
f2 = build(x, [0.0, 12.870861541436838, 9.400266410116624, 1.0],
           [0.0, 0.21726162435306018, 0.4564847693916873, 0.334],
           fname=None, use_protein=False, qoff=-0.0005197557108984115, sig=3.4)
check_close('2-box lipid', logL(f2, y, e), -1212.2127, 1e-3)

x3, y3, e3 = rrf('3box.rrf')
f3 = build(x3, [0.0, 26.886814178891647, 18.579820992512868, 11.532122447496487, 1.0],
           [0.2574, 0.30077908545466, 0.25596604523845007, 0.31662370213285806, 0.334],
           fname=None, use_protein=False, qoff=0.0008998796447718044,
           sig=2.511818916572768)
check_close('3-box lipid (rho_top, fitted sigma)', logL(f3, y3, e3), -600.1768, 1e-3)

x4, y4, e4 = rrf('4box.rrf')
f4 = build(x4, [0.0, 8.0, 8.0, 8.0, 8.0, 1.0],
           [0.0, 0.1, 0.2, 0.3, 0.4, 0.334],
           fname=None, use_protein=False, qoff=0.0, yscale=1.05, bkg=1e-6)
check_close('4-box (fitted yscl + bkg)', logL(f4, y4, e4), -62404.0635, 1e-2)

xp, yp, ep = rrf('protein_2box.rrf')
pdb = os.path.join(DATA, 'protein.pdb')
# THE INSERTION CONVENTION. protins is centroid-referenced; upstream's demo now
# quotes the physically-corresponding pose directly, so this is a like-for-like
# comparison. The old top-referenced value was 13.358030329245691, and
# maxh(theta, phi) converts between the two.
PROTINS_REF = -13.717
MAXH_REF = pg.com_extent(pdb, 32.92088491028436, 112.59475254401123)[0]
prot = dict(fname=pdb, use_protein=True, vol_calc='approx',
            qoff=0.0002054238403330481, theta=32.92088491028436,
            phi=112.59475254401123, protins=PROTINS_REF,
            cov=2.20516005726808)
fp = build(xp, [0.0, 11.66753477743085, 9.643473851409357, 1.0],
           [0.0, 0.21702095531305382, 0.4470646192673328, 0.334], **prot)
check_close('protein + 2-box', logL(fp, yp, ep), -464.6397, 1e-3)
check_close('  and maxh itself', MAXH_REF, 27.0751317, 1e-5)
check_close('  protins_top reported back', fp.protins_top(), PROTINS_REF + MAXH_REF, 1e-9)
check_close('  scaler output protins_top',
      fp.output_params['scaler_parameters']['protins_top'],
      PROTINS_REF + MAXH_REF, 1e-6)
# the old top-referenced value must still round-trip through the converters
check_close('  protins_from_top(13.358030329)', fp.protins_from_top(13.358030329245691),
      13.358030329245691 - MAXH_REF, 1e-9)

# --- the two insertion conventions, end to end -----------------------------------
print('\n--- insertion convention: centroid-referenced protins vs top-referenced ---')
# Julia computes protins_top as a by-product of the rotation; Python recomputes it
# from the structure so a chain can be converted without re-running the model. They
# must agree bit for bit, or the reported number and the fitted one drift apart.
_ins = PROTINS_REF          # already centroid-referenced
_full0 = fp._full_params(fp.__d__['Model'], fp.__rho__['Model'], 2)
xb.load(xp, 2, pdbfile=pdb, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
        max_box_length=30.0, sig_max=5.0, d_protein_bounds=(-40.0, 20.0),
        data_R=yp, data_err=ep, force=True)
_lj, _Rj, _phj = xb.sim_ref(_full0)
_hj, _dnj, _kj = xb.maxh(prot['theta'], prot['phi'])
_hp, _dnp, _kp = pg.com_extent(pdb, prot['theta'], prot['phi'])
print('  maxh    julia %.12f   python %.12f' % (_hj, _hp))
print('  mindepth julia %.12f  python %.12f' % (_dnj, _dnp))
print('  protein_height from julia %.12f   protins + maxh %.12f'
      % (_phj, _ins + _hp))
if abs(_hj - _hp) > 1e-12 or abs(_dnj - _dnp) > 1e-12 or _kj != _kp + 1:
    FAIL.append('python com_extent disagrees with XPR_maxh')
if abs(_phj - (_ins + _hp)) > 1e-9:
    FAIL.append("julia's protein_height != protins + maxh")
if abs(fp.protins_from_top(fp.protins_top()) - fp.protins) > 1e-9:
    FAIL.append('protins_top / protins_from_top do not round-trip')

# d(maxh)/d(theta,phi) drives the propagated error on protins_top. Exact wherever
# the topmost atom is unique; check it against central differences at poses well
# away from a tie, and confirm the ties ARE detected where they occur.
_worst_h = 0.0
for _th, _ph in [(32.92088491028436, 112.59475254401123), (7.0, 33.0),
                 (61.0, 217.0), (128.0, 301.0), (171.0, 88.0)]:
    _h, _dt, _dp, _k, _nt = pg.com_extent_grad(pdb, _th, _ph)
    _d = 1e-4
    _ft = (pg.com_extent(pdb, _th + _d, _ph)[0]
           - pg.com_extent(pdb, _th - _d, _ph)[0]) / (2 * _d)
    _fp2 = (pg.com_extent(pdb, _th, _ph + _d)[0]
            - pg.com_extent(pdb, _th, _ph - _d)[0]) / (2 * _d)
    _worst_h = max(_worst_h, abs(_dt - _ft) / max(abs(_ft), 1e-6),
                   abs(_dp - _fp2) / max(abs(_fp2), 1e-6))
print('  d(maxh)/d(angle): worst relative error vs central differences %.2e'
      % _worst_h)
if _worst_h > 1e-5:
    FAIL.append('com_extent_grad disagrees with finite differences: %.2e' % _worst_h)

# The smoothness claim itself: the forward model must be smooth THROUGH the angles
# where the topmost atom changes, because maxh no longer enters it. Under the old
# convention the protein was translated by maxh, so dlogL/dtheta jumped there.
_ths = np.arange(1.0, 179.0, 0.25)
_atoms = [pg.com_extent(pdb, _t, prot['phi'])[2] for _t in _ths]
_switch = [(_ths[i - 1] + _ths[i]) / 2 for i in range(1, len(_ths))
           if _atoms[i] != _atoms[i - 1]]
print('  topmost atom changes at %d angles along phi=%.2f; AD gradient vs one-sided'
      ' differences there:' % (len(_switch), prot['phi']))
_worst_k = 0.0
for _t0 in _switch:
    _q = _full0.copy(); _q[xb.IDX_THETA] = _t0
    _g = xb.grad(_q)[xb.IDX_THETA]
    _qm = _q.copy(); _qm[xb.IDX_THETA] -= 1e-4
    _qp = _q.copy(); _qp[xb.IDX_THETA] += 1e-4
    _l0 = xb.sim(_q)[0]
    _dl = (xb.sim(_qm)[0] - _l0) / -1e-4
    _dr = (xb.sim(_qp)[0] - _l0) / 1e-4
    _worst_k = max(_worst_k, abs(_dr - _dl) / max(abs(_g), 1e-9))
    _hl = pg.com_extent_grad(pdb, _t0 - 0.05, prot['phi'])[1]
    _hr = pg.com_extent_grad(pdb, _t0 + 0.05, prot['phi'])[1]
    print('    theta %7.3f : d(maxh)/dth kinks by %.1e | dlogL/dth L %+.4f R %+.4f'
          ' AD %+.4f' % (_t0, abs(_hr - _hl), _dl, _dr, _g))
print('  worst left/right disagreement in dlogL/dtheta at an atom switch: %.2e'
      ' (relative)' % _worst_k)
if _worst_k > 1e-3:
    FAIL.append('logL is kinked at an atom-switch angle: %.2e -- the insertion '
                'offset has leaked back into the forward model' % _worst_k)

# --- upstream demo parity: the CAL-mass divergence is GONE ----------------------
# Until 2026-09-10 `XPRsim._import_pdb` overrode the CAL atom's name to calcium for
# the electron count and the radius but took its MASS from `atom.element`, which
# Bio.PDB guesses as 'C' when the element columns are blank. Once `prealloc` started
# mass-centring the coordinates that shifted the zero of protins by 0.0345 A on
# `protein.pdb` -- worth 60 in logL at the demo point, and the whole reason our
# numbers used to disagree with the notebook. Upstream now uses `element_key` for
# all three and applies `.capitalize()`, so our reader and theirs must agree
# exactly. Assert that, rather than the old correction.
print('\n--- upstream demo parity: mass table agreement ---')
import Bio.PDB as _BP                                             # noqa: E402
from pdb_cache import _element_table as _et, pdbread as _pr        # noqa: E402
_atoms_bp = list(_BP.PDBParser(QUIET=True).get_structure('s', pdb).get_atoms())
_ai = _et()


def _xprsim_row(atom):
    """XPRsim._import_pdb as of 2026-09-10."""
    key = 'Ca' if atom.name == 'CAL' else atom.element.capitalize()
    return _ai[key]


_m_xprsim = np.array([_xprsim_row(a)[2] for a in _atoms_bp])
_e_xprsim = np.array([_xprsim_row(a)[0] for a in _atoms_bp])
_r_xprsim = np.array([_xprsim_row(a)[1] / 100.0 for a in _atoms_bp])
_xc, _yc, _zc, _e_ours, _r_ours, _m_ours = _pr(pdb)
print('  %d atoms; mass/electron/radius mismatches vs XPRsim: %d / %d / %d'
      % (len(_m_ours), int(np.count_nonzero(_m_xprsim != _m_ours)),
         int(np.count_nonzero(_e_xprsim != _e_ours)),
         int(np.count_nonzero(np.abs(_r_xprsim - _r_ours) > 1e-12))))
if np.count_nonzero(_m_xprsim != _m_ours):
    FAIL.append('our per-atom masses no longer match XPRsim -- the centroid, and '
                'therefore the zero of protins, will differ from upstream')
if (np.count_nonzero(_e_xprsim != _e_ours)
        or np.count_nonzero(np.abs(_r_xprsim - _r_ours) > 1e-12)):
    FAIL.append('our electrons/radii no longer match XPRsim')
_c = np.transpose(np.array([_xc, _yc, _zc]))
_com_shift = ((_c * _m_xprsim[:, None]).sum(0) / _m_xprsim.sum()
              - (_c * _m_ours[:, None]).sum(0) / _m_ours.sum())
print('  mass centroid offset vs XPRsim: %.3e A  (was 0.0345 A before the fix)'
      % float(np.linalg.norm(_com_shift)))
if float(np.linalg.norm(_com_shift)) > 1e-12:
    FAIL.append('mass centroid still differs from XPRsim by %.3g A'
                % float(np.linalg.norm(_com_shift)))

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
                rrf=False, max_box_length=30.0, sig_max=6.0,
                d_protein_bounds=(-40.0, 20.0))
        out[vc] = xb.edp(np.array([0.0, theta, phi, -13.0, 1.0, 3.4, 1.0, 0.0,
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
        max_box_length=30.0, sig_max=5.0, d_protein_bounds=(-40.0, 20.0),
        data_R=yp, data_err=ep, force=True)
check_close('protein logL from the backend', xb.sim(full)[0], -464.6397, 1e-3)
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
# The published parameter point was fitted on an older mesh, so on this one the
# optimum sits slightly BELOW 2*|published logL|. Assert the fit gets at least that
# good rather than equalling it -- equality would be the wrong assertion, and a
# large overshoot would mean the published point is not the optimum any more.
_chi2_fit = float(np.sum(fit.result.residual ** 2))
_chi2_pub = 2 * 1212.2127
print('%-42s got %13.4f  vs published %13.4f  (%+.4f)'
      % ('leastsq chi2 reaches the published optimum', _chi2_fit, _chi2_pub,
         _chi2_fit - _chi2_pub))
if not (_chi2_fit <= _chi2_pub + 1.0):
    FAIL.append('leastsq did not reach the published optimum: %.4f vs %.4f'
                % (_chi2_fit, _chi2_pub))
if _chi2_fit < _chi2_pub - 10.0:
    FAIL.append('leastsq beat the published optimum by %.1f in chi2 -- the '
                'published point may no longer be the optimum on this mesh'
                % (_chi2_pub - _chi2_fit))
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
for k, v in dict(qoff=0.0, theta=185.0, phi=-10.0, protins=-13.137,
                 cov=2.0).items():
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
# protins is centroid-referenced now, so its bounds straddle zero
NLO = np.array([-1e-3,  0.,   0.,-40.,  0., 1.0, 0.0, -1.0,  0.,  0.,  0.,  0.])
NHI = np.array([ 1e-3,180., 360., 10., 10., 6.0, 2.0,  1.0, 30., 30., 1.0, 2.0])
NP  = np.array([2.054e-4, 32.92, 112.59, -13.137, 2.205, 3.37, 1.0, 0.0,
                11.13, 9.07, 0.21, 0.44])
NU = (NP - NLO) / (NHI - NLO)
nkw = dict(pdbfile=pdb, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
           data_R=yp, data_err=ep, max_box_length=40.0, sig_max=8.0,
           d_protein_bounds=(-40.0, 10.0))
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
           phi=110.0, protins=-13.137, cov=2.0, sig=3.37)
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
    for k, v in dict(qoff=0.0, theta=30.0, phi=110.0, protins=-13.137,
                     cov=2.0).items():
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

# --- protins_top's propagated error and covariance -------------------------------
# derived_params linearises protins_top = protins + maxh(theta, phi) through the fit
# covariance. Two separate things to check, and conflating them is a trap:
#
#   1. Is the LINEARISATION right? Draw from the same covariance shrunk by 1e-4, so
#      the draws stay inside the linear regime, and scale the sampled sigma back up.
#      That must match to sampling error (~0.5% at 20000 draws) or the Jacobian is
#      wrong. This is the correctness test.
#   2. Is the linearisation ADEQUATE at the fitted uncertainty? Draw from the full
#      covariance. maxh is a max over atoms -- piecewise smooth, and curving -- so
#      over several degrees of angular uncertainty it need not agree, and on this
#      test problem it does not (~30%). That is physics, not a bug; it is why
#      derived_params documents protins_top() over an MCMC chain as the honest
#      interval, and why derived_report prints the tie count. Bounded loosely here
#      only so a real breakage still trips it.
print('\n--- derived protins_top: propagated error vs Monte Carlo ---')
fd_ = Fit(XLayersPDB_XPR, xp, y=yp, yerr=ep)
gd = fd_.func
gd.fname = pdb; gd.use_protein = True; gd.sig = 3.37
gd.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
gd.__mpar__['Model']['d'] = [0.0, 11.13, 9.07, 1.0]
gd.__mpar__['Model']['rho'] = [0.0, 0.21, 0.44, 0.334]
gd.init_params()
fd_.fit_params, fd_.params = gd.params, gd.__dict__
for k, v in dict(qoff=0.0, theta=30.0, phi=110.0, protins=-13.137, cov=2.0).items():
    fd_.fit_params[k].value = v; gd.__dict__[k] = v
for k in ('theta', 'phi', 'protins', 'cov', '__Model_d_001', '__Model_d_002'):
    fd_.fit_params[k].vary = 1
rep, _msg = fd_.perform_fit(xp.min(), xp.max(), fit_scale='Linear',
                            fit_method='leastsq', maxiter=800)
info = gd.derived_params(fd_.result)
print(gd.derived_report(fd_.result))
if info is None or info['protins_top_err'] is None:
    FAIL.append('derived_params produced no error for protins_top')
else:
    if '[[Derived]]' not in rep or 'protins_top' not in rep:
        FAIL.append('the derived block did not reach the fit report')
    cv = np.asarray(fd_.result.covar, dtype=float)
    nm = list(fd_.result.var_names)
    mu = np.array([fd_.result.params[k].value for k in nm])
    it, ith, iph = nm.index('protins'), nm.index('theta'), nm.index('phi')
    print('  fitted sigma: protins %.4f  theta %.4f deg  phi %.4f deg'
          % (np.sqrt(cv[it, it]), np.sqrt(cv[ith, ith]), np.sqrt(cv[iph, iph])))

    def _mc(scale, seed):
        rng = np.random.default_rng(seed)
        dr = rng.multivariate_normal(mu, cv * scale ** 2, size=20000)
        return float(np.std(gd.protins_top(dr[:, it], dr[:, ith],
                                           dr[:, iph]))) / scale

    mc_lin = _mc(1e-2, 11)          # inside the linear regime
    mc_full = _mc(1.0, 11)          # at the fitted uncertainty
    print('  linearised %.6f   MC at 1%% of sigma (rescaled) %.6f   ratio %.4f'
          % (info['protins_top_err'], mc_lin, info['protins_top_err'] / mc_lin))
    print('  MC at the FULL fitted sigma %.6f   ratio %.4f  <- maxh is nonlinear'
          ' over this much angular uncertainty' % (mc_full,
                                                   info['protins_top_err'] / mc_full))
    if not (0.98 < info['protins_top_err'] / mc_lin < 1.02):
        FAIL.append('the linearised error on protins_top is %.4g but the '
                    'small-step Monte Carlo says %.4g -- the Jacobian is wrong'
                    % (info['protins_top_err'], mc_lin))
    if not (0.25 < info['protins_top_err'] / mc_full < 4.0):
        FAIL.append('propagated error on protins_top (%.4g) is nowhere near the '
                    'full Monte Carlo (%.4g)' % (info['protins_top_err'], mc_full))
    # the value must equal what the model actually evaluated
    if abs(info['protins_top'] - gd.protins_top()) > 1e-9:
        FAIL.append('derived protins_top disagrees with the model')
    # cov(protins_top, protins) must exceed var(protins): the maxh term adds to it
    if abs(info['cov']['protins'] - cv[it, it]) < 1e-30:
        FAIL.append('cov(protins_top, .) looks like it ignored the maxh term')

# --- the mesh is part of the model: pin which knobs move the answer -------------
# Since 2026-09-09 the slab COUNT is a load-time constant and the SPACING varies
# with the parameters. That splits the sizing arguments in two, and confusing them
# is silent: max_box_length/spacez change chi2, d_protein_bounds/sig_max do not.
print('\n--- mesh: which sizing arguments change the answer? ---')
_MK = dict(pdbfile=pdb, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
           data_R=yp, data_err=ep)
_pf = fp._full_params(fp.__d__['Model'], fp.__rho__['Model'], 2)


def _L(**kw):
    kk = dict(max_box_length=30.0, sig_max=5.0, spacez=0.5,
              d_protein_bounds=(-40.0, 20.0))
    kk.update(kw)
    xb.load(xp, 2, force=True, **dict(_MK, **kk))
    return xb.sim(_pf)[0]


_base = _L()
print('  reference logL %.6f' % _base)
for lbl, kw in [('d_protein_bounds (-40,20) -> (-80,60)',
                 dict(d_protein_bounds=(-80.0, 60.0))),
                ('sig_max 5 -> 12', dict(sig_max=12.0)),
                ('min_box_length_sum 0 -> 5', dict(min_box_length_sum=5.0))]:
    _d = abs(_L(**kw) - _base)
    print('    %-38s dlogL %.2e  (must be ~0)' % (lbl, _d))
    if _d > 1e-6 * max(abs(_base), 1.0):
        FAIL.append('%s changed the answer by %.3g -- it should only change cost'
                    % (lbl, _d))
for lbl, kw in [('spacez 0.5 -> 0.25', dict(spacez=0.25, max_box_length=15.0)),
                ('box_ref_length 30 -> 120', dict(max_box_length=120.0))]:
    _d = abs(_L(**kw) - _base)
    print('    %-38s dlogL %.3f  (must be nonzero)' % (lbl, _d))
    if _d < 1e-3:
        FAIL.append('%s did NOT change the answer -- the mesh knob is not wired'
                    % lbl)

# mesh convergence: refining spacez and the box mesh together must converge, and
# O(h^2)-ish. This is the check that would catch a discretization regression.
print('\n--- mesh convergence in spacez (box mesh held at d/60) ---')
_seq = []
for _sp in (0.5, 0.25, 0.125):
    _seq.append(_L(spacez=_sp, max_box_length=30.0 * _sp / 0.5))
    print('    spacez %.3f  logL %.6f' % (_sp, _seq[-1]))
_d1, _d2 = _seq[1] - _seq[0], _seq[2] - _seq[1]
print('    successive changes %.4f then %.4f  -> ratio %.2f (O(h^2) gives ~4)'
      % (_d1, _d2, _d1 / _d2 if _d2 else float('inf')))
if not (_d1 * _d2 > 0 and 1.5 < abs(_d1 / _d2) < 8.0):
    FAIL.append('spacez refinement is not converging monotonically at a sane rate: '
                '%.4f then %.4f' % (_d1, _d2))

# --- upstream's out-of-bounds error, caught in Python first ----------------------
# Upstream raises error("protein out of bounds") when the protein leaves the grid.
# That is the right call, but a Julia exception unwinding mid-fit aborts the run, so
# check_headroom must catch it first with a message naming the numbers.
print('\n--- out-of-window protein: Python raises before Julia does ---')
xb.load(xp, 2, force=True, **dict(_MK, max_box_length=30.0, sig_max=5.0,
                                  spacez=0.5, d_protein_bounds=(-40.0, 20.0)))
_ok = True
_pin = _pf.copy()
_pin[xb.IDX_PROTINS] = 10.0
try:
    xb.check_headroom(_pin, 2, True)
    print('    protins=10, window (-40,20): accepted (correct)')
except ValueError as _e:
    _ok = False
    FAIL.append('check_headroom rejected an in-window protins: %s' % _e)
_pin[xb.IDX_PROTINS] = 100.0
try:
    xb.check_headroom(_pin, 2, True)
    FAIL.append('check_headroom accepted protins=100, outside the declared window')
except ValueError as _e:
    print('    protins=100: ValueError -> %s' % str(_e)[:96])
# and Julia really would have raised, i.e. we are mirroring a real constraint
try:
    xb.sim(_pin)
    FAIL.append('Julia accepted an out-of-window protins -- the mirror is now wrong')
except Exception as _e:
    print('    Julia on the same vector: %s (mirror confirmed)' % type(_e).__name__)

print('\n' + ('ALL CHECKS PASSED' if not FAIL else 'FAILURES:\n  ' + '\n  '.join(FAIL)))
for _m in FAIL:
    check(_m, False)   # registers the ad-hoc failure with the harness
sys.exit(finish())
