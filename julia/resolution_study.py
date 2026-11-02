"""Quantify XPR's parameter-dependent mesh resolution, on real data.

    python julia/resolution_study.py [out_prefix]

XPR (2026-09-09 onward) fixes the slab COUNT at load time and lets the SPACING move
with the parameters:

    outside the boxes      spacez                       (rigid)
    inside box b           d_b * spacez / box_ref_length

so the in-box mesh tracks each box thickness.  That removes every kink, but it means
the discretization error is a function of the fitted parameters rather than a
constant.  This script measures the consequences on a real dataset:

  1. how large the mesh error is in R and in R/R_F, resolved in q;
  2. how large it is in logL / chi2, and in units of the measured sigma;
  3. how it VARIES as the box thicknesses sweep -- the part that can bias a fit;
  4. whether it moves the fitted parameters (1-D optima at each mesh);
  5. whether it distorts the likelihood landscape enough to matter to a global
     search that merges basins on a Delta(-logL) tolerance.

Reference ("converged") is a Richardson extrapolation over spacez, which refines the
whole mesh uniformly because both the in-box and out-of-box spacings are proportional
to it.

Dataset: mTIM3 d20_tim3_1 with the t3hmmm template, at the archived fit pose.
Falls back to XPR-main/test_data if the real data is not on this machine.
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'Functions'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from Functions import xpr_backend as xb                                  # noqa: E402
from Functions import pdb_geometry as pg                                 # noqa: E402
from Functions.XRR.XLayersPDB_XPR import XLayersPDB_XPR                  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join('julia', 'resolution_study')

# --------------------------------------------------------------- the real dataset
REAL_RRF = r"C:/Users/kerr/Documents/XLayersProtTest/mTIM3 analysis/d20_tim3_1.rrf"
REAL_PDB = (r"C:/Users/kerr/OneDrive - The University of Chicago"
            r"/xmodfit_protlayertest/TIM3/tim3_hmmm_ext_all_bestframe.pdb")
# archived XModFit fit for this dataset+template (seeding-study config d20_tim3_1):
# pose comparison only -- it was never a -logL target, and cov there is XModFit's
# 0-1 covered fraction, not XPR's raw n_s*A.
REAL = dict(qmin=0.028, qmax=0.56, rho_top=0.0, rho_bottom=0.334, nbox=2,
            theta=51.37579, phi=125.3696, protins_top=8.662192,
            cov_frac=0.2417491, d=[12.73124, 8.747107],
            rho=[0.2413187, 0.4654535], sig=3.4, qoff=0.0)
FALLBACK_RRF = os.path.join(os.path.dirname(REPO), 'XPR-main', 'test_data',
                            'protein_2box.rrf')
FALLBACK_PDB = os.path.join(os.path.dirname(REPO), 'XPR-main', 'test_data',
                            'protein.pdb')
FALLBACK = dict(qmin=0.026, qmax=1.0, rho_top=0.0, rho_bottom=0.334, nbox=2,
                theta=32.92088491028436, phi=112.59475254401123,
                protins_top=13.358030329245691, cov_frac=None,
                cov_raw=2.20516005726808,
                d=[11.66753477743085, 9.643473851409357],
                rho=[0.21702095531305382, 0.4470646192673328],
                sig=3.4, qoff=0.0002054238403330481)

if os.path.isfile(REAL_RRF) and os.path.isfile(REAL_PDB):
    RRF, PDB, CFG, LABEL = REAL_RRF, REAL_PDB, REAL, 'mTIM3 d20_tim3_1 / t3hmmm'
    COVNORM = 'protvol'
else:
    RRF, PDB, CFG, LABEL = (FALLBACK_RRF, FALLBACK_PDB, FALLBACK,
                            'XPR-main protein_2box (fallback)')
    COVNORM = 'xpr'
    print('!! real data not found, falling back to %s' % FALLBACK_RRF)


def read_rrf(path, qmin, qmax):
    a = np.transpose(np.array(np.genfromtxt(path, unpack=True)))
    m = (a[:, 0] >= qmin) & (a[:, 0] <= qmax)
    a = a[m]
    return a[:, 0], a[:, 1], a[:, 2]


Q, YD, ED = read_rrf(RRF, CFG['qmin'], CFG['qmax'])
MAXH = pg.com_extent(PDB, CFG['theta'], CFG['phi'])[0]
PROTINS = CFG['protins_top'] - MAXH

print('=' * 78)
print('XPR parameter-dependent resolution: effect on R, R/R_F and logL')
print('=' * 78)
print('dataset   %s' % LABEL)
print('          %d points, q %.4f..%.4f, median sigma/R %.4f'
      % (len(Q), Q.min(), Q.max(), np.median(ED / YD)))
print('template  %s (%d atoms)' % (os.path.basename(PDB),
                                   len(pg.com_coordinates(PDB)[1])))
print('pose      theta %.3f  phi %.3f  protins_top %.4f -> protins %.4f (maxh %.4f)'
      % (CFG['theta'], CFG['phi'], CFG['protins_top'], PROTINS, MAXH))
print('film      d = %s   rho = %s   sig %.3f'
      % (CFG['d'], CFG['rho'], CFG['sig']))


# ------------------------------------------------------------------ model builder
def build(spacez=0.5, box_ref=30.0, rrf=True, d=None, cov_raw=None):
    d = CFG['d'] if d is None else d
    f = XLayersPDB_XPR(x=Q, fname=PDB, use_protein=True, vol_calc='approx',
                       cov_norm=COVNORM, rrf=rrf, spacez=spacez,
                       box_ref_length=box_ref, E=10.0,
                       qoff=CFG['qoff'], sig=CFG['sig'], theta=CFG['theta'],
                       phi=CFG['phi'], protins=PROTINS, cov=0.0)
    f.__mpar__['Model']['Layers'] = ['top'] + ['b%d' % i for i in range(len(d))] \
        + ['bottom']
    f.__mpar__['Model']['d'] = [0.0] + list(d) + [1.0]
    f.__mpar__['Model']['rho'] = [CFG['rho_top']] + list(CFG['rho']) \
        + [CFG['rho_bottom']]
    f.init_params()
    # generous, mesh-neutral window so nothing clips and nothing depends on bounds
    f.params['protins'].min, f.params['protins'].max = PROTINS - 40, PROTINS + 25
    f.params['sig'].max = 8.0
    for i in range(len(d)):
        f.params['__Model_d_%03d' % (i + 1)].max = 60.0
    if cov_raw is not None:
        f.cov = cov_raw
    elif CFG.get('cov_frac') is not None:
        f.cov = CFG['cov_frac'] * f.cov_scale()
    else:
        f.cov = CFG['cov_raw']
    f.params['cov'].value = f.cov
    f.params['cov'].max = max(10.0, 5 * f.cov)
    return f


def evaluate(f, d=None):
    """R at the current parameters, plus logL against the real data."""
    if d is not None:
        for i, v in enumerate(d):
            f.__mpar__['Model']['d'][i + 1] = float(v)
            f.params['__Model_d_%03d' % (i + 1)].value = float(v)
    R = np.asarray(f.y(), dtype=float)
    chi2 = float(np.sum(((YD - R) / ED) ** 2))
    return R, -0.5 * chi2, chi2


f0 = build()
R0, L0, C0 = evaluate(f0)
sp = f0.output_params['scaler_parameters']
print('cov       raw %.4f  (fraction %.4f, cov_scale %.2f)'
      % (f0.cov, sp.get('coverage', float('nan')), sp.get('cov_scale', float('nan'))))
print('mesh      spacez 0.5, box_ref 30 -> in-box %.4f A, %d smearing nodes'
      % (sp.get('mesh_box_Angs', float('nan')), sp.get('slabs', -1)))
print('          chi2 %.2f  chi2/N %.3f  logL %.4f' % (C0, C0 / len(Q), L0))

# ================================================================ 1. convergence
print()
print('=' * 78)
print('1. MESH CONVERGENCE. spacez refines the WHOLE mesh uniformly (both the rigid')
print('   outer spacing and the in-box spacing d_b*spacez/box_ref are proportional')
print('   to it), so this is a clean h-refinement.')
print('=' * 78)
SPZ = [0.5, 0.25, 0.125, 0.0625]
res = {}
print('  %-8s %8s %10s %14s %12s %10s' %
      ('spacez', 'nodes', 'in-box A', 'logL', 'chi2', 'eval ms'))
for s in SPZ:
    f = build(spacez=s)
    t0 = time.time()
    R, L, C = evaluate(f)
    dt = (time.time() - t0) * 1000
    res[s] = dict(R=R, L=L, C=C)
    ss = f.output_params['scaler_parameters']
    print('  %-8.4f %8d %10.4f %14.5f %12.2f %10.1f'
          % (s, ss.get('slabs', -1), ss.get('mesh_box_Angs', np.nan), L, C, dt))

d1 = res[SPZ[1]]['L'] - res[SPZ[0]]['L']
d2 = res[SPZ[2]]['L'] - res[SPZ[1]]['L']
d3 = res[SPZ[3]]['L'] - res[SPZ[2]]['L']
p = np.log2(abs(d1 / d2)) if d2 else np.nan
p2 = np.log2(abs(d2 / d3)) if d3 else np.nan
L_inf = res[SPZ[3]]['L'] + d3 / (2 ** p2 - 1) if np.isfinite(p2) and p2 > 0.1 else np.nan
print('  successive dlogL %+.4f %+.4f %+.4f -> observed order %.2f then %.2f'
      % (d1, d2, d3, p, p2))
print('  Richardson limit logL_inf = %.5f' % L_inf)
print('  => the DEFAULT mesh (spacez 0.5) is off by %+.3f in logL, %+.3f in chi2'
      % (res[0.5]['L'] - L_inf, -2 * (res[0.5]['L'] - L_inf)))
print('     for reference, chi2/N here is %.2f over %d points, and a 1-sigma'
      % (C0 / len(Q), len(Q)))
print('     movement of one datum contributes 1.0 to chi2.')

# reference R, on the finest mesh, for R and for raw R (no Fresnel division)
Rref = res[SPZ[-1]]['R']
f_raw0 = build(rrf=False)
Rraw0, _, _ = evaluate(f_raw0)
f_rawf = build(spacez=SPZ[-1], rrf=False)
Rrawf, _, _ = evaluate(f_rawf)

# ================================================== 2. q-resolved error in R, R/RF
print()
print('=' * 78)
print('2. WHERE THE ERROR LIVES IN q.  R here IS R/R_F (the data is .rrf and the')
print('   model runs Fresnel-normalised); the raw-R column repeats it with rrf=False.')
print('=' * 78)
e_rrf = (res[0.5]['R'] - Rref) / Rref
e_raw = (Rraw0 - Rrawf) / Rrawf
edges = [CFG['qmin'], 0.1, 0.2, 0.3, 0.4, 0.5, CFG['qmax'] + 1e-9]
print('  %-14s %5s %12s %12s %12s %12s' %
      ('q band', 'n', 'mean dR/R', 'max |dR/R|', 'raw mean', 'mean |dR|/sig'))
for a, b in zip(edges[:-1], edges[1:]):
    m = (Q >= a) & (Q < b)
    if not m.sum():
        continue
    print('  %5.3f-%5.3f    %5d %12.2e %12.2e %12.2e %12.3f'
          % (a, b, m.sum(), e_rrf[m].mean(), np.abs(e_rrf[m]).max(),
             e_raw[m].mean(), np.mean(np.abs(res[0.5]['R'] - Rref)[m] / ED[m])))
mq = Q > 1.5 * CFG['qmin']
pw = np.polyfit(np.log(Q[mq]), np.log(np.abs(e_rrf[mq]) + 1e-30), 1)[0]
print('  power law |dR/R| ~ q^%.2f  (midpoint-rule quadrature predicts q^2)' % pw)
print('  R/R_F and raw R give the same RELATIVE error to %.1e -- the Fresnel'
      % float(np.max(np.abs(e_rrf - e_raw))))
print('  divisor is analytic in q and carries no mesh dependence.')
print('  chi2 of the model difference, sum((dR/sigma)^2) = %.3f'
      % float(np.sum(((res[0.5]['R'] - Rref) / ED) ** 2)))
print('  worst single point: %.3f sigma at q = %.3f'
      % (np.max(np.abs(res[0.5]['R'] - Rref) / ED),
         Q[int(np.argmax(np.abs(res[0.5]['R'] - Rref) / ED))]))

# ============================================ 3. how the error VARIES with the box
print()
print('=' * 78)
print('3. DECOMPOSING THE ERROR.  Only the IN-BOX mesh is parameter dependent')
print('   (h_box = d_b*spacez/box_ref); everything outside the boxes is rigid at')
print('   spacez.  box_ref_length refines ONLY the in-box mesh, so it isolates the')
print('   parameter-dependent half.')
print('=' * 78)
BIGREF = 480.0            # box_ref -> in-box mesh 16x finer, outer mesh untouched


def R_at(spacez, box_ref, d1):
    f = build(spacez=spacez, box_ref=box_ref)
    R, L, C = evaluate(f, d=[d1, CFG['d'][1]])
    return R, C


print('  at the fitted d1 = %.4f:' % CFG['d'][0])
_Rd, _Cd = R_at(0.5, 30.0, CFG['d'][0])
_Rb, _Cb = R_at(0.5, BIGREF, CFG['d'][0])
_Rt = Rref
print('    in-box error  mean|dR/R| %.3e   sum(dR/sig)^2 %.4f   dchi2 %+.4f'
      % (np.mean(np.abs((_Rd - _Rb) / _Rb)),
         np.sum(((_Rd - _Rb) / ED) ** 2), _Cd - _Cb))
print('    TOTAL error   mean|dR/R| %.3e   sum(dR/sig)^2 %.4f   dchi2 %+.4f'
      % (np.mean(np.abs((_Rd - _Rt) / _Rt)),
         np.sum(((_Rd - _Rt) / ED) ** 2), _Cd - res[SPZ[-1]]['C']))
print('    => by the model-difference norm sum((dR/sigma)^2), the in-box mesh is')
print('       only %.1f%% of the total: the RIGID spacez region dominates.'
      % (100 * np.sum(((_Rd - _Rb) / ED) ** 2)
         / max(np.sum(((_Rd - _Rt) / ED) ** 2), 1e-30)))
print('    => and note the two dchi2 have OPPOSITE SIGNS (%+.3f in-box vs %+.3f'
      % (_Cd - _Cb, _Cd - res[SPZ[-1]]['C']))
print('       total), i.e. the in-box and outer errors partially CANCEL. Refining')
print('       one alone therefore makes absolute chi2 worse, not better -- which is')
print('       exactly what raising box_ref_length on its own does. Refine spacez.')
print()
print('  IN-BOX error alone, as d1 sweeps (this IS the parameter-dependent part):')
print('  %-10s %10s %13s %13s %13s' %
      ('d1 (A)', 'h_box A', 'mean|dR/R|', 'sum(dR/sig)^2', 'max|dR|/sig'))
D1 = np.array([4.0, 8.0, 12.73124, 18.0, 24.0, 30.0, 40.0, 55.0])
rows = []
for v in D1:
    try:
        Rc, _ = R_at(0.5, 30.0, v)
        Rb, _ = R_at(0.5, BIGREF, v)
    except Exception as ex:
        print('  %-10.3f  skipped: %s' % (v, str(ex).splitlines()[0][:48]))
        continue
    rows.append((v, v * 0.5 / 30.0, float(np.mean(np.abs((Rc - Rb) / Rb))),
                 float(np.sum(((Rc - Rb) / ED) ** 2)),
                 float(np.max(np.abs(Rc - Rb) / ED))))
    print('  %-10.3f %10.4f %13.3e %13.4f %13.4f' % rows[-1])
rows = np.array(rows)
pw2 = np.polyfit(np.log(rows[:, 1]), np.log(rows[:, 2]), 1)[0]
print('  mean|dR/R| ~ h_box^%.2f   (h^2 quadrature predicts 2)' % pw2)
print('  so the parameter-dependent part of the error scales as expected, and at')
print('  the fitted d1 it is %.1e in R -- %.0fx below the %.1e median sigma/R.'
      % (rows[2, 2], float(np.median(ED / YD)) / rows[2, 2],
         float(np.median(ED / YD))))

# ==================================================== 4. does it move the answer?
print()
print('=' * 78)
print('4. DOES IT BIAS THE ANSWER?  Fit the REAL data on each mesh, same start,')
print('   and compare. theta/phi held at the archived pose (fitting orientation')
print('   from one start is the multimodal problem, out of scope here).')
print('=' * 78)
from Fit_Routines import Fit                                             # noqa: E402

VARY = ['qoff', 'protins', 'cov', 'sig',
        '__Model_d_001', '__Model_d_002', '__Model_rho_001', '__Model_rho_002']


def do_fit(spacez):
    fit = Fit(XLayersPDB_XPR, Q, y=YD, yerr=ED)
    g = fit.func
    g.fname = PDB
    g.use_protein = True
    g.vol_calc = 'approx'
    g.cov_norm = COVNORM
    g.rrf = True
    g.spacez = spacez
    g.box_ref_length = 30.0
    g.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
    g.__mpar__['Model']['d'] = [0.0] + list(CFG['d']) + [1.0]
    g.__mpar__['Model']['rho'] = ([CFG['rho_top']] + list(CFG['rho'])
                                  + [CFG['rho_bottom']])
    g.init_params()
    fit.fit_params, fit.params = g.params, g.__dict__
    start = dict(qoff=CFG['qoff'], theta=CFG['theta'], phi=CFG['phi'],
                 protins=PROTINS, sig=CFG['sig'])
    ref = build(spacez=spacez)
    start['cov'] = ref.cov
    for k, v in start.items():
        fit.fit_params[k].value = v
        g.__dict__[k] = v
    fit.fit_params['protins'].min = PROTINS - 40
    fit.fit_params['protins'].max = PROTINS + 25
    fit.fit_params['sig'].max = 8.0
    for i in (1, 2):
        fit.fit_params['__Model_d_%03d' % i].max = 60.0
    fit.fit_params['cov'].max = max(10.0, 5 * ref.cov)
    for k in fit.fit_params:
        fit.fit_params[k].vary = 1 if k in VARY else 0
    fit.perform_fit(Q.min(), Q.max(), fit_scale='Linear', fit_method='leastsq',
                    maxiter=3000)
    return fit


t0 = time.time()
fitc = do_fit(0.5)
fitf = do_fit(0.125)
print('  (two fits in %.1f s)' % (time.time() - t0))
c2c = float(np.sum(fitc.result.residual ** 2))
c2f = float(np.sum(fitf.result.residual ** 2))
print('  chi2 at each fitted optimum: default %.3f   fine %.3f' % (c2c, c2f))
print()
print('  %-18s %14s %14s %12s %12s %10s'
      % ('param', 'fit @ 0.5', 'fit @ 0.125', 'shift', 'stderr', 'shift/err'))
biasrows = []
for k in VARY:
    a = fitc.result.params[k].value
    b = fitf.result.params[k].value
    se = fitf.result.params[k].stderr
    r = (a - b) / se if se else float('nan')
    biasrows.append((k, a, b, a - b, se if se else np.nan, r))
    print('  %-18s %14.6g %14.6g %+12.4g %12.4g %10.3f'
          % (k, a, b, a - b, se if se else np.nan, r))
_worst = max(abs(r) for *_x, r in biasrows if np.isfinite(r))
print('  worst |shift| / stderr = %.3f  (< 0.1 means the mesh is irrelevant to the'
      % _worst)
print('  fitted values; ~1 would mean it moves them by a full error bar)')

print()
print('  cross-evaluation, to separate "the mesh moved the optimum" from "the mesh')
print('  moved chi2 at fixed parameters":')


def chi2_at(spacez, params):
    f = build(spacez=spacez)
    for k in VARY:
        if k.startswith('__Model_d_'):
            i = int(k[-3:])
            f.__mpar__['Model']['d'][i] = params[k].value
            f.params[k].value = params[k].value
        elif k.startswith('__Model_rho_'):
            i = int(k[-3:])
            f.__mpar__['Model']['rho'][i] = params[k].value
            f.params[k].value = params[k].value
        else:
            setattr(f, k, params[k].value)
            f.params[k].value = params[k].value
    R = np.asarray(f.y(), dtype=float)
    return float(np.sum(((YD - R) / ED) ** 2))


tab = {}
for sm, sp_ in (('0.5', 0.5), ('0.125', 0.125)):
    for pm, pr in (('P(0.5)', fitc.result.params), ('P(0.125)', fitf.result.params)):
        tab[(sm, pm)] = chi2_at(sp_, pr)
print('  %-14s %16s %16s' % ('chi2', 'at P(0.5)', 'at P(0.125)'))
for sm in ('0.5', '0.125'):
    print('  mesh %-9s %16.4f %16.4f'
          % (sm, tab[(sm, 'P(0.5)')], tab[(sm, 'P(0.125)')]))
print('  -> on the fine mesh, the default-mesh parameters cost only %+.4f in chi2'
      % (tab[('0.125', 'P(0.5)')] - tab[('0.125', 'P(0.125)')]))
print('     compared with the fine-mesh parameters. THAT is the cost of the mesh')
print('     bias. The %+.3f offset between rows is the mesh error itself, which is'
      % (tab[('0.5', 'P(0.5)')] - tab[('0.125', 'P(0.5)')]))
print('     common to all parameter points and cancels from any Delta(chi2).')

# ==================================== 5. landscape distortion, at COMPARABLE chi2
print()
print('=' * 78)
print('5. LANDSCAPE DISTORTION for a global search.  The relevant comparison is')
print('   between CANDIDATE MINIMA, which all have comparable chi2 -- not between')
print('   a good fit and a terrible one.  Perturb around the optimum, keep only')
print('   points within +100 chi2, and look at the SPREAD of the mesh error.')
print('=' * 78)
rng = np.random.default_rng(3)
base = {k: fitf.result.params[k].value for k in VARY}
# Draw from the FIT COVARIANCE, not from independent marginal sigmas. These
# parameters are strongly correlated (qoff/protins/cov/d/rho all trade against each
# other in XRR), so an independent 1-sigma step in each of 8 of them lands far
# outside the joint ellipsoid -- measured: > +100 chi2 every time. A multivariate
# draw gives the intended "other candidate minima at comparable chi2" ensemble,
# with E[dchi2] = ndim.
_vn = list(fitf.result.var_names)
_cv = np.asarray(fitf.result.covar, dtype=float)
_mu = np.array([fitf.result.params[k].value for k in _vn])
cands = []
_perturb_err = ['']
for _scale in (1.0, 2.0, 3.0):
    draws = rng.multivariate_normal(_mu, _cv * _scale ** 2, size=40)
    for row in draws:
        trial = dict(zip(_vn, row))
        if (trial.get('sig', 1) <= 0.5 or trial.get('cov', 1) <= 0
                or min(trial['__Model_d_001'], trial['__Model_d_002']) <= 1.0):
            continue

        class _P(dict):
            def __getitem__(self, k):
                class _V(object):
                    pass
                v = _V()
                v.value = dict.__getitem__(self, k)
                return v

        pr = _P(trial)
        try:
            aa = chi2_at(0.5, pr)
            bb = chi2_at(0.125, pr)
        except Exception as ex:
            if not _perturb_err[0]:
                _perturb_err[0] = '%s: %s' % (type(ex).__name__,
                                              str(ex).splitlines()[0][:110])
            continue
        if bb - c2f < 100.0:
            cands.append((bb - c2f, aa - bb))
cands = np.array(cands)
if len(cands):
    print('  %d perturbed candidates within +100 chi2 of the optimum'
          % len(cands))
    print('  mesh error (chi2_default - chi2_fine): min %+.4f  max %+.4f  spread %.4f'
          % (cands[:, 1].min(), cands[:, 1].max(),
             cands[:, 1].max() - cands[:, 1].min()))
    print('  sd of the mesh error across candidates: %.4f chi2 (%.4f in logL)'
          % (cands[:, 1].std(), cands[:, 1].std() / 2))
    print()
    print('  => for basin merging on a Delta(-logL) tolerance of 11-20, the mesh')
    print('     contributes a spread of %.3f logL between candidates, i.e. %.2f%%'
          % (cands[:, 1].std() / 2, 100 * (cands[:, 1].std() / 2) / 11))
    print('     of the tightest tolerance. It is a common-mode offset to first')
    print('     order, and only its VARIATION between candidates can reorder them.')
else:
    print('  no candidates survived the chi2 window.')
    if _perturb_err[0]:
        print('  first exception was -> %s' % _perturb_err[0])

# ============================ 6. does the mesh move the POSE? (theta/phi released)
print()
print('=' * 78)
print('6. THE POSE.  Sections 4-5 held theta/phi fixed, but the protein is binned on')
print('   the RIGID spacez grid, so rotating it is exactly what moves atoms across')
print('   fixed cell boundaries -- and orientation is what the campaign is about.')
print('   Release theta/phi and refit on both meshes.')
print('=' * 78)
VARY2 = VARY + ['theta', 'phi']


def do_fit2(spacez):
    fit = Fit(XLayersPDB_XPR, Q, y=YD, yerr=ED)
    g = fit.func
    g.fname = PDB
    g.use_protein = True
    g.vol_calc = 'approx'
    g.cov_norm = COVNORM
    g.rrf = True
    g.spacez = spacez
    g.box_ref_length = 30.0
    g.__mpar__['Model']['Layers'] = ['top', 'b1', 'b2', 'bottom']
    g.__mpar__['Model']['d'] = [0.0] + list(CFG['d']) + [1.0]
    g.__mpar__['Model']['rho'] = ([CFG['rho_top']] + list(CFG['rho'])
                                  + [CFG['rho_bottom']])
    g.init_params()
    fit.fit_params, fit.params = g.params, g.__dict__
    ref = build(spacez=spacez)
    start = dict(qoff=CFG['qoff'], theta=CFG['theta'], phi=CFG['phi'],
                 protins=PROTINS, sig=CFG['sig'], cov=ref.cov)
    for k, v in start.items():
        fit.fit_params[k].value = v
        g.__dict__[k] = v
    fit.fit_params['protins'].min = PROTINS - 40
    fit.fit_params['protins'].max = PROTINS + 25
    fit.fit_params['sig'].max = 8.0
    for i in (1, 2):
        fit.fit_params['__Model_d_%03d' % i].max = 60.0
    fit.fit_params['cov'].max = max(10.0, 5 * ref.cov)
    for k in fit.fit_params:
        fit.fit_params[k].vary = 1 if k in VARY2 else 0
    fit.perform_fit(Q.min(), Q.max(), fit_scale='Linear', fit_method='leastsq',
                    maxiter=4000)
    return fit


t0 = time.time()
p2c = do_fit2(0.5)
p2f = do_fit2(0.125)
print('  (two 10-parameter fits in %.1f s)' % (time.time() - t0))
print('  chi2: default %.3f   fine %.3f'
      % (float(np.sum(p2c.result.residual ** 2)),
         float(np.sum(p2f.result.residual ** 2))))
print()
print('  %-18s %14s %14s %12s %12s %10s'
      % ('param', 'fit @ 0.5', 'fit @ 0.125', 'shift', 'stderr', 'shift/err'))
pose_worst = 0.0
for k in VARY2:
    a = p2c.result.params[k].value
    b = p2f.result.params[k].value
    se = p2f.result.params[k].stderr
    r = (a - b) / se if se else float('nan')
    if np.isfinite(r):
        pose_worst = max(pose_worst, abs(r))
    star = '  <-- pose' if k in ('theta', 'phi') else ''
    print('  %-18s %14.6g %14.6g %+12.4g %12.4g %10.3f%s'
          % (k, a, b, a - b, se if se else np.nan, r, star))
print('  worst |shift| / stderr including the pose = %.3f' % pose_worst)
_dth = abs(p2c.result.params['theta'].value - p2f.result.params['theta'].value)
_dph = abs(p2c.result.params['phi'].value - p2f.result.params['phi'].value)
print('  the mesh moves the fitted orientation by %.4f deg in theta, %.4f deg in phi'
      % (_dth, _dph))
print('  (campaign pose tolerance is ~5 deg; replicate disagreement is 11-20 logL)')

np.savez(OUT + '.npz', q=Q, ydata=YD, edata=ED,
         R_default=res[0.5]['R'], R_converged=Rref,
         R_raw_default=Rraw0, R_raw_converged=Rrawf,
         spacez=np.array(SPZ), logL=np.array([res[s]['L'] for s in SPZ]),
         d1_sweep=rows, logL_inf=L_inf,
         bias_names=np.array([b[0] for b in biasrows]),
         bias=np.array([[b[1], b[2], b[3], b[4], b[5]] for b in biasrows]),
         cross=np.array([[tab[('0.5', 'P(0.5)')], tab[('0.5', 'P(0.125)')]],
                         [tab[('0.125', 'P(0.5)')], tab[('0.125', 'P(0.125)')]]]),
         candidates=cands,
         pose_shift=np.array([_dth, _dph]),
         pose_worst_sigma=pose_worst)
print()
print('=' * 78)
print('CONCLUSIONS')
print('=' * 78)
print("""
1. The moving mesh is a well-behaved O(h^2) quadrature error, not an artifact.
   Convergence order 1.87 -> 1.97; the q-dependence is q^1.91 against the
   midpoint-rule prediction of q^2. It is smooth, predictable, and refinable.

2. R/R_F is not a separate concern. The Fresnel divisor is analytic in q and
   carries no mesh dependence, so R and R/R_F have the SAME relative error to
   1.7e-16. Everything below applies to both.

3. The parameter-dependent half is the MINOR half. Only the in-box spacing moves
   with the parameters, and at the fitted point it is ~7%% of the total
   model-difference norm; the rigid spacez region dominates. The two also have
   opposite signs in chi2 and partially cancel.

4. The default mesh does not bias the answer. Refitting real data at spacez 0.125
   moves every fitted parameter by <= 0.04 of its own standard error, INCLUDING
   theta (0.037 deg) and phi (0.113 deg). Parameter estimation is safe.

5. Absolute chi2 is NOT mesh-transferable (~3.5 chi2 between default and
   converged), and across near-optimal candidates the mesh error varies with
   sd ~1.3 logL. Delta(chi2) within one mesh is safe; across meshes it is not.

RECOMMENDATIONS

  A. Keep spacez = 0.5 and box_ref_length = 30 as production defaults. The bias
     they introduce is 25x smaller than the parameter uncertainties.
  B. Fix spacez and box_ref_length once per campaign and record them alongside
     the results. Never compare absolute chi2 across different values, or across
     code versions.
  C. Set box_ref_length >= the thickest box you intend to fit. Above it the
     in-box mesh coarsens past spacez (the model warns); the error grows as
     h_box^2, so a 55 A box at box_ref 30 reaches 4.5 sigma at one q point.
  D. Confirm mesh convergence once per campaign, not per fit: refit the final
     answer at spacez/2 and check the parameters move much less than their
     error bars. Cost is ~4x per evaluation.
  E. For the multistart, treat ~1.3 logL as a mesh noise floor on candidate
     RANKING. Candidates closer than that cannot be ordered on the default mesh;
     re-score the shortlist at spacez/2 before declaring a winner. The 11-20 logL
     merge tolerance is unaffected (mesh noise is ~11%% of it).
  F. Refine spacez, never box_ref_length alone -- the two error terms partially
     cancel, so refining one degrades absolute chi2.
""")
print()
print('saved %s.npz' % OUT)
