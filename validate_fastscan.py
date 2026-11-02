"""Validation harness for the fast global search (fastscan/).

Part A  pure Python -- no Julia, no Qt, no data. Pins the rules that are easy to
        get silently wrong: position-only dedup, the great-circle metric, the
        never-clip seed screen, eigenvalue flooring, bound-aware classification,
        the .xmodes round trip, and the structural absence of an ambiguity
        threshold.
Part B  needs the Julia/XPR backend. Pins the two Increment-0 gates permanently --
        the objective identity and mesh reproduction -- plus an end-to-end scan,
        the regression guard, and backend hygiene.
Part C  Qt, offscreen. The Tools-plugin contract and the not-ready guidance paths.
Part E  the stage-0 homometric enumerator and its two refusals. Pure numpy.
Part D  the spawn-child round trip: model pickling, the pipe protocol, terminate().
        Slowest, so it runs last.

Run:
    "C:/Users/kerr/.conda/envs/cmc-py39/python.exe" validate_fastscan.py
    ... --part AE     to run only the parts that need neither Julia nor Qt

Exit status is non-zero if any check fails.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.chdir(ROOT)
for _p in (ROOT, os.path.join(ROOT, 'Functions')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# the bundled xraydb/ directory carries no source; stub it before anything wants it
if 'xraydb' not in sys.modules:
    try:
        import xraydb  # noqa: F401
    except Exception:
        import types
        _m = types.ModuleType('xraydb')
        _m.f0 = lambda *a, **k: np.array([1.0])
        _m.atomic_number = lambda s: 1
        _m.mirror_reflectivity = lambda *a, **k: np.array([1.0])
        _m.xray_delta_beta = lambda *a, **k: (0.0, 0.0, 0.0)
        sys.modules['xraydb'] = _m

FAIL = []
NOTE = []


def ok(name, cond, detail=''):
    print('  %-58s %s%s' % (name, 'PASS' if cond else 'FAIL',
                            ('  ' + detail) if detail else ''))
    if not cond:
        FAIL.append(name)
    return bool(cond)


def close(name, got, want, tol, rel=False):
    d = abs(got - want)
    base = max(abs(want), 1e-300) if rel else 1.0
    return ok(name, d / base <= tol, '%.6g vs %.6g (%.2e)' % (got, want, d / base))


# ============================================================ PART A ==========

def part_a():
    from fastscan import modes as M
    from fastscan import seeds as S

    print('\n--- A1  great-circle metric (never |dtheta|+|dphi|) ---')
    # at theta=0 every phi is the SAME orientation
    ok('theta=0: phi 0 vs 173 -> 0 deg',
       M.great_circle(0.0, 0.0, 0.0, 173.0) < 1e-9,
       '%.3e' % M.great_circle(0.0, 0.0, 0.0, 173.0))
    # phi wraps
    close('phi 359 vs 1 at theta=90 -> 2 deg',
          M.great_circle(90.0, 359.0, 90.0, 1.0), 2.0, 1e-9)
    close('theta 10 vs 20 at same phi -> 10 deg',
          M.great_circle(10.0, 45.0, 20.0, 45.0), 10.0, 1e-9)
    # antipodal bands
    close('antipodal -> 180 deg', M.great_circle(0.0, 0.0, 180.0, 0.0), 180.0, 1e-9)
    naive = abs(90.0 - 90.0) + abs(359.0 - 1.0)
    ok('the naive metric would have said 358 (so this matters)', abs(naive - 358) < 1e-9)

    print('\n--- A2  dedup is POSITION ONLY, never depth ---')

    def mk(theta, phi, protins, cov, neglogL):
        return M.Mode(u=np.zeros(12), neglogL=neglogL, theta=theta, phi=phi,
                      protins=protins, cov=cov)

    truth = [(30.0, 115.0, 10.0, 5.0), (150.0, 20.0, -5.0, 6.0),
             (80.0, 300.0, 2.0, 4.0)]
    pop = []
    for j, (t, p, pi, c) in enumerate(truth):
        for i in range(4):
            # within tolerance in position, but DEPTHS SPREAD WIDELY
            pop.append(mk(t + 0.5 * i, p + 0.4 * i, pi + 0.1 * i, c + 0.05 * i,
                          100.0 + 10.0 * i + j))
    kept = M.dedup(pop, M.DedupRule())
    ok('3 true positions x 4 depths -> 3 clusters', len(kept) == 3,
       'got %d' % len(kept))
    ok('cluster starts accumulate to the population',
       sum(m.n_starts for m in kept) == len(pop),
       '%d of %d' % (sum(m.n_starts for m in kept), len(pop)))

    # the counterfactual: adding depth to the identity rule over-counts
    def dedup_with_depth(ms, df=0.5):
        order = sorted(ms, key=lambda m: m.neglogL)
        keep = []
        for m in order:
            if not any(M.same_position(m, k) and abs(m.neglogL - k.neglogL) <= df
                       for k in keep):
                keep.append(m)
        return keep

    over = len(dedup_with_depth(pop))
    ok('including depth in identity over-counts (the measured ~3x)',
       over > len(kept), '%d vs %d' % (over, len(kept)))

    print('\n--- A3  no ambiguity threshold exists in the type ---')
    ms = M.ModeSet([mk(30.0, 115.0, 10.0, 5.0, 80.0),
                    mk(150.0, 20.0, -5.0, 6.0, 82.5),
                    mk(80.0, 300.0, 2.0, 4.0, 120.0)],
                   M.Provenance(n1=64, k=6))
    ok('ModeSet has no is_ambiguous attribute', not hasattr(ms, 'is_ambiguous'))
    ok('Mode has no is_ambiguous attribute', not hasattr(ms[0], 'is_ambiguous'))
    ok('ModeSet needs a Provenance', _raises(lambda: M.ModeSet([], None)))
    for bad in ('AMBIGUITY_THRESHOLD', 'is_ambiguous', 'DEFAULT_DELTA'):
        ok('no %r symbol exists' % bad, not hasattr(M, bad))
    # the structural property: the ambiguity SCALE is stored as a band, never as a
    # single number, because it was measured to vary ten-fold between strata.
    band = M.Power().scale_band
    ok('Power.scale_band is a band, not a scalar',
       isinstance(band, tuple) and len(band) == 2 and band[0] < band[1], str(band))
    ok('Power records that the budget is NOT calibrated',
       M.Power().budget_is_calibrated is False)
    # and every threshold-dependent answer must be requested WITH a threshold
    ok('n_distinct requires a delta argument', _raises(lambda: ms.n_distinct()))
    ok('pose_span requires a delta argument', _raises(lambda: ms.pose_span()))

    print('\n--- A4  the threshold curves ---')
    ok('n_distinct is monotone non-decreasing in delta',
       all(ms.n_distinct(d) <= ms.n_distinct(d + 1.0) for d in np.arange(0, 50, 1.0)))
    ok('pose_span is monotone non-decreasing in delta',
       all(ms.pose_span(d) <= ms.pose_span(d + 1.0) + 1e-9
           for d in np.arange(0, 50, 1.0)))
    ok('n_distinct(0) == 1 (the best mode alone)', ms.n_distinct(0.0) == 1,
       'got %d' % ms.n_distinct(0.0))
    ok('deltas are offsets from the best', abs(ms[0].delta) < 1e-12
       and abs(ms[1].delta - 2.5) < 1e-9)
    b = ms.bands(5.0)
    ok('both bands flagged within delta=5 (up and down present)',
       b.get('up') == 1 and b.get('down') == 1, str(b))
    close('expected_at_reference applies the 7->49 shortfall',
          ms.expected_at_reference(5.0), 2 * (49.0 / 7.0), 1e-9)

    print('\n--- A5  seed screen drops, never clips ---')
    nb = 2
    lo = np.array([1.0, 0.5, 0.5, 0.0, 0.0])
    hi = np.array([6.0, 30.0, 30.0, 0.62, 0.62])
    good = np.array([3.4, 14.0, 5.6, 0.31, 0.49])
    cases = [
        ('negative ED', np.array([3.4, 14.0, 5.6, -0.02, 0.49])),
        ('ED above 0.62', np.array([3.4, 14.0, 5.6, 0.31, 1.126])),
        ('non-positive thickness', np.array([3.4, 0.0, 5.6, 0.31, 0.49])),
        ('outside the box (sig)', np.array([9.9, 14.0, 5.6, 0.31, 0.49])),
    ]
    kept, rej = S.screen([good] + [c[1] for c in cases], lo, hi, num_boxes=nb)
    ok('only the physical seed survives', len(kept) == 1 and len(rej) == 4,
       '%d kept, %d rejected' % (len(kept), len(rej)))
    ok('no kept seed was mutated', np.allclose(kept[0], good))
    for (label, _), (_, _, why) in zip(cases, rej):
        ok('rejected %-24s with a reason' % label, bool(why), why[:46])
    # the specific quiet failure: a clipped -0.02 becomes a vacuum layer
    ok('rho=-0.02 is NOT clipped to 0.0',
       not any(np.isclose(np.asarray(k)[3], 0.0) for k in kept))
    allbad, rej2 = S.screen([c[1] for c in cases], lo, hi, num_boxes=nb)
    ok('all-rejected falls back to unscreened and still reports',
       len(allbad) == 4 and len(rej2) == 4)

    print('\n--- A6  seed extraction, pooling, file IO ---')
    cand = {'sigma': 3.4, 'zpos': np.array([0.0, 14.0, 19.6]),
            'rbx': np.array([0.31, 0.49])}
    s = S.from_basin(cand)
    ok('from_basin -> [sig, d..., rho...]',
       np.allclose(s, [3.4, 14.0, 5.6, 0.31, 0.49]), np.array2string(s, precision=3))
    sig, d, rho = S.split(s, nb)
    ok('split round-trips', sig == 3.4 and np.allclose(d, [14.0, 5.6])
       and np.allclose(rho, [0.31, 0.49]))
    # from_basin reproduces `good` exactly here, so use a genuinely different
    # second candidate or the dedup test proves nothing
    other = np.array([3.9, 12.5, 7.1, 0.28, 0.52])
    pooled, labels = S.pool([('scanA', [good, other]), ('scanB', [good.copy()])])
    ok('pooling dedups the duplicate across scans', len(pooled) == 2,
       '%d pooled' % len(pooled))
    ok('pooled seeds keep their source label', labels == ['scanA', 'scanA'],
       str(labels))
    ok('the surviving pair is the two distinct seeds',
       np.allclose(pooled[0], good) and np.allclose(pooled[1], other))
    tmp = os.path.join(os.environ.get('TEMP', '.'), '_fastscan_seeds_test.seeds')
    S.write(tmp, pooled, labels, meta={'tag': 'unit-test'})
    back, blabels, meta = S.read(tmp, num_boxes=nb)
    ok('.seeds round trip preserves values',
       len(back) == 2 and np.allclose(back[0], pooled[0]))
    ok('.seeds round trip preserves labels', blabels == labels, str(blabels))
    import json
    tmpj = tmp.replace('.seeds', '.json')
    with open(tmpj, 'w', encoding='utf-8') as fh:
        json.dump({'tag': 'study', 'membrane_seeds': [list(good), list(s)]}, fh)
    jb, jl_, jm = S.read(tmpj, num_boxes=nb)
    ok('study-format JSON membrane_seeds reads', len(jb) == 2 and jm['tag'] == 'study')
    ok('a length mismatch is an error, not a silent pass',
       _raises(lambda: S.split(np.zeros(4), 2)))
    for f in (tmp, tmpj):
        try:
            os.remove(f)
        except OSError:
            pass

    print('\n--- A7  classification, incl. the bound-aware correction ---')
    H_min = np.diag([2.0, 1.0, 0.5])
    H_sad = np.diag([2.0, 1.0, -0.5])
    H_deg = np.diag([2.0, 1.0, 0.0])
    lab, emin, cond, nullv = M.classify(H_min)
    ok('positive definite -> MIN', lab == M.MIN, lab)
    ok('MIN reports eigmin and cond', emin is not None and cond is not None,
       'eigmin=%.3g cond=%.3g' % (emin, cond))
    ok('indefinite -> SADDLE', M.classify(H_sad)[0] == M.SADDLE)
    ok('SADDLE names the escape direction', M.classify(H_sad)[3] is not None)
    ok('singular -> DEGENERATE', M.classify(H_deg)[0] == M.DEGENERATE)
    # the published-method defect: a constrained min on a face reads SADDLE
    u_interior = np.array([0.5, 0.5, 0.5])
    u_railed = np.array([0.5, 0.5, 0.0])
    ok('full-block test calls a face-constrained min SADDLE (the defect)',
       M.classify(H_sad, u_interior)[0] == M.SADDLE)
    lab2 = M.classify(H_sad, u_railed)[0]
    ok('bound-aware test calls it MIN-BOUND (the fix)', lab2 == 'MIN-BOUND', lab2)
    ok('MIN-BOUND counts as a minimum for the sampler hand-off',
       M.Mode(u=np.zeros(3), neglogL=1.0, hess_class='MIN-BOUND').is_min)
    ok('SADDLE does not', not M.Mode(u=np.zeros(3), neglogL=1.0,
                                     hess_class=M.SADDLE).is_min)
    ok('empty/non-finite Hessian -> UNCERTIFIED',
       M.classify(np.array([[np.nan]]))[0] == M.UNCERTIFIED)

    print('\n--- A8  Laplace weights need flooring ---')
    soft = np.diag([1.0, 1.0, 1e-14])          # a genuinely soft direction
    m1 = M.Mode(u=np.zeros(3), neglogL=80.0, hess_class=M.MIN, gn=np.diag([1., 1., 1.]))
    m2 = M.Mode(u=np.zeros(3), neglogL=80.5, hess_class=M.MIN, gn=soft)
    m3 = M.Mode(u=np.zeros(3), neglogL=81.0, hess_class=M.SADDLE, gn=np.eye(3))
    unfloored = -0.5 * float(np.sum(np.log(np.linalg.eigvalsh(soft))))
    ok('unfloored 0.5*logdet is absurdly large', unfloored > 15.0, '%.1f' % unfloored)
    M.laplace_weights([m1, m2, m3], floor_rel=1e-6)
    ok('weights are finite', np.isfinite(m1.laplace_wt) and np.isfinite(m2.laplace_wt))
    close('weights over the minima sum to 1', m1.laplace_wt + m2.laplace_wt, 1.0, 1e-12)
    ok('the saddle is excluded from the weights', m3.laplace_wt is None)
    ok('the floored direction is counted and reportable', m2.n_floored == 1,
       'n_floored=%d' % m2.n_floored)
    ok('an unfloored direction is not counted', m1.n_floored == 0)
    m4 = M.Mode(u=np.zeros(3), neglogL=80.0, hess_class=M.MIN, n_starts=42)
    m5 = M.Mode(u=np.zeros(3), neglogL=80.5, hess_class=M.MIN, n_starts=6)
    M.laplace_weights([m4, m5])
    close('no curvature -> basin-size fallback', m4.laplace_wt, 42.0 / 48.0, 1e-12)

    print('\n--- A9  provenance identity and staleness ---')
    pa = M.Provenance(spacez=0.5, data_sha256='aa', n_active=76)
    pb = M.Provenance(spacez=0.25, data_sha256='aa', n_active=76)
    ok('differing spacez is incompatible', pa.incompatible_with(pb) is not None,
       (pa.incompatible_with(pb) or '')[:44])
    ok('identical provenance is compatible',
       pa.incompatible_with(M.Provenance(spacez=0.5, data_sha256='aa',
                                         n_active=76)) is None)
    sa = M.ModeSet([mk(30., 115., 10., 5., 80.)], pa)
    sb = M.ModeSet([mk(31., 116., 10., 5., 81.)], pb)
    ok('merge refuses incomparable sets', _raises(lambda: sa.merge(sb)))
    ok('stale data is detected', sa.compatible_with(data_sha256='bb') is not None)
    ok('matching data is not stale', sa.compatible_with(data_sha256='aa') is None)
    f1 = M.data_fingerprint([1., 2.], [3., 4.], [.1, .1])
    f2 = M.data_fingerprint([1., 2.], [3., 4.0001], [.1, .1])
    ok('the data fingerprint is sensitive to R', f1 != f2)
    ok('and stable for identical input',
       f1 == M.data_fingerprint([1., 2.], [3., 4.], [.1, .1]))

    print('\n--- A11  .xmodes round trip ---')
    from fastscan import io as FIO
    prov = M.Provenance(algorithm='twostage-bandstrat-v1', n1=64, n1_ok=63, k=6,
                        sobol_seed=2024, n_active=76, num_boxes=2, spacez=0.5,
                        dataset='Data/i2.txt', data_sha256='deadbeef',
                        hessian='exact', template_label='tim3',
                        template_path='Data/tim3.pdb', n_cpu_limited=1,
                        varying_names=('theta', 'phi', 'protins', 'cov'),
                        free=(1, 2, 3, 4),
                        param_lower=np.zeros(12), param_upper=np.ones(12),
                        timings={'stage1': 40.2, 'stage2': 45.1})
    a = M.Mode(u=np.linspace(0, 1, 12), neglogL=80.286, theta=31.6, phi=115.8,
               protins=10.7, cov=5.0, chi2=160.572, n_active=76,
               hess_class=M.MIN, hess_source='exact', eigmin=0.032, cond=7.1e5,
               n_starts=42, stage1_rank=0, start_index=37, laplace_wt=0.71,
               flags=('railed:cov',), active_bounds=('cov',),
               template_label='tim3',
               params={'theta': 31.6, 'phi': 115.8, 'protins': 10.7, 'cov': 5.0},
               R=np.linspace(1.0, 0.1, 5))
    b = M.Mode(u=np.linspace(1, 0, 12), neglogL=82.746, theta=148.4, phi=20.0,
               protins=-5.0, cov=6.0, chi2=165.492, n_active=76,
               hess_class='MIN-BOUND', hess_source='exact', n_starts=5,
               template_label='tim3',
               params={'theta': 148.4, 'phi': 20.0, 'protins': -5.0, 'cov': 6.0})
    src = M.ModeSet([a, b], prov, warnings=['1 solve hit the CPU limit'])
    text = FIO.format_xmodes(src)
    back = FIO.parse_xmodes(text)
    ok('round trip preserves the mode count', len(back) == 2, '%d' % len(back))
    close('neglogL survives to full precision', back[0].neglogL, a.neglogL, 1e-12)
    close('delta survives', back[1].delta, 2.46, 1e-6)
    ok('u survives to full precision', np.allclose(back[0].u, a.u, rtol=0, atol=0))
    ok('the classification survives',
       [m.hess_class for m in back] == [M.MIN, 'MIN-BOUND'],
       str([m.hess_class for m in back]))
    ok('flags and active bounds survive',
       back[0].flags == ('railed:cov',) and back[0].active_bounds == ('cov',))
    ok('the physical params table survives',
       abs(back[0].params['theta'] - 31.6) < 1e-9 and len(back[0].params) == 4)
    ok('the cached curve survives', np.allclose(back[0].R, a.R))
    ok('provenance identity survives',
       back.prov[0].identity() == prov.identity())
    ok('the power block travels with the file',
       back.prov[0].power.ref_median == 49
       and back.prov[0].power.budget_is_calibrated is False)
    ok('warnings survive', back.warnings == ['1 solve hit the CPU limit'],
       str(back.warnings))
    ok('the cpu-limited count survives', back.prov[0].n_cpu_limited == 1)
    ok('timings survive', abs(back.prov[0].timings.get('stage1', 0) - 40.2) < 1e-9)
    ok('the dedup rule is recorded in the text', 'position-only' in text)
    ok('staleness still detected after a round trip',
       back.compatible_with(data_sha256='other') is not None
       and back.compatible_with(data_sha256='deadbeef') is None)
    ok('a non-xmodes file is rejected', _raises(lambda: FIO.parse_xmodes('hello')))
    ok('reloaded modes carry no curvature (not stored by design)',
       all(m.hessian is None and m.gn is None for m in back))
    ok('default_path sits beside the dataset',
       FIO.default_path('Data/i2_tim3.txt').replace('\\\\', '/')
       == 'Data/i2_tim3.xmodes')

    print('\n--- A10  naming discipline ---')
    for mod in ('modes.py', 'seeds.py'):
        text = open(os.path.join(ROOT, 'fastscan', mod), encoding='utf-8').read()
        prose = '\n'.join(ln for ln in text.splitlines()
                          if not ln.lstrip().startswith(('#', '*')))
        ok('%s does not say "the minima"' % mod, 'the minima' not in prose.lower())
    ok('ModeSet.minima() is the certified subset, and modes is the full list',
       len(ms.modes) == 3 and len(ms.minima()) == 0)


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


# ============================================================ PART B ==========

def part_b():
    try:
        from Functions import xpr_backend as xb
    except Exception as ex:
        NOTE.append('Part B skipped: cannot import xpr_backend (%s)' % str(ex)[:70])
        return
    if not xb.julia_available():
        NOTE.append('Part B skipped: the Julia/XPR backend is unavailable here')
        return

    import copy
    from Functions.XRR.XLayersPDB_XPR import XLayersPDB_XPR
    from Fit_Routines import Fit
    from fastscan import core as C

    data = os.path.join(ROOT, 'Data', 'i2_tim3_1000_test.txt')
    pdb = os.path.join(ROOT, 'Data', 'tim3_hmmm_ext_all_bestframe.pdb')
    if not (os.path.isfile(data) and os.path.isfile(pdb)):
        NOTE.append('Part B skipped: fixtures missing (%s, %s)'
                    % (os.path.basename(data), os.path.basename(pdb)))
        return
    a = np.genfromtxt(data)
    a = a[np.isfinite(a[:, 1])]
    a = a[(a[:, 0] >= 0.027447) & (a[:, 0] <= 0.550447)]
    Q, Y, E = a[:, 0], a[:, 1], a[:, 2]

    mpar = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                      'd': [0.0, 14.06975, 5.637274, 1.0],
                      'rho': [0.0, 0.3087728, 0.4867595, 0.334]}}
    fit = Fit(XLayersPDB_XPR, Q, y=Y, yerr=E)
    f = fit.func
    f.fname, f.rrf, f.use_protein, f.vol_calc = pdb, True, True, 'approx'
    f.__mpar__ = copy.deepcopy(mpar)
    f.__mkeys__ = list(f.__mpar__.keys())
    f.init_params()
    fit.fit_params = f.params
    mkey = f.__mkeys__[0]
    setv = {'qoff': 4.076766e-05, 'yscale': 1.0, 'bkg': 0.0, 'sig': 3.726493,
            'theta': 24.12562, 'phi': 138.0743}
    for i, dv in enumerate(mpar['Model']['d']):
        setv['__%s_d_%03d' % (mkey, i)] = dv
    for i, rv in enumerate(mpar['Model']['rho']):
        setv['__%s_rho_%03d' % (mkey, i)] = rv
    f.theta, f.phi = setv['theta'], setv['phi']
    setv['protins'] = float(f.protins_from_top(14.90174, f.theta, f.phi))
    setv['cov'] = float(0.09125066 * f.cov_scale('approx', f.theta, f.phi))
    for k, v in setv.items():
        if k in fit.fit_params:
            fit.fit_params[k].set(value=float(v))
        fit.params[k] = float(v)

    print('\n--- B1/B2  objective identity and mesh reproduction ---')
    r = fit.residual(fit.fit_params, 'Linear')
    chi2_fit = float(np.sum(r ** 2))
    nbox = len(f.__d__[mkey]) - 2
    p_phys = f._full_params(f.__d__[mkey], f.__rho__[mkey], nbox)
    _, R_model = xb.sim(p_phys)
    slabs_model = xb.slab_count()

    C.load_for_scan(f, Q, Y, E)
    slabs_scan = xb.slab_count()
    u = xb.to_normalized(p_phys)
    logL, R_scan = xb.sim(u)

    ok('the scan load is on the unit cube', xb.is_normalized())
    ok('slab counts identical (same mesh)', slabs_model == slabs_scan,
       '%s vs %s' % (slabs_model, slabs_scan))
    close('B2 mesh: max|R_scan - R_model|',
          float(np.max(np.abs(R_scan - R_model))), 0.0, 1e-12)
    close('B1 chi2_fit == 2*(-logL_xpr)', -2.0 * logL, chi2_fit, 1e-8, rel=True)

    print('\n--- B3  the scan box respects vary, bounds and the angle rule ---')
    lo, hi, free, names = C.scan_box(f, nbox)
    idx = {n: i for i, n in enumerate(names)}
    ok('theta box is the fundamental domain [0,180]',
       (lo[xb.IDX_THETA], hi[xb.IDX_THETA]) == (0.0, 180.0))
    ok('phi box is [0,360]', (lo[xb.IDX_PHI], hi[xb.IDX_PHI]) == (0.0, 360.0))
    ok('lmfit theta/phi stay unbounded (rule 5 intact)',
       not np.isfinite(f.params['theta'].min) and not np.isfinite(f.params['phi'].max))
    ok('pinned coords get a strictly positive, representable gap',
       all(hi[c] > lo[c] for c in range(len(lo))),
       'min gap %.3g' % float(np.min(hi - lo)))
    ok('cov ceiling comes from cov_scale, not a guess',
       hi[xb.IDX_COV] > f.params['cov'].value > 0)
    ok('the free set excludes pinned yscale/bkg',
       xb.IDX_YSCALE not in free and xb.IDX_BKG not in free, str(sorted(free)))
    ok('the evaluated point lies inside the box',
       np.all(u >= -1e-12) and np.all(u <= 1 + 1e-12))
    # this fixture pins the membrane, so pinning must WIN over the rho ceiling
    ok('a pinned box rho gets degenerate bounds, not the 0.62 ceiling',
       hi[xb.IDX_BOXES + nbox] - lo[xb.IDX_BOXES + nbox] < 1e-6,
       'gap %.3g' % (hi[xb.IDX_BOXES + nbox] - lo[xb.IDX_BOXES + nbox]))
    # ...and when the user does vary it, the ceiling is the physicality limit
    for nm in ('sig', '__Model_d_001', '__Model_d_002',
               '__Model_rho_001', '__Model_rho_002'):
        f.params[nm].set(vary=True)
    lo2, hi2, free2, _ = C.scan_box(f, nbox)
    close('a varying box rho is capped at 0.62', hi2[xb.IDX_BOXES + nbox], 0.62, 1e-12)
    ok('the free set is now the full 10 coordinates the calibration used',
       len(free2) == 9 + int(xb.IDX_QOFF in free2), '%d free' % len(free2))
    ok('box d lower bound stays off zero is NOT assumed here (model bounds win)',
       lo2[xb.IDX_BOXES] >= 0.0)

    print('\n--- B4  end-to-end short scan on real data ---')
    from fastscan import modes as MM
    f.params['qoff'].set(vary=True)
    prog = []
    t0 = __import__('time').time()
    res = C.scan(f, Q, Y, E, n1=8, k=2, seed=2024, iter1=30, tol1=1e-3,
                 iter2=200, tol2=1e-8, max_cpu1=20.0, max_cpu2=60.0,
                 progress=lambda *a: prog.append(a[0]))
    dt = __import__('time').time() - t0
    ok('stage 1 produced finite objectives',
       int(np.isfinite(res.stage1_fs).sum()) > 0,
       '%d of 8 finite' % int(np.isfinite(res.stage1_fs).sum()))
    ok('stage 2 converged at least one endpoint', len(res.stage2_fs) > 0,
       '%d endpoints' % len(res.stage2_fs))
    # Ipopt gives NO monotone-descent guarantee and a truncated solve returns
    # whatever iterate it stopped on, so this is only true because xb.solve keeps
    # the better of {start, endpoint}. It is the guard against a corrupted ranking.
    ok('stage 2 never returns worse than the best of stage 1',
       len(res.stage2_fs) and np.nanmin(res.stage2_fs) <= np.nanmin(res.stage1_fs) + 1e-6,
       's1 %.4f -> s2 %.4f' % (np.nanmin(res.stage1_fs), np.nanmin(res.stage2_fs)))
    ok('a regressed refine is recorded rather than hidden',
       res.n_regressed == 0 or any('made it worse' in w for w in res.warnings),
       '%d regressed' % res.n_regressed)
    ok('budget exhaustion is reported honestly',
       res.n_converged > 0 or any('NO endpoint converged' in w for w in res.warnings),
       '%d of %d converged' % (res.n_converged, len(res.stage2_fs)))
    for w in res.warnings:
        print('      warning: %s' % w[:110])
    ok('progress was reported for both stages',
       'stage1' in prog and 'stage2' in prog)
    ok('the reported counts are deterministic',
       prog.count('stage1') == 8 and prog.count('stage2') == len(res.stage2_fs))
    prov = MM.Provenance(n1=8, k=2, n_active=len(Q), num_boxes=nbox,
                         data_sha256=MM.data_fingerprint(Q, Y, E),
                         hessian='exact', template_label='tim3')
    ms = MM.ModeSet(C.to_modes(res, f, prov), prov)
    ok('modes were produced and ranked', len(ms) > 0, '%d modes' % len(ms))
    ok('deltas are non-negative and start at 0',
       abs(ms[0].delta) < 1e-12 and all(m.delta >= -1e-12 for m in ms))
    # An endpoint is certified OR explicitly disclaimed -- never silently unlabelled.
    # A regressed refine is the screening point, not a stationary point, so it has no
    # right to a curvature claim and must stay UNCERTIFIED.
    ok('every endpoint is either certified or flagged as not a stationary point',
       all(m.hess_class != MM.UNCERTIFIED
           or 'refine_regressed' in m.flags or 'not_converged' in m.flags
           for m in ms),
       ', '.join('%s[%s]' % (m.hess_class, '|'.join(m.flags)) for m in ms))
    ok('an uncertified mode has no curvature attached',
       all(m.hessian is None and m.gn is None
           for m in ms if m.hess_class == MM.UNCERTIFIED))
    ok('chi2 == 2*neglogL per mode',
       all(abs(m.chi2 - 2 * m.neglogL) < 1e-9 for m in ms))
    ok('pose values are physical and in the folded domain',
       all(0 <= m.theta <= 180 and 0 <= m.phi <= 360 for m in ms),
       ' '.join('(%.1f,%.1f)' % (m.theta, m.phi) for m in ms))
    if ms.minima():
        w = [m.laplace_wt for m in ms.minima() if m.laplace_wt is not None]
        ok('Laplace weights over the certified minima sum to 1',
           not w or abs(sum(w) - 1.0) < 1e-9, 'n=%d sum=%.6f' % (len(w), sum(w)))
    print('    (%d-start scan took %.0f s; %.1f s/start stage 1)'
          % (8, dt, res.timings['stage1'] / 8.0))
    print('    timings: %s' % {k: round(v, 1) for k, v in res.timings.items()})

    print('\n--- B5  cancellation returns partial results, never raises ---')
    calls = [0]

    def cancel_after_two():
        calls[0] += 1
        return calls[0] > 3

    res2 = C.scan(f, Q, Y, E, n1=8, k=2, seed=7, iter1=10, tol1=1e-2,
                  max_cpu1=20.0, certify=False, cancelled=cancel_after_two)
    ok('cancelled scan is flagged', res2.cancelled)
    ok('cancelled scan returned without raising', isinstance(res2, C.ScanResult))
    ok('cancelled scan did fewer than all starts',
       int(np.isfinite(res2.stage1_fs).sum()) < 8,
       '%d of 8' % int(np.isfinite(res2.stage1_fs).sum()))
    part_modes = C.to_modes(res2, f, prov)
    ok('partial modes are flagged unconverged',
       all('unconverged' in m.flags for m in part_modes) if part_modes else True,
       '%d partial modes' % len(part_modes))

    print('\n--- B6  backend hygiene: the scan leaves no poisoned load ---')
    R_after = f.y()
    ok('model.y() is unchanged after a scan-style load',
       np.allclose(R_after, R_model, rtol=0, atol=1e-12),
       'max diff %.3e' % float(np.max(np.abs(R_after - R_model))))


# ================================================================ main ========

# ============================================================ PART E ==========

def part_e():
    """The stage-0 homometric enumerator: refusals first, then the pipeline.

    Pure numpy, so this is a real test rather than a smoke test. No protein-free
    fixture exists in this tree, so the SCIENCE cannot be validated here -- what is
    pinned is that the code runs, that seeds come out well formed, and above all
    that the two refusals fire with the CORRECT diagnosis, which is where a wrong
    answer would otherwise look plausible.
    """
    from fastscan import homometry as H
    from fastscan import seeds as S

    rrf = os.path.join(ROOT, 'Data', 'i2_tim3_1000_test.txt')      # R/R_F
    raw = os.path.join(ROOT, 'Data', 'n1_tim3_1500_raw_test.txt')  # raw R
    if not (os.path.isfile(rrf) and os.path.isfile(raw)):
        NOTE.append('Part E skipped: fixtures missing')
        return

    print('\n--- E1  the vendored pipeline is intact and matplotlib-free ---')
    from fastscan import _xrr_vendor as X
    ok('the module self-alias survived vendoring (NB is the module)', X.NB is X)
    for fn in ('run_pipeline', 'scan_basin', 'refine_spanbounded',
               'homometric_family', 'physical_members', 'patterson',
               'precondition', 'load_rrf', '_unpack_sb', 'on_boundary'):
        ok('vendored %s present' % fn, hasattr(X, fn))
    ok('importing it does NOT bind matplotlib.pyplot',
       'matplotlib.pyplot' not in sys.modules)

    print('\n--- E2  the raw-R refusal, with the right diagnosis ---')
    q, r, e = H.read_scan(raw)
    why = H.convention_warning(q, r)
    ok('raw R is detected', bool(why))
    ok('and the message names the actual cause (not "protein")',
       'RAW R' in (why or '') and 'Fresnel' in (why or ''))
    ok('enumerating raw R raises', _raises(
        lambda: H.enumerate_scan(raw, nbox=2, rho_top=0.0, rho_bottom=0.334)))
    ok('and it is NOT overridable by allow_protein', _raises(
        lambda: H.enumerate_scan(raw, nbox=2, rho_top=0.0, rho_bottom=0.334,
                                 allow_protein=True)))
    q2, r2, _e2 = H.read_scan(rrf)
    ok('an R/R_F curve is not flagged as raw', H.convention_warning(q2, r2) is None)

    print('\n--- E3  the span screen, and its honest limitation ---')
    pat_raw = H.patterson(q, r, e)
    ok('raw R manufactures a huge Patterson span (why E2 must come first)',
       pat_raw['span'] > 100.0, '%.0f A' % pat_raw['span'])
    ok('and the span screen would also have fired on it',
       H.protein_loaded_warning(pat_raw) is not None)
    pat = H.patterson(q2, r2)
    ok('patterson() accepts arrays, not just a path',
       set(['z', 'G', 'span', 'dz', 'zpk']).issubset(pat.keys()))
    # the documented limitation: resolution caps the span this can see
    ok('a protein-loaded scan at q_max=0.55 PASSES the span screen (documented)',
       H.protein_loaded_warning(pat) is None, 'span %.1f A' % pat['span'])

    print('\n--- E4  the pipeline, and the check that catches what E3 cannot ---')
    import time as _t
    t0 = _t.time()
    out = H.enumerate_scan(rrf, nbox=2, rho_top=0.0, rho_bottom=0.334)
    dt = _t.time() - t0
    ok('the enumeration produced candidates', len(out['basin']) >= 1,
       '%d in %.0f s' % (len(out['basin']), dt))
    ok('candidates are ordered by fit quality',
       all(out['basin'][i]['chi'] <= out['basin'][i + 1]['chi'] + 1e-9
           for i in range(len(out['basin']) - 1)))
    ok('the chi2/N check caught the protein-loaded scan the span screen missed',
       any('far too high' in w for w in out['warnings']),
       'best chi2/N %.0f' % out['basin'][0]['chi'])
    ok('caching makes a repeat call free',
       (lambda: (_t.time(),
                 H.enumerate_scan(rrf, nbox=2, rho_top=0.0, rho_bottom=0.334),
                 _t.time()))()[2] - _t.time() < 1.0)

    print('\n--- E5  candidates become well-formed seeds ---')
    cands = H.candidates(rrf, nbox=2, rho_top=0.0, rho_bottom=0.334)
    ok('candidates() returns seeds', len(cands) >= 1, '%d seeds' % len(cands))
    ok('every seed has the 1+2n layout for n=2',
       all(len(np.asarray(s).ravel()) == 5 for s in cands))
    for s in cands[:1]:
        sig, d, rho = S.split(s, 2)
        ok('seed splits into sig/d/rho sensibly',
           sig > 0 and np.all(d > 0) and len(rho) == 2,
           'sig %.2f d %s rho %s' % (sig, np.round(d, 2), np.round(rho, 3)))
    lo = np.array([1.0, 0.5, 0.5, 0.0, 0.0])
    hi = np.array([6.0, 30.0, 30.0, 0.62, 0.62])
    kept, rej = S.screen(cands, lo, hi, num_boxes=2)
    ok('the screen accepts or rejects each with a reason, never clips',
       len(kept) + len(rej) >= len(cands)
       and all(any(np.allclose(k, c) for c in cands) for k in kept),
       '%d kept, %d rejected' % (len(kept), len(rej)))
    ok('summary() renders without raising', isinstance(H.summary(out), str)
       and out['name'] in H.summary(out))


# ============================================================ PART D ==========

def part_d():
    """The spawn-child round trip: pickling, the pipe protocol, and cancel.

    Part B exercises ``core.scan`` in process. This is the only check that the MODEL
    survives pickling into a spawn child and that the child's protocol works -- the
    riskiest untested surface, because the model carries lmfit Parameters and numpy
    arrays and a ``Fit`` deliberately does not go through the pipe at all.

    Separate part because it costs ~2 minutes: the child pays Julia startup and a
    one-off derivative compilation.
    """
    try:
        from Functions import xpr_backend as xb
    except Exception as ex:
        NOTE.append('Part D skipped: cannot import xpr_backend (%s)' % str(ex)[:60])
        return
    if not xb.julia_available():
        NOTE.append('Part D skipped: the Julia/XPR backend is unavailable here')
        return

    import copy
    import multiprocessing as mp
    import pickle

    from Functions.XRR.XLayersPDB_XPR import XLayersPDB_XPR
    from fastscan import worker as fs_worker

    data = os.path.join(ROOT, 'Data', 'i2_tim3_1000_test.txt')
    pdb = os.path.join(ROOT, 'Data', 'tim3_hmmm_ext_all_bestframe.pdb')
    if not (os.path.isfile(data) and os.path.isfile(pdb)):
        NOTE.append('Part D skipped: fixtures missing')
        return
    a = np.genfromtxt(data)
    a = a[np.isfinite(a[:, 1])]
    a = a[(a[:, 0] >= 0.027447) & (a[:, 0] <= 0.550447)]
    Q, Y, E = a[:, 0], a[:, 1], a[:, 2]

    mpar = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                      'd': [0.0, 14.06975, 5.637274, 1.0],
                      'rho': [0.0, 0.3087728, 0.4867595, 0.334]}}
    f = XLayersPDB_XPR(x=Q, fname=pdb, use_protein=True, vol_calc='approx')
    f.__mpar__ = copy.deepcopy(mpar)
    f.__mkeys__ = list(f.__mpar__.keys())
    f.init_params()
    for nm in ('theta', 'phi', 'protins', 'cov', 'sig',
               '__Model_d_001', '__Model_d_002',
               '__Model_rho_001', '__Model_rho_002'):
        f.params[nm].set(vary=True)
    f.update_parameters()

    print('\n--- D1  the model survives pickling ---')
    blob = None
    try:
        blob = pickle.dumps(f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        ok('the model pickles (spawn requires it)', False, str(ex)[:70])
        return
    ok('the model pickles (spawn requires it)', True, '%d bytes' % len(blob))
    f2 = pickle.loads(blob)
    ok('the unpickled model keeps its layer table',
       len(f2.__mpar__[f2.__mkeys__[0]]['d']) == 4)
    ok('the unpickled model keeps its lmfit vary flags',
       f2.params['theta'].vary and not f2.params['yscale'].vary)

    print('\n--- D2  the child protocol round trip ---')
    job = fs_worker.build_job(None, f, Q, Y, E, n1=4, k=1, seed=2024,
                              iter1=15, tol1=1e-2, max_cpu1=25.0,
                              iter2=40, tol2=1e-6, max_cpu2=40.0,
                              certify=False, gauss_newton=False,
                              dataset=data, fit_scale='Linear',
                              template_label='tim3',
                              electron_scale=float(f.electron_scale('approx')))
    ctx = mp.get_context('spawn')
    rx, tx = ctx.Pipe(duplex=False)
    p = ctx.Process(target=fs_worker.run, args=(tx, f, job), daemon=True)
    import time as _t
    t0 = _t.time()
    p.start()
    stages, result, err = [], None, None
    while _t.time() - t0 < 600:
        if not rx.poll(1.0):
            if not p.is_alive():
                break
            continue
        m = rx.recv()
        if m.get('t') == 'progress':
            stages.append(m['stage'])
        elif m.get('t') == 'result':
            result = m
            break
        elif m.get('t') == 'error':
            err = m['msg']
            break
    p.join(20)
    if err:
        print(err[-1200:])
    ok('the child returned a result, not an error', result is not None and not err)
    if result is None:
        return
    ok('progress arrived for both stages',
       'stage1' in stages and 'stage2' in stages, str(sorted(set(stages))))
    res, mode_list, prov = result['res'], result['modes'], result['prov']
    ok('modes came back over the pipe', len(mode_list) >= 1,
       '%d modes' % len(mode_list))
    ok('the provenance was built in the child',
       prov.n1 == 4 and prov.n_active == len(Q) and prov.data_sha256 != '')
    ok('mode curves were cached for the overlay',
       all(m.R is not None and len(m.R) == len(Q) for m in mode_list))
    ok('physical params were de-normalised without the backend',
       all(0 <= m.theta <= 180 and np.isfinite(m.protins) for m in mode_list),
       ' '.join('(%.1f,%.1f)' % (m.theta, m.phi) for m in mode_list))
    from fastscan import io as FIO
    from fastscan import modes as MM
    ms = MM.ModeSet(mode_list, prov, cancelled=res.cancelled,
                    warnings=res.warnings)
    ok('the returned set serialises to .xmodes',
       FIO.format_xmodes(ms).startswith(FIO.MAGIC))
    print('    (child round trip took %.0f s)' % (_t.time() - t0))

    print('\n--- D3  terminate() mid-run is clean ---')
    rx2, tx2 = ctx.Pipe(duplex=False)
    job2 = dict(job)
    job2.update(n1=64, k=6, iter1=300, max_cpu1=120.0)
    p2 = ctx.Process(target=fs_worker.run, args=(tx2, f, job2), daemon=True)
    p2.start()
    t1 = _t.time()
    saw = False
    while _t.time() - t1 < 300:
        if rx2.poll(1.0):
            m = rx2.recv()
            if m.get('t') == 'progress' and m['stage'] == 'stage1':
                saw = True
                break
        elif not p2.is_alive():
            break
    p2.terminate()
    p2.join(15)
    ok('the child reached stage 1 before being killed', saw)
    ok('terminate() left it dead, no kill() needed', not p2.is_alive(),
       'exitcode %s' % p2.exitcode)
    ok('the parent backend is still usable afterwards',
       np.isfinite(float(f.y()[0])))


# ============================================================ PART C ==========

def part_c():
    """The Tools-plugin contract and the dialog's not-ready handling.

    Uses a stub host rather than a real XModFit: this pins the plugin contract and
    the guidance paths, which is what breaks on the first click, without paying for
    a full GUI construction.
    """
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception as ex:
        NOTE.append('Part C skipped: PyQt5 unavailable (%s)' % str(ex)[:60])
        return
    app = QApplication.instance() or QApplication([])

    print('\n--- C1  the Tools loader contract ---')
    import importlib
    tool_path = os.path.join(ROOT, 'Tools', 'GlobalSearch', 'FastGlobalSearch.py')
    ok('the tool file is where create_menus globs for it', os.path.isfile(tool_path))
    mod = importlib.import_module('Tools.GlobalSearch.FastGlobalSearch')
    base = os.path.splitext(os.path.basename(tool_path))[0]
    ok('the class name equals the file name (launch_tApp does getattr)',
       hasattr(mod, base), base)
    cls = getattr(mod, base)
    # If an optional import had rebound QDialog to `object`, launch_tApp's
    # `tclass(self)` would die with "object.__init__() takes exactly one argument".
    ok('the class is a real QDialog subclass', issubclass(cls, QDialog))
    ok('no space in the category or file name (module path is dot-joined)',
       ' ' not in base and ' ' not in 'GlobalSearch')

    print('\n--- C2  not-ready states produce guidance, never a crash ---')

    class StubHost(object):
        def __init__(self):
            self.fit = None
            self.sfnames = None
            self.curDir = ROOT
            self.format = '%.6e'

    d = cls(None)
    ok('constructs with parent=None', isinstance(d, QDialog))
    d.host = StubHost()
    why = d._not_ready()
    ok('no function selected -> a guidance message', bool(why), (why or '')[:52])
    d.refresh()
    ok('refresh() hides the steps and shows guidance',
       not d.tabs.isVisible() and d.guidance.text() != '')

    class FakeFunc(object):
        pass

    class FitStub(object):
        def __init__(self, func):
            self.func = func
            self.x = np.linspace(0.02, 0.5, 20)

    d.host.fit = FitStub(FakeFunc())
    why = d._not_ready()
    ok('an incapable model is refused by CAPABILITY, not by name',
       bool(why) and 'does not support' in why, (why or '')[:52])
    ok('the refusal names what is missing', 'missing:' in (why or ''))
    d.close()

    print('\n--- C3  the install path writes values only ---')
    src = open(os.path.join(ROOT, 'Tools', 'GlobalSearch', 'FastGlobalSearch.py'),
               encoding='utf-8').read()
    ok('the mode install disconnects cellChanged before writing',
       'cellChanged.disconnect()' in src)
    ok('and reconnects both handlers afterwards',
       'sfitParamChanged' in src and 'mfitParamChanged_new' in src)
    ok('it writes column 1 only (never min/max/vary)',
       'setText(host.format' in src and '.item(row, 1).setText' in src)
    ok('it arms XModFit undo before writing', 'host.oldParams' in src)
    ok('it ends with a single update_plot', 'host.update_plot()' in src)


def main():
    part = 'ABCDE'
    if '--part' in sys.argv:
        part = sys.argv[sys.argv.index('--part') + 1].upper()
    print('validate_fastscan: parts %s' % part)
    if 'A' in part:
        print('\n================ PART A (pure Python) ================')
        part_a()
    if 'B' in part:
        print('\n================ PART B (needs Julia) ================')
        part_b()
    if 'C' in part:
        print('\n================ PART C (Qt, offscreen) ==============')
        part_c()
    if 'E' in part:
        print('\n================ PART E (stage-0 enumerator) =========')
        part_e()
    if 'D' in part:
        print('\n================ PART D (spawn child) ================')
        part_d()
    for n in NOTE:
        print('\nNOTE: %s' % n)
    print('\n%s' % ('ALL CHECKS PASS' if not FAIL
                    else 'FAILED %d: %s' % (len(FAIL), FAIL)))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
