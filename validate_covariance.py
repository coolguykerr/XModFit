"""
validate_covariance.py -- general covariance fitting: data model and loaders (Qt-free).

    C1  v1 JSON loader: every field, and every load-time refusal of spec 7.1
    C2  match_problems (x and y, rtol 1e-4) and subset()
    C3  Peelle's puzzle: data mode 0.8823529, model-based outer iteration 1.1538462
    C4  blocks: private terms block-diagonal; shared terms couple blocks; conflicts warn
    C5  a mask is a submatrix of the full C (linear and log)
    C6  factorize: plain, jitter and eigenvalue-floor paths and their warnings
    C7  Log formulas for multiplicative and additive terms
    C8  legacy loaders (npz, c_factors + raw, bundle) and from_legacy_meta
    J1  mixed joint fit (one dataset with a covariance, one without), Linear and Log
    J2  JointFit downgrades: Log y <= 0 point, w/o error scale, v1 mismatch, Sigma_logc conflict

    python validate_covariance.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

from tests.harness import check, finish

FIX = os.path.join(ROOT, 'tests', 'fixtures', 'cov_v1')
GOOD = os.path.join(FIX, 'good.cov.json')
REF = os.path.join(FIX, 'good_ref.txt')


def _mutations(doc):
    n = len(doc['x'])
    out = {}

    def case(name, word, fn):
        d = copy.deepcopy(doc)
        fn(d)
        out[name] = (word, d)

    case('no_format', 'format', lambda d: d.pop('format'))
    case('bad_version', 'version', lambda d: d.__setitem__('version', 2))
    case('len_sigma', 'sigma_meas', lambda d: d['sigma_meas'].pop())
    case('neg_sigma', 'sigma_meas', lambda d: d['sigma_meas'].__setitem__(3, 0.0))
    case('w_shape', 'weights',
         lambda d: d['terms'][0].__setitem__('weights', [r[:-1] for r in d['terms'][0]['weights']]))
    case('cov_asym', 'symmetric', lambda d: d['terms'][0]['cov'][0].__setitem__(
        1, d['terms'][0]['cov'][0][1] + 1e-3))
    case('cov_not_psd', 'positive',
         lambda d: d['terms'][0].__setitem__('cov', [[1, 2, 0], [2, 1, 0], [0, 0, 1]]))

    def dense_w(d):
        t = d['terms'][0]
        t['kind'] = 'dense'
        t['cov'] = np.eye(n).tolist()          # weights left in place

    case('dense_w', 'dense', dense_w)
    case('bad_kind', 'kind', lambda d: d['terms'][0].__setitem__('kind', 'mult'))
    return out


def section_c1(cc):
    d = cc.load_covariance(GOOD)
    with open(GOOD) as f:
        doc = json.load(f)
    check('C1 loader name', d.loader == 'xmodfit-covariance-v1', d.loader)
    check('C1 n == 12', d.n == 12, str(d.n))
    check('C1 x/y/sigma exact', np.array_equal(d.x, doc['x']) and np.array_equal(d.y, doc['y'])
          and np.array_equal(d.sigma_meas, doc['sigma_meas']))
    check('C1 data_file kept', d.data_file == 'good_ref.txt', str(d.data_file))
    check('C1 one term', len(d.terms) == 1 and d.replaces_y is None)
    t = d.terms[0]
    check('C1 term fields', t.name == 'stitch' and t.kind == 'multiplicative'
          and t.scope == 'dataset' and t.labels == ['S12', 'S13', 'S14'])
    check('C1 term arrays', t.weights.shape == (12, 3) and t.cov.shape == (3, 3)
          and t.weights[4].tolist() == [0.5, 0.5, 0.0] and t.cov[2, 2] == 9e-4)
    check('C1 provenance kept', 'fac' in t.provenance)
    check('C1 y_for_fit without replaces_y', d.y_for_fit(d.x) is d.x)

    tmp = tempfile.mkdtemp()
    try:
        for name, (word, md) in _mutations(doc).items():
            p = os.path.join(tmp, name + '.cov.json')
            with open(p, 'w') as f:
                json.dump(md, f)
            try:
                cc.load_covariance(p)
                check('C1 %s refused' % name, False, 'no error raised')
            except cc.CovarianceError as e:
                check('C1 %s refused' % name, word in str(e), str(e))
        p = os.path.join(tmp, 'junk.txt')
        with open(p, 'w') as f:
            f.write('hello\n')
        try:
            cc.load_covariance(p)
            check('C1 unrecognised refused', False, 'no error raised')
        except cc.CovarianceError as e:
            check('C1 unrecognised refused', True, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def section_c2(cc):
    d = cc.load_covariance(GOOD)
    xf, yf = np.loadtxt(REF, usecols=(0, 1), unpack=True)
    check('C2 match with %.4e file', d.match_problems(xf, yf) == [],
          str(d.match_problems(xf, yf)))
    check('C2 x only', d.match_problems(xf) == [])
    p = d.match_problems(xf, yf * 1.01)
    check('C2 y_mismatch names y and index 0',
          len(p) == 1 and 'y' in p[0] and '0' in p[0], str(p))
    p = d.match_problems(np.r_[xf[1:], xf[-1] * 1.1], yf)
    check('C2 x shifted by one point', len(p) >= 1 and 'x' in p[0], str(p))
    p = d.match_problems(xf[:-1], yf[:-1])
    check('C2 length mismatch', len(p) >= 1, str(p))

    s = d.subset(np.array([2, 3, 4]))
    t, t0 = s.terms[0], d.terms[0]
    check('C2 subset n', s.n == 3)
    check('C2 subset rows', np.array_equal(s.x, d.x[2:5]) and np.array_equal(s.y, d.y[2:5])
          and np.array_equal(s.sigma_meas, d.sigma_meas[2:5])
          and np.array_equal(t.weights, t0.weights[2:5]))
    check('C2 subset K x K cov untouched', np.array_equal(t.cov, t0.cov))
    check('C2 original untouched', d.n == 12 and d.terms[0].weights.shape == (12, 3))

    dn = cc.DatasetCovariance(x=np.arange(5.), sigma_meas=np.ones(5),
                              terms=[cc.CovTerm('d', 'dense', np.arange(25.).reshape(5, 5), None, None)],
                              y=np.arange(5.) + 10, replaces_y=np.arange(5.) + 20)
    sd = dn.subset(np.array([1, 3]))
    check('C2 dense subset rows and columns',
          np.array_equal(sd.terms[0].cov, np.array([[6., 8.], [16., 18.]])))
    check('C2 replaces_y subset + y_for_fit', np.array_equal(sd.replaces_y, [21., 23.])
          and np.array_equal(sd.y_for_fit(np.zeros(2)), [21., 23.]))
    check('C2 without_terms', dn.without_terms().terms == [] and len(dn.terms) == 1)
    check('C2 data_file mismatch not an error at this level',
          d.match_problems(xf, yf) == [] and d.data_file != 'other.txt')


def _peelle_block(cc, intensity):
    t = cc.CovTerm('norm', 'multiplicative', np.array([[0.04]]), np.ones((2, 1)), ['n'])
    dc = cc.DatasetCovariance(x=np.array([1., 2.]), sigma_meas=np.array([0.15, 0.10]), terms=[t])
    y = np.array([1.5, 1.0])
    return cc.Block(cov=dc, y=y, yerr=np.array([0.15, 0.10]), intensity=intensity)


def _gls_const(cc, blocks):
    C, _ = cc.assemble(blocks, 'Linear')
    L, _ = cc.factorize(C)
    y = np.concatenate([b.y for b in blocks])
    a, o = cc.whiten(L, y), cc.whiten(L, np.ones_like(y))
    return float(a @ o / (o @ o)), float(1.0 / np.sqrt(o @ o))


def section_c3(cc):
    from lmfit import Parameters
    y = np.array([1.5, 1.0])
    mu, _ = _gls_const(cc, [_peelle_block(cc, y)])
    check('C3 Peelle data mode 0.8823529', abs(mu - 0.8823529) < 1e-6, '%.9f' % mu)
    sig = np.array([0.15, 0.10])
    wm = float(np.sum(y / sig ** 2) / np.sum(1 / sig ** 2))
    state = {'blk': _peelle_block(cc, y)}

    class Res:
        pass

    def fit_once():
        m, err = _gls_const(cc, [state['blk']])
        r = Res()
        r.params = Parameters()
        r.params.add('mu', value=m, vary=True)
        r.params['mu'].stderr = err
        return r

    def rebuild(r):
        state['blk'] = _peelle_block(cc, r.params['mu'].value * np.ones(2))

    res, rounds, conv = cc.outer_iterate(fit_once, rebuild, max_rounds=50, tol=1e-6)
    m = res.params['mu'].value
    check('C3 Peelle model mode 1.1538462', abs(m - 1.1538462) < 1e-6, '%.9f' % m)
    check('C3 model mode equals sigma-weighted mean', abs(m - wm) < 1e-6, '%.9f vs %.9f' % (m, wm))
    check('C3 converged', conv and rounds >= 1, '%s %s' % (conv, rounds))
    state['blk'] = _peelle_block(cc, y)
    _, r1, c1 = cc.outer_iterate(fit_once, rebuild, max_rounds=1, tol=1e-12)
    check('C3 round limit reports not converged', r1 == 1 and c1 is False, '%s %s' % (r1, c1))


def _two_blocks(cc, shared, cov_b=0.01):
    rng = np.random.RandomState(1)
    out = []
    for i, n in enumerate((6, 5)):
        w = np.zeros((n, 2))
        w[np.arange(n), np.arange(n) % 2] = 1.0
        w[2] = [0.5, 0.5]
        cov = np.array([[0.01, 0.002], [0.002, cov_b if i else 0.01]])
        t = cc.CovTerm('stitch', 'multiplicative', cov, w, ['a', 'b'],
                       'shared' if shared else 'dataset')
        dc = cc.DatasetCovariance(np.arange(n, dtype=float), 0.1 * np.ones(n), [t],
                                  source='d%d.json' % i)
        y = 1 + rng.rand(n)
        out.append(cc.Block(cov=dc, y=y, yerr=0.1 * np.ones(n), intensity=y))
    return out


def section_c4(cc):
    bl = _two_blocks(cc, shared=False)
    C, warn = cc.assemble(bl, 'Linear')
    check('C4 private terms: off-block exactly 0',
          np.max(np.abs(C[:6, 6:])) == 0.0 and warn == [], str(warn))
    check('C4 symmetric', np.array_equal(C, C.T))
    bl = _two_blocks(cc, shared=True)
    C, warn = cc.assemble(bl, 'Linear')
    a, b = bl
    ta = a.cov.terms[0]
    want = (a.intensity[:, None] * ta.weights) @ ta.cov @ (b.intensity[:, None] * b.cov.terms[0].weights).T
    check('C4 shared identical: off-block = D_a W S W_b^T D_b',
          np.max(np.abs(C[:6, 6:] - want)) < 1e-14 and warn == [], str(warn))
    check('C4 shared: off-block nonzero', np.max(np.abs(want)) > 0)
    Cp, _ = cc.assemble(_two_blocks(cc, shared=False), 'Linear')
    check('C4 shared: diagonal blocks equal private ones',
          np.max(np.abs(C[:6, :6] - Cp[:6, :6])) < 1e-14
          and np.max(np.abs(C[6:, 6:] - Cp[6:, 6:])) < 1e-14)
    bl = _two_blocks(cc, shared=True, cov_b=0.02)
    C, warn = cc.assemble(bl, 'Linear')
    check('C4 shared conflict: block-diagonal', np.max(np.abs(C[:6, 6:])) == 0.0)
    check('C4 shared conflict: one warning naming the term',
          len(warn) == 1 and 'stitch' in warn[0], str(warn))
    check('C4 shared conflict: warning names the datasets',
          'd0.json' in warn[0] and 'd1.json' in warn[0], str(warn))
    bl = _two_blocks(cc, shared=True)
    C, _ = cc.assemble(bl, 'Log')
    check('C4 log shared off-block = W S W^T',
          np.max(np.abs(C[:6, 6:] - bl[0].cov.terms[0].weights @ bl[0].cov.terms[0].cov
                        @ bl[1].cov.terms[0].weights.T)) < 1e-14)
    d = cc.CovTerm('dn', 'dense', 0.01 * np.eye(3), None, None, 'shared')
    dc = cc.DatasetCovariance(np.arange(3.), 0.1 * np.ones(3), [d])
    bk = [cc.Block(dc, np.ones(3), 0.1 * np.ones(3), np.ones(3)) for _ in range(2)]
    C, warn = cc.assemble(bk, 'Linear')
    check('C4 dense shared is private, no warning',
          warn == [] and np.max(np.abs(C[:3, 3:])) == 0.0)
    check('C4 needs_intensity: mult linear', cc.needs_intensity(_two_blocks(cc, False), 'Linear'))
    check('C4 needs_intensity: mult log False',
          not cc.needs_intensity(_two_blocks(cc, False), 'Log'))
    check('C4 needs_intensity: dense log True, linear False',
          cc.needs_intensity(bk, 'Log') and not cc.needs_intensity(bk, 'Linear'))
    nocov = [cc.Block(None, np.ones(3), 0.1 * np.ones(3), np.ones(3))]
    check('C4 no cov: diag yerr^2', np.allclose(cc.assemble(nocov, 'Linear')[0], 0.01 * np.eye(3), rtol=1e-14, atol=0))


def section_c5(cc):
    n = 9
    rng = np.random.RandomState(3)
    ta = cc.CovTerm('m', 'multiplicative', 0.01 * np.eye(3) + 0.001, rng.rand(n, 3), None)
    tb = cc.CovTerm('a', 'additive', np.array([[0.02, 0.0], [0.0, 0.03]]), rng.rand(n, 2), None)
    g = rng.rand(n, n)
    td = cc.CovTerm('d', 'dense', 0.01 * (g @ g.T) / n, None, None)
    dc = cc.DatasetCovariance(np.arange(n, dtype=float), 0.1 + 0.1 * rng.rand(n), [ta, tb, td])
    y = 1 + rng.rand(n)
    inten = 1 + rng.rand(n)
    idx = np.array([0, 2, 3, 6, 8])
    for scale in ('Linear', 'Log'):
        full, _ = cc.assemble([cc.Block(dc, y, 0.1 * np.ones(n), inten)], scale)
        sub, _ = cc.assemble([cc.Block(dc.subset(idx), y[idx], 0.1 * np.ones(len(idx)),
                                       inten[idx])], scale)
        dev = np.max(np.abs(sub - full[np.ix_(idx, idx)]))
        check('C5 submatrix %s' % scale, dev < 1e-14, '%.3g' % dev)


def section_c6(cc):
    C = np.diag([1.0, 2.0, 3.0]) + 0.1
    L, warn = cc.factorize(C)
    check('C6 PD: no warning, L L^T = C', warn == [] and np.max(np.abs(L @ L.T - C)) < 1e-12,
          str(warn))
    t = cc.CovTerm('big', 'multiplicative', np.array([[1e2]]), np.ones((6, 1)), None)
    dc = cc.DatasetCovariance(np.arange(6.), 1e-9 * np.ones(6), [t])
    Cr, _ = cc.assemble([cc.Block(dc, np.ones(6), 1e-9 * np.ones(6), np.ones(6))], 'Linear')
    L, warn = cc.factorize(Cr)
    check('C6 rank-deficient: one warning', len(warn) == 1 and np.all(np.isfinite(L)), str(warn))
    L, warn = cc.factorize(np.array([[1.0, 2.0], [2.0, 1.0]]))
    check('C6 indefinite: eigenvalues floored', len(warn) == 1 and 'floored' in warn[0], str(warn))
    r = np.array([1.0, 2.0, 3.0])
    L = np.linalg.cholesky(np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]]))
    check('C6 whiten solves L z = r', np.allclose(L @ cc.whiten(L, r), r))
    J = np.arange(6.).reshape(3, 2)
    check('C6 whiten_jac solves L Z = J', np.allclose(L @ cc.whiten_jac(L, J), J))


def section_c7(cc):
    n = 4
    y = np.array([1., 2., 4., 8.])
    inten = np.array([2., 2., 5., 5.])
    sig = np.array([0.1, 0.2, 0.3, 0.4])
    w = np.eye(n)
    S = np.diag([0.01, 0.02, 0.03, 0.04])
    tm = cc.CovTerm('m', 'multiplicative', S, w, None)
    dc = cc.DatasetCovariance(np.arange(n, dtype=float), sig, [tm])
    C, _ = cc.assemble([cc.Block(dc, y, sig, inten)], 'Log')
    want = np.diag((sig / y) ** 2) + w @ S @ w.T
    check('C7 log multiplicative exact', np.max(np.abs(C - want)) < 1e-15)
    wa = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    Sa = np.array([[0.5, 0.1], [0.1, 0.6]])
    dc = cc.DatasetCovariance(np.arange(n, dtype=float), sig, [cc.CovTerm('a', 'additive', Sa, wa, None)])
    C, _ = cc.assemble([cc.Block(dc, y, sig, inten)], 'Log')
    E = np.diag(1 / inten)
    want = np.diag((sig / y) ** 2) + E @ wa @ Sa @ wa.T @ E
    check('C7 log additive E W S W^T E', np.max(np.abs(C - want)) < 1e-15)
    C, _ = cc.assemble([cc.Block(dc, y, sig, inten)], 'Linear')
    check('C7 linear additive W S W^T (no intensity)',
          np.max(np.abs(C - (np.diag(sig ** 2) + wa @ Sa @ wa.T))) < 1e-15)


def _legacy_fixture(tmp, footprint):
    """Two datasets in the bundle layout (shared K2 / SIG2 / V_FP / VAR_FWHM), plus the
    single-dataset npz and c_factors+raw routes for dataset 'a'."""
    from validate_joint_gls import X, K2, SIG2, V_FP, VAR_FWHM, E, make_data
    ys = [np.abs(y) + 1.0 for y in make_data(2)]
    sf_list = np.array([1.0, 10.0])
    c = np.array([1.2, 0.9])
    names = ['a', 'b']
    z = {'Sigma_logc': SIG2, 'SF_LIST': sf_list, 'c': c}
    for nm, y in zip(names, ys):
        z[nm + '__x'] = X
        z[nm + '__R_corr'] = y
        z[nm + '__sigma_meas'] = E * np.ones_like(X)
        z[nm + '__v'] = V_FP
        z[nm + '__k_index'] = K2
        z[nm + '__scalars'] = np.array([1.0, 0.0, VAR_FWHM, 0.1, 0.01])
    bundle = os.path.join(tmp, 'bundle.npz')
    np.savez(bundle, **z)
    one = dict(x=X, y_corr=ys[0], sigma_meas=E * np.ones_like(X), k_index=K2,
               SF_LIST=sf_list, Sigma_logc=SIG2, c=c)
    if footprint:
        one.update(v=V_FP, var_fwhm=VAR_FWHM)
    single = os.path.join(tmp, 'a_corrected_cov%d.npz' % int(footprint))
    np.savez(single, **one)
    # c_factors + raw route: raw y chosen so y_raw * c[k] == ys[0]
    ck = c[K2]
    raw = os.path.join(tmp, 'a_raw.txt')
    with open(raw, 'w') as f:
        for i in range(len(X)):
            f.write('%.17g\t%.17g\t%.17g\t%.17g\n'
                    % (X[i], ys[0][i] / ck[i], E / ck[i], sf_list[K2[i]]))
    cf = os.path.join(tmp, 'a_c_factors.txt')
    with open(cf, 'w') as f:
        f.write('# SF c sigma(log c)\n')
        for i in range(2):
            f.write('%.17g %.17g %.17g\n' % (sf_list[i], c[i], np.sqrt(SIG2[i, i])))
        f.write('# Sigma_logc matrix\n')
        for i in range(2):
            f.write('%.17g %.17g\n' % tuple(SIG2[i]))
    return dict(bundle=bundle, single=single, raw=raw, cf=cf, ys=ys, names=names)


def _eq_dc(a, b, terms=None):
    ok = (np.array_equal(a.x, b.x) and np.allclose(a.sigma_meas, b.sigma_meas, rtol=1e-12)
          and np.allclose(a.replaces_y, b.replaces_y, rtol=1e-12))
    ta_, tb_ = (a.terms, b.terms) if terms is None else ([t for t in a.terms if t.name in terms],
                                                         [t for t in b.terms if t.name in terms])
    ok = ok and [t.name for t in ta_] == [t.name for t in tb_]
    for ta, tb in zip(ta_, tb_):
        ok = ok and np.allclose(ta.cov, tb.cov, rtol=1e-12) and np.allclose(ta.weights, tb.weights)
        ok = ok and ta.labels == tb.labels
    return ok


def section_c8(cc):
    from validate_joint_gls import X, K2, SIG2, V_FP, VAR_FWHM, E
    from joint_covariance import build_full_covariance
    tmp = tempfile.mkdtemp()
    try:
        fx = _legacy_fixture(tmp, True)
        ds = cc.load_bundle(fx['bundle'])
        check('C8 load_bundle returns both datasets', list(ds) == ['a', 'b'])
        for foot in (True, False):
            dcs = []
            for nm in fx['names']:
                d = ds[nm]
                if not foot:
                    d = cc.DatasetCovariance(d.x, d.sigma_meas,
                                             [t for t in d.terms if t.name != 'beam_fwhm'],
                                             d.y, d.replaces_y, d.source, d.loader)
                dcs.append(d)
            blocks = [cc.Block(d, d.replaces_y, d.sigma_meas, d.replaces_y) for d in dcs]
            C, warns = cc.assemble(blocks, 'Linear')
            ys = np.concatenate(fx['ys'])
            ref = build_full_covariance(
                ys, np.concatenate([E * np.ones_like(X)] * 2), np.concatenate([K2, K2]), SIG2,
                v=(np.concatenate([V_FP, V_FP]) if foot else None),
                var_fwhm=(VAR_FWHM if foot else 0.0))
            err = np.max(np.abs(C - ref)) / np.max(np.abs(ref))
            tag = 'with' if foot else 'without'
            check('C8 assemble == build_full_covariance (%s footprint)' % tag,
                  err < 1e-12 and warns == [], 'rel err %.3g, warns %s' % (err, warns))
            n = len(X)
            check('C8 cross-dataset block present (%s footprint)' % tag,
                  np.max(np.abs(C[:n, n:])) > 0)
        check('C8 bundle terms: attenuator_cal then beam_fwhm, both shared multiplicative',
              [t.name for t in ds['a'].terms] == ['attenuator_cal', 'beam_fwhm']
              and all(t.scope == 'shared' and t.kind == 'multiplicative' for t in ds['a'].terms))
        check('C8 labels from SF_LIST', ds['a'].terms[0].labels == ['1.0', '10.0'],
              str(ds['a'].terms[0].labels))
        check('C8 legacy fields: y None, replaces_y = y_corr, source = path',
              ds['a'].y is None and np.allclose(ds['a'].replaces_y, fx['ys'][0])
              and ds['a'].loader != '' and ds['a'].source == fx['bundle'])
        a_npz = cc.load_covariance(fx['single'])
        a_cf = cc.load_covariance(fx['cf'], raw_path=fx['raw'])
        check('C8 legacy-npz loader name', a_npz.loader == 'legacy-npz', a_npz.loader)
        check('C8 legacy-c-factors loader name', a_cf.loader == 'legacy-c-factors', a_cf.loader)
        check('C8 npz and c_factors+raw routes give equal arrays (attenuator_cal)',
              _eq_dc(a_npz, a_cf, terms=['attenuator_cal']))
        check('C8 npz route equals the bundle dataset', _eq_dc(a_npz, ds['a']))
        try:
            cc.load_covariance(fx['cf'])
            check('C8 c_factors without raw_path raises', False, 'no error')
        except cc.CovarianceError as e:
            check('C8 c_factors without raw_path raises, naming raw_path',
                  'raw_path' in str(e), str(e))
        try:
            cc.load_covariance(fx['bundle'])
            check('C8 bundle path to load_covariance raises', False, 'no error')
        except cc.CovarianceError as e:
            check('C8 bundle path to load_covariance names load_bundle',
                  'load_bundle' in str(e), str(e))
        nofoot = cc.load_covariance(_legacy_fixture(tmp, False)['single'])
        check('C8 no v/var_fwhm: only attenuator_cal',
              [t.name for t in nofoot.terms] == ['attenuator_cal'])
        bad = dict(y_corr=fx['ys'][0].copy(), sigma_meas=E * np.ones_like(X), k_index=K2,
                   Sigma_logc=SIG2, v=V_FP, var_fwhm=VAR_FWHM)
        bad['y_corr'][3] = 0.0
        try:
            cc.from_legacy_meta(bad, X, 'p', 'legacy-npz')
            check('C8 y_corr == 0 with v != 0 raises', False, 'no error')
        except cc.CovarianceError as e:
            check('C8 y_corr == 0 with v != 0 names beam_fwhm and index 3',
                  'beam_fwhm' in str(e) and '3' in str(e), str(e))
        zv = dict(bad, v=V_FP.copy())
        zv['v'][3] = 0.0                       # y_corr 0 where v is 0 too: allowed
        try:
            dz = cc.from_legacy_meta(zv, X, 'p', 'legacy-npz')
            wz = dz.terms[1].weights
            check('C8 beam_fwhm W is 0 (not NaN) where v == 0 and y_corr == 0',
                  np.all(np.isfinite(wz)) and wz[3, 0] == 0.0, str(wz[3]))
        except Exception as e:
            check('C8 beam_fwhm W is 0 (not NaN) where v == 0 and y_corr == 0', False, repr(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- JointFit ----
def _j_line(a=2.5, b=0.3):
    from validate_joint_gls import Line, X
    f = Line(x=X, a=a, b=b)
    for nm in ('a', 'b'):
        f.params[nm].vary = True
    return f


def _j_data():
    """Two noisy lines on validate_joint_gls's X, and a v1-style covariance for the first
    (private stitch term, two regions with averaged overlap rows, plus a private additive
    background term so Log model mode iterates as well)."""
    from validate_joint_gls import X, E
    import covariance_core as cc
    rng = np.random.default_rng(5)
    n = len(X)
    ya = 3.0 * X + 0.5 + E * rng.standard_normal(n)
    yb = 2.0 * X + 1.0 + E * rng.standard_normal(n)
    w = np.zeros((n, 2))
    w[X <= 1.0, 0] = 1.0
    w[X > 1.0, 1] = 1.0
    w[19:21] = [0.5, 0.5]
    stitch = cc.CovTerm('stitch', 'multiplicative', np.array([[4e-4, 1e-4], [1e-4, 9e-4]]),
                        w, ['r0', 'r1'], 'dataset')
    bkg = cc.CovTerm('bkg', 'additive', np.array([[1e-3]]), np.ones((n, 1)), None, 'dataset')
    cov = cc.DatasetCovariance(X.copy(), E * np.ones(n), [stitch, bkg], ya.copy(),
                               source='A.cov.json', loader='xmodfit-covariance-v1')
    return X, E, ya, yb, cov


def _j_fit(rows, sc, mode='model', **kw):
    from Joint_Fit_Routines import JointFit
    from validate_joint_gls import X, E
    jf = JointFit()
    for nm, y, cov in rows:
        jf.add_dataset(nm, func=_j_line(), x=X, y=y, yerr=E * np.ones_like(X), cov=cov)
    jf.build_master_params()
    rep, msg = jf.perform_fit(X[0], X[-1], fit_scale=sc, fit_method='leastsq', maxiter=2000,
                              covariance_mode=mode, **kw)
    return jf, rep


def section_j1(cc):
    from Fit_Routines import Fit
    from validate_joint_gls import Line, rel
    X, E, ya, yb, cov = _j_data()
    for sc in ('Linear', 'Log'):
        try:
            jf, rep = _j_fit([('A', ya, cov), ('B', yb, None)], sc)
        except Exception as e:
            check('J1 mixed joint (%s) runs' % sc, False, repr(e))
            continue
        check('J1 mixed joint (%s) runs' % sc, True)
        ja, _ = _j_fit([('A', ya, cov)], sc)
        fb = Fit(lambda: Line(a=2.5, b=0.3), X, y=yb, yerr=E * np.ones_like(X))
        for nm in ('a', 'b'):
            fb.fit_params[nm].vary = True
        fb.perform_fit(X[0], X[-1], fit_scale=sc, fit_method='leastsq', maxiter=2000)
        want = ja.result.chisqr + fb.result.chisqr
        d = rel(jf.result.chisqr, want)
        check('J1 (%s) joint chi2 == A one-dataset cov chi2 + B stock chi2 to 1e-8' % sc,
              d < 1e-8, 'joint=%.10e sum=%.10e rel=%.2e' % (jf.result.chisqr, want, d))
        db = max(rel(jf.result.params['d1_' + nm].value, fb.result.params[nm].value)
                 for nm in ('a', 'b'))
        check('J1 (%s) B params == stock Fit to 1e-6' % sc, db < 1e-6, 'max rel=%.2e' % db)
        da = max(rel(jf.result.params['d0_' + nm].value, ja.result.params['d0_' + nm].value)
                 for nm in ('a', 'b'))
        check('J1 (%s) A params == one-dataset cov fit to 1e-6' % sc, da < 1e-6,
              'max rel=%.2e' % da)
        info = jf.cov_info
        check('J1 (%s) cov_info: model mode, scale, iterated, converged' % sc,
              info.get('mode') == 'model' and info.get('scale') == sc
              and info.get('rounds', 0) >= 1 and info.get('converged') is True, str(info))
        check('J1 (%s) absolute errors: scale_covar off' % sc,
              jf.fitter.scale_covar is False)
        check('J1 (%s) report: Covariance block lists A source, both terms, and B as none'
              % sc, 'Covariance:' in rep and 'A.cov.json' in rep and 'stitch' in rep
              and 'bkg' in rep and 'chi2_red =' in rep and 'not rescaled' in rep, rep[-600:])
        check('J1 (%s) report: not degraded' % sc, not rep.startswith('DEGRADED'))

    # the pre-T4 crash path: a legacy dict on A, nothing on B (data mode)
    from validate_joint_gls import K2, SIG2
    leg = dict(y_corr=ya, sigma_meas=E * np.ones_like(X), k_index=K2, Sigma_logc=SIG2)
    try:
        jf, _ = _j_fit([('A', ya, dict(leg)), ('B', yb, None)], 'Linear', mode='data')
        ja, _ = _j_fit([('A', ya, dict(leg))], 'Linear', mode='data')
        fb = Fit(lambda: Line(a=2.5, b=0.3), X, y=yb, yerr=E * np.ones_like(X))
        for nm in ('a', 'b'):
            fb.fit_params[nm].vary = True
        fb.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq', maxiter=2000)
        want = ja.result.chisqr + fb.result.chisqr
        d = rel(jf.result.chisqr, want)
        check('J1 legacy dict + no-cov: chi2 == sum of the separate fits to 1e-8', d < 1e-8,
              'rel=%.2e' % d)
    except Exception as e:
        check('J1 legacy dict + no-cov: chi2 == sum of the separate fits to 1e-8', False,
              repr(e))


def section_j2(cc):
    from validate_joint_gls import rel
    X, E, ya, yb, cov = _j_data()
    yneg = ya.copy()
    yneg[7] = -0.05                      # y <= 0 inside the window
    cneg = cc.DatasetCovariance(cov.x, cov.sigma_meas, cov.terms, yneg.copy(),
                                source='A.cov.json', loader='xmodfit-covariance-v1')
    try:
        jf, rep = _j_fit([('A', yneg, cneg), ('B', yb, None)], 'Log')
        ok = True
    except Exception as e:
        ok = False
        check('J2 Log with a y <= 0 point and a covariance: fit completes', False, repr(e))
    if ok:
        check('J2 Log with a y <= 0 point and a covariance: fit completes', True)
        m = jf._cov_mask
        check('J2 the y <= 0 point is not an active covariance row',
              m is not None and len(m) == 2 * len(X) and not m[7] and m.sum() == 2 * len(X) - 1,
              'len=%s n_false=%s' % (None if m is None else len(m),
                                      None if m is None else int((~m).sum())))
        check('J2 report lists the scale as Log', 'scale: Log' in rep, rep[-600:])

    # 'w/o error' scale: covariance ignored, same fit as no covariance, report says so first
    jf, rep = _j_fit([('A', ya, cov), ('B', yb, None)], 'Linear w/o error')
    j0, _ = _j_fit([('A', ya, None), ('B', yb, None)], 'Linear w/o error')
    d = rel(jf.result.chisqr, j0.result.chisqr)
    check('J2 Linear w/o error with a covariance == no-covariance fit', d < 1e-12,
          'rel=%.2e' % d)
    check('J2 Linear w/o error: report first line DEGRADED FIT, names the ignored covariance',
          rep.startswith('DEGRADED FIT:') and 'ignored' in rep.splitlines()[0],
          rep.splitlines()[0])

    # v1 covariance whose y does not match at add_dataset: dataset fits without it
    bad = cc.DatasetCovariance(cov.x, cov.sigma_meas, cov.terms, ya * 1.01,
                               source='A.cov.json', loader='xmodfit-covariance-v1')
    jf, rep = _j_fit([('A', ya, bad), ('B', yb, None)], 'Linear')
    j0, _ = _j_fit([('A', ya, None), ('B', yb, None)], 'Linear')
    check('J2 mismatched v1 covariance: dataset added without it',
          jf.objectives[0]['cov'] is None and not jf._use_cov)
    check('J2 mismatched v1 covariance: fit == no-covariance fit',
          rel(jf.result.chisqr, j0.result.chisqr) < 1e-12)
    check('J2 mismatched v1 covariance: DEGRADED FIT first line naming the mismatch',
          rep.startswith('DEGRADED FIT:') and 'y mismatch' in rep.splitlines()[0],
          rep.splitlines()[0])
    jf.build_master_params()             # resets build-time warnings only
    rep2, _ = jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq',
                             maxiter=2000)
    check('J2 mismatch note survives build_master_params and a re-fit',
          rep2.startswith('DEGRADED FIT:') and 'y mismatch' in rep2.splitlines()[0])

    # legacy dicts with different Sigma_logc: private terms + a warning (was a ValueError)
    from validate_joint_gls import K2, SIG2
    leg = [dict(y_corr=y, sigma_meas=E * np.ones_like(X), k_index=K2, Sigma_logc=S)
           for y, S in ((ya, SIG2), (yb, 2 * SIG2))]
    try:
        jf, rep = _j_fit([('A', ya, leg[0]), ('B', yb, leg[1])], 'Linear', mode='data')
        n = len(X)
        check('J2 different Sigma_logc: fit runs, C block-diagonal',
              np.max(np.abs((jf._L @ jf._L.T)[:n, n:])) == 0.0)
        check('J2 different Sigma_logc: DEGRADED FIT naming attenuator_cal and both datasets',
              rep.startswith('DEGRADED FIT:') and 'attenuator_cal' in rep.splitlines()[0]
              and 'A' in rep.splitlines()[0] and 'B' in rep.splitlines()[0],
              rep.splitlines()[0])
    except Exception as e:
        check('J2 different Sigma_logc: fit runs, C block-diagonal', False, repr(e))

    # chi2_red band: central 99 % of chi2(nfree)/nfree
    from scipy.stats import chi2 as _chi2

    class _R:
        def __init__(self, redchi, nfree):
            self.redchi, self.nfree = redchi, nfree
    info = {'mode': 'data', 'scale': 'Linear', 'rounds': 0, 'converged': True}
    hi = _chi2.ppf(0.995, 50) / 50
    lo = _chi2.ppf(0.005, 50) / 50
    txt = [cc.report_block([('A', cov)], info, [], _R(v, 50))
           for v in (1.0, hi * 1.001, hi * 0.999, lo * 0.999)]
    check('J2 report_block: chi2_red line always, band warning only outside the 99% band',
          all('chi2_red = ' in t for t in txt)
          and 'WARNING' not in txt[0] and 'misfit exceeds' in txt[1]
          and 'WARNING' not in txt[2] and 'below the central 99%' in txt[3],
          '%.4f %.4f' % (lo, hi))

    # emcee with a covariance: C frozen at the best-fit model, serial, reported as such
    try:
        jf, _ = _j_fit([('A', ya, cov), ('B', yb, None)], 'Linear')
        rep, _ = jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='emcee',
                                emcee_walkers=16, emcee_steps=20, emcee_cores=1)
        check('J2 emcee with a covariance: C frozen at the best-fit model and reported',
              jf.cov_info.get('mcmc_frozen') == 'model' and 'C frozen for the run' in rep
              and not rep.startswith('DEGRADED'), str(jf.cov_info))
    except Exception as e:
        check('J2 emcee with a covariance: C frozen at the best-fit model and reported',
              False, repr(e))


def main():
    try:
        import covariance_core as cc
    except ModuleNotFoundError as e:
        check('import covariance_core', False, str(e))
        return finish()
    section_c1(cc)
    section_c2(cc)
    for sec in (section_c3, section_c4, section_c5, section_c6, section_c7, section_c8):
        sec(cc)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    for sec in (section_j1, section_j2):
        try:
            sec(cc)
        except Exception as e:
            check('%s runs' % sec.__name__, False, repr(e))
    return finish()


if __name__ == '__main__':
    sys.exit(main())
