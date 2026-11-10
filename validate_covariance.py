"""
validate_covariance.py -- general covariance fitting: data model and loaders (Qt-free).

    C1  v1 JSON loader: every field, and every load-time refusal of spec 7.1
    C2  match_problems (x and y, rtol 1e-4) and subset()
    C3  Peelle's puzzle: data mode 0.8823529, model-based outer iteration 1.1538462
    C4  blocks: private terms block-diagonal; shared terms couple blocks; conflicts warn
    C5  a mask is a submatrix of the full C (linear and log)
    C6  factorize: plain, jitter and eigenvalue-floor paths and their warnings
    C7  Log formulas for multiplicative and additive terms

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


def main():
    try:
        import covariance_core as cc
    except ModuleNotFoundError as e:
        check('import covariance_core', False, str(e))
        return finish()
    section_c1(cc)
    section_c2(cc)
    for sec in (section_c3, section_c4, section_c5, section_c6, section_c7):
        sec(cc)
    return finish()


if __name__ == '__main__':
    sys.exit(main())
