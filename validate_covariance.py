"""
validate_covariance.py -- general covariance fitting: data model and loaders (Qt-free).

    C1  v1 JSON loader: every field, and every load-time refusal of spec 7.1
    C2  match_problems (x and y, rtol 1e-4) and subset()

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


def main():
    try:
        import covariance_core as cc
    except ModuleNotFoundError as e:
        check('import covariance_core', False, str(e))
        return finish()
    section_c1(cc)
    section_c2(cc)
    return finish()


if __name__ == '__main__':
    sys.exit(main())
