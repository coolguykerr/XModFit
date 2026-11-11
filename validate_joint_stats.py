"""
validate_joint_stats.py -- WS2 reporting harness (sections Q, P, K, M, R).

Section Q (this file, Task 1): JointFit plumbing -- quiet evaluation, row ranges,
prior rows, last / model covariance blocks.

The shared fixtures (Line, X, make_jf, private_cov, shared_cov) are defined at module
top so validate_joint_diag.py can `from validate_joint_stats import ...`; importing
this module runs nothing.

    QT_QPA_PLATFORM=offscreen python validate_joint_stats.py
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Functions'))

import numpy as np
from lmfit import Parameters
from PyQt5.QtWidgets import QApplication

import covariance_core as cc
from tests.harness import check, finish


# ----------------------------------------------------------------- fixtures --
X = np.linspace(0.1, 2.0, 40)
A_TRUE, B_TRUE, E = 3.0, 0.5, 0.05


class Line:
    def __init__(self, x=0, a=1.0, b=0.0, mpar={}):
        """y = a*x + b"""
        self.x = x
        self.a, self.b = a, b
        self.__mpar__ = mpar
        self.choices = {}
        self.filepaths = {}
        self.__fit__ = False
        self.output_params = {'scaler_parameters': {}}
        self.init_params()

    def init_params(self):
        self.params = Parameters()
        self.params.add('a', value=self.a, vary=0, min=-np.inf, max=np.inf)
        self.params.add('b', value=self.b, vary=0, min=-np.inf, max=np.inf)

    def y(self):
        return self.a * np.asarray(self.x, float) + self.b


def _onehot(mask):
    """N x 2 weights: column 0 = ~mask, column 1 = mask."""
    mask = np.asarray(mask, bool)
    return np.stack([(~mask).astype(float), mask.astype(float)], axis=1)


def private_cov(y, k):
    """Per-dataset multiplicative stitching term on the x > 1 segment."""
    return cc.DatasetCovariance(
        X, E * np.ones_like(X),
        [cc.CovTerm('stitch', 'multiplicative', np.diag([0, 4e-4]),
                    weights=_onehot(X > 1.0), scope='dataset')], y)


def shared_cov(y):
    """One shared multiplicative 'beam' term, identical on every dataset."""
    return cc.DatasetCovariance(
        X, E * np.ones_like(X),
        [cc.CovTerm('beam', 'multiplicative', np.array([[9e-4]]),
                    weights=np.ones((len(X), 1)), labels=['beam'], scope='shared')], y)


def make_jf(nds, cov=None, share='a', scale_err=None, seed=0):
    """nds datasets D0.. of a*x + b + noise; `a` shared when `share`, `b` per dataset.
    cov: None, or a callable (y) / (y, k) -> DatasetCovariance (private_cov, shared_cov).
    scale_err[i] multiplies dataset i's true noise (the yerr column stays E)."""
    from Joint_Fit_Routines import JointFit
    rng = np.random.default_rng(seed)
    jf = JointFit()
    for i in range(nds):
        s = 1.0 if scale_err is None else scale_err[i]
        y = A_TRUE * X + B_TRUE + s * E * rng.standard_normal(len(X))
        c = None
        if cov is not None:
            try:
                c = cov(y, i)
            except TypeError:
                c = cov(y)
        jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=y, yerr=E * np.ones_like(X), cov=c)
    groups = [jf.share_all(share)] if share else None
    jf.build_master_params(groups=groups)
    for k in jf.master_params:
        jf.master_params[k].vary = True
    return jf


def _close(name, got, want, rtol=1e-12, atol=0.0):
    got, want = np.asarray(got, float), np.asarray(want, float)
    ok = got.shape == want.shape and np.allclose(got, want, rtol=rtol, atol=atol)
    return check(name, ok, 'max|diff|=%.3e' % (np.max(np.abs(got - want)) if got.shape == want.shape
                                                 and got.size else float('nan')))


# ----------------------------------------------------------------------- Q --
def test_q1_quiet_restores():
    jf = make_jf(2); jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    n0 = jf.Niter
    before = [(o['func'].x.copy(), o['func'].a, o['func'].params['a'].value,
               o['func'].__fit__, o.get('yfit_masked')) for o in jf.objectives]
    p = jf.result.params.copy(); p['a'].value += 1.0
    with jf.quiet():
        jf.residual(p, 'Linear')
    check('Q1 Niter unchanged', jf.Niter == n0)
    check('Q1 models restored', all(np.array_equal(b[0], o['func'].x) and b[1] == o['func'].a
                                    and b[2] == o['func'].params['a'].value
                                    and b[3] == o['func'].__fit__ and b[4] == o.get('yfit_masked')
                                    for b, o in zip(before, jf.objectives)))
    # restored on exception too
    try:
        with jf.quiet():
            jf.residual(p, 'Linear')
            raise ValueError('boom')
    except ValueError:
        pass
    check('Q1 restored after exception', jf.Niter == n0 and all(
        b[1] == o['func'].a for b, o in zip(before, jf.objectives)))
    jf.residual(jf.result.params, 'Linear')
    check('Q1 outside quiet() Niter counts again', jf.Niter == n0 + 1)


def test_q2_no_emission():
    jf = make_jf(2); hits = []
    jf.functionCalled.connect(lambda *a: hits.append(1))
    jf.perform_fit(X[0], X[-1], fit_scale='Linear'); n = len(hits)
    check('Q2 fit emitted', n > 0)
    with jf.quiet():
        jf.residual(jf.result.params, 'Linear')
    check('Q2 no functionCalled inside quiet()', len(hits) == n)


def test_q3_row_ranges():
    for case in ('nocov', 'cov_log_drop', 'window_mask'):
        scale, lo, hi = 'Linear', X[0], X[-1]
        if case == 'nocov':
            jf = make_jf(2)
        elif case == 'cov_log_drop':
            jf = make_jf(2, cov=private_cov)
            jf.objectives[1]['y'][5] = -1.0          # one Log y <= 0 row in D1
            scale = 'Log'
        else:                                         # window + column mask
            jf = make_jf(2, cov=private_cov)
            m = np.ones(len(X), bool); m[[3, 4, 20]] = False
            jf.objectives[0]['mask'] = m
            lo, hi = X[2], X[35]
        jf.perform_fit(lo, hi, fit_scale=scale)
        r = jf.residual(jf.result.params, scale)
        rr = jf.row_ranges()
        check('Q3 %s: ranges tile the data rows' % case,
              rr[0][0] == 0 and all(a[1] == b[0] for a, b in zip(rr, rr[1:]))
              and rr[-1][1] == len(r) - len(jf.prior_rows(jf.result.params)),
              str(rr))
        if case == 'cov_log_drop':
            check('Q3 cov_log_drop: D1 is one row short',
                  rr[1][1] - rr[1][0] == len(jf.active_index(1)) - 1
                  and rr[0][1] - rr[0][0] == len(jf.active_index(0)))
        if case == 'window_mask':
            check('Q3 window_mask: D0 count = window minus masked',
                  rr[0][1] - rr[0][0] == len(jf.active_index(0)) < 34)
    jf = make_jf(2, cov=private_cov); jf._fit_scale = 'Linear'   # as after a covariance fit
    try:
        jf.row_ranges(); ok = False
    except RuntimeError:
        ok = True
    check('Q3 covariance path without _cov_mask raises RuntimeError', ok)


def test_q4_priors_unchanged():
    jf = make_jf(2); jf.set_priors({'d0_b': (0.5, 0.1)}); jf.perform_fit(X[0], X[-1])
    r = jf.residual(jf.result.params, 'Linear')
    pr = jf.prior_rows(jf.result.params)
    check('Q4 prior rows are the tail', np.array_equal(r[-1:], pr) and pr.dtype == float)
    jf2 = make_jf(2)
    check('Q4 no priors -> empty', len(jf2.prior_rows(jf2.master_params)) == 0)


def test_q5_last_blocks():
    jf = make_jf(2, cov=private_cov); jf.covariance_mode = 'model'; jf.perform_fit(X[0], X[-1])
    C, _ = cc.assemble(jf._last_blocks, 'Linear')
    _close('Q5 last blocks rebuild L', np.linalg.cholesky(C), jf._L)
    check('Q5 fit scale/method recorded', jf._fit_scale == 'Linear' and jf._fit_method == 'leastsq')


def test_q6_model_blocks():
    jf = make_jf(2, cov=private_cov); jf.covariance_mode = 'model'; jf.perform_fit(X[0], X[-1])
    state = (jf._L.copy(), jf._cov_mask.copy(), list(jf._cov_warnings), jf.Niter, jf._jitter)
    blocks, notes = jf.model_blocks(jf.result.params, 'Linear')
    check('Q6 no state touched', np.array_equal(state[0], jf._L) and
          np.array_equal(state[1], jf._cov_mask) and state[2] == jf._cov_warnings
          and state[3] == jf.Niter and state[4] == jf._jitter)
    jf.push_params(jf.result.params); jf._build_chol('Linear', model=True)
    _close('Q6 equals _build_chol(model=True)',
           np.linalg.cholesky(cc.assemble(blocks, 'Linear')[0]), jf._L)
    check('Q6 no notes on a clean model', notes == [])
    # a model returning NaN on one active row: data substituted there, one note
    func = jf.objectives[0]['func']; orig = func.y
    def y_nan():
        v = orig().copy(); v[7] = np.nan; return v
    func.y = y_nan
    wb = list(jf._cov_warnings)
    blocks, notes = jf.model_blocks(jf.result.params, 'Linear')
    func.y = orig
    b0 = blocks[0]
    check('Q6 NaN row: intensity equals y there', b0.intensity[7] == b0.y[7]
          and np.all(np.isfinite(b0.intensity)))
    check('Q6 NaN row: one note, _cov_warnings untouched', len(notes) == 1
          and wb == jf._cov_warnings, str(notes))
    jf2 = make_jf(2, cov=private_cov)
    try:
        jf2.model_blocks(jf2.master_params, 'Linear'); ok = False
    except RuntimeError:
        ok = True
    check('Q6 without _cov_mask raises RuntimeError', ok)


def main():
    app = QApplication.instance() or QApplication([])
    for t in (test_q1_quiet_restores, test_q2_no_emission, test_q3_row_ranges,
              test_q4_priors_unchanged, test_q5_last_blocks, test_q6_model_blocks):
        t()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
