"""
validate_joint_stats.py -- WS2 reporting harness (sections Q, P, K, M, R).

Section Q (Task 1): JointFit plumbing -- quiet evaluation, row ranges, prior rows,
last / model covariance blocks.
Section P (Task 2): joint_fit_stats -- the JointSnapshot, the sequential / marginal
split, the exact Gaussian -2 ln L (log-determinant included, exact in model mode),
objective labels, and JointFit.last_snapshot.

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


def make_jf(nds, cov=None, share='a', scale_err=None, seed=0, func=None):
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
        jf.add_dataset('D%d' % i, funcClass=func or Line, x=X, y=y, yerr=E * np.ones_like(X), cov=c)
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


# ----------------------------------------------------------------------- P --
def _rel(name, got, want, rtol):
    """Scalar relative check (rtol relative to |want|, floored at 1e-300)."""
    got, want = float(got), float(want)
    d = abs(got - want)
    return check(name, d <= rtol * max(abs(want), 1e-300),
                 'got %.15g want %.15g rel %.2e' % (got, want, d / max(abs(want), 1e-300)))


def _hand_f(jf, i, params=None):
    """a*x + b of dataset i on its active rows, from the master parameters."""
    params = jf.result.params if params is None else params
    loc = {}
    for mk, targets in jf.param_map.items():
        for (k, lk) in targets:
            if k == i:
                loc[lk] = params[mk].value
    x = np.asarray(jf.objectives[i]['x'], float)[jf.active_index(i)]
    return loc['a'] * x + loc['b']


def _cat_yf(jf):
    ys = [np.asarray(o['y'], float)[jf.active_index(i)] for i, o in enumerate(jf.objectives)]
    fs = [_hand_f(jf, i) for i in range(len(jf.objectives))]
    return ys, fs


def _np_shared_C(ys, intens, scale):
    """Independent numpy build of the shared-beam C: diag(sigma^2) + D w S w^T D."""
    y = np.concatenate(ys); d = np.concatenate(intens)
    if scale == 'Linear':
        return np.diag(np.full(len(y), E ** 2)) + 9e-4 * np.outer(d, d)
    return np.diag((E / y) ** 2) + 9e-4 * np.ones((len(y), len(y)))


def _make_jf_order(nds, cov, order, seed=0):
    """make_jf, with the datasets registered in `order` (same data per name)."""
    from Joint_Fit_Routines import JointFit
    rng = np.random.default_rng(seed)
    data = [A_TRUE * X + B_TRUE + E * rng.standard_normal(len(X)) for _ in range(nds)]
    jf = JointFit()
    for i in order:
        jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=data[i],
                       yerr=E * np.ones_like(X), cov=None if cov is None else cov(data[i]))
    jf.build_master_params(groups=[jf.share_all('a')])
    for k in jf.master_params:
        jf.master_params[k].vary = True
    return jf


def test_p1_nocov_identity():
    import joint_fit_stats as jfs
    for scale in ('Linear', 'Log'):
        jf = make_jf(2); jf.set_priors({'d0_b': (0.45, 0.1)})
        jf.perform_fit(X[0], X[-1], fit_scale=scale)
        terms, ex = jfs.per_dataset_objective(jfs.snapshot(jf))
        for i, t in enumerate(terms):
            y = np.asarray(jf.objectives[i]['y'], float); f = _hand_f(jf, i)
            hand = (np.sum(((y - f) / E) ** 2) if scale == 'Linear'
                    else np.sum(((np.log(y) - np.log(f)) * y / E) ** 2))
            _rel('P1 %s D%d sequential == hand' % (scale, i), t.sequential, hand, 1e-12)
            _rel('P1 %s D%d marginal == sequential' % (scale, i), t.marginal, t.sequential, 1e-12)
        _rel('P1 %s sum + prior == chisqr' % scale,
             sum(t.sequential for t in terms) + ex['prior'], jf.result.chisqr, 1e-12)
        check('P1 %s prior term present' % scale, ex['prior'] > 0)


def test_p2_private_mixed():
    import joint_fit_stats as jfs
    jf = make_jf(2, cov=lambda y, k: private_cov(y, k) if k == 0 else None)
    jf.covariance_mode = 'data'; jf.set_priors({'d1_b': (0.45, 0.1)})
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    snap = jfs.snapshot(jf)
    terms, ex = jfs.per_dataset_objective(snap)
    check('P2 cov in use, data mode', snap.cov_in_use and snap.cov_mode == 'data')
    for t in terms:
        _rel('P2 %s sequential == marginal' % t.name, t.marginal, t.sequential, 1e-10)
    check('P2 coupling == 0', abs(ex['coupling']) <= 1e-10 * jf.result.chisqr,
          '%.3e' % ex['coupling'])
    _rel('P2 sum + prior == chisqr', sum(t.sequential for t in terms) + ex['prior'],
         jf.result.chisqr, 1e-10)


def test_p3_shared_split():
    import joint_fit_stats as jfs
    jf = make_jf(3, cov=shared_cov); jf.covariance_mode = 'data'
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    snap = jfs.snapshot(jf)
    terms, ex = jfs.per_dataset_objective(snap)
    tot = sum(t.sequential for t in terms) + ex['prior']
    _rel('P3 sum + prior == chisqr', tot, jf.result.chisqr, 1e-10)
    _rel('P3 first dataset sequential == marginal', terms[0].marginal, terms[0].sequential, 1e-10)
    ys, fs = _cat_yf(jf)
    C = _np_shared_C(ys, ys, 'Linear')
    off = 0
    for i, t in enumerate(terms):
        n = len(ys[i]); r = ys[i] - fs[i]
        Cii = C[off:off + n, off:off + n]; off += n
        _rel('P3 D%d marginal == r C_ii^-1 r' % i, t.marginal, r @ np.linalg.solve(Cii, r), 1e-10)
    _rel('P3 coupling == sum(seq) - sum(marg)', ex['coupling'],
         sum(t.sequential for t in terms) - sum(t.marginal for t in terms), 1e-10)
    check('P3 coupling nonzero with a shared term', abs(ex['coupling']) > 1e-6, '%.3e' % ex['coupling'])
    check('P3 order recorded', ex['order'] == ['D0', 'D1', 'D2'])
    jr = _make_jf_order(3, shared_cov, [2, 1, 0]); jr.covariance_mode = 'data'
    jr.perform_fit(X[0], X[-1], fit_scale='Linear')
    tr, er = jfs.per_dataset_objective(jfs.snapshot(jr))
    _rel('P3 reversed order: same total', sum(t.sequential for t in tr) + er['prior'], tot, 1e-10)
    check('P3 reversed order: different first sequential',
          abs(tr[0].sequential - terms[0].sequential) > 1e-6 * tot,
          '%.6g vs %.6g' % (tr[0].sequential, terms[0].sequential))


def test_p4_model_mode():
    import joint_fit_stats as jfs
    jf = make_jf(2, cov=private_cov); jf.covariance_mode = 'model'
    jf.set_priors({'d0_b': (0.45, 0.1)})
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    check('P4 model-mode outer iteration ran', jf.cov_info.get('rounds', 0) >= 1, str(jf.cov_info))
    snap = jfs.snapshot(jf)
    nd = len(snap.z_fit)
    _rel('P4 sum z_fit^2 + prior == chisqr', np.sum(snap.z_fit ** 2) + np.sum(snap.prior ** 2),
         jf.result.chisqr, 1e-10)
    Lf = jfs.rebuild_L(snap, 'fit')
    rcat = np.concatenate([d.r for d in snap.datasets])
    _rel('P4 rebuild_L(fit) reproduces chisqr',
         np.sum(cc.whiten(Lf, rcat) ** 2) + np.sum(snap.prior ** 2), jf.result.chisqr, 1e-10)
    dmax = max(np.max(np.abs(d.f - d.fit_intensity)) for d in snap.datasets)
    check('P4 f != fit_intensity (final round used the previous C)', dmax > 0, '%.3e' % dmax)
    for d in snap.datasets:
        _close('P4 %s intensity == f on used rows' % d.name, d.intensity, d.f)
    blocks, _ = jf.model_blocks(jf.result.params, 'Linear')
    _close('P4 rebuild_L(exact) == chol(assemble(model_blocks))', jfs.rebuild_L(snap),
           np.linalg.cholesky(cc.assemble(blocks, 'Linear')[0]))
    terms, ex = jfs.per_dataset_objective(snap)
    check('P4 fit_gap present', ex['fit_gap'] is not None)
    if ex['fit_gap'] is not None:
        check('P4 fit_gap == sum seq + prior - chisqr',
              abs(ex['fit_gap'] - (sum(t.sequential for t in terms) + ex['prior']
                                   - jf.result.chisqr)) <= 1e-12 * jf.result.chisqr,
              '%.6g' % ex['fit_gap'])
    check('P4 z length == data rows', len(snap.z) == nd)


def test_p12_exact_likelihood():
    import joint_fit_stats as jfs
    from scipy.stats import multivariate_normal
    for scale in ('Linear', 'Log'):
        for mode in ('data', 'model'):
            jf = make_jf(3, cov=shared_cov); jf.covariance_mode = mode
            jf.perform_fit(X[0], X[-1], fit_scale=scale)
            terms, ex = jfs.per_dataset_objective(jfs.snapshot(jf))
            ys, fs = _cat_yf(jf)
            C = _np_shared_C(ys, ys if mode == 'data' else fs, scale)
            tag = 'P12 %s %s' % (scale, mode)
            _rel(tag + ' sum logdet_seq == slogdet(C)', sum(t.logdet_seq for t in terms),
                 np.linalg.slogdet(C)[1], 1e-10)
            yc, fc = np.concatenate(ys), np.concatenate(fs)
            if scale == 'Log':
                yc, fc = np.log(yc), np.log(fc)
            want = -2 * multivariate_normal(mean=fc, cov=C).logpdf(yc)
            _rel(tag + ' nll2 == -2 logpdf', ex['nll2'], want, 1e-10)
            _rel(tag + ' first nll2_seq == nll2_marg', terms[0].nll2_marg, terms[0].nll2_seq, 1e-10)
            off = 0
            for i, t in enumerate(terms):
                n = len(ys[i]); Cii = C[off:off + n, off:off + n]
                _rel(tag + ' D%d logdet_marg == slogdet(C_ii)' % i, t.logdet_marg,
                     np.linalg.slogdet(Cii)[1], 1e-10)
                _rel(tag + ' D%d nll2_marg == -2 logpdf(y_i)' % i, t.nll2_marg,
                     -2 * multivariate_normal(mean=fc[off:off + n], cov=Cii).logpdf(yc[off:off + n]),
                     1e-10)
                off += n
            check(tag + ' is a likelihood', ex['likelihood'] is True)


def test_p13_logdet_nocov_unweighted():
    import joint_fit_stats as jfs
    jf = make_jf(2); jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    terms, ex = jfs.per_dataset_objective(jfs.snapshot(jf))
    for i, t in enumerate(terms):
        _rel('P13 D%d logdet_seq == sum ln E^2' % i, t.logdet_seq,
             len(jf.active_index(i)) * np.log(E ** 2), 1e-12)
        _rel('P13 D%d logdet_marg == logdet_seq' % i, t.logdet_marg, t.logdet_seq, 1e-12)
    check('P13 no-cov fit_gap None', ex['fit_gap'] is None)
    jf = make_jf(2); jf.perform_fit(X[0], X[-1], fit_scale='Linear w/o error')
    terms, ex = jfs.per_dataset_objective(jfs.snapshot(jf))
    check("P13 'Linear w/o error': all nll2/logdet None",
          all(t.logdet_seq is None and t.logdet_marg is None and t.nll2_seq is None
              and t.nll2_marg is None for t in terms) and ex['nll2'] is None)
    check("P13 'Linear w/o error' not a likelihood", ex['likelihood'] is False)
    jf = make_jf(2, cov=shared_cov); jf.covariance_mode = 'data'
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    terms, ex = jfs.per_dataset_objective(jfs.snapshot(jf))
    check('P13 data mode: fit_gap None', ex['fit_gap'] is None)
    check('P13 data mode label clause',
          'covariance built from the measured data' in ex['objective_label'], ascii(ex['objective_label']))


def test_p5_labels():
    import joint_fit_stats as jfs
    cases = [
        (('Linear', False, None), 'χ² (Linear, weighted by the error column)', True),
        (('Linear', True, 'model'), 'GLS χ² (Linear, covariance model)', True),
        (('Log', False, None), 'χ² of ln y (Log, σ_ln = σ/y)', True),
        (('Log', True, 'model'), 'GLS χ² of ln y (Log, covariance model)', True),
        (('Log w/o error', False, None), 'unweighted sum of squares of log10 y', False),
        (('Linear w/o error', False, None),
         'unweighted sum of squares of y / max(y), per dataset', False),
    ]
    for args, want, lik in cases:
        got = jfs.objective_label(*args)
        check('P5 label %s' % (args,), got == (want, lik), ascii(got))
    for sc in ('Linear', 'Log'):
        lab, lik = jfs.objective_label(sc, True, 'data')
        check('P5 %s data mode appends the clause' % sc,
              lab.endswith('covariance data); covariance built from the measured data; approximate')
              and lik, ascii(lab))


def test_p6_emcee_label():
    import joint_fit_stats as jfs
    jf = make_jf(2); jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='emcee',
                   emcee_walkers=10, emcee_steps=20)
    snap = jf.last_snapshot
    check('P6 emcee snapshot label', snap is not None and snap.label == 'at posterior medians',
          None if snap is None else repr(snap.label))


def test_p7_aborted_label():
    import joint_fit_stats as jfs
    ref = make_jf(2, cov=private_cov); ref.covariance_mode = 'model'
    seen = {}
    orig = ref._build_chol
    def spy(fit_scale='Linear', model=False):
        if model and 'n0' not in seen:
            seen['n0'] = ref.Niter
        return orig(fit_scale, model)
    ref._build_chol = spy
    ref.perform_fit(X[0], X[-1], fit_scale='Linear')
    n0 = seen.get('n0')
    check('P7 round-0 evaluation count measured', n0 is not None and n0 > 0, str(n0))
    jf = make_jf(2, cov=private_cov); jf.covariance_mode = 'model'
    def slot(params, niter, resid, scale):
        if n0 is not None and niter > n0:
            jf.fit_abort = True
    jf.functionCalled.connect(slot)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    check('P7 fit recorded as aborted', jf.cov_info.get('aborted') is True, str(jf.cov_info))
    snap = jf.last_snapshot
    check('P7 aborted label', snap is not None
          and snap.label == 'aborted fit; not the minimised value',
          [str(w) for w in jf.warnings] if snap is None else repr(snap.label))


class MaskedLine(Line):
    __masked_x__ = True


def test_p8_masked_x():
    import joint_fit_stats as jfs
    from Joint_Fit_Routines import JointFit
    for case in ('nocov', 'cov'):
        rng = np.random.default_rng(3)
        jf = JointFit()
        for i in range(2):
            y = A_TRUE * X + B_TRUE + E * rng.standard_normal(len(X))
            m = np.ones(len(X), bool); m[[2, 9, 17 + i]] = False
            jf.add_dataset('D%d' % i, funcClass=MaskedLine, x=X, y=y, yerr=E * np.ones_like(X),
                           cov=private_cov(y, i) if case == 'cov' else None, mask=m)
        jf.build_master_params(groups=[jf.share_all('a')])
        for k in jf.master_params:
            jf.master_params[k].vary = True
        jf.covariance_mode = 'data'
        jf.perform_fit(X[1], X[-2], fit_scale='Linear')
        check('P8 %s: evaluate() masked the model' % case,
              all(o['yfit_masked'] for o in jf.objectives))
        snap = jfs.snapshot(jf)
        terms, ex = jfs.per_dataset_objective(snap)
        check('P8 %s: f has the used length' % case,
              all(len(d.f) == int(d.used.sum()) == len(jf.active_index(i)) == len(d.r)
                  for i, d in enumerate(snap.datasets)),
              str([(len(d.f), int(d.used.sum())) for d in snap.datasets]))
        for i, d in enumerate(snap.datasets):
            _close('P8 %s: %s f == hand f' % (case, d.name), d.f, _hand_f(jf, i))
        _rel('P8 %s: sum + prior == chisqr' % case,
             sum(t.sequential for t in terms) + ex['prior'], jf.result.chisqr, 1e-12)


def test_p9_empty_dataset():
    import joint_fit_stats as jfs
    jf = make_jf(2, cov=private_cov); jf.covariance_mode = 'data'
    jf.objectives[1]['y'][:] = -1.0
    jf.master_params['d1_b'].vary = False
    jf.perform_fit(X[0], X[-1], fit_scale='Log')
    snap = jfs.snapshot(jf)
    try:
        terms, ex = jfs.per_dataset_objective(snap); ok = True
    except Exception as e:                                  # noqa: BLE001
        ok = False; terms = None; print('P9 raised:', repr(e))
    check('P9 per_dataset_objective does not raise on an empty dataset', ok)
    if terms:
        t = terms[1]
        check('P9 empty dataset: n 0, sequential 0.0, chi2_per_point None',
              t.n_points == 0 and t.sequential == 0.0 and t.marginal == 0.0
              and t.chi2_per_point is None, str(t))
        check('P9 empty dataset: no NaN anywhere',
              all(v is None or np.isfinite(v) for v in
                  (t.logdet_seq, t.logdet_marg, t.nll2_seq, t.nll2_marg)))


def test_p10_no_side_effects():
    import joint_fit_stats as jfs
    for cov, mode in ((None, None), (private_cov, 'model')):
        jf = make_jf(2, cov=cov)
        if mode:
            jf.covariance_mode = mode
        hits = []
        jf.functionCalled.connect(lambda *a: hits.append(1))
        jf.perform_fit(X[0], X[-1], fit_scale='Linear')
        # leave the models in a state that is NOT the fitted one, so a snapshot that
        # pushes/evaluates without restoring shows up
        p = jf.result.params.copy()
        for k in p:
            p[k].value += 0.25
        jf.push_params(p)
        for o in jf.objectives:
            o['func'].x = X[:7].copy(); o['func'].__fit__ = False
        n0, h0 = jf.Niter, len(hits)
        before = [(o['func'].x.copy(), o['func'].a, o['func'].b, o['func'].params['a'].value,
                   o['func'].params['b'].value, o['func'].__fit__) for o in jf.objectives]
        L0 = None if jf._L is None else jf._L.copy()
        jfs.snapshot(jf)
        tag = 'P10 %s' % ('cov model' if cov else 'nocov')
        check(tag + ' Niter and emissions unchanged', jf.Niter == n0 and len(hits) == h0)
        check(tag + ' models unchanged', all(
            np.array_equal(b[0], o['func'].x) and b[1:] == (
                o['func'].a, o['func'].b, o['func'].params['a'].value,
                o['func'].params['b'].value, o['func'].__fit__)
            for b, o in zip(before, jf.objectives)))
        check(tag + ' _L unchanged', (L0 is None and jf._L is None)
              or np.array_equal(L0, jf._L))


def test_p11_last_snapshot():
    import joint_fit_stats as jfs
    jf = make_jf(2); jf.perform_fit(X[0], X[-1])
    check('P11 leastsq sets last_snapshot', isinstance(jf.last_snapshot, jfs.JointSnapshot))
    jm = make_jf(2, cov=private_cov); jm.covariance_mode = 'model'; jm.perform_fit(X[0], X[-1])
    check('P11 model-mode sets last_snapshot', isinstance(jm.last_snapshot, jfs.JointSnapshot)
          and jm.last_snapshot.cov_mode == 'model')
    jm.perform_fit(X[0], X[-1], fit_method='emcee', emcee_walkers=10, emcee_steps=20)
    check('P11 emcee sets last_snapshot', isinstance(jm.last_snapshot, jfs.JointSnapshot))
    check('P11 model-mode emcee label has the frozen-C clause',
          jm.last_snapshot is not None and jm.last_snapshot.label ==
          'at posterior medians; sampled with C frozen at the prior fit, not the exact likelihood',
          None if jm.last_snapshot is None else repr(jm.last_snapshot.label))
    ja = make_jf(2); ja.perform_fit(X[0], X[-1])
    jb = make_jf(2)
    orig = jfs.snapshot
    def boom(jf):
        raise ValueError('forced')
    jfs.snapshot = boom
    try:
        rep, _msg = jb.perform_fit(X[0], X[-1])
    finally:
        jfs.snapshot = orig
    check('P11 failing snapshot: last_snapshot None', jb.last_snapshot is None)
    check('P11 failing snapshot: result unchanged',
          jb.result.chisqr == ja.result.chisqr and all(
              jb.result.params[k].value == ja.result.params[k].value for k in ja.result.params)
          and jb.result.nfev == ja.result.nfev)
    check('P11 failing snapshot: warning added',
          'diagnostics unavailable: snapshot failed: forced' in [str(w) for w in jb.warnings],
          str(jb.warnings))


# ----------------------------------------------------------------------- K --
class Quad(Line):
    def y(self):
        return self.a * np.asarray(self.x, float) ** 2 + self.b


def _raises(a, b):
    import joint_fit_stats as jfs
    try:
        jfs.check_comparable(a, b)
    except jfs.NotComparable as e:
        return str(e)
    return None


def _fit(nds=2, cov=private_cov, mode=None, scale='Linear', xmax=None, func=None,
         method='leastsq'):
    jf = make_jf(nds, cov=cov, func=func)
    if mode:
        jf.covariance_mode = mode
    jf.perform_fit(X[0], X[-1] if xmax is None else xmax, fit_scale=scale, fit_method=method)
    return jf


def test_k1_fields():
    import dataclasses as dc
    import joint_fit_stats as jfs
    a = jfs.snapshot(_fit(mode='data'))
    check('K1 identical fits are comparable',
          _raises(a, jfs.snapshot(_fit(mode='data'))) is None)
    key = jfs.comparability_key(a)
    check('K1 key is a plain dict with the spec fields',
          isinstance(key, dict) and key['fit_scale'] == 'Linear' and len(key['datasets']) == 2
          and set(key['datasets'][0]) == {'name', 'data_hash', 'window', 'index_hash', 'cov'}
          and key['datasets'][0]['cov'][2] == 'data' and key['rescale'] is None)
    d0, d1 = a.datasets

    def rep(i, **kw):
        ds = list(a.datasets)
        ds[i] = dc.replace(ds[i], **kw)
        return dc.replace(a, datasets=tuple(ds))

    y2 = d1.y.copy()
    y2[3] += 1e-3
    cases = [
        ('window', jfs.snapshot(_fit(mode='data', xmax=X[-2]))),
        ('data', rep(1, y=y2)),
        ('index', rep(1, index_hash=jfs._hash(d1.active_index[1:]))),
        ('scale', jfs.snapshot(_fit(mode='data', scale='Log'))),
        ('covariance', jfs.snapshot(_fit(cov=None))),
        ('mode', jfs.snapshot(_fit(mode='model'))),
        ('order', dc.replace(a, datasets=(d1, d0))),
        ('rescale', dc.replace(a, rescale={'mode': 'per_dataset', 'inflate_only': False,
                                           's': (1.1, 0.9), 'nu': (40, 40), 'rounds': 1,
                                           'converged': True})),
    ]
    for word, b in cases:
        for lhs, rhs in ((a, b), (b, a)):
            msg = _raises(lhs, rhs)
            check('K1 %s change refused naming %r' % (word, word),
                  msg is not None and word in msg, repr(msg))
    msg = _raises(a, jfs.snapshot(_fit(mode='data', xmax=X[-2])))
    check('K1 window message shape', msg is not None and msg.startswith('dataset 1')
          and '0.100:2.000' in msg and '0.100:1.951' in msg, repr(msg))
    r1 = dc.replace(a, rescale={'mode': 'per_dataset', 'inflate_only': False, 's': (1., 1.)})
    r2 = dc.replace(a, rescale={'mode': 'per_dataset', 'inflate_only': True, 's': (1., 1.)})
    r3 = dc.replace(a, rescale={'mode': 'per_dataset', 'inflate_only': False,
                                's': (2., 3.), 'rounds': 4})
    check('K1 inflate_only differs -> refused', _raises(r1, r2) is not None)
    check('K1 fitted s / rounds do not matter', _raises(r1, r3) is None)
    c2 = dc.replace(d0.cov, source='other.cov')
    check('K1 covariance source differs -> refused',
          'covariance' in (_raises(a, rep(0, cov=c2, cov_source='other.cov')) or ''))


def test_k2_allowed():
    import joint_fit_stats as jfs
    a = jfs.snapshot(_fit(mode='data'))
    bq = jfs.snapshot(_fit(mode='data', func=Quad))
    check('K2 the model classes really differ', bq.datasets[0].model_class == 'Quad'
          and a.datasets[0].model_class == 'Line')
    check('K2 model class allowed', _raises(a, bq) is None, repr(_raises(a, bq)))
    bm = jfs.snapshot(_fit(mode='data', method='least_squares'))
    check('K2 fit method allowed', bm.method != a.method and _raises(a, bm) is None,
          repr(_raises(a, bm)))
    jf = make_jf(2, cov=private_cov)
    jf.covariance_mode = 'data'
    jf.set_priors({'d0_b': (0.45, 0.1)})
    jf.perform_fit(X[0], X[-1])
    check('K2 priors allowed', _raises(a, jfs.snapshot(jf)) is None)
    jf = make_jf(2, cov=private_cov)
    jf.covariance_mode = 'data'
    jf.master_params['d1_b'].vary = False
    jf.perform_fit(X[0], X[-1])
    b = jfs.snapshot(jf)
    check('K2 a fixed parameter really differs', a.vary != b.vary)
    check('K2 a parameter fixed in b allowed', _raises(a, b) is None, repr(_raises(a, b)))


def _cov_variant(sig=4e-4, dsig=1.0):
    def f(y, k):
        return cc.DatasetCovariance(
            X, E * dsig * np.ones_like(X),
            [cc.CovTerm('stitch', 'multiplicative', np.diag([0, sig]),
                        weights=_onehot(X > 1.0), scope='dataset')], y)
    return f


def test_k3_model_mode():
    import dataclasses as dc
    import joint_fit_stats as jfs
    ja = make_jf(2, cov=private_cov)
    ja.covariance_mode = 'model'
    ja.perform_fit(X[0], X[-1])
    jb = make_jf(2, cov=private_cov)
    jb.covariance_mode = 'model'
    for k in jb.master_params:
        jb.master_params[k].value = jb.master_params[k].value + 0.7
    jb.perform_fit(X[0], X[-1])
    a, b = jfs.snapshot(ja), jfs.snapshot(jb)
    check('K3 same data and covariance file, different start: comparable',
          _raises(a, b) is None, repr(_raises(a, b)))
    b2 = dc.replace(b, datasets=tuple(dc.replace(d, intensity=d.intensity * 1.05)
                                      for d in b.datasets))
    check('K3 intensities genuinely different',
          not np.allclose(a.datasets[0].intensity, b2.datasets[0].intensity))
    check('K3 different intensities do not matter', _raises(a, b2) is None)
    check('K3 cov_hash identical', jfs.comparability_key(a)['datasets'][0]['cov']
          == jfs.comparability_key(b2)['datasets'][0]['cov'])
    for tag, cf in (('one term sigma', _cov_variant(sig=9e-4)),
                    ('one sigma_meas', _cov_variant(dsig=1.5))):
        jc = make_jf(2, cov=cf)
        jc.covariance_mode = 'model'
        jc.perform_fit(X[0], X[-1])
        msg = _raises(a, jfs.snapshot(jc))
        check('K3 %s changed -> refused naming covariance' % tag,
              msg is not None and 'covariance' in msg, repr(msg))


def main():
    app = QApplication.instance() or QApplication([])
    for t in (test_q1_quiet_restores, test_q2_no_emission, test_q3_row_ranges,
              test_q4_priors_unchanged, test_q5_last_blocks, test_q6_model_blocks,
              test_p1_nocov_identity, test_p2_private_mixed, test_p3_shared_split,
              test_p4_model_mode, test_p12_exact_likelihood, test_p13_logdet_nocov_unweighted,
              test_p5_labels, test_p6_emcee_label, test_p7_aborted_label, test_p8_masked_x,
              test_p9_empty_dataset, test_p10_no_side_effects, test_p11_last_snapshot,
              test_k1_fields, test_k2_allowed, test_k3_model_mode):
        t()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
