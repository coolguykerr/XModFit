"""
validate_joint_diag.py -- WS2 reporting harness, sections D (Task 6), later I, F, L.

Section D: the "Joint-fit diagnostics:" block of JointFit._report() -- quality,
heterogeneity, q-range, shared-parameter warnings, degeneracy, failure safety.

Fixtures come from validate_joint_stats.  Report text contains non-ascii (chi, sigma,
sqrt): never print it raw, use ascii().

    QT_QPA_PLATFORM=offscreen python validate_joint_diag.py
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

from tests.harness import check, finish
from validate_joint_stats import (Line, make_jf, shared_cov, private_cov, X, E, A_TRUE, B_TRUE,
                                  _close)
import joint_fit_stats as jfs

HEAD = 'Joint-fit diagnostics:'


class Deg(Line):
    """y = (a + c)*x + b : a and c exactly degenerate (singular Hessian)."""
    def __init__(self, x=0, a=1.0, b=0.0, c=0.0, mpar={}):
        self.c = c
        super().__init__(x=x, a=a, b=b, mpar=mpar)

    def init_params(self):
        super().init_params()
        self.params.add('c', value=self.c, vary=0, min=-np.inf, max=np.inf)

    def y(self):
        return (self.a + self.c) * np.asarray(self.x, float) + self.b


class NearDeg(Deg):
    """y = (a + c)*x + b + 1e-3*c*x**2 : a and c degenerate to 1e-3 (finite Hessian).
    The unidentified direction is a -> a + t, c -> c - t, so rho(a, c) = -1 and the
    softest direction is 'a - 1*c' (opposite signs; the brief's 'same sign' is the sign
    of the junk covariance an exactly singular Hessian leaves, see test D5c)."""
    def y(self):
        x = np.asarray(self.x, float)
        return (self.a + self.c) * x + self.b + 1e-3 * self.c * x ** 2


def _fit(jf, scale='Linear', lo=None, hi=None, **kw):
    rep, _ = jf.perform_fit(X[0] if lo is None else lo, X[-1] if hi is None else hi,
                            fit_scale=scale, **kw)
    return rep


def _block(rep):
    i = rep.rfind('\n\n' + HEAD)
    return (rep[:i], rep[i + 2:]) if i >= 0 else (rep, '')


def _diag(jf, **kw):
    return jfs.diagnostics(jf, jf.last_snapshot, **kw)


def test_d1_block_and_identity():
    jf = make_jf(3, cov=private_cov)
    jf.covariance_mode = 'data'
    rep = _fit(jf)
    pre, blk = _block(rep)
    check('D1 report ends with the diagnostics block', blk.startswith(HEAD) and rep.endswith(blk))
    check('D1 block has the objective label', 'GLS χ² (Linear, covariance data)' in blk,
          ascii(blk[:200]))
    check('D1 one quality row per dataset', all(('D%d ' % i) in blk for i in range(3)), ascii(blk))
    check('D1 first line unchanged (lmfit header)', not rep.startswith(HEAD))
    # byte identity: patch diagnostics away, rebuild the report from the same fit
    od, of = jfs.diagnostics, jfs.format_diagnostics
    try:
        jfs.diagnostics = lambda *a, **k: {}
        jfs.format_diagnostics = lambda d: ''
        bare = jf._report()
    finally:
        jfs.diagnostics, jfs.format_diagnostics = od, of
    check('D1 everything before the block is byte-identical',
          bare == pre + '\n\n', 'len %d vs %d' % (len(bare), len(pre)))
    # no-covariance, plain fit too
    jf = make_jf(2)
    rep = _fit(jf)
    check('D1 nocov block present', _block(rep)[1].startswith(HEAD))


def test_d2_heterogeneity():
    jf = make_jf(3, scale_err=(1.0, 1.0, 4.0))
    _fit(jf)
    d = _diag(jf)
    txt = jfs.format_diagnostics(d)
    check('D2 E,E,4E flagged', d['heterogeneity']['flagged'], str(d['heterogeneity']))
    check('D2 flagged text', 'error bars or model adequacy differ between datasets' in txt)
    jf = make_jf(3)
    _fit(jf)
    d = _diag(jf)
    check('D2 E,E,E not flagged', not d['heterogeneity']['flagged'], str(d['heterogeneity']))
    check('D2 E,E,E no text', 'error bars or model adequacy differ' not in jfs.format_diagnostics(d))


def test_d3_qrange():
    jf = make_jf(2)
    _fit(jf, ranges={'D0': (0.1, 2.0), 'D1': (0.1, 1.6)})
    d = _diag(jf)
    q = d['qrange']
    qs = [dd.window[1] for dd in jf.last_snapshot.datasets]
    check('D3 spread flagged', q['flagged'] and abs(q['spread'] - (max(qs) / min(qs) - 1)) < 1e-12
          and q['spread'] > 0.2, str(q))
    check('D3 mode native windows', q['mode'] == 'native windows'
          and 'native windows' in jfs.format_diagnostics(d))
    jf = make_jf(2)
    _fit(jf, ranges={'D0': (0.1, 2.0), 'D1': (0.1, 2.0)})
    q = _diag(jf)['qrange']
    check('D3 identical windows not flagged', (not q['flagged']) and q['spread'] == 0.0, str(q))


def test_d4_shared_warnings():
    from Joint_Fit_Routines import JointFit
    rng = np.random.default_rng(3)
    jf = JointFit()
    for i, st in enumerate((2.5, 3.0)):
        y = A_TRUE * X + B_TRUE + E * rng.standard_normal(len(X))
        jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=y, yerr=E * np.ones_like(X))
        jf.objectives[i]['func'].params['a'].set(value=st, vary=True)
        jf.objectives[i]['func'].params['b'].set(vary=True)
    jf.build_master_params(groups=[jf.share_all('a')])
    rep = _fit(jf)
    d = _diag(jf)
    ws = d['shared_warnings']
    check('D4 captured', len(ws) == 1 and 'shared a: endpoints start at different values' in ws[0],
          ascii(ws))
    _, blk = _block(rep)
    check('D4 restated in the block', 'shared a: endpoints start at different values' in blk)
    check('D4 none when consistent', _diag(_fitted(make_jf(2)))['shared_warnings'] == [])


def _fitted(jf, **kw):
    _fit(jf, **kw)
    return jf


def test_d5_degeneracy():
    jf = make_jf(1, func=NearDeg, share=None)
    for k in ('d0_a', 'd0_b', 'd0_c'):
        jf.master_params[k].vary = True
    rep = _fit(jf)
    d = _diag(jf)['degeneracy']
    if 'unavailable' in d:
        check('D5 covariance available for the near-degenerate pair', False, ascii(d))
        return
    pr = [(a, b, r) for (a, b, r) in d['pairs'] if {a, b} == {'d0_a', 'd0_c'}]
    check('D5 pair (a, c) with |rho| > 0.9', len(pr) == 1 and abs(pr[0][2]) > 0.9, str(d['pairs']))
    sf = dict(d['softest'])
    check('D5 softest names a and c with |coef| 1, opposite signs (rho = -1)',
          'd0_a' in sf and 'd0_c' in sf and sf['d0_a'] * sf['d0_c'] < 0
          and abs(abs(sf['d0_a']) - 1) < 0.05 and abs(abs(sf['d0_c']) - 1) < 0.05, str(d['softest']))
    txt = jfs.format_diagnostics(_diag(jf))
    check('D5 text of the softest direction', 'softest direction: d0_a - 1·d0_c' in txt
          or 'softest direction: d0_c - 1·d0_a' in txt, ascii(txt))
    check('D5 block carries it', 'degenerate pair: d0_a, d0_c' in _block(rep)[1])


def test_d5c_exactly_singular():
    jf = make_jf(1, func=Deg, share=None)
    for k in ('d0_a', 'd0_b', 'd0_c'):
        jf.master_params[k].vary = True
    rep = _fit(jf)
    d = _diag(jf)
    txt = jfs.format_diagnostics(d)
    check('D5c exactly singular Hessian: no spurious pair, says so',
          'unavailable' in d['degeneracy'] and 'not positive definite' in txt and 'nan' not in txt.lower(),
          ascii(d['degeneracy']))
    check('D5c nothing raised', d['unavailable'] == [] and _block(rep)[1].startswith(HEAD))


def test_d5b_text_and_scaling():
    # hand-built snapshot: correlation 0.9 pair + independent -> coefficients scaled to 1
    import dataclasses
    jf = make_jf(1, share=None)
    _fit(jf)
    snap = jf.last_snapshot
    cov = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    s2 = dataclasses.replace(snap, covar=cov, var_names=('d_HGi', 'd_wat', 'z'))
    r = jfs._degeneracy(s2)
    check('D5b pairs below the threshold are omitted', r['pairs'] == [], str(r['pairs']))
    cov[0, 1] = cov[1, 0] = 0.95
    r = jfs._degeneracy(dataclasses.replace(snap, covar=cov, var_names=('d_HGi', 'd_wat', 'z')))
    check('D5b pair found', len(r['pairs']) == 1 and r['pairs'][0][:2] == ('d_HGi', 'd_wat'))
    check('D5b |coef| scaled to max 1, small terms omitted',
          max(abs(c) for _, c in r['softest']) == 1.0 and 'z' not in dict(r['softest']),
          str(r['softest']))
    cov[0, 1] = cov[1, 0] = -0.95
    r = jfs._degeneracy(dataclasses.replace(snap, covar=cov, var_names=('d_HGi', 'd_wat', 'z')))
    t = jfs._softest_text(r['softest'])
    check('D5b opposite sign written with a minus', ' - ' in t and t.startswith('d_HGi'), ascii(t))


def test_d6_no_covar():
    jf = make_jf(2)
    _fit(jf)
    jf.result.covar = None
    jf.last_snapshot = jfs.snapshot(jf)
    txt = jfs.format_diagnostics(_diag(jf))
    check('D6 snapshot covar None', jf.last_snapshot.covar is None)
    check('D6 no-covariance line', 'no parameter covariance' in txt, ascii(txt))
    check('D6 information criteria still printed', 'AIC' in txt and 'BIC' in txt)
    check('D6 nothing unavailable', _diag(jf)['unavailable'] == [])
    check('D6 report path survives', _block(jf._report())[1].startswith(HEAD))


def test_d7_section_failure():
    for cov in (None, private_cov):
        jf = make_jf(2, cov=cov)
        hits = []
        jf.functionCalled.connect(lambda *a: hits.append(1))
        rep0 = _fit(jf)
        pre0, _ = _block(rep0)
        params0 = {k: (p.value, p.stderr) for k, p in jf.result.params.items()}
        chi0 = jf.result.chisqr
        n0, h0 = jf.Niter, len(hits)
        orig = jfs._degeneracy

        def boom(*a, **k):
            raise ValueError('forced')
        jfs._degeneracy = boom
        try:
            rep1 = jf._report()
        finally:
            jfs._degeneracy = orig
        tag = 'D7 %s' % ('cov' if cov else 'nocov')
        pre1, blk = _block(rep1)
        check(tag + ' unavailable line', 'diagnostics unavailable (degeneracy): ValueError: forced' in blk,
              ascii(blk))
        check(tag + ' rest of block still present', 'AIC' in blk and 'objective:' in blk)
        check(tag + ' report before the block unchanged', pre1 == pre0)
        check(tag + ' params and chisqr unchanged',
              params0 == {k: (p.value, p.stderr) for k, p in jf.result.params.items()}
              and chi0 == jf.result.chisqr)
        check(tag + ' Niter and emissions unchanged', jf.Niter == n0 and len(hits) == h0)
        d = jfs.diagnostics(jf, jf.last_snapshot)
        check(tag + ' diagnostics() never raises, lists section',
              [u[0] for u in d['unavailable']] == [] and d['degeneracy'] is not None)


def test_d7b_all_sections_fail():
    jf = make_jf(2)
    _fit(jf)
    ops = jfs.per_dataset_objective, jfs.information_criteria
    def boom(*a, **k):
        raise RuntimeError('x')
    jfs.per_dataset_objective = boom
    jfs.information_criteria = boom
    try:
        d = _diag(jf)
        txt = jfs.format_diagnostics(d)
        rep = jf._report()
    finally:
        jfs.per_dataset_objective, jfs.information_criteria = ops
    secs = [u[0] for u in d['unavailable']]
    check('D7b failing sections listed',
          {'objective', 'quality', 'heterogeneity', 'information_criteria'} <= set(secs), str(secs))
    check('D7b unaffected sections computed', d['qrange'] is not None and d['degeneracy'] is not None)
    check('D7b text lines', 'diagnostics unavailable (quality): RuntimeError: x' in txt)
    check('D7b report returned', _block(rep)[1].startswith(HEAD))


def test_d8_snapshot_fails():
    jf = make_jf(2)
    orig = jfs.snapshot
    def boom(j):
        raise RuntimeError('snap boom')
    jfs.snapshot = boom
    try:
        rep = _fit(jf)
    finally:
        jfs.snapshot = orig
    check('D8 snapshot is None', jf.last_snapshot is None)
    pre, blk = _block(rep)
    check('D8 report returned with the unavailable line',
          blk.startswith(HEAD) and 'diagnostics unavailable' in blk and rep.endswith(blk), ascii(rep[-200:]))
    check('D8 fit unaffected', jf.result is not None and np.isfinite(jf.result.chisqr))


def test_d9_empty_dataset():            # ruling R1: P9's missing format_diagnostics check
    jf = make_jf(2, cov=private_cov)
    jf.covariance_mode = 'data'
    jf.objectives[1]['y'][:] = -1.0
    jf.master_params['d1_b'].vary = False
    rep = _fit(jf, scale='Log')
    try:
        d = _diag(jf)
        txt = jfs.format_diagnostics(d)
        ok = True
    except Exception as e:                      # noqa: BLE001
        ok = False; txt = ''
        print('D9 raised', repr(e))
    check('D9 diagnostics + format do not raise', ok)
    row = [l for l in txt.splitlines() if l.strip().startswith('D1 ')]
    check('D9 quality row says no points', len(row) == 1 and 'no points' in row[0], ascii(txt))
    check('D9 no NaN in text', 'nan' not in txt.lower(), ascii(txt))
    check('D9 nothing unavailable', d['unavailable'] == [], str(d['unavailable']))
    check('D9 report carries it', 'no points' in _block(rep)[1])


def test_d10_objective_extras():
    jf = make_jf(2, cov=shared_cov)
    jf.covariance_mode = 'data'
    _fit(jf)
    o = _diag(jf)['objective']
    check('D10 coupled fit lists dataset order', o.get('order') == ['D0', 'D1'], ascii(o))
    check('D10 nll2 carried', o['nll2'] is not None and np.isfinite(o['nll2']))
    jf = make_jf(2, cov=private_cov)
    jf.covariance_mode = 'model'
    _fit(jf)
    o = _diag(jf)['objective']
    check('D10 model mode carries fit_gap and notes list',
          'fit_gap' in o and isinstance(o['notes'], list), ascii(o))
    jf = make_jf(2)
    _fit(jf)
    d = _diag(jf)
    check('D10 Task 8 keys exist as None',
          all(k in d and d[k] is None for k in ('fisher', 'tension', 'optimum_distance')))
    check('D10 uncoupled: no order', 'order' not in d['objective'])


def test_d11_rescaled_heterogeneity():
    # rescaling must not hide misfit: quality and heterogeneity use UNRESCALED chi2/n
    from validate_joint_stats import _rfit
    jf = _rfit(2, scale_err=(1.0, 4.0))
    snap = jf.last_snapshot
    d = _diag(jf)
    q = d['quality']
    v = [r['chi2_per_point'] for r in q]
    check('D11 snapshot is rescaled', snap.rescale is not None)
    check('D11 unrescaled rows ~1 and ~16', 0.5 < v[0] < 2.0 and 8.0 < v[1] < 32.0, str(v))
    terms, _ = jfs.per_dataset_objective(snap)
    check('D11 the rescaled terms really are ~1 (the bug it guards)',
          all(0.5 < t.chi2_per_point < 1.6 for t in terms), str([t.chi2_per_point for t in terms]))
    check('D11 heterogeneity flagged on a rescaled fit', d['heterogeneity']['flagged']
          and d['heterogeneity']['ratio'] > 3, str(d['heterogeneity']))
    txt = jfs.format_diagnostics(d)
    check('D11 flag text and (unrescaled) column label',
          'error bars or model adequacy differ between datasets' in txt
          and 'chi2 per point (unrescaled)' in txt, ascii(txt))
    check('D11 rows flagged unrescaled', all(r['unrescaled'] for r in q))
    jf = make_jf(2)
    _fit(jf)
    check('D11 no label without rescaling',
          '(unrescaled)' not in jfs.format_diagnostics(_diag(jf)))


def test_d12_render_failure():
    jf = make_jf(2)
    _fit(jf)
    d = _diag(jf)
    d['quality'] = [{'name': 'x'}]            # malformed: rendering raises KeyError
    txt = jfs.format_diagnostics(d)
    check('D12 a failing renderer degrades to a line, rest intact',
          'diagnostics unavailable (quality, rendering)' in txt and 'AIC' in txt
          and 'objective:' in txt, ascii(txt))


def main():
    app = QApplication.instance() or QApplication([])
    for t in (test_d1_block_and_identity, test_d2_heterogeneity, test_d3_qrange,
              test_d4_shared_warnings, test_d5_degeneracy, test_d5c_exactly_singular, test_d5b_text_and_scaling,
              test_d6_no_covar, test_d7_section_failure, test_d7b_all_sections_fail,
              test_d8_snapshot_fails, test_d9_empty_dataset, test_d10_objective_extras, test_d11_rescaled_heterogeneity,
              test_d12_render_failure):
        t()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
