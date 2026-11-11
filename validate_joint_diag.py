"""
validate_joint_diag.py -- WS2 reporting harness, sections D (Task 6), I (Task 7), later F, L.

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


# ----------------------------------------------------------------- section I --
def _indiv_fits(jf, scale='Linear', skip=(), scales=None):
    """Real Fit_Routines.Fit objects on Line, one per joint dataset, fitted with the same
    scale and no covariance (a and b free)."""
    from Fit_Routines import Fit
    out = {}
    for k, obj in enumerate(jf.objectives):
        if obj['name'] in skip:
            continue
        sc = scale if scales is None else scales.get(obj['name'], scale)
        fit = Fit(Line, np.array(obj['x'], float), y=np.array(obj['y'], float),
                  yerr=np.array(obj['yerr'], float))
        for key in ('a', 'b'):
            fit.fit_params[key].vary = True
        fit.perform_fit(X[0], X[-1], fit_scale=sc, maxiter=200)
        out[obj['name']] = fit
    return out


def test_i1_i2_joint_vs_individual():
    jf = make_jf(3)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    fits = _indiv_fits(jf)
    check('I1 userargs carry the scale', fits['D0'].fitter.userargs[0] == 'Linear')
    rows = jfs.individual_terms(jf, jf.last_snapshot, fits)
    check('I1 one row per dataset, available',
          len(rows) == 3 and all(r['unavailable'] is None for r in rows), str(rows))
    check('I1 delta >= -1e-8 for all', all(r['delta'] >= -1e-8 for r in rows),
          str([r['delta'] for r in rows]))
    check('I1 none flagged', not any(r['flagged'] for r in rows))
    check('I1 delta == joint_nll2 - indiv_nll2', all(
        abs(r['delta'] - (r['joint_nll2'] - r['indiv_nll2'])) < 1e-9 for r in rows))
    d = jfs.diagnostics(jf, jf.last_snapshot, individual_fits=fits)
    q = d['quality']
    check('I1 quality rows carry the individual columns',
          all({'indiv_chi2_per_point', 'delta', 'flagged'} <= set(r) for r in q), str(q[0]))
    check('I1 indiv chi2 per point = indiv_m / n',
          all(abs(r['indiv_chi2_per_point'] - rr['indiv_m'] / r['n']) < 1e-12
              for r, rr in zip(q, rows)))
    check('I1 plain diagnostics keep the old row keys',
          'delta' not in jfs.diagnostics(jf, jf.last_snapshot)['quality'][0])
    check('I1 text shows the columns', 'indiv chi2/n' in jfs.format_diagnostics(d),
          ascii(jfs.format_diagnostics(d)))
    # I2: D2 generated with a = 4.0
    jf = make_jf(3)
    rng = np.random.default_rng(5)
    jf.objectives[2]['y'] = 4.0 * X + B_TRUE + E * rng.standard_normal(len(X))
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    fits = _indiv_fits(jf)
    rows = jfs.individual_terms(jf, jf.last_snapshot, fits)
    fl = [r['flagged'] for r in rows]
    check('I2 only D2 flagged', fl == [False, False, True],
          str([(r['name'], round(r['delta'], 2), r['flagged']) for r in rows]))
    d = jfs.diagnostics(jf, jf.last_snapshot, individual_fits=fits)
    check('I2 flag reaches the quality rows',
          [r['flagged'] for r in d['quality']] == [False, False, True])


def test_i3_unavailable():
    jf = make_jf(3)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    fits = _indiv_fits(jf, skip=('D1',), scales={'D0': 'Log'})
    rows = jfs.individual_terms(jf, jf.last_snapshot, fits)
    r0, r1, r2 = rows
    check('I3 Log individual vs Linear joint unavailable',
          r0['unavailable'] and 'fit scale' in r0['unavailable'], str(r0['unavailable']))
    check('I3 missing entry', r1['unavailable'] and 'no individual fit' in r1['unavailable'],
          str(r1['unavailable']))
    check('I3 the third row is available', r2['unavailable'] is None and np.isfinite(r2['delta']))
    check('I3 unavailable rows carry no numbers',
          all(r[k] is None for r in (r0, r1) for k in ('indiv_m', 'indiv_nll2', 'delta'))
          and not r0['flagged'] and not r1['flagged'])
    from Fit_Routines import Fit
    f = Fit(Line, np.array(X), y=np.ones_like(X), yerr=E * np.ones_like(X))
    rows = jfs.individual_terms(jf, jf.last_snapshot, {'D0': f})
    check('I3 no result', rows[0]['unavailable'] and 'result' in rows[0]['unavailable'],
          str(rows[0]['unavailable']))
    d = jfs.diagnostics(jf, jf.last_snapshot, individual_fits=fits)
    txt = jfs.format_diagnostics(d)
    check('I3 diagnostics survive unavailable rows',
          d['quality'][0]['delta'] is None and d['quality'][2]['delta'] is not None
          and 'fit scale' in txt, ascii(txt))


def test_i4_side_effects():
    jf = make_jf(3)
    hits = []
    jf.functionCalled.connect(lambda *a: hits.append(1))
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    fits = _indiv_fits(jf)
    n0, nit = len(hits), jf.Niter
    # perturb the individuals' live attributes so a missing restore would show
    for f in fits.values():
        f.func.a = 99.0
        f.func.params['a'].value = 99.0
    before = {k: (f.func.x, np.array(f.func.x, float).copy(), f.func.a, f.func.b,
                  f.func.params['a'].value, f.func.params['a'].expr, f.func.__fit__,
                  {kk: (p.value, p.stderr) for kk, p in f.result.params.items()},
                  f.Niter, f.fitter) for k, f in fits.items()}
    jb = [(o['func'].x.copy(), o['func'].a, o['func'].params['a'].value, o['func'].__fit__)
          for o in jf.objectives]
    jfs.individual_terms(jf, jf.last_snapshot, fits)
    jfs.format_joint_vs_individuals(jf, fits)
    jfs.diagnostics(jf, jf.last_snapshot, individual_fits=fits)
    ok = True
    for k, f in fits.items():
        b = before[k]
        ok &= (f.func.x is b[0] and np.array_equal(f.func.x, b[1]) and f.func.a == b[2]
               and f.func.b == b[3] and f.func.params['a'].value == b[4]
               and f.func.params['a'].expr == b[5] and f.func.__fit__ == b[6]
               and {kk: (p.value, p.stderr) for kk, p in f.result.params.items()} == b[7]
               and f.Niter == b[8] and f.fitter is b[9])
    check('I4 individual Fit func.x/params/result unchanged', ok)
    check('I4 jf emission count and Niter unchanged', len(hits) == n0 and jf.Niter == nit)
    check('I4 jf models restored', all(
        np.array_equal(b[0], o['func'].x) and b[1] == o['func'].a
        and b[2] == o['func'].params['a'].value and b[3] == o['func'].__fit__
        for b, o in zip(jb, jf.objectives)))


def test_i5_private_cov():
    import covariance_core as cc
    jf = make_jf(3, cov=private_cov)
    jf.covariance_mode = 'data'
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    fits = _indiv_fits(jf)
    rows = jfs.individual_terms(jf, jf.last_snapshot, fits)
    p = fits['D0'].result.params
    d0 = jf.last_snapshot.datasets[0]
    r = d0.y[d0.used] - (p['a'].value * d0.x[d0.used] + p['b'].value)
    C = cc.assemble([jf._last_blocks[0]], 'Linear')[0]
    hand = float(r @ np.linalg.solve(C, r))
    check('I5 indiv_m == r C_00^-1 r (1e-10)', abs(rows[0]['indiv_m'] - hand) <= 1e-10 * hand,
          '%.12g vs %.12g' % (rows[0]['indiv_m'], hand))
    n = len(r)
    nll = hand + np.linalg.slogdet(C)[1] + n * np.log(2 * np.pi)
    check('I5 indiv_nll2 includes the log det and n ln 2pi',
          abs(rows[0]['indiv_nll2'] - nll) <= 1e-10 * abs(nll))
    check('I5 data mode: delta == joint_m - indiv_m',
          abs(rows[0]['delta'] - (rows[0]['joint_m'] - rows[0]['indiv_m'])) < 1e-8)


def test_i6_format():
    jf = make_jf(3)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    fits = _indiv_fits(jf)
    txt = jfs.format_joint_vs_individuals(jf, fits)
    check('I6 label', 'joint vs individual fits (marginal terms)' in txt, ascii(txt))
    check('I6 all dataset names', all(('D%d' % i) in txt for i in range(3)))
    check('I6 small-N sentence present, no z', jfs.NO_Z_NOTE in txt and '(z =' not in txt,
          ascii(txt))
    check('I6 explicit snap argument accepted',
          jfs.format_joint_vs_individuals(jf, fits, snap=jf.last_snapshot) == txt)
    txt2 = jfs.format_joint_vs_individuals(jf, {})
    check('I6 nothing available degrades to a message', 'no individual fit' in txt2, ascii(txt2))
    jf = make_jf(5)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    txt = jfs.format_joint_vs_individuals(jf, _indiv_fits(jf))
    check('I6 five datasets carry z', '(z =' in txt and jfs.NO_Z_NOTE not in txt, ascii(txt))


def test_i7_model_mode():
    import covariance_core as cc
    jf = make_jf(2, cov=private_cov)
    jf.covariance_mode = 'model'
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    snap = jf.last_snapshot
    check('I7 model mode, intensity needed',
          snap.cov_mode == 'model' and cc.needs_intensity(jfs._snap_blocks(snap), 'Linear'))
    fits = _indiv_fits(jf)
    rows = jfs.individual_terms(jf, snap, fits)
    p = fits['D0'].result.params
    d0 = snap.datasets[0]
    check('I7 individual theta differs from the joint',
          abs(p['a'].value - snap.params['a']) > 1e-6, '%g %g' % (p['a'].value, snap.params['a']))
    f = p['a'].value * d0.x[d0.used] + p['b'].value
    r = d0.y[d0.used] - f
    C = cc.assemble([cc.Block(d0.cov, d0.y[d0.used], d0.yerr[d0.used], f)], 'Linear')[0]
    n = len(r)
    hand = float(r @ np.linalg.solve(C, r)) + np.linalg.slogdet(C)[1] + n * np.log(2 * np.pi)
    check('I7 indiv_nll2 == hand value with C from the individual f (1e-10)',
          abs(rows[0]['indiv_nll2'] - hand) <= 1e-10 * abs(hand),
          '%.12g vs %.12g' % (rows[0]['indiv_nll2'], hand))
    Cj = jfs._block_of(jfs.rebuild_L(snap), *d0.rows)
    check('I7 that C_00 differs from the joint block', np.max(np.abs(C - Cj)) > 1e-9,
          '%.3e' % np.max(np.abs(C - Cj)))
    check('I7 delta == joint_nll2 - indiv_nll2',
          all(abs(r_['delta'] - (r_['joint_nll2'] - r_['indiv_nll2'])) < 1e-9 for r_ in rows))
    check('I7 joint_nll2 equals the per-dataset marginal nll2',
          abs(rows[0]['joint_nll2'] - jfs.per_dataset_objective(snap)[0][0].nll2_marg) < 1e-8)
    # the substitution rule agrees with JointFit._substitute_unusable (Log: <= 0 too)
    from Joint_Fit_Routines import JointFit
    yy = np.array([1., 2., 3.])
    ii = np.array([np.nan, 2., -1.])
    blocks = [cc.Block(None, yy, np.ones(3), ii.copy())]
    JointFit._substitute_unusable(blocks, ii, np.ones(3, bool), 'Log')
    check('I7 substitution rule agrees with _build_chol (Log)',
          np.array_equal(jfs._usable_intensity(ii, yy, 'Log'), blocks[0].intensity))
    blocks = [cc.Block(None, yy, np.ones(3), ii.copy())]
    JointFit._substitute_unusable(blocks, ii, np.ones(3, bool), 'Linear')
    check('I7 substitution rule agrees with _build_chol (Linear)',
          np.array_equal(jfs._usable_intensity(ii, yy, 'Linear'), blocks[0].intensity))


def main():
    app = QApplication.instance() or QApplication([])
    for t in (test_d1_block_and_identity, test_d2_heterogeneity, test_d3_qrange,
              test_d4_shared_warnings, test_d5_degeneracy, test_d5c_exactly_singular, test_d5b_text_and_scaling,
              test_d6_no_covar, test_d7_section_failure, test_d7b_all_sections_fail,
              test_d8_snapshot_fails, test_d9_empty_dataset, test_d10_objective_extras, test_d11_rescaled_heterogeneity,
              test_d12_render_failure,
              test_i1_i2_joint_vs_individual, test_i3_unavailable, test_i4_side_effects,
              test_i5_private_cov, test_i6_format, test_i7_model_mode):
        t()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
