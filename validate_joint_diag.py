"""
validate_joint_diag.py -- WS2 reporting harness, sections D (Task 6), I (Task 7), F (Task 8),
L (Task 9, leave-one-out).

Section D: the "Joint-fit diagnostics:" block of JointFit._report() -- quality,
heterogeneity, q-range, shared-parameter warnings, degeneracy, failure safety.
Section F: Fisher shares of the shared parameters, model-reduction tension, the exact
objective and the distance from the fitted point to the exact-likelihood optimum.

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
    row = [l for l in txt.splitlines() if l.strip().startswith('D1 ') and ' n = ' in l]
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
    check('D10 Task 8 keys: fisher filled; tension, optimum_distance None without a covariance',
          isinstance(d.get('fisher'), dict) and d['tension'] is None
          and 'optimum_distance' in d and d['optimum_distance'] is None)
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


class Boom(Line):
    armed = False

    def y(self):
        if Boom.armed:
            raise RuntimeError('boom')
        return super().y()


def test_i8_model_raises():
    from Fit_Routines import Fit
    jf = make_jf(3)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    fits = _indiv_fits(jf)
    obj = jf.objectives[1]
    f = Fit(Boom, np.array(obj['x'], float), y=np.array(obj['y'], float),
            yerr=np.array(obj['yerr'], float))
    for key in ('a', 'b'):
        f.fit_params[key].vary = True
    f.perform_fit(X[0], X[-1], fit_scale='Linear', maxiter=200)
    fits['D1'] = f
    f.func.a = 77.0
    f.func.params['a'].value = 77.0
    x0 = f.func.x
    xc = np.array(x0, float).copy()
    d0 = dict(f.func.__dict__)
    pv = {k: (p.value, p.expr) for k, p in f.func.params.items()}
    Boom.armed = True
    try:
        rows = jfs.individual_terms(jf, jf.last_snapshot, fits)
        txt = jfs.format_joint_vs_individuals(jf, fits)
    finally:
        Boom.armed = False
    check('I8 raising model: row unavailable with a reason',
          rows[1]['unavailable'] and 'boom' in rows[1]['unavailable'], str(rows[1]['unavailable']))
    check('I8 the other rows are unaffected',
          rows[0]['unavailable'] is None and rows[2]['unavailable'] is None)
    check('I8 formatter still renders', 'boom' in txt and 'D0' in txt, ascii(txt))
    check('I8 func.x restored', f.func.x is x0 and np.array_equal(f.func.x, xc))
    check('I8 func scalar __dict__ entries restored',
          all(f.func.__dict__[k] == v for k, v in d0.items()
              if k in ('a', 'b', '__fit__') and not isinstance(v, np.ndarray)))
    check('I8 func.params restored', all(f.func.params[k].value == v[0] and f.func.params[k].expr == v[1]
                                         for k, v in pv.items()))


def test_i9_log_cov_dropped_row():
    import covariance_core as cc
    jf = make_jf(2, cov=private_cov)
    jf.covariance_mode = 'data'
    jf.objectives[1]['y'][5] = -1.0              # dropped by _cov_mask in Log
    jf.perform_fit(X[0], X[-1], fit_scale='Log')
    snap = jf.last_snapshot
    d1 = snap.datasets[1]
    check('I9 one row dropped', int(np.sum(d1.used)) == len(d1.x) - 1)
    from Fit_Routines import Fit
    ypos = np.array(jf.objectives[1]['y'], float)
    ypos[5] = abs(ypos[5])
    fit = Fit(Line, np.array(X), y=ypos, yerr=E * np.ones_like(X))
    for key in ('a', 'b'):
        fit.fit_params[key].vary = True
    fit.perform_fit(X[0], X[-1], fit_scale='Log', maxiter=200)
    rows = jfs.individual_terms(jf, snap, {'D1': fit})
    row = rows[1]
    check('I9 row available and finite', row['unavailable'] is None and np.isfinite(row['delta'])
          and np.isfinite(row['indiv_nll2']), str(row))
    p = fit.result.params
    u = d1.used
    r = np.log(d1.y[u]) - np.log(p['a'].value * d1.x[u] + p['b'].value)
    check('I9 residual length == used rows', len(r) == int(np.sum(u)) == row['n'])
    C = cc.assemble([jf._last_blocks[1]], 'Log')[0]
    hand = float(r @ np.linalg.solve(C, r))
    check('I9 indiv_m equals the hand value on the used rows (1e-10)',
          abs(row['indiv_m'] - hand) <= 1e-10 * hand, '%.12g vs %.12g' % (row['indiv_m'], hand))
    hn = hand + np.linalg.slogdet(C)[1] + len(r) * np.log(2 * np.pi)
    check('I9 indiv_nll2 equals the hand value', abs(row['indiv_nll2'] - hn) <= 1e-10 * abs(hn))


def test_i10_rescaled():
    from validate_joint_stats import _rfit
    jf = _rfit(2, scale_err=(1.0, 2.0))
    snap = jf.last_snapshot
    check('I10 fit is rescaled', snap.rescale is not None)
    fits = _indiv_fits(jf)
    txt = jfs.format_joint_vs_individuals(jf, fits)
    check('I10 header says (unrescaled)',
          txt.split('\n')[0] == 'joint vs individual fits (marginal terms) (unrescaled)',
          ascii(txt.split('\n')[0]))
    rows = jfs.individual_terms(jf, snap, fits)
    m = jfs.marginal_terms(snap, scaled=False)
    check('I10 joint_m == marginal_terms(scaled=False)',
          all(abs(r['joint_m'] - mi) <= 1e-10 * mi for r, mi in zip(rows, m)),
          str([(r['joint_m'], mi) for r, mi in zip(rows, m)]))
    check('I10 differs from the rescaled marginal term',
          abs(rows[1]['joint_m'] - jfs.marginal_terms(snap)[1]) > 1e-6)
    d = jfs.diagnostics(jf, snap, individual_fits=fits)
    check('I10 diagnostics text says (unrescaled)', '(unrescaled)' in jfs.format_diagnostics(d))


def test_i11_column_mask():
    jf = make_jf(2)
    m = np.ones(len(X), bool)
    m[[3, 4, 20]] = False
    jf.objectives[0]['mask'] = m
    jf.perform_fit(X[2], X[35], fit_scale='Linear')
    snap = jf.last_snapshot
    d0 = snap.datasets[0]
    n_act = (35 - 2 + 1) - 3
    check('I11 imask removed rows', len(d0.x) == n_act and np.all(d0.used), '%d' % len(d0.x))
    fits = _indiv_fits(jf)
    rows = jfs.individual_terms(jf, snap, fits)
    p = fits['D0'].result.params
    r = d0.y - (p['a'].value * d0.x + p['b'].value)
    hand = float(np.sum((r / d0.yerr) ** 2))
    check('I11 indiv_m == hand chi2 on the active rows (1e-10)',
          rows[0]['n'] == n_act and abs(rows[0]['indiv_m'] - hand) <= 1e-10 * hand,
          '%.12g vs %.12g' % (rows[0]['indiv_m'], hand))


# ----------------------------------------------------------------------- F --
def _params_from(jf, snap, v):
    p = jf.result.params.copy()
    for k, val in zip(snap.var_names, v):
        p[k].value = float(val)
    return p


def _fisher_text(jf, **kw):
    return jfs.format_diagnostics(jfs.diagnostics(jf, jf.last_snapshot, **kw))


def test_f1_fisher_analytic():
    jf = make_jf(3, cov=private_cov)
    jf.covariance_mode = 'data'
    _fit(jf)
    snap = jf.last_snapshot
    fs = jfs.fisher_shares(jf, snap)
    check('F1 shared params are [a]', fs['params'] == ['a'], str(fs['params']))
    L = jfs.rebuild_L(snap)
    ok, det = True, []
    for k, d in enumerate(snap.datasets):
        Cii = jfs._block_of(L, *d.rows)
        x = d.x[d.used]
        Xd = np.stack([x, np.ones_like(x)], axis=1)
        Ia = Xd.T @ np.linalg.solve(Cii, Xd)
        got = fs['information'][d.name]['a']
        rel = abs(got - Ia[0, 0]) / Ia[0, 0]
        # the full per-dataset matrix over (a, this dataset's b) as well
        M = fs['matrix'][d.name]
        names = list(fs['var_names'])
        ii = [names.index('a'), names.index('d%d_b' % k)]
        relM = np.max(np.abs(M[np.ix_(ii, ii)] - Ia)) / np.max(np.abs(Ia))
        ok &= rel <= 1e-5 and relM <= 1e-5
        det.append('%s %.2e %.2e' % (d.name, rel, relM))
    check('F1 I_i[a,a] == x^T C_ii^-1 x (1e-5 rel), full block too', ok, '; '.join(det))


def test_f2_shares_sum_and_coupling():
    for cov in (None, private_cov):
        jf = make_jf(3, cov=cov, scale_err=[1.0, 2.0, 0.5])
        jf.covariance_mode = 'data'
        _fit(jf)
        fs = jfs.fisher_shares(jf, jf.last_snapshot)
        tot = sum(fs['shares'][n]['a'] for n in fs['shares'])
        tag = 'F2 %s' % ('private' if cov else 'nocov')
        check(tag + ' shares of a sum to 1 (1e-12)', abs(tot - 1.0) <= 1e-12, '%.3e' % (tot - 1))
        check(tag + ' not coupled', fs['coupled'] is False)
        txt = _fisher_text(jf)
        check(tag + ' no coupling sentence', 'shared errors are counted once' not in txt)
    jf = make_jf(3, cov=shared_cov)
    jf.covariance_mode = 'data'
    _fit(jf)
    fs = jfs.fisher_shares(jf, jf.last_snapshot)
    check('F2 shared cov: coupled', fs['coupled'] is True)
    txt = _fisher_text(jf)
    check('F2 shared cov: sentence present',
          'the joint information is smaller than the sum: shared errors are counted once' in txt,
          ascii(txt))
    check('F2 shares table rendered', 'D0' in txt and 'share' in txt, ascii(txt))


def test_f3_one_dataset():
    jf = make_jf(1)
    _fit(jf)
    fs = jfs.fisher_shares(jf, jf.last_snapshot)
    check('F3 no shared master: params empty', fs['params'] == [], str(fs['params']))
    d = jfs.diagnostics(jf, jf.last_snapshot)
    check('F3 diagnostics fisher filled, nothing unavailable',
          d['fisher'] is not None and d['unavailable'] == [], str(d['unavailable']))
    jfs.format_diagnostics(d)


def _tension_jf(scale=np.exp(0.03), cov=private_cov, mode='data'):
    jf = make_jf(3, cov=cov, seed=1)
    seg = X > 1.0
    o = jf.objectives[1]
    o['y'] = np.where(seg, o['y'] * scale, o['y'])
    if o['cov'] is not None and o['cov'].y is not None:
        o['cov'].y = np.asarray(o['y'], float).copy()
    jf.covariance_mode = mode
    _fit(jf)
    return jf


def test_f4_tension():
    jf = _tension_jf()
    rows = jfs.tension(jf.last_snapshot)
    r1 = [r for r in rows if r['dataset'] == 'D1']
    check('F4 one row per non-anchor component per dataset',
          len(rows) == 3 and len(r1) == 1, str([(r['dataset'], r['label']) for r in rows]))
    p = r1[0]['pull'] if r1 else float('nan')
    check('F4 scan-2 pull within 0.5 of 1.5', abs(p - 1.5) <= 0.5, '%.4g' % p)
    others = [abs(r['pull']) for r in rows if r['dataset'] != 'D1']
    check('F4 the other datasets show no tension (|pull| < 0.6)', max(others) < 0.6, str(others))
    txt = _fisher_text(jf)
    check('F4 heading model-reduction tension', 'model–reduction tension' in txt, ascii(txt))
    check('F4 never stitching quality', 'stitching quality' not in txt)
    check('F4 not presented as a fitted scale factor', 'fitted scale factor' not in txt)
    jf0 = make_jf(2)
    _fit(jf0)
    check('F4 no covariance: tension None', jfs.tension(jf0.last_snapshot) is None)
    d = jfs.diagnostics(jf, jf.last_snapshot, tension=False)
    check('F4 tension=False skips it', d['tension'] is None)


def test_f5_no_side_effects():
    for mode in ('data', 'model'):
        jf = make_jf(2, cov=private_cov)
        jf.covariance_mode = mode
        hits = []
        jf.functionCalled.connect(lambda *a: hits.append(1))
        _fit(jf)
        snap = jf.last_snapshot
        # a sentinel model state (the fit's own _report already ran the diagnostics once)
        sent = jf.result.params.copy()
        sent['a'].value = 123.0
        jf.push_params(sent)
        for o in jf.objectives:
            o['func'].x = np.array([7.0, 8.0])
        n0, nit = len(hits), jf.Niter
        res0 = {k: (p.value, p.stderr, p.vary, p.expr) for k, p in jf.result.params.items()}
        chi0, covar0 = jf.result.chisqr, np.array(jf.result.covar, float)
        mb = [(o['func'].x.copy(), o['func'].a, o['func'].b, o['func'].params['a'].value,
               o['func'].params['b'].value, o['func'].__fit__, o.get('yfit_masked'))
              for o in jf.objectives]
        L0 = np.array(jf._L, float)
        jfs.fisher_shares(jf, snap)
        jfs.tension(snap)
        p = jf.result.params.copy()
        p['a'].value += 0.1
        jfs.exact_objective(jf, snap, p)
        jfs.optimum_distance(jf, snap)
        jfs.diagnostics(jf, snap)
        tag = 'F5 %s' % mode
        check(tag + ' emissions and Niter unchanged', len(hits) == n0 and jf.Niter == nit)
        check(tag + ' result unchanged',
              res0 == {k: (q.value, q.stderr, q.vary, q.expr) for k, q in jf.result.params.items()}
              and chi0 == jf.result.chisqr and np.array_equal(covar0, jf.result.covar))
        check(tag + ' models restored', all(
            np.array_equal(b[0], o['func'].x) and b[1] == o['func'].a and b[2] == o['func'].b
            and b[3] == o['func'].params['a'].value and b[4] == o['func'].params['b'].value
            and b[5] == o['func'].__fit__ and b[6] == o.get('yfit_masked')
            for b, o in zip(mb, jf.objectives)))
        check(tag + ' fit L untouched', np.array_equal(L0, jf._L))


def _f6_jf():
    """Model mode, Linear, a private multiplicative term with Sigma = 0.05^2 per point
    (weights = identity, K = N) and data whose relative scatter is twice that: C(theta)
    moves with the model at every point and the score through C is O(N), so the reweighting
    fixed point sits visibly (~0.45 stderr in a) off the exact optimum. A rank-1 stitching
    term (one component per segment) moves it by only ~0.01 stderr: its score through C is
    O(1) against an O(N) curvature, too small for a non-vacuous test."""
    import covariance_core as cc
    from Joint_Fit_Routines import JointFit
    rng = np.random.default_rng(0)
    s = 0.05
    jf = JointFit()
    for i in range(2):
        f = A_TRUE * X + B_TRUE
        y = f * (1 + 2.0 * s * rng.standard_normal(len(X))) + E * rng.standard_normal(len(X))
        c = cc.DatasetCovariance(X, E * np.ones_like(X),
                                 [cc.CovTerm('pp', 'multiplicative', s ** 2 * np.eye(len(X)),
                                             weights=np.eye(len(X)), scope='dataset')], y)
        jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=y, yerr=E * np.ones_like(X), cov=c)
    jf.build_master_params(groups=[jf.share_all('a')])
    for k in jf.master_params:
        jf.master_params[k].vary = True
    jf.covariance_mode = 'model'
    _fit(jf)
    return jf


def test_f6_exact_objective_and_optimum():
    from scipy.optimize import minimize
    for cov, mode, scale, resc in ((private_cov, 'data', 'Linear', None),
                                   (private_cov, 'model', 'Linear', None),
                                   (private_cov, 'model', 'Linear', 'per_dataset'),
                                   (private_cov, 'data', 'Log', None),
                                   (shared_cov, 'model', 'Linear', None),
                                   (None, None, 'Linear', None), (None, None, 'Log', None),
                                   (None, None, 'Linear', 'per_dataset')):
        jf = make_jf(3, cov=cov, scale_err=[1.0, 1.5, 0.7])
        if mode:
            jf.covariance_mode = mode
        jf.rescale_errors = resc
        _fit(jf, scale=scale)
        snap = jf.last_snapshot
        _, ex = jfs.per_dataset_objective(snap)
        want = ex['nll2'] + ex['prior']
        got = jfs.exact_objective(jf, snap, jf.result.params)
        check('F6 exact_objective == nll2 + prior (1e-10) [%s %s %s %s]'
              % ('cov' if cov else 'nocov', mode, scale, resc),
              abs(got - want) <= 1e-10 * max(1.0, abs(want)), '%.15g vs %.15g' % (got, want))
    # the distance to the exact optimum, against Nelder-Mead on the exact objective
    jf = _f6_jf()
    snap = jf.last_snapshot
    od = jfs.optimum_distance(jf, snap)
    th = np.array([snap.params[k] for k in snap.var_names])
    sol = minimize(lambda v: jfs.exact_objective(jf, snap, _params_from(jf, snap, v)), th,
                   method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 40000, 'maxfev': 40000})
    check('F6 the Nelder-Mead reference converged', bool(sol.success), str(sol.message))
    true_d = sol.x - th
    ok, det, n_big = True, [], 0
    for k, name in enumerate(snap.var_names):
        se = snap.stderr[name]
        dd, est = true_d[k], od['delta'][name]
        det.append('%s true %.3e est %.3e (%.3g se)' % (name, dd, est, dd / se))
        if abs(dd) > 0.05 * se:
            n_big += 1
            ok &= np.sign(dd) == np.sign(est) and abs(est - dd) <= 0.2 * abs(dd)
    check('F6 some parameter moves > 0.05 stderr (the test is not vacuous)', n_big >= 1, '; '.join(det))
    check('F6 delta agrees with Nelder-Mead in sign and within 20 %', ok, '; '.join(det))
    check('F6 in_stderr = delta / stderr', all(
        abs(od['in_stderr'][n] - od['delta'][n] / snap.stderr[n]) <= 1e-12 * abs(od['in_stderr'][n]) + 1e-15
        for n in snap.var_names))
    wn, wv = od['worst']
    check('F6 worst is the largest |delta/stderr|',
          abs(wv) == max(abs(v) for v in od['in_stderr'].values()) and od['in_stderr'][wn] == wv)
    _, ex = jfs.per_dataset_objective(snap)
    check('F6 fit_gap carried', od['fit_gap'] == ex['fit_gap'])
    txt = _fisher_text(jf)
    check('F6 line printed', 'the fitted point is the reweighting fixed point; the exact-likelihood '
          'optimum is estimated' in txt and 'fit gap' in txt, ascii(txt))
    check('F6 WS6 clause iff |worst| > 0.3',
          ('the WS6 exact optimiser should settle it' in txt) == (abs(wv) > 0.3), '%.3g' % wv)


def test_f7_modes():
    import dataclasses
    for cov, mode in ((private_cov, 'data'), (None, 'model')):
        jf = make_jf(2, cov=cov)
        jf.covariance_mode = mode
        _fit(jf)
        tag = 'F7 %s' % ('data mode' if cov else 'no covariance')
        check(tag + ': optimum_distance None', jfs.optimum_distance(jf, jf.last_snapshot) is None)
        d = jfs.diagnostics(jf, jf.last_snapshot)
        txt = jfs.format_diagnostics(d)
        check(tag + ': no optimum line', 'exact-likelihood optimum' not in txt and d['optimum_distance'] is None)
        check(tag + ': no mean-information footnote', 'mean information only' not in txt, ascii(txt))
    jf = make_jf(2, cov=private_cov)
    jf.covariance_mode = 'model'
    _fit(jf)
    snap = jf.last_snapshot
    od = jfs.optimum_distance(jf, dataclasses.replace(snap, covar=None))
    check('F7 model mode, covar None: unavailable',
          od == {'unavailable': 'no parameter covariance (the Hessian failed)'}, str(od))
    fs = jfs.fisher_shares(jf, snap)
    check('F7 model mode: mean_information_only flag', fs.get('mean_information_only') is True)
    d = jfs.diagnostics(jf, snap)
    txt = jfs.format_diagnostics(d)
    check('F7 model mode: footnote present',
          'mean information only; the information the covariance carries about the parameters '
          'is omitted' in txt, ascii(txt))
    check('F7 model mode: optimum line present', 'exact-likelihood optimum' in txt, ascii(txt))
    # an emcee snapshot: posterior medians, not the fixed point -> no line, no evaluations
    esnap = dataclasses.replace(snap, method='emcee', label=jfs.LABEL_MCMC + jfs._FROZEN_CLAUSE)
    hits = []
    jf.functionCalled.connect(lambda *a: hits.append(1))
    calls = []
    orig_rp = jfs.raw_parts
    jfs.raw_parts = lambda *a, **k: (calls.append(1), orig_rp(*a, **k))[1]
    try:
        od_e = jfs.optimum_distance(jf, esnap)
        d_e = jfs.diagnostics(jf, esnap, fisher=False)
    finally:
        jfs.raw_parts = orig_rp
    txt_e = jfs.format_diagnostics(d_e)
    check('F7 emcee snapshot: optimum_distance None, no evaluations',
          od_e is None and d_e['optimum_distance'] is None and calls == [] and hits == [],
          '%d raw_parts calls' % len(calls))
    check('F7 emcee snapshot: no fixed-point line', 'reweighting fixed point' not in txt_e
          and 'exact-likelihood optimum' not in txt_e, ascii(txt_e))
    d = jfs.diagnostics(jf, snap, optimum=False, fisher=False)
    check('F7 optimum=False / fisher=False skip', d['optimum_distance'] is None and d['fisher'] is None)


# ----------------------------------------------------------------------- L --
def _closed_form_a(jf, keep, gls):
    """Closed-form (G)LS estimate of the shared 'a' on the datasets in ``keep`` (a shared,
    b per dataset): the normal equations solved directly. ``gls``: the data-mode covariance
    block of each dataset, else diag(1/E^2)."""
    import covariance_core as cc
    n = len(keep)
    AtWA, AtWy = np.zeros((n + 1, n + 1)), np.zeros(n + 1)
    for j, i in enumerate(keep):
        o = jf.objectives[i]
        x, y = np.asarray(o['x'], float), np.asarray(o['y'], float)
        A = np.zeros((len(x), n + 1)); A[:, 0] = x; A[:, 1 + j] = 1.0
        if gls:
            C, _ = cc.assemble([cc.Block(o['cov'], y, o['yerr'], y)], 'Linear')
            W = np.linalg.inv(C)
        else:
            W = np.diag(1.0 / np.asarray(o['yerr'], float) ** 2)
        AtWA += A.T @ W @ A
        AtWy += A.T @ W @ y
    return np.linalg.solve(AtWA, AtWy)[0]


def _l_closed(name, cov):
    jf = make_jf(4, cov=cov)
    jf.covariance_mode = 'data'
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    res = jfs.leave_one_out(jf)
    want = [_closed_form_a(jf, [i for i in range(4) if i != k], cov is not None)
            for k in range(4)]
    check('%s omitted names' % name, res.omitted == ['D0', 'D1', 'D2', 'D3'], str(res.omitted))
    _close('%s LOO estimates of a = closed form' % name, res.estimates['a'], want, rtol=1e-6)
    check('%s no failures' % name, res.failures == [], str(res.failures))
    sd = float(np.std(want, ddof=1))
    _close('%s spread = SD of the estimates' % name, res.spread['a'], sd, rtol=1e-5)
    full = jf.result.params['a']
    _close('%s influence in stderr units' % name, res.influence['a'],
           [(w - full.value) / full.stderr for w in want], rtol=1e-4, atol=1e-6)
    check('%s units stderr' % name, res.units['a'] == 'stderr', str(res.units))
    return jf, res


def test_l1_l2_closed_form():
    _l_closed('L1', None)
    _l_closed('L2', private_cov)
    # shared covariance terms between the remaining datasets stay shared (exact marginalisation)
    import covariance_core as cc
    jf = make_jf(3, cov=shared_cov)
    jf.covariance_mode = 'data'
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    res = jfs.leave_one_out(jf)
    keep = [0, 2]
    n = 2
    blocks = [cc.Block(jf.objectives[i]['cov'], np.asarray(jf.objectives[i]['y'], float),
                       jf.objectives[i]['yerr'], np.asarray(jf.objectives[i]['y'], float))
              for i in keep]
    C, _ = cc.assemble(blocks, 'Linear')
    Wm = np.linalg.inv(C)
    A = np.zeros((2 * len(X), n + 1)); yy = np.zeros(2 * len(X))
    for j, i in enumerate(keep):
        sl = slice(j * len(X), (j + 1) * len(X))
        A[sl, 0] = X; A[sl, 1 + j] = 1.0
        yy[sl] = jf.objectives[i]['y']
    a_want = np.linalg.solve(A.T @ Wm @ A, A.T @ Wm @ yy)[0]
    _close('L2 shared term stays shared across the remaining datasets',
           res.estimates['a'][1], a_want, rtol=1e-6)


def _l_state(jf):
    r = jf.result
    return dict(p={k: (v.value, v.stderr) for k, v in r.params.items()}, chi=r.chisqr,
                nit=jf.Niter,
                mp={k: (v.value, v.vary) for k, v in jf.master_params.items()},
                objs=[(o['func'].x.copy(), o['func'].a, o['func'].b,
                       o['func'].params['a'].value, o['func'].params['b'].value,
                       o['func'].__fit__, o['imin'], o['imax'], id(o['func']), id(o['cov']))
                      for o in jf.objectives],
                snap=jf.last_snapshot, nobj=len(jf.objectives), res=r, fitter=jf.fitter,
                warn=list(jf.warnings))


def test_l3_unchanged():
    jf = make_jf(3, cov=private_cov)
    jf.covariance_mode = 'data'
    hits = []
    jf.functionCalled.connect(lambda *a: hits.append(1))
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    n0, st = len(hits), _l_state(jf)
    jfs.leave_one_out(jf)
    now = _l_state(jf)
    ok = (st['p'] == now['p'] and st['chi'] == now['chi'] and st['nit'] == now['nit']
          and st['mp'] == now['mp'] and st['nobj'] == now['nobj'] and st['snap'] is now['snap']
          and st['res'] is now['res'] and st['fitter'] is now['fitter'] and st['warn'] == now['warn'])
    oo = all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(st['objs'], now['objs']))
    check('L3 result params, chisqr, Niter, master params, snapshot identity unchanged', ok)
    check('L3 every model x/params/window/identity unchanged', oo)
    check('L3 no functionCalled emissions', len(hits) == n0, '%d -> %d' % (n0, len(hits)))


def test_l4_no_covar():
    jf = make_jf(3)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    for p in jf.result.params.values():
        p.stderr = None
    jf.result.covar = None
    res = jfs.leave_one_out(jf)
    full = jf.result.params['a'].value
    check('L4 units raw', res.units['a'] == 'raw', str(res.units))
    _close('L4 influence is the raw shift', res.influence['a'],
           [e - full for e in res.estimates['a']], rtol=1e-9, atol=1e-12)
    check('L4 note present', any('raw' in n for n in res.notes), ascii(str(res.notes)))
    check('L4 format mentions raw', 'raw' in jfs.format_loo(res))


def test_l5_one_dataset():
    jf = make_jf(1)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    try:
        jfs.leave_one_out(jf)
        check('L5 ValueError', False, 'no exception')
    except ValueError as e:
        check('L5 ValueError mentions at least 2', 'at least 2' in str(e), str(e))


def test_l6_refit_raises():
    from lmfit import Minimizer
    jf = make_jf(4)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    orig, calls = Minimizer.minimize, []

    def boom(self, *a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('forced failure')
        return orig(self, *a, **k)
    Minimizer.minimize = boom
    try:
        res = jfs.leave_one_out(jf)
    finally:
        Minimizer.minimize = orig
    ok_est = [e for e in res.estimates['a'] if e is not None]
    check('L6 one failures row', len(res.failures) == 1 and res.failures[0][0] == 'D1'
          and 'forced failure' in res.failures[0][1], ascii(str(res.failures)))
    check('L6 the other estimates present', len(ok_est) == 3 and res.estimates['a'][1] is None,
          str(res.estimates['a']))
    txt = jfs.format_loo(res)
    check('L6 format lists the failure', 'forced failure' in txt and 'D1' in txt, ascii(txt))


def test_l7_subset_share_and_no_shared():
    # a subset-shared master is estimated whenever it keeps an endpoint
    jf = make_jf(3, share=None)
    jf.build_master_params(groups=[jf.share_within('a', ['D0', 'D1'])])
    for k in jf.master_params:
        jf.master_params[k].vary = True
    jf.perform_fit(X[0], X[-1], fit_scale='Linear')
    res = jfs.leave_one_out(jf)
    g = [k for k in res.estimates if len(jf.param_map[k]) == 2]
    check('L7 one shared master', len(g) == 1 and list(res.estimates) == g, str(list(res.estimates)))
    check('L7 estimates for every omission (one endpoint is enough)',
          all(e is not None for e in res.estimates[g[0]]), str(res.estimates[g[0]]))
    jf2 = make_jf(2, share=None)
    jf2.perform_fit(X[0], X[-1], fit_scale='Linear')
    r2 = jfs.leave_one_out(jf2)
    check('L7 nothing shared -> empty estimates', r2.estimates == {} and r2.failures == [])


def _guarded(t):
    try:
        t()
    except Exception as e:                       # noqa: BLE001 -- count, keep going
        check('%s raised' % t.__name__, False, '%s: %s' % (type(e).__name__, e))


def main():
    app = QApplication.instance() or QApplication([])
    for t in (test_d1_block_and_identity, test_d2_heterogeneity, test_d3_qrange,
              test_d4_shared_warnings, test_d5_degeneracy, test_d5c_exactly_singular, test_d5b_text_and_scaling,
              test_d6_no_covar, test_d7_section_failure, test_d7b_all_sections_fail,
              test_d8_snapshot_fails, test_d9_empty_dataset, test_d10_objective_extras, test_d11_rescaled_heterogeneity,
              test_d12_render_failure,
              test_i1_i2_joint_vs_individual, test_i3_unavailable, test_i4_side_effects,
              test_i5_private_cov, test_i6_format, test_i7_model_mode,
              test_i8_model_raises, test_i9_log_cov_dropped_row, test_i10_rescaled,
              test_i11_column_mask):
        t()
    for t in (test_f1_fisher_analytic, test_f2_shares_sum_and_coupling, test_f3_one_dataset,
              test_f4_tension, test_f5_no_side_effects, test_f6_exact_objective_and_optimum,
              test_f7_modes, test_l1_l2_closed_form, test_l3_unchanged, test_l4_no_covar,
              test_l5_one_dataset, test_l6_refit_raises, test_l7_subset_share_and_no_shared):
        _guarded(t)
    return finish()


if __name__ == '__main__':
    sys.exit(main())
