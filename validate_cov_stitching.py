"""
validate_cov_stitching.py -- statistical acceptance of the stitching covariance (spec 9.4).

Synthetic slab reflectivity is cut into K overlapping scans with random per-scan scale
errors, reduced by tests/lss_reference.py (a re-implementation of the spec 8 LSS_Reader
merge: chained factor fits, region weights, Sigma by the 4.3 recursion) and fitted by
Fit_Routines.Fit with and without the covariance.

    T6a  reduction sanity (weight rows sum to 1, overlap rows 0.5/0.5, previous-scan-only
         layout: diag Sigma = cumulative sum of delta^2, Sigma[k,l] = Sigma[min,min])
    T6b  recursion case (scan 3 reaches back into scan 1): Sigma == Monte-Carlo covariance
         of the realised region errors (un-inflated Sigma within 10 %; the inflate-only
         Sigma that LSS_Reader writes is reported and bounded)
    T6c  parameter pulls with the covariance, model mode, Linear: |mean| < 0.25, SD in
         [0.85, 1.15] for d and sigma over 200 seeds
    T6d  the covariance-free fit with sigma_meas only is over-confident (pull SD > 1.5 for
         d or sigma); the sigma_total variant is printed
    T6e  Log vs Linear model mode agree: mean |log - lin| / stderr < 0.2

    python validate_cov_stitching.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from tests.harness import check, finish

QC = 0.0217
TRUTH = dict(d=30.0, dr=0.3, sigma=3.0, scale=1.0)
START = dict(d=28.0, dr=0.27, sigma=3.4, scale=1.04)
N_SEEDS = 200
H = 0.48 / 161.0
LAYOUT_A = [(0, 45), (39, 45), (78, 45), (117, 45)]          # previous-scan-only overlaps
LAYOUT_B = [(0, 45), (20, 51), (35, 76), (100, 45)]         # scan 3 starts inside scan 1 (and 2)


def fresnel(q):
    q = np.asarray(q, float)
    root = np.sqrt((q * q - QC * QC).astype(complex))
    return np.abs((q - root) / (q + root)) ** 2


def slab(q, d, dr, sigma, scale):
    q = np.asarray(q, float)
    return (scale * fresnel(q) * np.abs(1 + dr * (np.exp(1j * q * d) - 1)) ** 2
            * np.exp(-q * q * sigma * sigma))


class Slab:
    """Harness-local model: Fresnel x slab form factor x Debye-Waller (fixed q_c)."""
    NAMES = ('d', 'dr', 'sigma', 'scale')

    def __init__(self, x=0, d=START['d'], dr=START['dr'], sigma=START['sigma'],
                 scale=START['scale'], mpar={}):
        self.x = x
        self.d, self.dr, self.sigma, self.scale = d, dr, sigma, scale
        self.__mpar__ = mpar
        self.choices = {}
        self.filepaths = {}
        self.__fit__ = False
        self.output_params = {'scaler_parameters': {}}
        self.init_params()

    def init_params(self):
        from lmfit import Parameters
        self.params = Parameters()
        for nm in self.NAMES:
            self.params.add(nm, value=getattr(self, nm), vary=0, min=-np.inf, max=np.inf)

    def y(self):
        return slab(self.x, self.d, self.dr, self.sigma, self.scale)


def simulate(rng, layout):
    """Scans of the true curve: y = R exp(eps_k) (1 + 0.01 n), counting sigma 1 % of
    R exp(eps_k); eps_1 = 0, eps_k ~ N(0, 0.03^2).  Each scan's grid is offset by a
    sub-step so the spline is not evaluated on its own nodes."""
    scans, eps = [], []
    for k, (start, n) in enumerate(layout):
        x = 0.02 + H * (start + 0.13 * k + 0.1 + np.arange(n))
        e = 0.0 if k == 0 else 0.03 * rng.standard_normal()
        r = slab(x, **TRUTH) * np.exp(e)
        scans.append(dict(x=x, y=r * (1 + 0.01 * rng.standard_normal(n)), sigma=0.01 * r))
        eps.append(e)
    return scans, np.array(eps)


def fit_once(red, scale, cov=None, yerr=None):
    from Fit_Routines import Fit
    x, y = red['x'], red['y']
    f = Fit(Slab, x, y=y, yerr=red['sigma_meas'] if yerr is None else yerr)
    for nm in Slab.NAMES:
        f.fit_params[nm].vary = True
    if cov is not None:
        f.attach_covariance(cov)
    f.perform_fit(x[0], x[-1], fit_scale=scale, fit_method='leastsq', maxiter=2000,
                  covariance_mode='model')
    p = f.result.params
    return {nm: (abs(p[nm].value) if nm == 'sigma' else p[nm].value, p[nm].stderr)
            for nm in Slab.NAMES}


def pulls(res):
    """res: list of fit dicts (or None) -> {name: array of pulls} over the good seeds."""
    out = {}
    for nm in ('d', 'sigma'):
        v = [(r[nm][0] - TRUTH[nm]) / r[nm][1] for r in res
             if r is not None and r[nm][1] not in (None, 0.0) and np.isfinite(r[nm][1])]
        out[nm] = np.array(v)
    return out


def summarise(label, res):
    pl = pulls(res)
    for nm, v in pl.items():
        print('    %-8s %-5s n=%3d  mean pull %+.3f  SD %.3f'
              % (label, nm, len(v), v.mean(), v.std(ddof=1)))
    return pl


def section_a(lr):
    rng = np.random.default_rng(12345)
    red = lr.reduce(simulate(rng, LAYOUT_A)[0])
    W, S, a, K = red['weights'], red['Sigma'], red['a'], 4
    check('T6a every weight row sums to 1', np.allclose(W.sum(axis=1), 1.0, atol=1e-14))
    ov = np.isclose(W.max(axis=1), 0.5)
    rows_ok = all(np.allclose(np.sort(W[i][W[i] > 0]), [0.5, 0.5]) for i in np.where(ov)[0])
    check('T6a overlap rows are 0.5/0.5 (%d of them)' % int(ov.sum()),
          ov.sum() > 0 and rows_ok and np.all(W[~ov].max(axis=1) == 1.0))
    check('T6a previous-scan-only layout: a_m = e_(m-1)',
          all(np.allclose(a[m], np.eye(K)[m - 1]) for m in range(1, K)),
          str([np.round(v, 3) for v in a[1:]]))
    delta2 = (red['facerr'] / red['fac']) ** 2
    check('T6a diag(Sigma) == cumulative sum of delta^2',
          np.allclose(np.diag(S), np.cumsum(delta2), rtol=1e-12, atol=0))
    ok = all(np.isclose(S[k, l], S[min(k, l), min(k, l)], rtol=1e-12, atol=0)
             for k in range(K) for l in range(K))
    check('T6a Sigma[k,l] == Sigma[min,min]', ok)
    first = W[:, 0] == 1.0
    check('T6a sigma_meas <= sigma_total, equal on scan 1',
          np.all(red['sigma_meas'] <= red['sigma_total'] * (1 + 1e-14))
          and np.allclose(red['sigma_meas'][first], red['sigma_total'][first]))
    # dropping overlap points removes their rows and the sums still hold
    red2 = lr.reduce(simulate(np.random.default_rng(12345), LAYOUT_A)[0], drop={1: [0, 1]})
    check('T6a drop: rows removed, sums still 1',
          len(red2['x']) == len(red['x']) - 2 and np.allclose(red2['weights'].sum(axis=1), 1.0))


def section_b(lr):
    rng = np.random.default_rng(777)
    ndraw = 4000
    e_all, s_infl, s_abs = [], [], []
    a3 = None
    for _ in range(ndraw):
        scans, eps = simulate(rng, LAYOUT_B)
        red = lr.reduce(scans)
        e_all.append(np.log(red['fac']) + eps)       # realised region errors (scan 1 = 0)
        s_infl.append(red['Sigma'])
        s_abs.append(red['Sigma_abs'])
        a3 = red['a'][2]
    mc = np.cov(np.array(e_all).T)
    sa, si = np.mean(s_abs, axis=0), np.mean(s_infl, axis=0)
    print('    a_3 = %s' % np.round(a3, 3))
    check('T6b scan 3 reaches back: a_3 mixes rows of scans 1 and 2',
          a3[0] > 0.01 and a3[1] > 0.01, str(np.round(a3, 3)))
    dm, da, di = np.diag(mc)[1:], np.diag(sa)[1:], np.diag(si)[1:]
    print('    diag MC        %s' % np.array2string(dm, precision=4))
    print('    diag Sigma_abs %s  ratio %s' % (np.array2string(da, precision=4),
                                              np.array2string(da / dm, precision=3)))
    print('    diag Sigma_inf %s  ratio %s' % (np.array2string(di, precision=4),
                                              np.array2string(di / dm, precision=3)))
    print('    offdiag Sigma_abs/MC [2,1],[3,1],[3,2] = %s' % np.round(
        [sa[2, 1] / mc[2, 1], sa[3, 1] / mc[3, 1], sa[3, 2] / mc[3, 2]], 3))
    check('T6b un-inflated Sigma diagonal within 10 %% of Monte Carlo (%d draws)' % ndraw,
          np.all(np.abs(da / dm - 1) < 0.10), 'ratios %s' % np.round(da / dm, 3))
    check('T6b inflated Sigma / MC diagonal in [0.95, 1.5]',
          np.all((di / dm >= 0.95) & (di / dm <= 1.5)), 'ratios %s' % np.round(di / dm, 3))


def section_c(lr):
    keys = ('cov_lin', 'cov_log', 'meas', 'total')
    res = {k: [] for k in keys}
    failed = {k: 0 for k in keys}
    t0 = time.time()
    for seed in range(N_SEEDS):
        red = lr.reduce(simulate(np.random.default_rng(seed), LAYOUT_A)[0])
        cov = lr.to_cov(red)
        jobs = {'cov_lin': lambda: fit_once(red, 'Linear', cov=cov),
                'cov_log': lambda: fit_once(red, 'Log', cov=cov),
                'meas': lambda: fit_once(red, 'Linear'),
                'total': lambda: fit_once(red, 'Linear', yerr=red['sigma_total'])}
        for k, job in jobs.items():
            try:
                res[k].append(job())
            except Exception as ex:
                failed[k] += 1
                res[k].append(None)
                print('    seed %d %s failed: %r' % (seed, k, ex))
    print('    fit time %.1f s for %d seeds x 4 variants' % (time.time() - t0, N_SEEDS))
    print('  pulls (theta_hat - truth) / stderr:')
    pl = {k: summarise(k, res[k]) for k in keys}
    check('T6c at most 2 of %d seeds fail (cov, Linear, model)' % N_SEEDS,
          failed['cov_lin'] <= 2, 'failed %d' % failed['cov_lin'])
    for nm in ('d', 'sigma'):
        v = pl['cov_lin'][nm]
        check('T6c %s: |mean pull| < 0.25 and SD in [0.85, 1.15]' % nm,
              len(v) >= N_SEEDS - 2 and abs(v.mean()) < 0.25 and 0.85 <= v.std(ddof=1) <= 1.15,
              'mean %+.3f SD %.3f' % (v.mean(), v.std(ddof=1)))
    sds = {nm: pl['meas'][nm].std(ddof=1) for nm in ('d', 'sigma')}
    check('T6d covariance-free fit (sigma_meas only) is over-confident: pull SD > 1.5 for d or sigma',
          max(sds.values()) > 1.5, 'SDs %s' % {k: round(v, 3) for k, v in sds.items()})
    print('    (print only) sigma_total, no covariance: SD d %.3f sigma %.3f'
          % (pl['total']['d'].std(ddof=1), pl['total']['sigma'].std(ddof=1)))
    diffs = {nm: [] for nm in Slab.NAMES}
    for rl, rg in zip(res['cov_lin'], res['cov_log']):
        if rl is None or rg is None:
            continue
        for nm in Slab.NAMES:
            if rl[nm][1]:
                diffs[nm].append(abs(rg[nm][0] - rl[nm][0]) / rl[nm][1])
    for nm in Slab.NAMES:
        print('    Log vs Linear mean |diff|/stderr  %-6s %.3f' % (nm, np.mean(diffs[nm])))
    for nm in ('d', 'sigma'):
        check('T6e Log vs Linear (model) %s: mean |diff| / stderr < 0.2' % nm,
              np.mean(diffs[nm]) < 0.2, '%.3f' % np.mean(diffs[nm]))


def main():
    t0 = time.time()
    try:
        from tests import lss_reference as lr
        import covariance_core  # noqa: F401
    except ModuleNotFoundError as e:
        check('import covariance_core / lss_reference', False, str(e))
        return finish()
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    for sec in (section_a, section_b, section_c):
        try:
            sec(lr)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check('%s runs' % sec.__name__, False, repr(e))
    print('  total runtime %.1f s' % (time.time() - t0))
    return finish()


if __name__ == '__main__':
    sys.exit(main())
