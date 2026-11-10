"""
lss_reference.py -- Python re-implementation of the LSS_Reader stitching of spec 8.

A FIXTURE for validate_cov_stitching.py, not a harness (the name does not match
tests/test_*.py, so tests/run_all.py never runs it).  It follows spec 8.1-8.5:

  * scans are joined in x order; the overlap of join m is the already-merged points that
    lie inside the new scan's x range (LSS_Reader's overlap1);
  * a cubic spline through the new scan is evaluated at those points, fac is the weighted
    mean of existing/spline with ratio errors from sigma_meas only, and
    facerr = facerr_abs * max(1, sqrt(chi2_red))  (spec 8.3, inflate-only);
  * the new scan is scaled by fac; overlap points are the 0.5/0.5 average of the existing
    point and the scaled spline; the new scan's own overlap points are discarded;
  * sigma_meas never carries facerr, sigma_total does (spec 8.2);
  * W (N x K) carries the per-point region weights (spec 8.4);
  * Sigma follows the 4.3 recursion in natural-log units with scan 1 the anchor.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def reduce(scans, drop=None):
    """scans: list of dicts with x, y, sigma (counting), in patch order (sorted by x here).
    drop: optional {m: positions} -- positions index the overlap list of join m (the
    merged points inside scan m's x range, in x order); those merged points are removed
    from the data before the factor fit (the patch-dialog drop of spec 8.4/8.5).
    Returns x, y, sigma_meas, sigma_total, weights (N x K), Sigma (K x K), Sigma_abs (the
    same with the un-inflated facerr_abs), fac, facerr, facerr_abs, a (list of vectors)."""
    scans = sorted(scans, key=lambda s: float(np.min(s['x'])))
    K = len(scans)
    s0 = scans[0]
    x = np.asarray(s0['x'], float).copy()
    y = np.asarray(s0['y'], float).copy()
    sm = np.asarray(s0['sigma'], float).copy()
    st = sm.copy()
    W = np.zeros((len(x), K))
    W[:, 0] = 1.0
    fac, facerr, facerr_abs = [1.0], [0.0], [0.0]
    a_list = [np.zeros(K)]
    for m in range(1, K):
        xn = np.asarray(scans[m]['x'], float)
        yn = np.asarray(scans[m]['y'], float)
        sn = np.asarray(scans[m]['sigma'], float)
        ov = np.where((x >= xn.min()) & (x <= xn.max()))[0]
        if drop and m in drop:
            gone = ov[np.asarray(drop[m], int)]
            keep = np.ones(len(x), bool)
            keep[gone] = False
            x, y, sm, st, W = x[keep], y[keep], sm[keep], st[keep], W[keep]
            ov = np.where((x >= xn.min()) & (x <= xn.max()))[0]
        if len(ov) == 0:
            raise ValueError('scan %d does not overlap the merged data' % (m + 1))
        s = CubicSpline(xn, yn)(x[ov])
        sig_new = np.interp(x[ov], xn, sn)
        r = y[ov] / s
        var_r = (sm[ov] / s) ** 2 + (y[ov] * sig_new / s ** 2) ** 2
        w = 1.0 / var_r
        f = float(np.sum(w * r) / np.sum(w))
        fe_abs = float(np.sum(w) ** -0.5)
        chi2_red = float(np.sum(w * (r - f) ** 2) / (len(ov) - 1)) if len(ov) > 1 else 1.0
        fe = fe_abs * max(1.0, np.sqrt(chi2_red))
        fac.append(f); facerr.append(fe); facerr_abs.append(fe_abs)
        a_list.append(W[ov].mean(axis=0))
        # overlap points: average of the existing point and the scaled spline
        y_new = f * s
        sm_new = f * sig_new
        st_new = np.sqrt((f * sig_new) ** 2 + (s * fe) ** 2)
        e_m = np.zeros(K); e_m[m] = 1.0
        y_o = 0.5 * (y[ov] + y_new)
        sm_o = 0.5 * np.sqrt(sm[ov] ** 2 + sm_new ** 2)
        st_o = 0.5 * np.sqrt(st[ov] ** 2 + st_new ** 2)
        W_o = 0.5 * W[ov] + 0.5 * e_m
        y[ov], sm[ov], st[ov], W[ov] = y_o, sm_o, st_o, W_o
        # the new scan's points beyond the merged data are appended, scaled by fac
        tail = xn > x.max()
        x = np.concatenate([x, xn[tail]])
        y = np.concatenate([y, f * yn[tail]])
        sm = np.concatenate([sm, f * sn[tail]])
        st = np.concatenate([st, np.sqrt((f * sn[tail]) ** 2 + (yn[tail] * fe) ** 2)])
        W = np.vstack([W, np.tile(e_m, (int(tail.sum()), 1))])
    fac, facerr, facerr_abs = np.array(fac), np.array(facerr), np.array(facerr_abs)
    return dict(x=x, y=y, sigma_meas=sm, sigma_total=st, weights=W,
                Sigma=_sigma(a_list, facerr / fac), Sigma_abs=_sigma(a_list, facerr_abs / fac),
                fac=fac, facerr=facerr, facerr_abs=facerr_abs, a=a_list)


def _sigma(a_list, delta):
    """The 4.3 recursion: Sigma[m,j] = a_m^T Sigma[:,j] (j < m); Sigma[m,m] = a_m^T Sigma a_m + d_m^2."""
    K = len(a_list)
    S = np.zeros((K, K))
    for m in range(1, K):
        a = a_list[m]
        S[m, :m] = a @ S[:, :m]
        S[:m, m] = S[m, :m]
        S[m, m] = a @ S @ a + delta[m] ** 2
    return S


def to_cov(red, inflated=True, source='lss_reference'):
    """A DatasetCovariance with one 'stitch' multiplicative term (Sigma or Sigma_abs)."""
    import covariance_core as cc
    K = red['weights'].shape[1]
    term = cc.CovTerm('stitch', 'multiplicative',
                      np.array(red['Sigma' if inflated else 'Sigma_abs']),
                      np.array(red['weights']), ['S%d' % (k + 1) for k in range(K)],
                      'dataset', {'fac': [float(v) for v in red['fac']],
                                  'facerr': [float(v) for v in red['facerr']]})
    return cc.DatasetCovariance(np.array(red['x']), np.array(red['sigma_meas']), [term],
                                y=np.array(red['y']), source=source, loader='lss_reference')
