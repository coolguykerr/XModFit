"""
validate_joint_gls.py -- synthetic regression for cross-dataset covariance (GLS)
joint fitting. Replaces the real-data validate_jointfit.py, whose data is not in
the repository.

Every expectation here is built INDEPENDENTLY of the engine's intermediates: the
covariance C comes from joint_covariance.build_full_covariance over arrays this
script concatenates itself, the residual r = d - f from the known linear model,
and the GLS parameter covariance from the exact design matrix of that model.

    G1  whitening_identity_with_footprint   L L^T == C and |L^-1 r|^2 == r^T C^-1 r
    G2  n1_nocov_equals_stock_fit            JointFit(1 dataset, no cov) == stock Fit
    G3  gls_covar_matches_analytic           stderr(a) == sqrt((X^T C^-1 X)^-1 * redchi)
    G4  different_masks_with_cross_blocks    masked whitening == chol(C_active) solve
    G4b same, plus per-dataset fit windows cutting into both masked regions
        (slice -> imask order under covariance)

    QT_QPA_PLATFORM=offscreen python validate_joint_gls.py
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
from scipy.linalg import solve_triangular

from tests.harness import check, finish


# ----------------------------------------------------------------- fixed data --
X = np.linspace(0.1, 2.0, 40)
A_TRUE, B_TRUE, E = 3.0, 0.5, 0.05
K2 = (X > 1.0).astype(int)                       # two scale-factor segments
SIG2 = np.array([[4e-4, 2e-4], [2e-4, 4e-4]])
V_FP = 0.3 * np.exp(-X / 0.5)                    # footprint sensitivity
VAR_FWHM = 4e-4


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


def make_data(nds):
    """nds noisy copies of a*x + b, all seeded from default_rng(0)."""
    rng = np.random.default_rng(0)
    return [A_TRUE * X + B_TRUE + E * rng.standard_normal(len(X)) for _ in range(nds)]


def sidecar(y, k_index, Sigma_logc, footprint=False):
    cov = dict(y_corr=y, sigma_meas=E * np.ones_like(X), k_index=k_index,
               Sigma_logc=Sigma_logc)
    if footprint:
        cov['v'] = V_FP
        cov['var_fwhm'] = VAR_FWHM
    return cov


def variant(name, nds):
    """(k_index per dataset, Sigma_logc) for the 'cross' or 'block' layout."""
    if name == 'cross':
        return [K2] * nds, SIG2
    S = np.zeros((2 * nds, 2 * nds))
    for i in range(nds):
        S[2 * i:2 * i + 2, 2 * i:2 * i + 2] = SIG2
    return [K2 + 2 * i for i in range(nds)], S


def independent_C(ys, ks, Sigma_logc, footprint):
    from joint_covariance import build_full_covariance
    return build_full_covariance(
        np.concatenate(ys), np.concatenate([E * np.ones_like(X)] * len(ys)),
        np.concatenate(ks), Sigma_logc,
        v=(np.concatenate([V_FP] * len(ys)) if footprint else None),
        var_fwhm=(VAR_FWHM if footprint else 0.0))


def rel(a, b):
    return abs(a - b) / abs(b)


# ------------------------------------------------------------------------ G1 --
def test_g1():
    from Joint_Fit_Routines import JointFit
    ys = make_data(2)
    ks, S = variant('cross', 2)
    jf = JointFit()
    for i, (y, k) in enumerate(zip(ys, ks)):
        jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=y, yerr=E * np.ones_like(X),
                       cov=sidecar(y, k, S, footprint=True))
    jf.build_master_params()
    jf._set_ranges(X[0], X[-1])
    w = jf.residual(jf.master_params, 'Linear')     # builds the Cholesky

    C = independent_C(ys, ks, S, footprint=True)
    dL = np.linalg.norm(jf._L @ jf._L.T - C) / np.linalg.norm(C)
    check("G1 whitening_identity_with_footprint: L L^T == C", dL < 1e-12,
          'rel=%.3e' % dL)

    p = jf.master_params
    r = np.concatenate([ys[i] - (p['d%d_a' % i].value * X + p['d%d_b' % i].value)
                        for i in range(2)])
    q = float(r @ np.linalg.solve(C, r))
    s = float(np.sum(np.asarray(w) ** 2))
    d = abs(s - q) / q
    check("G1 whitening_identity_with_footprint: |L^-1 r|^2 == r^T C^-1 r",
          d < 1e-10, 'sum=%.10e quad=%.10e rel=%.3e' % (s, q, d))


# ------------------------------------------------------------------------ G2 --
def test_g2():
    from Joint_Fit_Routines import JointFit
    from Fit_Routines import Fit
    y = make_data(1)[0]
    err = E * np.ones_like(X)
    a0, b0 = 2.5, 0.3
    for sc in ('Linear', 'Log'):
        fit = Fit(lambda: Line(a=a0, b=b0), X, y=y, yerr=err)
        for nm in ('a', 'b'):
            fit.fit_params[nm].vary = True
        fit.perform_fit(X[0], X[-1], fit_scale=sc, fit_method='leastsq', maxiter=2000)
        rs = fit.result

        jf = JointFit()
        f = Line(x=X, a=a0, b=b0)
        for nm in ('a', 'b'):
            f.params[nm].vary = True
        jf.add_dataset('D0', func=f, x=X, y=y, yerr=err)
        jf.build_master_params()
        jf.perform_fit(X[0], X[-1], fit_scale=sc, fit_method='leastsq', maxiter=2000)
        rj = jf.result

        dc = rel(rj.chisqr, rs.chisqr)
        check("G2 n1_nocov_equals_stock_fit (%s): chisqr" % sc, dc < 1e-8,
              'stock=%.10e joint=%.10e rel=%.2e' % (rs.chisqr, rj.chisqr, dc))
        for nm in ('a', 'b'):
            ps, pj = rs.params[nm], rj.params['d0_' + nm]
            dv = rel(pj.value, ps.value)
            check("G2 n1_nocov_equals_stock_fit (%s): %s value" % (sc, nm), dv < 1e-6,
                  'stock=%.10f joint=%.10f rel=%.2e' % (ps.value, pj.value, dv))
            de = rel(pj.stderr, ps.stderr)
            check("G2 n1_nocov_equals_stock_fit (%s): %s stderr" % (sc, nm), de < 1e-5,
                  'stock=%.6e joint=%.6e rel=%.2e' % (ps.stderr, pj.stderr, de))


# ------------------------------------------------------------------------ G3 --
def test_g3():
    from Joint_Fit_Routines import JointFit
    nds = 3
    ys = make_data(nds)
    stderr_a = {}
    for vname in ('cross', 'block'):
        ks, S = variant(vname, nds)
        jf = JointFit()
        for i, (y, k) in enumerate(zip(ys, ks)):
            f = Line(x=X, a=2.5, b=0.3)
            for nm in ('a', 'b'):
                f.params[nm].vary = True
            jf.add_dataset('D%d' % i, func=f, x=X, cov=sidecar(y, k, S))
        jf.build_master_params(groups=[jf.share_all('a')])
        jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq',
                       maxiter=2000)
        res = jf.result

        # analytic GLS for y = a*x + b_i, design columns [x | 1_{dataset i}]
        C = independent_C(ys, ks, S, footprint=False)
        n = len(X)
        Xd = np.zeros((nds * n, 1 + nds))
        Xd[:, 0] = np.tile(X, nds)
        for i in range(nds):
            Xd[i * n:(i + 1) * n, 1 + i] = 1.0
        d = np.concatenate(ys)
        CiX = np.linalg.solve(C, Xd)
        F = Xd.T @ CiX                                   # X^T C^-1 X
        Finv = np.linalg.inv(F)
        beta = Finv @ (CiX.T @ d)                        # GLS estimate
        rr = d - Xd @ beta
        chi2 = float(rr @ np.linalg.solve(C, rr))

        names = ['a'] + ['d%d_b' % i for i in range(nds)]
        db = max(rel(res.params[nm].value, beta[j]) for j, nm in enumerate(names))
        check("G3 gls_covar_matches_analytic (%s): params == GLS estimate" % vname,
              db < 1e-6, 'max rel=%.2e' % db)
        dc = rel(res.chisqr, chi2)
        check("G3 gls_covar_matches_analytic (%s): chisqr == GLS chi2" % vname,
              dc < 1e-8, 'engine=%.8e analytic=%.8e rel=%.2e' % (res.chisqr, chi2, dc))

        # redchi built independently of the engine: chi2 / (N - n_free)
        redchi_an = chi2 / (nds * n - (1 + nds))
        dr = rel(res.redchi, redchi_an)
        check("G3 gls_covar_matches_analytic (%s): redchi == chi2/(N-n_free)" % vname,
              dr < 1e-8, 'engine=%.8e analytic=%.8e rel=%.2e'
              % (res.redchi, redchi_an, dr))

        # lmfit scale_covar=True form (redchi-scaled). Whether reported GLS errors
        # should be scaled or unscaled for an absolute C is a WS2 decision.
        se_an = np.sqrt(Finv[0, 0] * redchi_an)
        se_en = res.params['a'].stderr
        ds = rel(se_en, se_an)
        check("G3 gls_covar_matches_analytic (%s): stderr(a)" % vname, ds < 1e-4,
              'engine=%.8e analytic=%.8e rel=%.2e redchi=%.4f'
              % (se_en, se_an, ds, redchi_an))
        stderr_a[vname] = se_en

    dv = rel(stderr_a['cross'], stderr_a['block'])
    check("G3 gls_covar_matches_analytic: stderr(a) cross vs block differ > 5%",
          dv > 0.05, 'cross=%.6e block=%.6e rel=%.3f'
          % (stderr_a['cross'], stderr_a['block'], dv))


# ------------------------------------------------------------------------ G4 --
def test_g4():
    from Joint_Fit_Routines import JointFit
    ys = make_data(2)
    ks, S = variant('cross', 2)
    n = len(X)
    mA = np.ones(n, bool); mA[5:12] = False      # points 5-11 out of A
    mB = np.ones(n, bool); mB[20:25] = False     # points 20-24 out of B
    jf = JointFit()
    for i, (y, k, m) in enumerate(zip(ys, ks, (mA, mB))):
        jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=y, yerr=E * np.ones_like(X),
                       cov=sidecar(y, k, S, footprint=True), mask=m)
    jf.build_master_params()
    jf._set_ranges(X[0], X[-1])
    w = np.asarray(jf.residual(jf.master_params, 'Linear'))

    check("G4 different_masks_with_cross_blocks: residual length",
          len(w) == 2 * n - 7 - 5, '%d vs %d' % (len(w), 2 * n - 7 - 5))

    # slice (full window: no-op) -> imask -> concatenate -> _cov_mask (all finite)
    C = independent_C(ys, ks, S, footprint=True)
    act = np.concatenate([mA, mB])
    C_active = C[np.ix_(act, act)]
    p = jf.master_params
    r = np.concatenate([ys[i] - (p['d%d_a' % i].value * X + p['d%d_b' % i].value)
                        for i in range(2)])
    want = solve_triangular(np.linalg.cholesky(C_active), r[act], lower=True)
    if len(w) == len(want):
        dw = np.linalg.norm(w - want) / np.linalg.norm(want)
    else:
        dw = np.inf
    check("G4 different_masks_with_cross_blocks: whitened residual == chol(C_active) solve",
          dw < 1e-10, 'rel=%.3e' % dw)


# ----------------------------------------------------------------------- G4b --
def test_g4b():
    from Joint_Fit_Routines import JointFit
    ys = make_data(2)
    ks, S = variant('cross', 2)
    n = len(X)
    mA = np.ones(n, bool); mA[5:12] = False      # same masks as G4
    mB = np.ones(n, bool); mB[20:25] = False
    jf = JointFit()
    for i, (y, k, m) in enumerate(zip(ys, ks, (mA, mB))):
        jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=y, yerr=E * np.ones_like(X),
                       cov=sidecar(y, k, S, footprint=True), mask=m)
    jf.build_master_params()
    # windows cut into both masked regions: D0 keeps 3..30, D1 keeps 8..n-4
    jf._set_ranges(X[0], X[-1], ranges={'D0': (X[3], X[30]), 'D1': (X[8], X[-4])})
    w = np.asarray(jf.residual(jf.master_params, 'Linear'))

    wins = [(3, 30), (8, n - 4)]
    got = [(o['imin'], o['imax']) for o in jf.objectives]
    check("G4b windowed_masks_with_cross_blocks: engine windows", got == wins, str(got))
    check("G4b windowed_masks_with_cross_blocks: _cov_mask all True",
          bool(jf._cov_mask.all()), 'n_false=%d' % int((~jf._cov_mask).sum()))

    # reference: per dataset (window AND mask), concatenated, then restrict full C
    C = independent_C(ys, ks, S, footprint=True)
    act = []
    for (lo, hi), m in zip(wins, (mA, mB)):
        a = np.zeros(n, bool); a[lo:hi + 1] = True
        act.append(a & m)
    act = np.concatenate(act)
    p = jf.master_params
    r = np.concatenate([ys[i] - (p['d%d_a' % i].value * X + p['d%d_b' % i].value)
                        for i in range(2)])
    want = solve_triangular(np.linalg.cholesky(C[np.ix_(act, act)]), r[act], lower=True)
    check("G4b windowed_masks_with_cross_blocks: residual length",
          len(w) == len(want), '%d vs %d' % (len(w), len(want)))
    if len(w) == len(want):
        dw = np.linalg.norm(w - want) / np.linalg.norm(want)
    else:
        dw = np.inf
    check("G4b windowed_masks_with_cross_blocks: whitened residual == chol(C_active) solve",
          dw < 1e-10, 'rel=%.3e' % dw)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    for t in (test_g1, test_g2, test_g3, test_g4, test_g4b):
        try:
            t()
        except Exception as ex:
            import traceback; traceback.print_exc()
            check("%s runs" % t.__name__, False, repr(ex))
    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
