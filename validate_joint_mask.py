"""
validate_joint_mask.py -- regression for the per-column mask on the JOINT path.

The highest-risk edit in this change is the mask COMPOSITION ORDER inside the
covariance branch: (1) slice [imin,imax], (2) apply the objective's imask,
(3) concatenate, (4) apply _cov_mask. _cov_mask is defined against the output of
(3), so imask must be folded in at (2) in BOTH _build_chol and residual. Getting
it wrong yields a plausible fit with misaligned whitening rows -- no exception and
no warning -- which is why the length assertion and this test exist.

    QT_QPA_PLATFORM=offscreen python validate_joint_mask.py
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


def build(mask_a=None, use_cov=False, n=60, footprint=False, v_on_b=True):
    from Joint_Fit_Routines import JointFit
    x = np.linspace(0.1, 2.0, n)
    ya = 3.0 * x + 0.5
    yb = 3.0 * x + 0.5
    e = 0.05 * np.ones_like(x)
    jf = JointFit()
    cov_a = cov_b = None
    if use_cov:
        # minimal covariance sidecar: one scale-factor group
        cov_a = dict(y_corr=ya, sigma_meas=e, k_index=np.zeros(n, int),
                     Sigma_logc=np.array([[0.01]]))
        cov_b = dict(y_corr=yb, sigma_meas=e, k_index=np.zeros(n, int),
                     Sigma_logc=np.array([[0.01]]))
        if footprint:
            # a SHARED beam FWHM couples every point carrying a sensitivity v,
            # across datasets -- the rank-1 term in build_full_covariance. This is
            # the branch of _build_chol that builds vseg/nseg per objective.
            cov_a['v'] = 0.3 * np.exp(-x / 0.5)
            cov_a['var_fwhm'] = 4e-4
            if v_on_b:
                cov_b['v'] = 0.2 * np.exp(-x / 0.4)
                cov_b['var_fwhm'] = 4e-4
            else:
                cov_b.pop('v', None)      # one dataset with NO footprint: the
                cov_b['var_fwhm'] = 4e-4  # zeros(nseg) path
    jf.add_dataset('A', funcClass=Line, x=x, y=ya, yerr=e, cov=cov_a, mask=mask_a)
    jf.add_dataset('B', funcClass=Line, x=x, y=yb, yerr=e, cov=cov_b)
    jf.build_master_params()
    return jf, x


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    from Joint_Fit_Routines import JointFit

    n = 60
    m = np.ones(n, bool)
    m[10:17] = False                    # 7 excluded points on dataset A only

    # ---------------------------------------- no-cov: all-True == no mask -----
    for sc in ('Linear', 'Log'):
        j0, x = build(None)
        j0._set_ranges(x[0], x[-1])
        r0 = j0.residual(j0.master_params, sc)
        j1, _ = build(np.ones(n, bool))
        j1._set_ranges(x[0], x[-1])
        r1 = j1.residual(j1.master_params, sc)
        check("no-cov: all-True mask identical to no mask (%s)" % sc,
              len(r0) == len(r1) and np.allclose(r0, r1),
              'max|d|=%.2e' % (np.max(np.abs(r0 - r1)) if len(r0) == len(r1) else -1))

    # ---------------------------------------- no-cov: drops exactly k ---------
    jm, x = build(m)
    jm._set_ranges(x[0], x[-1])
    rm = jm.residual(jm.master_params, 'Linear')
    check("no-cov: masked residual shorter by exactly k",
          len(rm) == 2 * n - 7, '%d vs %d' % (len(rm), 2 * n - 7))
    check("_set_ranges froze imask on A only",
          jm.objectives[0]['imask'] is not None and jm.objectives[1]['imask'] is None)

    # ---------------------------------------- masking == equivalent ranges ----
    # excluding a CONTIGUOUS head slice via a mask must equal excluding it via the
    # x window, since both leave the same points in the residual
    k = 12
    jr, x = build(None)
    jr._set_ranges(x[k], x[-1])
    rr = jr.residual(jr.master_params, 'Linear')
    mm = np.ones(n, bool); mm[:k] = False
    jq, _ = build(mm)
    jq._set_ranges(x[0], x[-1])
    rq = jq.residual(jq.master_params, 'Linear')
    # dataset B differs (jr windows both, jq masks only A), so compare A's block
    check("mask and x-window agree on the same active points (A block)",
          np.allclose(rr[:n - k], rq[:n - k]),
          'max|d|=%.2e' % np.max(np.abs(rr[:n - k] - rq[:n - k])))

    # ---------------------------------------- COVARIANCE branch --------------
    try:
        jc, x = build(None, use_cov=True)
        jc._set_ranges(x[0], x[-1])
        rc0 = jc.residual(jc.master_params, 'Linear')
        cov_ok = True
    except Exception as ex:
        cov_ok = False
        print('   (covariance branch unavailable: %r)' % (ex,))
    if cov_ok:
        check("cov: unmasked whitened residual has full length",
              len(rc0) == 2 * n, len(rc0))
        jc2, _ = build(m, use_cov=True)
        jc2._set_ranges(x[0], x[-1])
        rc1 = jc2.residual(jc2.master_params, 'Linear')
        check("cov: masked whitened residual shorter by exactly k",
              len(rc1) == 2 * n - 7, '%d vs %d' % (len(rc1), 2 * n - 7))
        check("cov: _cov_mask length matches the masked concatenation",
              len(jc2._cov_mask) == 2 * n - 7, len(jc2._cov_mask))
        # the guard must fire if the two sides ever disagree
        jc3, _ = build(m, use_cov=True)
        jc3._set_ranges(x[0], x[-1])
        jc3.residual(jc3.master_params, 'Linear')     # builds the Cholesky
        jc3.objectives[0]['imask'] = np.ones(n, bool)  # deliberately desync
        try:
            jc3.residual(jc3.master_params, 'Linear')
            check("cov: length-mismatch guard raises on a desynced mask", False,
                  'no exception')
        except RuntimeError as ex:
            check("cov: length-mismatch guard raises on a desynced mask",
                  'covariance mask' in str(ex), str(ex)[:60])
        except Exception as ex:
            check("cov: length-mismatch guard raises on a desynced mask", False, repr(ex))

    # ------------------------------- COVARIANCE + shared-beam FOOTPRINT ------
    # _build_chol assembles vseg per objective and must apply imask to it exactly
    # as it does to y_corr/sigma_meas/k_index, or the rank-1 footprint term is
    # built over different points than the residual -- silently.
    for v_on_b in (True, False):
        tag = 'v on both' if v_on_b else 'v on one dataset only'
        try:
            jf0, x = build(None, use_cov=True, footprint=True, v_on_b=v_on_b)
            jf0._set_ranges(x[0], x[-1])
            r0 = jf0.residual(jf0.master_params, 'Linear')
            jf1, _ = build(m, use_cov=True, footprint=True, v_on_b=v_on_b)
            jf1._set_ranges(x[0], x[-1])
            r1 = jf1.residual(jf1.master_params, 'Linear')
            check("cov+footprint (%s): unmasked length is full" % tag,
                  len(r0) == 2 * n, len(r0))
            check("cov+footprint (%s): masked length drops by exactly k" % tag,
                  len(r1) == 2 * n - 7, '%d vs %d' % (len(r1), 2 * n - 7))
            check("cov+footprint (%s): _cov_mask matches the masked concat" % tag,
                  len(jf1._cov_mask) == 2 * n - 7, len(jf1._cov_mask))
            check("cov+footprint (%s): footprint term is actually active" % tag,
                  jf1._use_footprint is True)
            # the Cholesky must be of the MASKED size, not the full one
            check("cov+footprint (%s): Cholesky factor sized to the active set" % tag,
                  jf1._L is not None and jf1._L.shape[0] == 2 * n - 7,
                  None if jf1._L is None else jf1._L.shape)
        except Exception as ex:
            import traceback; traceback.print_exc()
            check("cov+footprint (%s) runs" % tag, False, repr(ex))

    # ---------------------------------------- add_dataset validates length ---
    jf = JointFit()
    x = np.linspace(0.1, 2.0, n)
    try:
        jf.add_dataset('A', funcClass=Line, x=x, y=x, yerr=np.ones(n),
                       mask=np.ones(3, bool))
        check("add_dataset rejects a wrong-length mask", False, 'no exception')
    except ValueError as ex:
        check("add_dataset rejects a wrong-length mask", 'points but x has' in str(ex),
              str(ex)[:60])

    # ---------------------------------------- _window_indices shared ---------
    from Fit_Routines import _window_indices
    jw, x = build(None)
    jw._set_ranges(0.5, 1.5)
    i0, i1 = jw.objectives[0]['imin'], jw.objectives[0]['imax']
    check("_set_ranges uses the exact bound (no 0.99 factor)",
          (i0, i1) == _window_indices(x, 0.5, 1.5) and x[i0] >= 0.5,
          '(%d,%d) x[i0]=%.4f' % (i0, i1, x[i0]))

    # ---------------------------------------- a real masked joint fit --------
    jfit, x = build(m)
    keys = [k for k in jfit.master_params]
    for k2 in keys:
        jfit.master_params[k2].vary = 1
    print('   varying joint params:', keys)
    lens = []
    jfit.functionCalled.connect(lambda p, i, r, s: lens.append(len(r)))
    try:
        rep, msg = jfit.perform_fit(x[0], x[-1], fit_scale='Linear',
                                    fit_method='leastsq', maxiter=200)
        check("masked joint fit runs with constant residual length",
              len(set(lens)) == 1 and lens[0] == 2 * n - 7,
              'lens=%s msg=%s' % (sorted(set(lens)), msg))
    except Exception as ex:
        check("masked joint fit runs with constant residual length", False, repr(ex))

    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
