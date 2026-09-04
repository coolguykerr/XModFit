"""
End-to-end validation of parallel MCMC sampling (Path B):
  1. objective identity: PdbLogProb == -0.5*sum(Fit.residual**2) (the real Fit path)
  2. parallel emcee (cores>1) completes without the pickling crash; finite chain
  3. serial emcee (cores=1) still works (regression)
  4. per-step progress hook fires; user-abort stops the run early
  5. wall-clock: sampling speedup
Guarded for Windows spawn.
"""
import os, sys, time, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
from PyQt5.QtWidgets import QApplication
_app = QApplication.instance() or QApplication([])

from importlib import import_module
XLayersPDB = import_module("Functions.XRR.XLayersPDB").XLayersPDB
from Fit_Routines import Fit
from lmfit.customhdf import CustomHDFBackend
from pdb_logprob import PdbLogProb

RRF = r"C:\Users\kerr\Documents\XLayersProtTest\mTIM3 analysis\i2_tim3_1000.rrf"
raw = np.loadtxt(RRF); m = (raw[:, 0] >= 0.026) & (raw[:, 0] <= 0.550)
XD, YD, ED = raw[m, 0], raw[m, 1], raw[m, 2]
VARY = ['qoff', 'theta', 'phi', 'protins', 'cov',
        '__Model_d_001', '__Model_rho_001', '__Model_d_002', '__Model_rho_002']


def make_fit():
    mpar = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                      'd':   [0.0, 11.3182806, 8.75928561, 1.0],
                      'rho': [0.0, 0.23396671, 0.46419943, 0.334],
                      'mu':  [0.0, 0.0, 0.0, 0.0],
                      'sig': [3.4, 3.4, 3.4, 3.4]}}
    func = XLayersPDB(x=XD, E=10.0, fname='./Data/tim3_hmmm_ext_all_bestframe.pdb',
                      mpar=mpar, dz=0.5, fslice=0.5, rrf=True, fix_sig=True,
                      qoff=9.830348e-05, yscale=1.0, bkg=0.0,
                      theta=40.88, phi=74.08, protins=6.094141, cov=0.1582754)
    func.params['__Model_d_001'].value = 11.3182806
    func.params['__Model_d_002'].value = 8.75928561
    func.params['__Model_rho_001'].value = 0.23396671
    func.params['__Model_rho_002'].value = 0.46419943
    fit = Fit(XLayersPDB, XD, y=YD, yerr=ED)
    fit.func = func; fit.func.x = XD
    fit.params = func.__dict__; fit.fit_params = func.params
    for k in func.params:
        func.params[k].set(vary=1 if k in VARY else 0)
    return fit


def emcee_run(fit, cores, steps=40, backend_path=None, progress_counter=None, abort_after=None):
    if backend_path is None:
        backend_path = tempfile.mktemp(suffix='.h5')
    if os.path.exists(backend_path):
        os.remove(backend_path)
    if progress_counter is not None or abort_after is not None:
        state = {'n': 0}
        def slot(*a):
            state['n'] += 1
            if progress_counter is not None:
                progress_counter[0] = state['n']
            if abort_after is not None and state['n'] >= abort_after:
                fit.fit_abort = True
        fit.functionCalled.connect(slot)
    fit.fit_abort = False
    be = CustomHDFBackend(backend_path)
    t0 = time.perf_counter()
    fit.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='emcee', maxiter=1,
                    emcee_walkers=20, emcee_steps=steps, emcee_burn=0, emcee_cores=cores,
                    reuse_sampler=False, emcee_thin=1, backend=be)
    dt = time.perf_counter() - t0
    return fit.result.chain, dt


def main():
    ncpu = os.cpu_count() or 2
    workers = max(2, min(8, ncpu - 2))
    print("cpu:", ncpu, " parallel workers:", workers)

    # --- baseline leastsq to establish fit.result / imin,imax ---
    fit0 = make_fit()
    fit0.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=50)
    print("baseline leastsq redchi:", getattr(fit0.result, 'redchi', None))

    # --- (1) objective identity vs the REAL Fit.residual ---
    bounds = np.array([[-np.inf, np.inf]] * len(VARY))
    lp = PdbLogProb(VARY, bounds, 'Linear', True, model=fit0.func, x=XD, y=YD, err=ED)
    base = np.array([fit0.func.params[k].value for k in VARY])
    rng = np.random.RandomState(0)
    maxdiff = 0.0
    for _ in range(25):
        th = base * (1 + 0.05 * rng.randn(len(base)))
        # reference: drive the real Fit.residual
        for k, v in zip(VARY, th):
            fit0.fit_params[k].value = v
        resid = fit0.residual(fit0.fit_params, 'Linear')
        ref = -0.5 * np.sum(np.asarray(resid) ** 2)
        got = lp(th)
        maxdiff = max(maxdiff, abs(ref - got))
    print("(1) objective identity: max|PdbLogProb - (-0.5*sum(Fit.residual^2))| = %.3e" % maxdiff)

    # --- (2/5) parallel run + (4) progress hook ---
    prog = [0]
    fitp = make_fit(); fitp.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=50)
    chain_par, t_par = emcee_run(fitp, workers, steps=40, progress_counter=prog)
    print("(2) PARALLEL completed: chain shape %s finite=%s  time=%.1fs  progress_ticks=%d"
          % (chain_par.shape, bool(np.all(np.isfinite(chain_par))), t_par, prog[0]))

    # --- (3) serial run (regression) ---
    fits = make_fit(); fits.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=50)
    chain_ser, t_ser = emcee_run(fits, 1, steps=40)
    print("(3) SERIAL   completed: chain shape %s finite=%s  time=%.1fs"
          % (chain_ser.shape, bool(np.all(np.isfinite(chain_ser))), t_ser))
    # sampling-only speedup (subtract ~equal seeding? seeding differs; report totals)
    print("(5) total-run speedup (incl seeding): %.2fx" % (t_ser / t_par))

    # --- (4) abort test ---
    fita = make_fit(); fita.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=50)
    chain_ab, _ = emcee_run(fita, workers, steps=200, abort_after=6)
    print("(4) ABORT after ~6 steps: chain steps=%d (of 200 requested) aborted=%s"
          % (chain_ab.shape[0], getattr(fita.result, 'aborted', None)))


if __name__ == '__main__':
    main()
