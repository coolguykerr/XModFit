"""Reproduce: joint emcee, aborted partway (like Stop), inspect result for the GUI path."""
import os, sys, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
from tests.harness import check, finish
import numpy as np
from PyQt5.QtWidgets import QApplication
_app = QApplication.instance() or QApplication([])
from importlib import import_module
XLayersPDB = import_module("Functions.XRR.XLayersPDB").XLayersPDB
from Joint_Fit_Routines import JointFit
from lmfit.customhdf import CustomHDFBackend

RRF = os.path.join(REPO, 'Data', 'i2_tim3_1000_test.txt')
raw = np.loadtxt(RRF); m = (raw[:, 0] >= 0.026) & (raw[:, 0] <= 0.550)
XD, YD, ED = raw[m, 0], raw[m, 1], raw[m, 2]
MPAR = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                  'd': [0.0, 11.3182806, 8.75928561, 1.0],
                  'rho': [0.0, 0.23396671, 0.46419943, 0.334],
                  'mu': [0.0, 0.0, 0.0, 0.0], 'sig': [3.4, 3.4, 3.4, 3.4]}}
VARY = ['qoff', 'theta', 'phi', 'protins', 'cov',
        '__Model_d_001', '__Model_rho_001', '__Model_d_002', '__Model_rho_002']


def build_func():
    f = XLayersPDB(x=XD, E=10.0, fname='./Data/1ubq.pdb', mpar=MPAR,
                   dz=0.5, fslice=0.5, rrf=True, fix_sig=True, qoff=9.83e-05, yscale=1.0,
                   bkg=0.0, theta=40.88, phi=74.08, protins=6.094141, cov=0.1582754)
    for k, v in [('__Model_d_001', 11.3182806), ('__Model_d_002', 8.75928561),
                 ('__Model_rho_001', 0.23396671), ('__Model_rho_002', 0.46419943)]:
        f.params[k].value = v
    for k in f.params:
        f.params[k].set(vary=1 if k in VARY else 0)
    return f


def make_joint():
    jf = JointFit()
    jf.add_dataset('dsA', func=build_func(), x=XD, y=YD, yerr=ED)
    jf.add_dataset('dsB', func=build_func(), x=XD, y=YD, yerr=ED)
    jf.build_master_params(shared=VARY)
    jf.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=100)
    return jf


def run_abort(cores, abort_after_emits):
    jf = make_joint()
    state = {'n': 0}
    def slot(*a):
        state['n'] += 1
        if state['n'] >= abort_after_emits:
            jf.fit_abort = True
    jf.functionCalled.connect(slot)
    jf.fit_abort = False
    bp = tempfile.mktemp(suffix='.h5')
    if os.path.exists(bp):
        os.remove(bp)
    be = CustomHDFBackend(bp)
    jf.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='emcee',
                   emcee_walkers=20, emcee_steps=1000, emcee_burn=100, emcee_cores=cores,
                   reuse_sampler=False, emcee_thin=1, backend=be)
    r = jf.result
    print("  aborted=%s  chain.shape=%s  var_names(n)=%d" %
          (getattr(r, 'aborted', None), getattr(r, 'chain', np.empty(0)).shape, len(r.var_names)))
    fc = r.flatchain
    print("  flatchain: %s  keys=%d" % (type(fc).__name__, (len(fc.keys()) if fc is not None else -1)))
    # emulate perform_post_sampling_tasks' backend autocorr + sampler autocorr
    ang = np.array([1 if n == 'theta' else 2 if n == 'phi' else 0 for n in r.var_names])
    try:
        t_be = be.get_autocorr_time(angleindex=ang, tol=0)
        print("  backend.get_autocorr_time OK, mean=%.2f" % np.nanmean(t_be))
    except Exception as e:
        print("  backend.get_autocorr_time FAILED:", type(e).__name__, str(e)[:100])
    try:
        t_s = jf.fitter.sampler.get_autocorr_time(quiet=True)
        print("  sampler.get_autocorr_time OK")
    except Exception as e:
        print("  sampler.get_autocorr_time raised:", type(e).__name__, str(e)[:80])


def run_clean(cores, steps):
    jf = make_joint()
    jf.fit_abort = False
    bp = tempfile.mktemp(suffix='.h5')
    if os.path.exists(bp):
        os.remove(bp)
    jf.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='emcee',
                   emcee_walkers=20, emcee_steps=steps, emcee_burn=10, emcee_cores=cores,
                   reuse_sampler=False, emcee_thin=1, backend=CustomHDFBackend(bp))
    r = jf.result
    print("  clean: aborted=%s chain.shape=%s flatchain_keys=%d" %
          (getattr(r, 'aborted', None), r.chain.shape,
           (len(r.flatchain.keys()) if r.flatchain is not None else -1)))


def safe(label, fn):
    print(label)
    try:
        fn()
        check(label + " completes without raising", True)
    except Exception as e:
        import traceback
        print("  >> RAISED:", type(e).__name__, str(e)[:120])
        check(label + " completes without raising", False, '%s: %s' % (type(e).__name__, str(e)[:120]))


if __name__ == '__main__':
    safe("=== SERIAL joint, CLEAN 50 steps (CustomHDFBackend) ===", lambda: run_clean(1, 50))
    safe("=== SERIAL joint, LATE abort (~step 25) ===", lambda: run_abort(1, 500))
    safe("=== SERIAL joint, EARLY abort (~mid step 1) ===", lambda: run_abort(1, 30))
    safe("=== PARALLEL joint (cores=6), LATE abort (~step 40) ===", lambda: run_abort(6, 40))
    safe("=== PARALLEL joint (cores=6), EARLY abort (step 1) ===", lambda: run_abort(6, 1))
    sys.exit(finish())
