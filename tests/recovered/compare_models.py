"""
Compare MCMC behaviour of XLayersPDB vs XLayersPDB_dev_smooth on i2_tim3, through
the identical seeding + parallel-sampling harness. Reports per-step time, mean
acceptance fraction, integrated autocorrelation time (tau) per parameter, and
effective sample size. dz/fslice matched so the difference isolates the
discretization (hard voxel) vs erf-smoothed objective.
"""
import os, sys, time, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
from PyQt5.QtWidgets import QApplication
_app = QApplication.instance() or QApplication([])
from importlib import import_module
from Fit_Routines import Fit
from lmfit.customhdf import CustomHDFBackend

RRF = os.path.join(REPO, 'Data', 'i2_tim3_1000_test.txt')
raw = np.loadtxt(RRF); m = (raw[:, 0] >= 0.026) & (raw[:, 0] <= 0.550)
XD, YD, ED = raw[m, 0], raw[m, 1], raw[m, 2]
VARY = ['qoff', 'theta', 'phi', 'protins', 'cov',
        '__Model_d_001', '__Model_rho_001', '__Model_d_002', '__Model_rho_002']
MPAR = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                  'd': [0.0, 11.3182806, 8.75928561, 1.0],
                  'rho': [0.0, 0.23396671, 0.46419943, 0.334],
                  'mu': [0.0, 0.0, 0.0, 0.0], 'sig': [3.4, 3.4, 3.4, 3.4]}}
WALKERS, STEPS, WORKERS, BURN = 20, 5000, 8, 500


def make_fit(clsname):
    cls = import_module("Functions.XRR.%s" % clsname).__getattribute__(clsname)
    func = cls(x=XD, E=10.0, fname='./Data/1ubq.pdb', mpar=MPAR,
               dz=0.5, fslice=0.5, rrf=True, fix_sig=True, qoff=9.83e-05, yscale=1.0,
               bkg=0.0, theta=40.88, phi=74.08, protins=6.094141, cov=0.1582754)
    for k, v in [('__Model_d_001', 11.3182806), ('__Model_d_002', 8.75928561),
                 ('__Model_rho_001', 0.23396671), ('__Model_rho_002', 0.46419943)]:
        func.params[k].value = v
    fit = Fit(cls, XD, y=YD, yerr=ED)
    fit.func = func; fit.func.x = XD; fit.params = func.__dict__; fit.fit_params = func.params
    for k in func.params:
        func.params[k].set(vary=1 if k in VARY else 0)
    return fit


def run(clsname):
    print("\n=== %s ===" % clsname, flush=True)
    fit = make_fit(clsname)
    t0 = time.perf_counter()
    fit.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='leastsq', maxiter=200)
    print("leastsq redchi=%.4f (%.1fs)" % (fit.result.redchi, time.perf_counter() - t0), flush=True)
    bp = tempfile.mktemp(suffix='.h5'); fit.fit_abort = False
    t0 = time.perf_counter()
    fit.perform_fit(0.026, 0.550, fit_scale='Linear', fit_method='emcee', maxiter=1,
                    emcee_walkers=WALKERS, emcee_steps=STEPS, emcee_burn=BURN,
                    emcee_cores=WORKERS, reuse_sampler=False, emcee_thin=1,
                    backend=CustomHDFBackend(bp))
    dt = time.perf_counter() - t0
    sampler = fit.fitter.sampler
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception as e:
        tau = np.full(len(VARY), np.nan); print("tau err:", e)
    acc = np.mean(sampler.acceptance_fraction)
    chain = fit.result.chain  # (steps-burn, walkers, ndim)
    var_names = list(fit.result.var_names)
    med = np.median(chain.reshape(-1, chain.shape[-1]), axis=0)
    std = np.std(chain.reshape(-1, chain.shape[-1]), axis=0)
    print("emcee %d steps x %d walkers, %d workers: %.1fs (%.1f ms/step)"
          % (STEPS, WALKERS, WORKERS, dt, 1e3 * dt / STEPS), flush=True)
    print("mean acceptance fraction: %.3f" % acc, flush=True)
    tau = np.asarray(tau, dtype=float)
    print("tau: mean=%.1f max=%.1f  ESS(min per-param)=%.0f"
          % (np.nanmean(tau), np.nanmax(tau),
             (STEPS - BURN) * WALKERS / np.nanmax(tau)), flush=True)
    return dict(clsname=clsname, dt=dt, acc=acc, tau=tau, med=med, std=std, var_names=var_names)


def main():
    res = {}
    for cls in ['XLayersPDB', 'XLayersPDB_dev_smooth']:
        try:
            res[cls] = run(cls)
        except Exception as e:
            import traceback; print("FAILED %s:\n%s" % (cls, traceback.format_exc()), flush=True)

    if len(res) == 2:
        a, b = res['XLayersPDB'], res['XLayersPDB_dev_smooth']
        print("\n================ SUMMARY ================", flush=True)
        print("%-16s %12s %12s" % ("metric", "XLayersPDB", "dev_smooth"), flush=True)
        print("%-16s %12.1f %12.1f" % ("ms/step", 1e3*a['dt']/STEPS, 1e3*b['dt']/STEPS), flush=True)
        print("%-16s %12.3f %12.3f" % ("acceptance", a['acc'], b['acc']), flush=True)
        print("%-16s %12.1f %12.1f" % ("tau mean", np.nanmean(a['tau']), np.nanmean(b['tau'])), flush=True)
        print("%-16s %12.1f %12.1f" % ("tau max", np.nanmax(a['tau']), np.nanmax(b['tau'])), flush=True)
        print("\nper-param tau (name: XLayersPDB / dev_smooth):", flush=True)
        for i, nm in enumerate(a['var_names']):
            j = b['var_names'].index(nm) if nm in b['var_names'] else None
            bt = b['tau'][j] if j is not None else np.nan
            print("  %-18s %7.1f / %7.1f" % (nm, a['tau'][i], bt), flush=True)
        print("\nposterior median +/- std (note: dev_smooth redefines protins ~ -fslice):", flush=True)
        for i, nm in enumerate(a['var_names']):
            j = b['var_names'].index(nm) if nm in b['var_names'] else None
            print("  %-18s  %12.5g +/- %-10.4g | %12.5g +/- %-10.4g" %
                  (nm, a['med'][i], a['std'][i],
                   (b['med'][j] if j is not None else np.nan),
                   (b['std'][j] if j is not None else np.nan)), flush=True)


if __name__ == '__main__':
    main()
