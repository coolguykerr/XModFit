"""End-to-end smoke fit of GID2D through the real Fit_Routines.Fit path."""
import os, sys, time
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./Functions'))
import numpy as np
from PyQt5.QtWidgets import QApplication
app = QApplication([])

import gid2d_core as GC
from Fit_Routines import Fit
from Functions.GISAXS.GID2D import GID2D

PACKED = os.path.join('Data', 'dppc_q2d_gid_indexed.txt')
mp = GC.load_map(PACKED)
x = mp['idx'].astype(float)
fit = Fit(GID2D, x, y=mp['I'], yerr=mp['Ierr'])
f = fit.func
f.mapfile = os.path.abspath(PACKED)
f.filepaths['mapfile'] = f.mapfile
f.mode = 'map'
f.fisher = True

cfg = dict(E=15.126, alpha_i=0.10, qc=0.0217,
           a0=4.95, t=30.5, psi=0.0, eps=0.041,
           L_dom=400.0, L_ch=35.0, sig_t=0.0, sig_z=0.0,
           yscale=50.0, bkg0=40.0, bkg1=0.0, bkg2=0.0, bkg3=0.0,
           qxymin=1.40, qxymax=1.54, qzmin=0.02, qzmax=0.60)
for k, v in cfg.items():
    setattr(f, k, v)
    if k in f.params:
        f.params[k].value = v
fit.sync_param()

t0 = time.perf_counter()
f.__fit__ = False
y0 = fit.evaluate()
t1 = time.perf_counter()
print("one evaluation (%d rows, reports ON):  %.3f s" % (len(y0), t1 - t0))
t0 = time.perf_counter()
y0 = fit.evaluate()
print("one evaluation (reports OFF, in-fit):  %.3f s" % (time.perf_counter() - t0))
s = f.output_params['scaler_parameters']
print("\nstart point: chi2/N reported %.1f, active-only %.1f, %d active px"
      % (s['chi2_per_N_reported'], s['chi2_per_N_active_only'], s['n_active_pixels']))
print("chi2 concentration: top 0.1%% carry %.1f%%, top 10 px carry %.1f%%"
      % (100 * s['chi2_frac_in_top_0.1pct_pixels'],
         100 * s['chi2_frac_in_top_10_pixels']))
print("min sigma / median sigma = %.3e" % s['min_sigma_over_median_sigma'])

for k in ('yscale', 'bkg0', 'bkg2', 'a0', 'L_dom', 'L_ch'):
    f.params[k].vary = 1
print("\nvarying:", [k for k in f.params if f.params[k].vary])
t0 = time.perf_counter()
rep, msg = fit.perform_fit(0.0, float(mp['n'] - 1), fit_scale='Linear',
                           fit_method='leastsq', maxiter=200)
print("fit: %.1f s, %d function calls, message=%s"
      % (time.perf_counter() - t0, fit.Niter, msg))
for k in ('yscale', 'bkg0', 'bkg2', 'a0', 'L_dom', 'L_ch'):
    p = fit.result.params[k]
    print("   %-8s %12.5f  +/- %s" % (k, p.value,
          ('%.5f' % p.stderr) if p.stderr else 'None'))

f.__fit__ = False
fit.evaluate()
s = f.output_params['scaler_parameters']
print("\nafter fit: chi2/N active-only %.2f (was %.1f)"
      % (s['chi2_per_N_active_only'], 2312.0))
for k in sorted(s):
    if k.startswith('varinflation_') or k in ('fisher_condition_number',
                                              'A_per_molecule', 'qxy_11',
                                              'qxy_02', 'qz_11'):
        print("   %-30s %s" % (k, s[k]))
print("\noutput_params curves:", sorted(k for k in f.output_params
                                        if k != 'scaler_parameters'))
