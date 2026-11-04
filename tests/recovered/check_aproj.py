"""Does our Python grid projected area equal the one XPR's 'exact' uses internally?

XPR never exposes tempvars.A, but it can be extracted exactly. Set rho_top, every
box rho and rho_bottom to ZERO; then

    rho_hard[j] = C*rho_prot[j] + (1 - C*alpha[j])*0 = C*rho_prot[j]

and rho_prot[j] = sum_i Z_i f_ij / (A * dz) uses the SAME numerator in both
vol_calc modes (electron_densities! is shared) -- only A differs. So the pointwise
ratio of the two EDPs is exactly A_xpr / A_exact, with no other approximation.
"""
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); XP = os.path.join(os.path.dirname(REPO), 'XPR-main')
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'Functions'))
from Functions.pdb_cache import pdbread
from Functions import pdb_geometry as pg
from Functions import xpr_backend as xb

pdb = os.path.join(XP, 'test_data', 'protein.pdb')
xc, yc, zc, el, rad, _m = pdbread(pdb)
coord = np.transpose(np.array([xc, yc, zc]))
N, A_xpr, _V = pg.sphere_sums(pdb)

jl = xb.get_jl()
from juliacall import convert as jc
q = np.linspace(0.03, 0.5, 60)


def load(vol_calc):
    def A(a):
        return jc(jl.Array, np.ascontiguousarray(np.asarray(a, float)))
    jl.load_data(A(q), A(np.zeros_like(q)), A(np.ones_like(q)), 2, True, True, True, True,
                 coordinates_py=A(coord), electrons_py=A(el), radii_py=A(rad),
                 vol_calc=vol_calc, rho_top=0.0, rho_bottom=0.0,   # <- everything zero
                 xray_energy=10.0, fres=False,                     # no Fresnel 0/0
                 masses_py=A(_m),
                 # 2026-09-10 sizing API: max_sig -> sig_max, and slab_headroom is
                 # gone (the slab count no longer depends on the parameters).
                 # spacez/min_box_length_sum/d_protein_bounds keep upstream defaults,
                 # which is what this ratio test ran with before.
                 max_box_length=30.0, sig_max=6.0)


def edp(th, ph):
    p = np.array([0.0, th, ph, 13.0, 1.0, 3.4, 1.0, 0.0, 12.0, 10.0, 0.0, 0.0])
    z, r = jl.XPR_edp(p)
    return np.array(z), np.array(r)


print('%-8s %-8s %12s %12s %12s %8s' %
      ('theta', 'phi', 'A_xpr/A_edp', 'A_exact', 'A_grid(0.5)', 'diff'))
for th, ph in [(0.0, 0.0), (32.92, 112.59), (60.0, 200.0), (90.0, 45.0), (150.0, 300.0)]:
    load('exact');  ze, re_ = edp(th, ph)
    load('approx'); za, ra = edp(th, ph)
    m = (re_ > 1e-6) & (ra > 1e-6)
    ratio = ra[m] / re_[m]                      # = A_exact/A_xpr inverted below
    r_med = float(np.median(ratio))
    A_exact = A_xpr * r_med                     # ra/re = A_exact/A_xpr
    A_grid = pg.projected_area(pdb, th, ph, 0.5)
    print('%-8.2f %-8.2f %12.4f %12.1f %12.1f %7.2f%%'
          % (th, ph, 1.0 / r_med, A_exact, A_grid, 100 * (A_grid - A_exact) / A_exact))

print('\ngrid step convergence at theta=32.92, phi=112.59:')
load('exact');  _z, re_ = edp(32.92, 112.59)
load('approx'); _z, ra = edp(32.92, 112.59)
m = (re_ > 1e-6) & (ra > 1e-6)
A_exact = A_xpr * float(np.median(ra[m] / re_[m]))
for fs in (1.0, 0.5, 0.25, 0.125):
    A_grid = pg.projected_area(pdb, 32.92, 112.59, fs)
    print('   fslice=%.3f A : A_grid = %8.1f   vs XPR %8.1f   (%+.2f%%)'
          % (fs, A_grid, A_exact, 100 * (A_grid - A_exact) / A_exact))
