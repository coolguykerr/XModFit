import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from lmfit import minimizer as M

rng = np.random.RandomState(0)

# ---- build the SAME ragged grid the seeder uses (thetares=10) ----
thetares = 10
thetameshrange = np.arange(0, 180 + thetares, thetares)
phires = np.zeros(len(thetameshrange))
phimeshrange = [None] * len(thetameshrange)
cells = []
for ti, thetamesh in enumerate(thetameshrange):
    if thetamesh == 0 or thetamesh == 180:
        phires[ti] = 360
    else:
        phires[ti] = int(360 / np.round((np.cos((thetamesh - thetares/2)*np.pi/180) -
                                          np.cos((thetamesh + thetares/2)*np.pi/180)) /
                                         (1 - np.cos((thetares/2)*np.pi/180))))
    phimeshrange[ti] = np.arange(0, 360, phires[ti])
    for pi, phimesh in enumerate(phimeshrange[ti]):
        cells.append((ti, pi, float(thetamesh), float(phimesh)))

cell_ti = np.array([c[0] for c in cells])
cell_pi = np.array([c[1] for c in cells])
cell_theta = np.array([c[2] for c in cells], float)
cell_phi = np.array([c[3] for c in cells], float)
cell_dphi = np.array([phires[c[0]] for c in cells], float)
n_cells = len(cells)
print('n_cells:', n_cells)

# ---- 1. solid angles sum to ~4*pi ----
area = M._cell_solid_angles(cell_theta, thetares, cell_dphi)
tot = area.sum()
print('sum(area) = %.4f  (4*pi = %.4f)' % (tot, 4*np.pi))
assert abs(tot - 4*np.pi) < 0.15, tot
assert np.all(area > 0)

# ---- 2. two-well synthetic cost map ----
def well(theta0, phi0, depth, width):
    dth = cell_theta - theta0
    dph = ((cell_phi - phi0 + 180) % 360) - 180
    return -depth * np.exp(-0.5 * ((dth/width)**2 + (dph/width)**2))
base = 100.0
chi2 = base + well(41, 74, 40, 12) + well(31, 149, 34, 6)  # primary deep+wide, secondary shallow+narrow
chimin = np.min(chi2) / 70.0

# ---- 2a. probs normalize; temperature limits ----
p1 = M._seed_cell_probs(chi2, area, chimin, temperature=1.0, uniform_frac=0.1)
assert abs(p1.sum() - 1.0) < 1e-9
p_hot = M._seed_cell_probs(chi2, area, chimin, temperature=1e6, uniform_frac=0.0)
unif = area / area.sum()
print('T->inf max|p-uniform| = %.3e' % np.max(np.abs(p_hot - unif)))
assert np.max(np.abs(p_hot - unif)) < 1e-3
p_cold = M._seed_cell_probs(chi2, area, chimin, temperature=1e-3, uniform_frac=0.0)
print('T->0 argmax == argmin(chi2)?', np.argmax(p_cold) == np.argmin(chi2))
assert np.argmax(p_cold) == np.argmin(chi2)

# ---- 2b. inverse-CDF reproduces the categorical ----
N = 400000
draws = M._inverse_cdf_draw(p1, N, rng)
counts = np.bincount(draws, minlength=n_cells) / float(N)
tv = 0.5 * np.sum(np.abs(counts - p1))          # total-variation distance
top = np.argsort(p1)[-5:]
print('p1[top5]   :', np.round(p1[top], 4))
print('count[top5]:', np.round(counts[top], 4))
print('total-variation distance: %.4f' % tv)
assert tv < 0.01, tv
# peak cell reproduced closely
pk = int(np.argmax(p1))
assert abs(counts[pk] - p1[pk]) / p1[pk] < 0.05, (counts[pk], p1[pk])

# ---- 3. basin detection finds BOTH wells ----
basins = M._detect_orientation_basins(cell_ti, cell_pi, cell_theta, cell_phi, chi2,
                                      len(thetameshrange), rel_tol=0.5)
btheta = [(round(cell_theta[b]), round(cell_phi[b])) for b in basins]
print('detected basins (theta,phi):', btheta)
def near(bl, t, p, tol_t=12, tol_p=25):
    return any(abs(bt-t) <= tol_t and abs(((bp-p+180)%360)-180) <= tol_p for (bt,bp) in bl)
assert near(btheta, 41, 74), 'primary well missed'
assert near(btheta, 31, 149), 'secondary (narrow) well missed'

# one-well map -> exactly the one well among detected
chi2b = base + well(41, 74, 40, 12)
basins2 = M._detect_orientation_basins(cell_ti, cell_pi, cell_theta, cell_phi, chi2b,
                                       len(thetameshrange), rel_tol=0.5)
bt2 = [(round(cell_theta[b]), round(cell_phi[b])) for b in basins2]
print('one-well basins:', bt2)
assert near(bt2, 41, 74)

print('SEED HELPERS OK')
