####Please do not remove lines below####
import math
import os
import sys

import numpy as np
from lmfit import Parameters

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines/'))
from functools import lru_cache

####Please do not remove lines above####

####Import your modules below if needed####
import Bio.PDB
from scipy.spatial.transform import Rotation as R

try:
    from mendeleev.fetch import fetch_table
except ImportError:
    from mendeleev import get_table

from numba import njit

# ---------------------------------------------------------------------------
# This file is deliberately self-contained: the Parratt recursion, the erf
# profile builder, the sphere/slab overlap, the projected-area quadrature and
# the union-volume quadrature are all inlined below.  Drop it into XModFit's
# Functions directory and it runs with no other project files present.  The
# only external requirements are numpy, scipy, lmfit, numba, biopython and
# mendeleev.
# ---------------------------------------------------------------------------

# ------------------------------- area presets ------------------------------
# h  : quadrature lattice spacing (A)
# ws : kernel widths evaluated on that lattice (A)
# terms : powers of w removed by extrapolation
# Measured over five poses against the converged hard-grid area:
#   name        mean bias   pose spread   cost (serial)
#   reference     -0.002%      0.091%       65 ms
#   default       +0.212%      0.118%       25 ms
#   fast          +0.907%      0.355%       14 ms
# A uniform bias only rescales the fitted coverage; the *spread* across poses
# is what distorts the pose landscape, and even 'fast' is ~27x tighter on that
# than the covariance-ellipse proxy (which spans 1.027-1.184, a 15% spread).
AREA_PRESETS = {
    'reference': dict(h=0.40, ws=(0.30, 0.45, 0.60, 0.90), terms=(2, 3, 4)),
    'default':   dict(h=0.60, ws=(0.40, 0.60, 0.90),        terms=(2, 3)),
    'fast':      dict(h=0.65, ws=(0.40, 0.80),              terms=(2,)),
}
AREA_NPAD = 5.0     # kernel truncation in units of w.  This is a SMOOTHNESS knob,
                    # not a cost knob: 5 -> 6e-9, 4 -> 2e-7, 3.5 -> 2e-6, because
                    # truncation reintroduces a step as cells cross the cutoff.
                    # Do not lower it.
VOL_H = 0.125       # union-volume lattice spacing (A), deliberately decoupled
                    # from fslice so protvol stops drifting with it.


# ============================ numba kernels ================================

@njit(parallel=False, cache=True)
def parratt_numba(q, lam, d, rho, beta):
    ref = np.ones_like(q)
    refc = np.ones_like(q) * complex(1.0, 0.0)
    f1 = 16.0 * np.pi * 2.818e-5
    f2 = -32.0 * np.pi ** 2 / lam ** 2
    Nl = len(d)
    for j in range(len(q)):
        refc[j] = complex(0.0, 0.0)
        for it in range(1, Nl):
            i = Nl - it
            qc1 = f1 * (rho[i - 1] - rho[0])
            qc2 = f1 * (rho[i] - rho[0])
            k1 = np.sqrt(complex(q[j] ** 2 - qc1, f2 * beta[i - 1]))
            k2 = np.sqrt(complex(q[j] ** 2 - qc2, f2 * beta[i]))
            X = (k1 - k2) / (k1 + k2)
            fact1 = complex(np.cos(k2.real * d[i]), np.sin(k2.real * d[i]))
            fact2 = np.exp(-k2.imag * d[i])
            fact = fact1 * fact2
            refc[j] = (X + refc[j] * fact) / (1.0 + X * refc[j] * fact)
        ref[j] = np.abs(refc[j]) ** 2
    return ref, refc


@njit(parallel=False, cache=True)
def sldCalFun_numba(pos, y, sigma, x):
    wholesld = []
    for j in range(len(x)):
        sld = 0
        for i in range(len(pos) - 1):
            sld += math.erf((x[j] - pos[i]) / sigma[i + 1] / math.sqrt(2)) * (y[i + 1] - y[i])
        wholesld.append(max((sld + y[0] + y[-1]) / 2, 0))
    return wholesld


@njit(parallel=False, cache=True)
def sphere_slab_nonuniform(ua, radius, electrons, w, bco, edges, Elec, Vol, VolU):
    """Exact sphere/slab overlap on an arbitrary ascending edge list.

    For a sphere of radius r centred at depth ua, the volume inside [a, b] is
    pi*[ r^2*(hi-lo) - ((hi-ua)^3 - (lo-ua)^3)/3 ] with lo = max(a, ua-r) and
    hi = min(b, ua+r).  Continuous in ua and identically zero once a slab clears
    the sphere, so slabs entering or leaving the support do so smoothly.

    Three accumulators:
      Elec[j]  electrons, apportioned by each atom's OWN sphere fraction, so
               sum_j Elec[j] = sum_i Z_i exactly with no overlap correction
               needed (verified to 1.0000000000).
      Vol[j]   additive occupied volume, overlaps double-counted, so
               sum_j Vol[j] = sum_i (4/3) pi r_i^3 exactly.
      VolU[j]  overlap-corrected (union) volume, sum_i [s_i + b_i*t] over the
               zone, where t is the depth offset from the atom centre.  The
               extra analytic term is q = int t*pi*(r^2-t^2) dt
               = pi*[r^2 t^2/2 - t^4/4], which integrates to zero over a whole
               sphere, so sum_j VolU[j] = protvol exactly regardless of b.
    """
    M = len(edges) - 1
    for i in range(len(radius)):
        r = radius[i]
        vol = (4.0 / 3.0) * np.pi * r ** 3
        lo_a = ua[i] - r
        hi_a = ua[i] + r
        jlo = np.searchsorted(edges, lo_a) - 1
        jhi = np.searchsorted(edges, hi_a)
        if jlo < 0:
            jlo = 0
        if jhi > M - 1:
            jhi = M - 1
        for j in range(jlo, jhi + 1):
            lo = edges[j]
            hi = edges[j + 1]
            if lo < lo_a:
                lo = lo_a
            if hi > hi_a:
                hi = hi_a
            if hi <= lo:
                continue
            dl = lo - ua[i]
            dh = hi - ua[i]
            v = np.pi * (r * r * (hi - lo) - (dh ** 3 - dl ** 3) / 3.0)
            q = np.pi * (r * r * (dh * dh - dl * dl) / 2.0 - (dh ** 4 - dl ** 4) / 4.0)
            Vol[j] += v
            VolU[j] += w[i] * v + bco[i] * q
            Elec[j] += electrons[i] * v / vol
    return Elec, Vol, VolU


@njit(parallel=False, cache=True)
def _area_multi(xr, yr, rad, h, ws, npad, L, P):
    """Accumulate prod_i (1 - k_i) for every smoothing width simultaneously.

    Membership is k_i = Phi((rad_i - dist)/w) with Phi the standard normal CDF,
    and the union indicator is 1 - prod_i (1 - k_i).  The cell loop, the index
    arithmetic and the distance evaluation are shared across widths, which is
    why the widths are fused into one sweep rather than run as separate passes.

    Serial by design: this model is intended to run inside a parallelised MCMC,
    where internal threading would oversubscribe the cores.  The row-block
    structure is kept so prange could be reinstated if ever wanted.
    """
    nw = len(ws)
    N = P.shape[1]
    wmax = ws[0]
    for iw in range(nw):
        if ws[iw] > wmax:
            wmax = ws[iw]
    nblk = 16
    blk = (N + nblk - 1) // nblk
    for b in range(nblk):
        j0 = b * blk
        j1 = min(N, j0 + blk)
        if j0 >= j1:
            continue
        for i in range(len(rad)):
            r = rad[i]
            reach = r + npad * wmax
            jlo = int(math.floor((xr[i] - reach + L) / h - 0.5))
            jhi = int(math.ceil((xr[i] + reach + L) / h - 0.5))
            if jlo < j0:
                jlo = j0
            if jhi > j1 - 1:
                jhi = j1 - 1
            if jlo > jhi:
                continue
            klo = int(math.floor((yr[i] - reach + L) / h - 0.5))
            khi = int(math.ceil((yr[i] + reach + L) / h - 0.5))
            if klo < 0:
                klo = 0
            if khi > N - 1:
                khi = N - 1
            reach2 = reach * reach
            for j in range(jlo, jhi + 1):
                dx = -L + (j + 0.5) * h - xr[i]
                dx2 = dx * dx
                if dx2 > reach2:
                    continue
                for k in range(klo, khi + 1):
                    dy = -L + (k + 0.5) * h - yr[i]
                    d2 = dx2 + dy * dy
                    if d2 > reach2:
                        continue
                    gap = r - math.sqrt(d2)
                    for iw in range(nw):
                        if P[iw, j, k] == 0.0:
                            continue                    # already saturated
                        s = gap / (ws[iw] * math.sqrt(2.0))
                        if s > 6.0:
                            P[iw, j, k] = 0.0           # erf(s) == 1 in float64
                        elif s > -npad / math.sqrt(2.0):
                            P[iw, j, k] *= (1.0 - 0.5 * (1.0 + math.erf(s)))
    return P


@njit(parallel=False, cache=True)
def _shield_mom(xc, yc, zc, rad, h, L, N, nblk, share, mom, cells):
    """Per-atom share of the UNION volume, and its first moment.

    Two passes per block: first the multiplicity m of each occupied cell (how
    many spheres contain it), then each covering sphere is credited h^3/m of
    that cell.  By construction sum_i share_i = h^3 * (#occupied cells) = the
    union volume, so the correction is exactly total-preserving.  The first
    moment records WHERE within each sphere the shielding sits, which is what
    lets the correction resolve depth rather than being a single global factor.
    Both are properties of the template alone, so this runs once.
    """
    blk = (N + nblk - 1) // nblk
    for b in range(nblk):
        j0 = b * blk
        j1 = min(N, j0 + blk)
        if j0 >= j1:
            continue
        m = np.zeros((j1 - j0, N, N), dtype=np.int16)
        for rep in range(2):
            for i in range(len(rad)):
                r = rad[i]
                r2 = r * r
                jlo = int(math.floor((xc[i] - r + L) / h - 0.5))
                jhi = int(math.ceil((xc[i] + r + L) / h - 0.5))
                if jlo < j0:
                    jlo = j0
                if jhi > j1 - 1:
                    jhi = j1 - 1
                if jlo > jhi:
                    continue
                klo = max(0, int(math.floor((yc[i] - r + L) / h - 0.5)))
                khi = min(N - 1, int(math.ceil((yc[i] + r + L) / h - 0.5)))
                llo = max(0, int(math.floor((zc[i] - r + L) / h - 0.5)))
                lhi = min(N - 1, int(math.ceil((zc[i] + r + L) / h - 0.5)))
                a0 = 0.0; mx = 0.0; my = 0.0; mz = 0.0
                for j in range(jlo, jhi + 1):
                    dx = -L + (j + 0.5) * h - xc[i]
                    for k in range(klo, khi + 1):
                        dy = -L + (k + 0.5) * h - yc[i]
                        pq = dx * dx + dy * dy
                        if pq > r2:
                            continue
                        for l in range(llo, lhi + 1):
                            dz = -L + (l + 0.5) * h - zc[i]
                            if pq + dz * dz <= r2:
                                if rep == 0:
                                    m[j - j0, k, l] += 1
                                else:
                                    f = 1.0 / m[j - j0, k, l]
                                    a0 += f; mx += f * dx; my += f * dy; mz += f * dz
                if rep == 1:
                    share[i] += a0 * h ** 3
                    mom[i, 0] += mx * h ** 3
                    mom[i, 1] += my * h ** 3
                    mom[i, 2] += mz * h ** 3
        c = 0
        for j in range(j1 - j0):
            for k in range(N):
                for l in range(N):
                    if m[j, k, l] > 0:
                        c += 1
        cells[b] = c
    return share, mom, cells


# ============================ area and volume ==============================

def _extrap_weights(ws, terms):
    """Row of the pseudo-inverse mapping [A(w_1)..A(w_n)] -> A(0)."""
    ws = np.asarray(ws, dtype=float)
    M = np.vstack([np.ones(len(ws))] + [ws ** t for t in terms]).T
    return np.linalg.pinv(M)[0]


def soft_area(xr, yr, rad, rmax, preset='default'):
    """Projected area of the union of vdW discs, smooth in the atom coordinates.

    Still a pixel quadrature -- no analytic union-of-discs geometry, no
    Green's-theorem boundary integral -- but on a lattice fixed in the lab frame
    and with a smooth cell-membership weight instead of a hard 0/1 test.

    A hard test cannot work at any spacing: the measured area roughness is
    1.2e-3 with develop's pose-dependent lattice, 6.9e-4 with a fixed lattice at
    h=0.5, and still 2.0e-4 at h=0.15, because the step size falls as h^2 while
    the number of steps rises as 1/h.  Smoothing the membership is what removes
    the staircase, giving <=1e-8.

    For an isolated disc the smoothed integral is exactly pi*(r^2 + w^2), so
    there is no O(w) bias.  For a union the bias is a boundary integral,
    empirically A(w) = A + a2 w^2 + a3 w^3 + a4 w^4 (the odd term comes from the
    reflex corners where discs intersect), removed by evaluating several widths
    on the same lattice and solving for A.  Because the extrapolation is a fixed
    linear combination of individually smooth quantities it preserves smoothness.

    ``rmax`` must be a rotation-invariant bound on max_i(|r_i| + rad_i) computed
    from the unrotated, CENTRED coordinates.  That is what makes the lattice
    pose-independent, and it is why pdbread centres the template.
    """
    p = AREA_PRESETS[preset]
    h = p['h']
    ws = np.asarray(p['ws'], dtype=np.float64)
    xr = np.ascontiguousarray(xr, dtype=np.float64)
    yr = np.ascontiguousarray(yr, dtype=np.float64)
    rad = np.ascontiguousarray(rad, dtype=np.float64)
    L = rmax + AREA_NPAD * ws.max() + h
    N = int(np.ceil(2 * L / h))
    L = N * h / 2.0
    P = np.ones((len(ws), N, N))
    _area_multi(xr, yr, rad, h, ws, AREA_NPAD, L, P)
    A = h * h * np.sum(1.0 - P, axis=(1, 2))
    return float(_extrap_weights(ws, p['terms']) @ A)


def atom_shielding(xc, yc, zc, rad, h=VOL_H, nblk=16):
    """Union volume of the vdW spheres, plus the per-atom overlap correction.

    Returns (protvol, s, mom) where s_i is atom i's share of the union volume as
    a fraction of its own sphere volume, and mom_i is the first moment of that
    share about the atom centre.  Computed on unrotated coordinates at fixed
    spacing h, so pose-independent, cached, and decoupled from fslice.
    Measured on TIM-3: protvol 11127.66 A^3, s spanning 0.374-0.873 with mean
    0.526, one-time cost 0.23 s.
    """
    xc = np.ascontiguousarray(xc, dtype=np.float64)
    yc = np.ascontiguousarray(yc, dtype=np.float64)
    zc = np.ascontiguousarray(zc, dtype=np.float64)
    rad = np.ascontiguousarray(rad, dtype=np.float64)
    L = max(np.max(np.abs(a) + rad) for a in (xc, yc, zc)) + h
    N = int(np.ceil(2 * L / h))
    L = N * h / 2.0
    share = np.zeros(len(rad))
    mom = np.zeros((len(rad), 3))
    cells = np.zeros(nblk, dtype=np.int64)
    _shield_mom(xc, yc, zc, rad, h, L, N, nblk, share, mom, cells)
    protvol = h ** 3 * float(cells.sum())
    return protvol, share / ((4.0 / 3.0) * np.pi * rad ** 3), mom


# ======================= template and pose caches ==========================
# Keyed on the PDB path and on scalars, NOT on the model instance.  lru_cache
# on a bound method keys on `self`, so N model instances sharing one template
# occupy N slots; with the previous maxsize=3 a joint fit over more than three
# datasets thrashed the cache and recomputed the 0.13 s shielding pass on every
# single evaluation (measured: 6.8 ms/eval at N=1 rising to 170 ms at N>=5).
# Keying on the path instead means all instances share, and instances sitting at
# a common pose -- which is exactly the joint-fit case, where theta and phi are
# shared -- also share the projected area.
# Cache keys carry a file stamp (absolute path, mtime, size) so that editing a
# PDB in place -- same filename -- invalidates the template automatically.  The
# previous @lru_cache(maxsize=2) on pdbread was keyed on (self, fname), so a new
# model instance was a cache miss and re-read from disk; keying on the path
# alone silently reused the stale template.  One os.stat per call, ~2 us.
_TEMPLATE = {}
_SHIELD = {}


def _stamp(fname):
    st = os.stat(fname)
    return (os.path.abspath(fname), st.st_mtime_ns, st.st_size)


def _read_template(key):
    fname = key[0]
    if key not in _TEMPLATE:
        try:
            ptable = fetch_table('elements')[['symbol', 'atomic_number', 'vdw_radius', 'atomic_weight']]
        except NameError:
            ptable = get_table('elements')[['symbol', 'atomic_number', 'vdw_radius', 'atomic_weight']]
        atominfo = ptable.set_index('symbol').T.to_dict('list')
        struct = Bio.PDB.PDBParser(QUIET=True).get_structure('struct', fname)
        atoms = list(struct.get_atoms())
        n = len(atoms)
        radius = np.zeros(n); electrons = np.zeros(n); mass = np.zeros(n)
        coord = np.zeros((n, 3))
        for ind, atom in enumerate(atoms):
            coord[ind, :] = atom.coord
            sym = 'Ca' if atom.name == 'CAL' else atom.element
            electrons[ind] = atominfo[sym][0]
            radius[ind] = atominfo[sym][1] / 100
            mass[ind] = atominfo[sym][2]
        """Centre the template once, on the unrotated coordinates.  The model is
        translation invariant, but centring makes max_i(|r_i| + rad_i) rotation
        invariant, which is what lets a single lab-frame lattice serve every
        pose."""
        coord = coord - np.mean(coord, axis=0)
        out = (np.ascontiguousarray(coord[:, 0]), np.ascontiguousarray(coord[:, 1]),
               np.ascontiguousarray(coord[:, 2]), electrons, radius, mass)
        for a in out:
            a.setflags(write=False)      # handed out by reference; fail loudly on mutation
        _TEMPLATE[key] = out
    return _TEMPLATE[key]


def _shielding(key, h=VOL_H):
    ck = (key, h)
    if ck not in _SHIELD:
        xc, yc, zc, el, rad, mass = _read_template(key)
        pv, sh, mom = atom_shielding(xc, yc, zc, rad, h)
        sh.setflags(write=False); mom.setflags(write=False)
        _SHIELD[ck] = (pv, sh, mom)
    return _SHIELD[ck]


@lru_cache(maxsize=64)
def _protarea(key, theta, phi, mode, preset, softbeta):
    """Rotation, smooth projected area, anchor and rotated shielding moment.
    Cached on scalars so that every model instance at this pose shares it."""
    xc, yc, zc, el, rad, mass = _read_template(key)
    coord = np.transpose(np.array([xc, yc, zc]))
    rot = R.from_euler('zx', [[phi, theta]], degrees=True)
    rotcoord = rot.apply(coord)
    if mode == 'ellipse':
        """The iso_areacov / smooth proxy: the 95% ellipse of the bivariate
        normal fitted to the projected ATOM CENTRES, 5.991*pi*sqrt(det Sigma).
        vdW radii are not used at all.  Exactly a coverage reparametrisation --
        A enters the model only through cov/A -- so the pose landscape at
        profiled cov is unchanged, but the fitted coverage is inflated by
        A_ellipse/A_grid, which spans 1.024-1.215."""
        C = np.cov(rotcoord[:, :2].T)
        minarea = 5.991 * np.pi * np.sqrt(max(float(np.linalg.det(C)), 0.0))
    else:
        rmax = np.max(np.sqrt(np.sum(coord ** 2, axis=1)) + rad)
        minarea = soft_area(rotcoord[:, 0], rotcoord[:, 1], rad, rmax, preset)
    apex = rotcoord[:, 2] + rad
    ztop = float(np.max(apex))
    ztop_soft = float(ztop + np.log(np.sum(np.exp(softbeta * (apex - ztop)))) / softbeta)
    """Only the depth component of the rotated shielding moment is needed, and
    it is linear in the rotation matrix, so it is exactly as smooth in
    (theta, phi) as the coordinates are."""
    momz = rot.apply(np.array(_shielding(key)[2]))[:, 2]   # scipy needs a writeable buffer
    rz = np.ascontiguousarray(rotcoord[:, 2]); mz = np.ascontiguousarray(momz)
    rz.setflags(write=False); mz.setflags(write=False)
    return rz, float(minarea), ztop, ztop_soft, mz


# ================================= model ===================================

class XLayersPDB_dev_smooth:  # Please put the class name same as the function name
    def __init__(self, x=0.1, E=10.0, fname='./Data/tim3_hmmm_ext_all_bestframe.pdb', mpar={
        'Model': {'Layers': ['top', 'bottom'], 'd': [0.0, 1.0], 'rho': [0.0, 0.333],
                  'mu': [0.0, 0.0], 'sig': [3.0, 3.0]}},
                 dz=0.25, fslice=0.5, rrf=True, fix_sig=True, qoff=0.0, yscale=1, bkg=0.0,
                 theta=0.0, phi=0.0, protins=0.0, cov=0.1,
                 anchor='softtop', softbeta=10.0, area_mode='grid', area_preset='default',
                 overlap='dipole', zpad=25.0):
        """
        X-ray reflectivity from a stack of layers by the Parratt formalism, with a
        rigid protein template folded into the stack.

        This is the develop-branch XLayersPDB with its pose-dependent
        discretization removed.  The algorithm is unchanged in kind: the
        projected area and the union volume are still pixel quadratures on a
        lattice, the z-profile is still built from the exact analytic sphere/slab
        overlap, the sharp protein+membrane profile is still assembled first and
        then erf-smoothed once as a whole.  What changed is listed under
        WHAT_CHANGED.

        x         : array of wave-vector transfer along z
        E         : energy of x-rays in inverse units of x
        dz        : Parratt slab thickness (A).  See RECOMMENDED SETTINGS.
        fslice    : target protein slab thickness (A).  See RECOMMENDED SETTINGS.
        rrf       : True for Fresnel-normalized reflectivity, False for raw R
        fix_sig   : tie every roughness to sig[1], including sig[0] (see SIGMA)
        qoff      : q-offset to correct the instrument zero
        yscale    : scale factor on R or R/Rf
        bkg       : incoherent background
        theta/phi : Euler angles of the protein template (degrees, 'zx')
        protins   : insertion depth (A).  The topmost atom SURFACE of the rotated
                    template sits at -sum(d[:-1]) + protins.  See ANCHOR and the
                    REGISTRATION warning.
        cov       : surface coverage of the protein, 0 to 1
        anchor    : 'softtop' (default), 'top', or 'centroid'.  See ANCHOR.
        softbeta  : sharpness of the softtop anchor, 1/A.  10 is calibrated.
        area_preset : 'reference' | 'default' | 'fast'.  The only real speed knob.
        zpad      : one-sided pad on the Parratt lattice for the erf tails (A).
                    Raised automatically to 6*max(sig) if that is larger, which
                    is an exact no-op (see WHAT_CHANGED 3).
        mpar      : dictionary of layers, with
                    Layers: names, d: thickness (A), rho: electron density
                    (el/A^3), mu: absorption (1/cm), sig: interfacial roughness
                    (A).  The upper and lower thicknesses should always be fixed.

        WHAT_CHANGED relative to develop XLayersPDB
        -------------------------------------------
        1. Area_gridcalc -> soft_area.  Still a pixel quadrature, but on a
           lab-frame-fixed lattice with a smooth membership weight and the
           smoothing bias removed by extrapolation in the kernel width.  This was
           the dominant source of the theta/phi roughness.

        2. The protein slab lattice is CONFORMAL TO THE MEMBRANE: each finite
           layer is divided into a fixed integer number of sub-slabs, so the
           interfaces ARE slab boundaries, identically and at all times.  This is
           the key fix.  The crossing kink is not intrinsic to the merge; it
           comes from pinning the slab lattice to one reference while the
           interfaces are pinned to another.  Whenever a slab boundary crosses an
           interface the erf amplitudes of the two nodes jump by
           +-cov*d(layarea)*d(rho_memb), equal and opposite, so R stays
           continuous but its derivative kinks.  develop pinned the lattice to
           -sum(d[:-1]), which moves with the interior interfaces, so d was
           smooth (2e-11) while protins/theta/phi were not (1e-4); pinning to
           z = 0 instead reverses that.  Pinning to every interface at once
           removes it for all of them.

           Consequences, all simplifications: np.unique and the protmembcomb
           pointer chase both disappear, since the merged node set IS the edge
           list and each slab's layer is known by construction (this also removes
           an off-by-one where a membrane node inside the protein span advanced
           the protein pointer); protein steps land on slab BOUNDARIES rather
           than centres; and "sigma of the containing layer" becomes well defined
           and never jumps, because an interior boundary never leaves its layer.

           Price: fslice is a target, not an exact thickness.  Sub-slab thickness
           inside layer j is d_j/N_j with N_j fixed at construction, so it drifts
           smoothly if d_j is fitted.  A fit that moves d far from its starting
           value will drift the effective fslice.

        3. The Parratt output lattice is anchored: nodes at dz*(n+0.5) for
           integer n with a fixed pad, instead of a linspace whose endpoints and
           node count moved with the pose, with protins, with d and with sig.
           Node count now changes only by adding or removing cells in the
           zero-contrast tails, which is exact in Parratt (the recursion starts
           at refc = 0, and X = 0 across a zero-contrast interface).  This also
           fixes an inconsistency in develop where the spacing implied by
           Nlayers and the offset implied by halfstep came from different
           expressions, so the lattice spacing was only approximately dz.

        4. protins is anchored on the topmost atom SURFACE via a smooth
           surrogate.  See ANCHOR and REGISTRATION.

        5. protvol is computed at fixed spacing VOL_H, decoupled from fslice.

        ANCHOR
        ------
        max_i(z_i + r_i) is non-differentiable wherever the argmax atom changes,
        and it slides the whole protein through the lattice, so it injects a kink
        into theta and phi.  'softtop' replaces it with the log-sum-exp
        surrogate beta^-1 log sum exp(beta*(z_i+r_i)), which is analytic in the
        coordinates.  At beta = 10 /A the bias against the true apex was under
        0.001 A while the phi roughness fell from 4.6e-6 to 3.0e-8.  The bias is
        bounded by ln(N_near)/beta and is pose-dependent -- up to about 0.07 A
        where two atoms are co-apical -- but it varies smoothly, which is the
        point.  'top' is the exact hard max (kinked); 'centroid' anchors on the
        template centroid, which is exactly smooth but redefines protins.

        REGISTRATION -- warning for comparing against old fits
        ------------------------------------------------------
        develop's effective protins is deep by exactly ONE fslice, measured as
        -0.996, -0.992, -0.982, -0.973 x fslice at fslice = 2, 1, 0.5, 0.25.  Two
        effects add: the +fslice envelope padding on the anchor (fslice/2), and
        the merge consuming protrho[i] as the value BELOW node i while node i is
        a slab CENTRE (another fslice/2).  To reproduce an old develop fit here,
        use protins_new = protins_old - fslice.  Equivalently, historical develop
        insertion depths are biased deep by ~fslice, i.e. 1-2 A at the fslice =
        1-2 settings used previously.

        SIGMA -- roughness of the bulk medium above the film
        ---------------------------------------------------
        With fix_sig=True, sig[0] is tied to sig[1] rather than held at 0, and
        the default mpar gives sig[0] a nonzero value.  All structure in the
        model is associated with the interface and so inherits its roughness; in
        particular a protein that protrudes above the film is still bound to it
        and must be smeared by the same sigma.  Leaving sig[0] = 0 makes the
        protruding slabs effectively sharp, and the residual staircase then
        survives into R: for a fully protruding case (theta=90, phi=0,
        protins=30, cov=0.5) at fslice=0.5 the agreement with develop was 3.55%
        in the EDP and 12.6% in R with sig[0]=0, versus 0.19% and 0.89% with
        sig[0]=sig[1].  With fix_sig=False you must set sig[0] yourself.
        This changes nothing while the protein is submerged: the lattice only
        extends above the film when the protein protrudes, so sig[0] is otherwise
        unused.

        RECOMMENDED SETTINGS
        --------------------
        fslice = 0.5, dz = 0.25.  Both errors go as the square of the spacing,
        with err% ~ 0.31*fslice^2 and err% ~ 1.15*dz^2 at cov = 0.5, so dz
        matters 3.7x more per Angstrom and the balanced choice is fslice ~ 2*dz.
        That pair gives 0.081% total, about an eighth of a typical dR/R of 0.65%,
        and these are cov = 0.5 figures; at cov = 0.1 they are 5-15x smaller.
        dz = 1.0 is not usable (1.1%).

        Note that the roughness does NOT wash this error out.  It washes out the
        high-frequency staircase, which is lattice-dependent, but not the
        systematic O(h^2) slab-averaging bias: box-averaging over width h and
        then convolving with a Gaussian gives f*G + (h^2/24)*f''*G + O(h^4), and
        the correction term is as smooth as the profile, so it passes straight
        through.  Measured: the fslice error is 1.2287 / 1.2240 / 1.2160 % at
        fslice = 2 for sigma = 2.0 / 3.4 / 5.0, i.e. independent of sigma.

        fslice is NOT a speed knob -- cost is flat within noise from fslice 1 to
        0.125, because the area quadrature dominates.  The only real speed knob
        is area_preset (about 45 -> 30 ms per pose, serial).

        MEASURED SMOOTHNESS
        -------------------
        Local-cubic residual relative to |R|, median over q, swept in each
        parameter separately (develop -> this model):
            protins 2.23e-4 -> 6.9e-9      cov  3.5e-13 -> 3.4e-13
            theta   1.14e-4 -> 7.0e-8      d    7.45e-6 -> 2.0e-10
            phi     1.68e-4 -> 2.1e-8      rho  4.4e-13 -> 2.8e-13
                                           sig  7.38e-6 -> 5.3e-11
        The residual 7e-8 in theta is unattributed; the area estimator and the
        anchor have both been ruled out by measurement.

        REJECTED ALTERNATIVES, recorded so they are not retried
        ------------------------------------------------------
        * Analytic union-of-discs area by Green's theorem: excluded by design,
          the area stays a grid quadrature.
        * Hard-indicator area on a fixed or refined lattice: cannot beat 2e-4.
        * Pointwise superposition, i.e. smoothing rho_memb, the excluded-area
          fraction and rho_prot separately and then multiplying: rejected on
          physical grounds, because roughness is a property of the whole system
          and not of its parts.  It is also not a discretization variant -- it
          differs by ~9% in R and 0.94 A in the protein-excess centroid,
          independent of fslice, since smoothing a product is not the product of
          smoothings.
        * Uniform (non-conformal) protein lattice: leaves 1e-4 in d.
        """
        if type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        if os.path.exists(fname):
            self.fname = fname
        else:
            self.fname = None
        self.E = E
        self.__mpar__ = mpar
        self.dz = dz
        self.fslice = fslice
        self.rrf = rrf
        self.fix_sig = fix_sig
        self.qoff = qoff
        self.bkg = bkg
        self.yscale = yscale
        self.theta = theta
        self.phi = phi
        self.protins = protins
        self.cov = cov
        self.anchor = anchor
        self.softbeta = softbeta
        self.area_mode = area_mode
        self.overlap = overlap
        self.area_preset = area_preset
        self.zpad = zpad
        self.choices = {'rrf': [True, False], 'fix_sig': [True, False],
                        'anchor': ['softtop', 'top', 'centroid'],
                        'area_mode': ['grid', 'ellipse'],
                        'area_preset': ['default', 'fast', 'reference'],
                        'overlap': ['dipole', 'shielded', 'global']}
        self.filepaths = {'fname': self.fname}

        self.__d__ = {}
        self.__rho__ = {}
        self.__mu__ = {}
        self.__sig__ = {}
        self.__fit__ = False
        self.__mkeys__ = list(self.__mpar__.keys())
        self.__fix_sig_changed__ = 0
        self.__nsub__ = None
        self.init_params()
        self.output_params = {'scaler_parameters': {}}
        if self.fname is not None:
            xc, yc, zc, el, rad, mass = self.pdbread(self.fname)
            self.__protvol__, self.__shield__, self.__mom__ = self.protvolcalc(self.fname)
            self.update_parameters()
            self._refresh_profile(xc, yc, zc, rad, el)

    def init_params(self):
        self.params = Parameters()
        self.params.add('qoff', self.qoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('yscale', self.yscale, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('bkg', self.bkg, vary=0, min=0, max=1, expr=None, brute_step=0.1)
        self.params.add('theta', self.theta, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=1)
        self.params.add('phi', self.phi, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=1)
        self.params.add('protins', self.protins, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('cov', self.cov, vary=1, min=0, max=1, expr=None, brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key != 'Layers':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i),
                                        value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0, max=np.inf, expr=None, brute_step=0.05)

    # ------------------------------ template -------------------------------

    def pdbread(self, fname):
        return _read_template(_stamp(fname))

    def protvolcalc(self, fname):
        return _shielding(_stamp(fname))

    def protarea(self, fname, theta, phi, mode, preset, softbeta):
        return _protarea(_stamp(fname), theta, phi, mode, preset, softbeta)

    # ------------------------- conformal lattice ---------------------------

    def _nsub(self, d):
        """Fixed integer subdivision count per finite membrane layer.  Set once,
        from the d in force at construction, then held, so the lattice deforms
        continuously with d instead of re-quantising."""
        if self.__nsub__ is None:
            self.__nsub__ = tuple(max(1, int(round(dj / self.fslice))) for dj in d[1:-1])
        return self.__nsub__

    def _lattice(self, d, nsub, utop, ubot):
        """Ascending depth edges, plus each slab's membrane layer index.

        Interfaces sit at cumsum(d)[:-1].  Inside finite layer j the layer is cut
        into nsub[j] equal pieces.  Above the first interface and below the last
        the lattice is uniform at fslice, anchored on that interface, extended
        far enough to cover [utop, ubot].
        """
        fs = self.fslice
        uint = np.cumsum(np.asarray(d))[:-1]
        n = len(d)
        edges = []
        klay = []
        nup = max(0, int(np.ceil((uint[0] - utop) / fs)))
        for i in range(nup, 0, -1):
            edges.append(uint[0] - i * fs)
            klay.append(0)
        edges.append(uint[0])
        for j in range(len(uint) - 1):
            lo, hi = uint[j], uint[j + 1]
            N = nsub[j] if j < len(nsub) else 1
            for i in range(1, N + 1):
                edges.append(lo + (hi - lo) * i / N)
                klay.append(j + 1)
        ndn = max(1, int(np.ceil((ubot - uint[-1]) / fs)))
        for i in range(1, ndn + 1):
            edges.append(uint[-1] + i * fs)
            klay.append(n - 1)
        return np.array(edges, dtype=np.float64), np.array(klay, dtype=np.int64)

    def _refresh_profile(self, xcoord, ycoord, zcoord, radius, electrons):
        mkey = self.__mkeys__[0]
        d = np.array(self.__d__[mkey])
        rotz, minarea, ztop, ztop_soft, momz = self.protarea(
            self.fname, self.theta, self.phi, self.area_mode, self.area_preset, self.softbeta)
        self.__minarea__ = minarea
        self.__ztop__ = ztop
        self.__ztop_soft__ = ztop_soft
        zref = {'centroid': 0.0, 'top': ztop, 'softtop': ztop_soft}[self.anchor]
        zoffset = -np.sum(d[:-1]) + self.protins - zref
        ua = -(np.asarray(rotz) + zoffset)
        rad = np.asarray(radius)
        edges, klay = self._lattice(tuple(d), self._nsub(tuple(d)),
                                    float(np.min(ua - rad)), float(np.max(ua + rad)))
        """Overlap correction.  'global' applies one factor protvol/sum(Vol) to
        every slab, which assumes the overlap fraction is depth-independent; it
        is not, and that was the largest remaining error in the model.
        'shielded' weights each atom by its own share s_i of the union.
        'dipole' additionally lets the shielding vary linearly across each atom,
        which matters because a surface atom is shielded on its buried side, not
        uniformly.  All three preserve the total exactly."""
        if self.overlap == 'global':
            w = np.ones(len(rad))
            bco = np.zeros(len(rad))
        else:
            w = np.asarray(self.__shield__)
            if self.overlap == 'dipole':
                bco = -np.asarray(momz) * 15.0 / (4.0 * np.pi * rad ** 5)
            else:
                bco = np.zeros(len(rad))
        Elec = np.zeros(len(edges) - 1)
        Vol = np.zeros(len(edges) - 1)
        VolU = np.zeros(len(edges) - 1)
        sphere_slab_nonuniform(ua, rad, np.asarray(electrons, dtype=np.float64),
                               w, bco, edges, Elec, Vol, VolU)
        if self.overlap == 'global':
            VolU = Vol * (self.__protvol__ / np.sum(Vol))
        cell = minarea * np.diff(edges)
        self.__edges__ = edges
        self.__klay__ = klay
        self.__protrho__ = Elec / cell
        self.__layarea__ = 1.0 - VolU / cell
        self.__protpos__ = -0.5 * (edges[:-1] + edges[1:])

    def calcProfComb(self, d, rho, mu, sig, protins, cov, phase, dz, zpad):
        """Assemble the sharp protein+membrane profile, then smooth it ONCE.

        The sharp profile is piecewise constant on the conformal edge list, so no
        merge, no np.unique and no pointer chase are needed: each slab's membrane
        layer is klay[i] by construction.  Outside the protein's support
        protrho = 0 and layarea = 1 identically, so V reduces to rho[k] and the
        edge jump is exactly zero, making slabs that enter or leave the support
        exact no-ops (verified to 5e-12 in R).
        """
        d = np.array(d); rho = np.array(rho); mu = np.array(mu); sig = np.array(sig)
        n = len(d)
        edges = self.__edges__
        klay = self.__klay__
        pr = self.__protrho__
        la = self.__layarea__

        V = cov * pr + (1.0 - cov * (1.0 - la)) * rho[klay]
        y = np.concatenate(([rho[0]], V, [rho[n - 1]]))
        ymu = np.concatenate(([mu[0]], mu[klay], [mu[n - 1]]))
        """sigma of the containing layer -- develop's rule.  An interior boundary
        never leaves its layer here, so this assignment is fixed."""
        sigma = np.concatenate(([sig[0]], sig[klay], [sig[n - 1]]))
        pos = np.append(edges, edges[-1] + dz)

        """Anchored Parratt lattice.  The pad is widened if sig demands it, which
        is exact: added cells sit in the zero-contrast tails."""
        pad = max(zpad, 6.0 * float(np.max(sig)))
        klo = int(np.floor((edges[0] - pad) / dz - 0.5))
        khi = int(np.ceil((edges[-1] + pad) / dz - 0.5))
        __z__ = dz * (np.arange(klo, khi + 1) + 0.5)
        __d__ = np.full(len(__z__), dz)
        __rho__ = self.sldCalFun(pos, y, sigma, __z__)
        __mu__ = self.sldCalFun(pos, ymu, sigma, __z__)
        return (n, __z__, __d__, __rho__, __mu__, -self.__protpos__, np.zeros(len(pr)),
                edges, np.append(V, rho[n - 1]), np.append(mu[klay], mu[n - 1]))

    def sldCalFun(self, pos, y, sigma, x):
        """No lru_cache here.  The arguments are arrays of a few hundred floats
        that change every evaluation, so the cache essentially never hit while
        building and hashing the key cost ~0.25 ms per call."""
        return np.asarray(sldCalFun_numba(np.asarray(pos, dtype=np.float64),
                                          np.asarray(y, dtype=np.float64),
                                          np.asarray(sigma, dtype=np.float64),
                                          np.asarray(x, dtype=np.float64)))

    @lru_cache(maxsize=10)
    def stepFun(self, zmin, zmax, d, rho, mu):
        tdata = [[zmin, rho[0], mu[0]]]
        z = np.cumsum(d)
        for i, td in enumerate(d[:-1]):
            tdata.append([z[i], rho[i], mu[i]])
            tdata.append([z[i], rho[i + 1], mu[i + 1]])
        tdata.append([zmax, rho[-1], mu[-1]])
        tdata = np.array(tdata)
        return tdata[:, 0], tdata[:, 1], tdata[:, 2]

    @lru_cache(maxsize=10)
    def protstepFun(self, zmin, zmax, z, rho, mu):
        tdata = [[zmin, rho[0], mu[0]]]
        for i, td in enumerate(z[:-1]):
            tdata.append([z[i], rho[i], mu[i]])
            tdata.append([z[i], rho[i + 1], mu[i + 1]])
        tdata.append([zmax, rho[-1], mu[-1]])
        tdata = np.array(tdata)
        return tdata[:, 0], tdata[:, 1], tdata[:, 2]

    def py_parratt(self, x, lam, d, rho, mu):
        return parratt_numba(np.asarray(x, dtype=np.float64), lam,
                             np.asarray(d, dtype=np.float64),
                             np.asarray(rho, dtype=np.float64),
                             np.asarray(mu, dtype=np.float64))

    def update_parameters(self):
        for mkey in self.__mpar__.keys():
            Nlayers = len(self.__mpar__[mkey]['d'])
            self.__d__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'd', i)].value for i in range(Nlayers)])
            self.__rho__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'rho', i)].value for i in range(Nlayers)])
            self.__mu__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'mu', i)].value for i in range(Nlayers)])
            sig = np.array([self.params['__%s_%s_%03d' % (mkey, 'sig', i)].value for i in range(Nlayers)])
            sig = np.where(sig < 1e-5, 1e-5, sig)
            self.__sig__[mkey] = tuple(sig)

    def y(self):
        x = self.x + self.qoff
        lam = 6.62607004e-34 * 2.99792458e8 * 1e10 / self.E / 1e3 / 1.60217662e-19
        phitemp = self.params['phi'].value % 360
        thetatemp = self.params['theta'].value % 360
        if thetatemp > 180:
            self.params['theta'].value = 360 - thetatemp
            self.params['phi'].value = (phitemp + 180) % 360
        else:
            self.params['phi'].value = phitemp

        if not self.__fit__:
            for mkey in self.__mpar__.keys():
                Nlayers = len(self.__mpar__[mkey]['sig'])
                if self.fix_sig:
                    """Index 0 -- the bulk medium above the film -- is tied in as
                    well.  All structure is associated with the interface and so
                    inherits its roughness; protruding protein is still bound to
                    the film and must be smeared by the same sigma."""
                    for i in [0] + list(range(2, Nlayers)):
                        self.params['__%s_%s_%03d' % (mkey, 'sig', i)].expr = '__%s_%s_%03d' % (mkey, 'sig', 1)
                    self.__fix_sig_changed__ = 1
                if not self.fix_sig and self.__fix_sig_changed__ > 0:
                    for i in [0] + list(range(2, Nlayers)):
                        self.params['__%s_%s_%03d' % (mkey, 'sig', i)].expr = None
                    self.__fix_sig_changed__ = 0

        xc, yc, zc, el, rad, mass = self.pdbread(self.fname)
        self.__protvol__, self.__shield__, self.__mom__ = self.protvolcalc(self.fname)
        self.update_parameters()
        self._refresh_profile(xc, yc, zc, rad, el)
        mkey = self.__mkeys__[0]
        n, z, d, rho, mu, zprot, protmu, zprotmemb, protmembrho, protmembmu = self.calcProfComb(
            self.__d__[mkey], self.__rho__[mkey], self.__mu__[mkey], self.__sig__[mkey],
            self.protins, self.cov, mkey, self.dz, self.zpad)

        if not self.__fit__:
            mk = self.__mkeys__[0]
            tz, trho, tmu = self.stepFun(z[0], z[-1], self.__d__[mkey], self.__rho__[mkey], self.__mu__[mkey])
            pz, prho, pmu = self.protstepFun(zprot[0], zprot[-1], tuple(zprot),
                                             tuple(self.__protrho__), tuple(protmu))
            ptz, ptrho, ptmu = self.protstepFun(z[0], z[-1], tuple(zprotmemb),
                                                tuple(protmembrho), tuple(protmembmu))
            self.output_params['%s_EDP' % mk] = {'x': z, 'y': rho, 'names': ['z (Angs)', 'rho (el/Angs^3)'], 'plotType': 'step'}
            self.output_params['%s_dEDP' % mk] = {'x': z[:-1], 'y': np.diff(rho) / self.dz, 'names': ['z (Angs)', 'drho/dz (el/Angs^4)'], 'plotType': 'step'}
            self.output_params['%s_step_layEDP' % mk] = {'x': tz, 'y': trho, 'names': ['z (Angs)', 'rho (el/Angs^3)'], 'plotType': 'step'}
            self.output_params['%s_step_protEDP' % mk] = {'x': pz, 'y': prho, 'names': ['z (Angs)', 'rho (el/Angs^3)'], 'plotType': 'step'}
            self.output_params['%s_step_EDP' % mk] = {'x': ptz, 'y': ptrho, 'names': ['z (Angs)', 'rho (el/Angs^3)'], 'plotType': 'step'}
            self.output_params['%s_ADP' % mk] = {'x': z, 'y': mu, 'names': ['z (Angs)', 'mu (1/cm)'], 'plotType': 'step'}
            self.output_params['%s_dADP' % mk] = {'x': z[:-1], 'y': np.diff(mu) / self.dz, 'names': ['z (Angs)', 'dmu/dz (1/cm^2)'], 'plotType': 'step'}
            self.output_params['%s_step_ADP' % mk] = {'x': tz, 'y': tmu, 'names': ['z (Angs)', 'mu (1/cm)'], 'plotType': 'step'}
            self.output_params['scaler_parameters']['proj_area_Angs2'] = self.__minarea__
            self.output_params['scaler_parameters']['prot_volume_Angs3'] = self.__protvol__
            self.output_params['scaler_parameters']['ztop_apex_Angs'] = self.__ztop__
            self.output_params['scaler_parameters']['nsub'] = str(self.__nsub__)
            self.output_params['scaler_parameters']['n_prot_slabs'] = len(self.__protrho__)

        refq, r2 = self.py_parratt(x, lam, d, rho, mu)
        if self.rrf:
            rhos = (self.params['__%s_rho_000' % mkey].value,
                    self.params['__%s_rho_%03d' % (mkey, n - 1)].value)
            mus = (0, 0)
            ref, r2 = self.py_parratt(x - self.qoff, lam, (0.0, 1.0), rhos, mus)
            refq = refq / ref
        return refq * self.yscale + self.bkg


if __name__ == '__main__':
    x = np.linspace(0.033, 0.55, 120)
    mpar = {'Model': {'Layers': ['air', 'tail', 'head', 'buffer'],
                      'd': [0.0, 13.4, 10.0, 1.0], 'rho': [0.0, 0.243, 0.453, 0.334],
                      'mu': [0.0, 0.0, 0.0, 0.0], 'sig': [3.4, 3.4, 3.4, 3.4]}}
    fun = XLayersPDB_dev_smooth(x=x, E=10.0, fname='./tim3_hmmm_ext_all_bestframe.pdb',
                                mpar=mpar, dz=0.25, fslice=0.5, rrf=False, cov=0.1)
    fun.theta, fun.phi, fun.protins = 90.0, 240.0, 18.0
    fun.params['theta'].value, fun.params['phi'].value = 90.0, 240.0
    fun.params['protins'].value = 18.0
    print('R[:4] =', np.asarray(fun.y())[:4])
    print('proj area =', fun.__minarea__, ' protvol =', fun.__protvol__,
          ' nsub =', fun.__nsub__, ' slabs =', len(fun.__protrho__))
