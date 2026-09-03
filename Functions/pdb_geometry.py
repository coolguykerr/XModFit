"""
pdb_geometry.py -- union volume and projected area of a PDB's atoms, on a grid.

Both are the quantities XLayersPDB uses to keep its ``cov`` parameter on a [0, 1]
scale while still summing per-atom spherical segments (which double counts every
overlap):

  * ``union_volume``   -- the true volume of the union of the atomic spheres.
    Invariant under rotation, so it is computed ONCE per structure and cached.
  * ``projected_area`` -- the true area of the xy-shadow of that union, i.e. the
    protein's footprint on the interface.  This one DOES depend on orientation and
    is recomputed per (theta, phi).

XLayersPDB carries its own copies of these grid kernels (``Vol_gridcalc`` and
``Area_gridcalc``).  They are re-implemented rather than imported because
``Functions/XRR/XLayersPDB.py`` already imports ``Functions/pdb_cache.py``, and
having a shared helper import back into a category folder would be both circular
and a violation of the rule that category folders contain only model classes.

Lives in ``Functions/`` root, not a category folder: XModFit lists every ``.py``
in a category folder as a fittable model named after the file.
"""

from functools import lru_cache

import numpy as np
from numba import njit

try:
    from Functions.pdb_cache import pdbread
except Exception:  # running with Functions/ on sys.path
    from pdb_cache import pdbread


@njit(parallel=False, cache=True)
def _mark_volume(x, y, z, radius, sx, sy, sz, x0, y0, z0, grid):
    """Set grid[i,j,k] where the cell centre lies inside any atom."""
    nx, ny, nz = grid.shape
    for a in range(len(radius)):
        r = radius[a]
        r2 = r * r
        ilo = max(0, int(np.floor((x[a] - r - x0) / sx)))
        ihi = min(nx - 1, int(np.ceil((x[a] + r - x0) / sx)))
        jlo = max(0, int(np.floor((y[a] - r - y0) / sy)))
        jhi = min(ny - 1, int(np.ceil((y[a] + r - y0) / sy)))
        klo = max(0, int(np.floor((z[a] - r - z0) / sz)))
        khi = min(nz - 1, int(np.ceil((z[a] + r - z0) / sz)))
        for i in range(ilo, ihi + 1):
            dx = x0 + i * sx - x[a]
            dx2 = dx * dx
            if dx2 > r2:
                continue
            for j in range(jlo, jhi + 1):
                dy = y0 + j * sy - y[a]
                d2 = dx2 + dy * dy
                if d2 > r2:
                    continue
                for k in range(klo, khi + 1):
                    dz = z0 + k * sz - z[a]
                    if d2 + dz * dz <= r2:
                        grid[i, j, k] = True
    return grid


@njit(parallel=False, cache=True)
def _mark_area(x, y, radius, sx, sy, x0, y0, grid):
    """Set grid[i,j] where the cell centre lies inside any atom's xy disc."""
    nx, ny = grid.shape
    for a in range(len(radius)):
        r = radius[a]
        r2 = r * r
        ilo = max(0, int(np.floor((x[a] - r - x0) / sx)))
        ihi = min(nx - 1, int(np.ceil((x[a] + r - x0) / sx)))
        jlo = max(0, int(np.floor((y[a] - r - y0) / sy)))
        jhi = min(ny - 1, int(np.ceil((y[a] + r - y0) / sy)))
        for i in range(ilo, ihi + 1):
            dx = x0 + i * sx - x[a]
            dx2 = dx * dx
            if dx2 > r2:
                continue
            for j in range(jlo, jhi + 1):
                dy = y0 + j * sy - y[a]
                if dx2 + dy * dy <= r2:
                    grid[i, j] = True
    return grid


def xpr_rotation_matrix(theta_deg, phi_deg):
    """XPR's rotation matrix -- rows r1, r2, r3 of ``rotate_protein!``.

    NOT the scipy ``'zx'`` Euler convention XLayersPDB uses; angles are not
    interchangeable between the two models.
    """
    t = np.radians(float(theta_deg))
    p = np.radians(float(phi_deg))
    return np.array([[np.cos(p), -np.sin(p), 0.0],
                     [np.cos(t) * np.sin(p), np.cos(t) * np.cos(p), -np.sin(t)],
                     [np.sin(t) * np.sin(p), np.sin(t) * np.cos(p), np.cos(t)]])


def _axis_grid(lo, hi, step):
    n = max(1, int(np.round((hi - lo) / step)))
    return n, (hi - lo) / n


@lru_cache(maxsize=16)
def sphere_sums(fname):
    """(N, sum pi r^2, sum 4/3 pi r^3) for the atoms of ``fname``.

    ``sum pi r^2`` is the overlap-blind projected area XPR's ``vol_calc='approx'``
    normalises by (``tempvars.A``); the third entry is what the per-atom spherical
    segments integrate to before any union correction.
    """
    _x, _y, _z, _e, radius, _m = pdbread(fname)
    radius = np.asarray(radius, dtype=float)
    return (len(radius),
            float(np.sum(np.pi * radius ** 2)),
            float(np.sum((4.0 / 3.0) * np.pi * radius ** 3)))


@lru_cache(maxsize=16)
def union_volume(fname, fslice=0.5):
    """Volume of the union of the atomic spheres, in A^3.

    Rotation invariant, so the unrotated coordinates are used and the result is
    cached per (file, grid step).
    """
    xc, yc, zc, _e, radius, _m = pdbread(fname)
    coord = np.ascontiguousarray(np.transpose(np.array([xc, yc, zc], dtype=float)))
    radius = np.ascontiguousarray(np.asarray(radius, dtype=float))
    pad = radius.max() + fslice
    lo = coord.min(axis=0) - pad
    hi = coord.max(axis=0) + pad
    n, s = zip(*[_axis_grid(lo[k], hi[k], fslice) for k in range(3)])
    # cell centres at lo + s*(i + 0.5)
    origin = lo + 0.5 * np.array(s)
    grid = np.zeros((n[0], n[1], n[2]), dtype=np.bool_)
    grid = _mark_volume(coord[:, 0], coord[:, 1], coord[:, 2], radius,
                        s[0], s[1], s[2], origin[0], origin[1], origin[2], grid)
    return float(s[0] * s[1] * s[2] * np.count_nonzero(grid))


@lru_cache(maxsize=512)
def projected_area(fname, theta, phi, fslice=0.5):
    """Area of the xy-shadow of the union of spheres after XPR's rotation, in A^2.

    Cached on the exact (theta, phi) floats, so a fit that holds the orientation
    fixed pays for this once while one that varies it recomputes -- which is the
    right way round. Deliberately NOT rounded: quantising the angles would put a
    staircase into the objective.
    """
    xc, yc, zc, _e, radius, _m = pdbread(fname)
    coord = np.transpose(np.array([xc, yc, zc], dtype=float))
    radius = np.ascontiguousarray(np.asarray(radius, dtype=float))
    rot = np.ascontiguousarray(coord @ xpr_rotation_matrix(theta, phi).T)
    pad = radius.max() + fslice
    lo = rot[:, :2].min(axis=0) - pad
    hi = rot[:, :2].max(axis=0) + pad
    n, s = zip(*[_axis_grid(lo[k], hi[k], fslice) for k in range(2)])
    origin = lo + 0.5 * np.array(s)
    grid = np.zeros((n[0], n[1]), dtype=np.bool_)
    grid = _mark_area(np.ascontiguousarray(rot[:, 0]),
                      np.ascontiguousarray(rot[:, 1]), radius,
                      s[0], s[1], origin[0], origin[1], grid)
    return float(s[0] * s[1] * np.count_nonzero(grid))
