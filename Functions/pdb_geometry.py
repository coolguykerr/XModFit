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

    Hand-written in Julia, but the SAME rotation ``XLayersPDB`` applies: it equals
    ``Rotation.from_euler('zx', [phi, theta], degrees=True).as_matrix()``
    (extrinsic, so ``R_x(theta) @ R_z(phi)``), which is exactly what
    ``XLayersPDB.ED_profrot`` uses, and ``M @ v`` reproduces its ``rot.apply(v)``
    -- 4e-16 on the matrix and 2e-14 on 20 A coordinates, over 475 (theta, phi)
    pairs covering theta 0-180 and phi 0-359.  Both take the angles in DEGREES
    (``XPR.jl`` converts to radians inside ``XPR!``), so theta/phi transfer between
    the two models: safe to seed one from the other and to hand either to the NGL
    viewer.  An earlier version of this docstring claimed they do not; that was
    retracted 2026-09-02, see ``handoff/05-xpr-julia.md`` 4.8.

    The ORIGIN of the rotation differs since XPR's 2026-09-08 revision: XLayersPDB
    rotates the raw PDB coordinates about the file origin, XPR about the mass
    centroid (``prealloc`` subtracts it at load time).  That is a pure translation,
    so it does not touch the orientation and the angles still transfer -- but it is
    absorbed into the insertion parameter, which is why XPR's ``protins`` is no
    longer XLayersPDB's.  Use :func:`com_extent` to convert.
    """
    t = np.radians(float(theta_deg))
    p = np.radians(float(phi_deg))
    return np.array([[np.cos(p), -np.sin(p), 0.0],
                     [np.cos(t) * np.sin(p), np.cos(t) * np.cos(p), -np.sin(t)],
                     [np.sin(t) * np.sin(p), np.sin(t) * np.cos(p), np.cos(t)]])


@lru_cache(maxsize=16)
def com_coordinates(fname):
    """``(coords - mass centroid, radii)`` for ``fname``, as XPR stores them.

    XPR's ``prealloc`` subtracts the MASS-weighted centroid from the coordinates at
    load time and measures its insertion parameter from there, so anything that
    reasons about where the protein's ends fall has to use the same origin.
    """
    xc, yc, zc, _e, radius, mass = pdbread(fname)
    coord = np.transpose(np.array([xc, yc, zc], dtype=float))
    mass = np.asarray(mass, dtype=float)
    com = (coord * mass[:, None]).sum(axis=0) / mass.sum()
    return coord - com, np.asarray(radius, dtype=float)


def com_extent(fname, theta, phi):
    """``(maxh, mindepth, iatom)``: the protein's reach either side of its centroid.

    Along the rotated z axis, ``maxh`` is the distance from the mass centroid to the
    top of the topmost atomic sphere and ``mindepth`` the same downwards, so the
    protein occupies ``[protins - mindepth, protins + maxh]`` above the bottom of
    the box stack. ``maxh`` is the offset between XPR's two insertion conventions::

        protins_top = protins + maxh(theta, phi)

    with ``protins`` placing the CENTRE OF MASS (what XPR fits, post-2026-09-08) and
    ``protins_top`` the TOP OF THE PROTEIN (the old XPR convention, and still the
    one every numba ``XLayersPDB*`` model uses). Reproduces ``XPR.XPR_maxh``.

    ``iatom`` is the 0-based index of the atom attaining the maximum. ``maxh`` is a
    maximum over atoms, hence only piecewise smooth in (theta, phi) -- which is
    precisely why the offset was taken out of the forward model -- and a derivative
    needs to know which atom is active; see :func:`com_extent_grad`.
    """
    coord, radius = com_coordinates(fname)
    proj = coord @ xpr_rotation_matrix(theta, phi)[2]
    hi = proj + radius
    k = int(np.argmax(hi))
    return float(hi[k]), float(-np.min(proj - radius)), k


def com_extent_grad(fname, theta, phi):
    """``(maxh, d maxh/d theta, d maxh/d phi, iatom, nties)``, angles in DEGREES.

    Exact wherever the topmost atom is unique -- ``maxh`` is then just
    ``r3(theta, phi) . c_k + rad_k`` for that one atom -- which is almost everywhere.
    ``nties`` counts the atoms whose top comes within 1e-6 A of the maximum: more
    than one means (theta, phi) sits on a kink where ``maxh`` has a one-sided
    derivative on each side and no two-sided one, so a propagated error bar there is
    only indicative. Reported so a caller can say so rather than guess.
    """
    coord, radius = com_coordinates(fname)
    t, p = np.radians(float(theta)), np.radians(float(phi))
    st, ct, sp, cp = np.sin(t), np.cos(t), np.sin(p), np.cos(p)
    r3 = np.array([st * sp, st * cp, ct])
    dr3_dt = np.array([ct * sp, ct * cp, -st])
    dr3_dp = np.array([st * cp, -st * sp, 0.0])
    hi = coord @ r3 + radius
    k = int(np.argmax(hi))
    nties = int(np.count_nonzero(hi >= hi[k] - 1e-6))
    rad = np.pi / 180.0
    return (float(hi[k]), float(coord[k] @ dr3_dt) * rad,
            float(coord[k] @ dr3_dp) * rad, k, nties)


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
