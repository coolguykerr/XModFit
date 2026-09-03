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
"""from timeit import default_timer
from scipy import special"""

####Please do not remove lines above####

####Import your modules below if needed####
# from xr_ref import parratt
import Bio.PDB.Atom
from scipy.spatial.transform import Rotation as R
try:
    from mendeleev.fetch import fetch_table
except:
    from mendeleev import get_table
from numba import njit, prange

# Shared, input-keyed PDB read cached across ALL XLayersPDB* instances/variants
# (keyed on fname, not self) -- see Functions/pdb_cache.py.
try:
    from Functions.pdb_cache import pdbread
except Exception:
    from pdb_cache import pdbread


@njit(parallel=False, cache=True)
def parratt_numba(q, lam, d, rho, beta):
    ref = np.ones_like(q)
    refc = np.ones_like(q) * complex(1.0, 0.0)
    f1 = 16.0 * np.pi * 2.818e-5
    f2 = -32.0 * np.pi ** 2 / lam ** 2
    Nl = len(d)
    for j in prange(len(q)):
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
def protmembcomb(protmembpos, membpos, d, rho, mu, sig, protpos, protrho, layarea, cov):
    m = len(protmembpos)
    protmembrho = np.zeros(m)
    protmembsig = np.zeros(m)
    protmembmu = np.zeros(m)
    """Membrane position 0th index is 0 because the first layer is air and has d[0]=0 thickness, so membrane
           electron density needs to start at rho[1].
           Protein position 0th index is a real thickness, so protein electron density needs to start at protrho[0].
           The position comparison should be < for protein and <= for membrane."""
    k = 0
    l = 0
    for j in range(m - 1):
        if protmembpos[j] <= membpos[k] and k < (len(d) - 1):
            k += 1
        if protmembpos[j] < protpos[l] and l < (len(protpos) - 1):
            l += 1

        protmembsig[j] = sig[k]
        protmembmu[j] = mu[k]
        if protmembpos[j] > protpos[0] or protmembpos[j] < protpos[-1]:
            protmembrho[j] = rho[k]
        else:
            protmembrho[j] = (cov * protrho[l] + (1 + cov * (layarea[l] - 1)) * rho[k])

    protmembsig[-1] = sig[k]
    protmembmu[-1] = mu[k]
    protmembrho[-1] = rho[k]

    return protmembrho, protmembmu, protmembsig


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
def SphereElecFrac(zrotcoord, radius, electrons, z_gridpos, zslice, ElecFracsum, Volfrac):
    for ind in range(len(radius)):
        vol = (4 / 3) * np.pi * (radius[ind] ** 3)

        botslice = z_gridpos - zrotcoord[ind] - 0.5 * zslice
        topslice = botslice + zslice

        zbotind = int(np.ceil((-radius[ind] - topslice[0]) / zslice))
        ztopind = int(np.floor((radius[ind] - botslice[0]) / zslice))

        inc = np.arange(zbotind, (ztopind + 1))

        d = np.maximum(botslice[inc], -radius[ind])
        h = np.minimum((topslice[inc] - d), (radius[ind] - botslice[inc]))

        ElecFracsum[inc] += electrons[ind] * (
                np.pi * h * (np.square(radius[ind]) - np.square(d) - h * d - (1 / 3) * np.square(h))) / vol

        Volfrac[inc] += (np.pi * h * (np.square(radius[ind]) - np.square(d) - h * d - (1 / 3) * np.square(h)))

    return ElecFracsum, Volfrac


@njit(parallel=False, cache=True)
def Vol_gridcalc(xrotcoord, yrotcoord, zrotcoord, radius, slices, x_gridpos, y_gridpos, z_gridpos, edges, Vol_grid):
    for atomind in range(len(radius)):

        xtopind = int(np.ceil((xrotcoord[atomind] + radius[atomind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        xbotind = int(np.floor((xrotcoord[atomind] - radius[atomind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        ytopind = int(np.ceil((yrotcoord[atomind] + radius[atomind] - edges[3] - 0.5 * slices[1]) / slices[1]))
        ybotind = int(np.floor((yrotcoord[atomind] - radius[atomind] - edges[3] - 0.5 * slices[1]) / slices[1]))
        ztopind = int(np.ceil((zrotcoord[atomind] + radius[atomind] - edges[5] - 0.5 * slices[2]) / slices[2]))
        zbotind = int(np.floor((zrotcoord[atomind] - radius[atomind] - edges[5] - 0.5 * slices[2]) / slices[2]))

        radius2 = radius[atomind] ** 2

        for xind in range(xbotind, xtopind + 1):
            for yind in range(ybotind, ytopind + 1):
                for zind in range(zbotind, ztopind + 1):
                    atomdist = (x_gridpos[xind] - xrotcoord[atomind]) ** 2 + (
                            y_gridpos[yind] - yrotcoord[atomind]) ** 2 + (
                                       z_gridpos[zind] - zrotcoord[atomind]) ** 2
                    if atomdist <= radius2:
                        Vol_grid[xind, yind, zind] = 1

    return Vol_grid


@njit(parallel=False, cache=True)
def Area_gridcalc(xrotcoord, yrotcoord, radius, slices, x_gridpos, y_gridpos, edges, Area_grid):
    for ind in range(len(radius)):

        xtopind = int(np.ceil((xrotcoord[ind] + radius[ind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        xbotind = int(np.floor((xrotcoord[ind] - radius[ind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        ytopind = int(np.ceil((yrotcoord[ind] + radius[ind] - edges[3] - 0.5 * slices[1]) / slices[1]))
        ybotind = int(np.floor((yrotcoord[ind] - radius[ind] - edges[3] - 0.5 * slices[1]) / slices[1]))

        radius2 = radius[ind] ** 2

        for xind in range(xbotind, xtopind + 1):
            for yind in range(ybotind, ytopind + 1):
                atomdist = (x_gridpos[xind] - xrotcoord[ind]) ** 2 + (y_gridpos[yind] - yrotcoord[ind]) ** 2
                if atomdist <= radius2:
                    Area_grid[xind, yind] = 1

    return Area_grid


############################################################################################################################################

@njit(parallel=False, cache=True)
def find_circle_intersections_grid_n(locations, radii, grid_xmax, grid_ymax, grid_count, grid, contained, adjacency,
                                     adjacency_count, intersects, intersecting_pairs, intersect_count):
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    # iterate through cells
    for gx in range(grid_xmax):
        for gy in range(grid_ymax):
            current_key = (gx, gy)

            # iterate through neighbors
            for dx, dy in neighbor_offsets:
                neighbor_key = (gx + dx, gy + dy)

                # cell not in grid
                if neighbor_key[0] < 0 or neighbor_key[0] >= grid_xmax:
                    continue
                if neighbor_key[1] < 0 or neighbor_key[1] >= grid_ymax:
                    continue
                    # already checked
                if neighbor_key < current_key:
                    continue

                # iterate through each ball in cell
                for gb in range(grid_count[gx, gy]):
                    i = grid[gx, gy, gb]
                    if contained[i]:
                        continue

                    for gc in range(grid_count[neighbor_key[0], neighbor_key[1]]):
                        j = grid[neighbor_key[0], neighbor_key[1], gc]
                        if contained[j] or j == i:
                            continue
                        if current_key == neighbor_key and j <= i:
                            continue  # avoid duplicate pairs within same cell

                        c0, r0 = locations[i], np.asarray(radii[i]).item()
                        c1, r1 = locations[j], np.asarray(radii[j]).item()

                        dx, dy = c1 - c0
                        dist2 = dx * dx + dy * dy
                        sum_r = r0 + r1
                        diff_r = abs(r0 - r1)
                        if dist2 >= sum_r * sum_r:
                            continue  # no intersection
                        elif dist2 <= diff_r * diff_r:
                            # one inside the other
                            if r0 < r1:
                                contained[i] = True
                                adjacency[i, adjacency_count[i]] = j
                                adjacency_count[i] += 1
                            else:
                                contained[j] = True
                                adjacency[j, adjacency_count[j]] = i
                                adjacency_count[j] += 1
                            continue
                        else:
                            intersects[i] = True
                            intersects[j] = True
                            adjacency[i, adjacency_count[i]] = j
                            adjacency_count[i] += 1
                            adjacency[j, adjacency_count[j]] = i
                            adjacency_count[j] += 1
                            intersecting_pairs[intersect_count, :] = np.array([i, j])
                            intersect_count += 1

    return intersecting_pairs, intersect_count, intersects, adjacency, adjacency_count, contained


@njit(parallel=False, cache=True)
def compute_ubp(intersecting_pairs, intersect_count, contained, locations, radii, ubp, counts):
    n = radii.shape[0]

    # process all circle pairs to obtain intersection points
    for k in range(intersect_count):
        vals = intersecting_pairs[k, :]
        i = vals[0]
        j = vals[1]

        if contained[i] or contained[j]:
            continue

        c0, r0 = locations[i], radii[i]
        c1, r1 = locations[j], radii[j]
        d_vec = c1 - c0
        d2 = d_vec @ d_vec
        d = np.sqrt(d2)

        # intersection geometry
        a = (r0 ** 2 - r1 ** 2 + d2) / (2 * d)
        h = np.sqrt(r0 ** 2 - a ** 2)
        p2 = c0 + a * d_vec / d
        offset = h * np.array([-d_vec[1], d_vec[0]]) / d

        ubp[i, counts[i]] = p2 + offset
        counts[i] += 1
        ubp[i, counts[i]] = p2 - offset
        counts[i] += 1
        ubp[j, counts[j]] = p2 + offset
        counts[j] += 1
        ubp[j, counts[j]] = p2 - offset
        counts[j] += 1

    return ubp, counts


@njit(parallel=False, cache=True)
def process_arcs(adjacency_arr, neighbor_counts, ubp, counts, locations, radii, tol):
    n = radii.shape[0]
    tol_area = 0.0

    for i in range(n):
        cand_count = neighbor_counts[i]
        if cand_count == 0 or counts[i] == 0:
            continue

        candidates = adjacency_arr[i, :cand_count]
        pts = ubp[i, :counts[i]]

        # distance check for UBPs
        cand_locs = locations[candidates]
        cand_rads = radii[candidates].reshape(-1)
        dists_squared = np.sum((pts[:, None, :] - cand_locs[None, :, :]) ** 2, axis=2)  # shape: (k, m)
        decision = dists_squared < (cand_rads[None, :] - tol) ** 2

        inside_any = np.zeros(dists_squared.shape[0], dtype=np.bool_)
        for k in range(dists_squared.shape[0]):
            inside_any[k] = np.any(decision[k])

        pts_arr = pts[~inside_any]

        if pts_arr.shape[0] == 0:
            continue

        # convert UBPs to angles and sort
        c = locations[i]
        cx = locations[i, 0]
        r = radii[i]
        angles = np.arctan2(pts_arr[:, 1] - c[1], pts_arr[:, 0] - c[0]) % (2 * np.pi)
        angles.sort()

        # arc processing
        m = len(angles)
        for j in range(m):
            a1 = angles[j]
            a2 = angles[(j + 1) % m]

            if a2 <= a1:
                a2 += 2 * np.pi

            amid = (a1 + a2) / 2 % (2 * np.pi)
            mid_pt = c + r * np.array([np.cos(amid), np.sin(amid)])

            # check if arc midpoint is inside any other circle
            d2s = np.sum((cand_locs - mid_pt) ** 2, axis=1)
            if np.any(d2s < (cand_rads - tol) ** 2):
                continue

            # CCW line integral for arc
            t1 = r ** 2 / 2 * (a2 - a1)
            t2 = r ** 2 / 4 * (np.sin(2 * a2) - np.sin(2 * a1))
            t3 = cx * r * (np.sin(a2) - np.sin(a1))
            tol_area += np.asarray(t1 + t2 + t3).item()

    return tol_area


def greens_grid_n(locations, radii):
    n = radii.shape[0]
    total_area = 0.0

    ########                                    USER INPUTS                                    ########
    ###################################################################################################

    # tolerance for a point to be inside a circle
    tol = 1e-8

    # GUESS ABOUT HOW MANY CIRCLES PER CELL
    if n < 50:
        cell_max = n
    else:
        cell_max = n // 4

    # GUESS ABOUT HOW MANY INTERSECTIONS
    max_intersect = int(30 * n)

    # GUESS ABOUT HOW MANY PROPER INTERSECTIONS A GIVEN CIRCLE HAS
    if n < 50:
        max_neighbor = n
    else:
        max_neighbor = n // 5

    ###################################################################################################

    max_r = np.max(radii)
    cell_size = 2.0 * max_r

    x_min = np.min(locations[:, 0])
    y_min = np.min(locations[:, 1])
    x_max = np.max(locations[:, 0])
    y_max = np.max(locations[:, 1])

    grid_xmax = int((x_max - x_min) // cell_size) + 1
    grid_ymax = int((y_max - y_min) // cell_size) + 1

    grid = np.zeros((grid_xmax, grid_ymax, cell_max), dtype=int)
    grid_count = np.zeros((grid_xmax, grid_ymax), dtype=int)

    # populate grid
    for idx, (x, y) in enumerate(locations):
        gx = int((x - x_min) // cell_size)
        gy = int((y - y_min) // cell_size)
        grid[gx, gy, grid_count[gx, gy]] = idx
        grid_count[gx, gy] += 1

    intersecting_pairs = np.full((max_intersect, 2), -1, dtype=int)
    intersect_count = 0

    adjacency = np.full((n, max_neighbor), -1, dtype=int)
    adjacency_count = np.zeros(n, dtype=int)

    intersects = np.zeros(n, dtype=bool)
    contained = np.zeros(n, dtype=bool)

    intersecting_pairs, intersect_count, intersects, adjacency_arr, neighbor_count, contained = find_circle_intersections_grid_n(
        locations, radii, grid_xmax, grid_ymax,
        grid_count, grid, contained, adjacency,
        adjacency_count, intersects, intersecting_pairs,
        intersect_count)

    max_neighbors = np.max(neighbor_count)

    ubp = np.empty((n, 2 * max_neighbors, 2), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    ubp, counts = compute_ubp(intersecting_pairs, intersect_count, contained, locations, radii, ubp, counts)

    # add area of circles that are not fully contained and do not properly intersect
    intersects_arr = intersects | contained
    total_area += np.pi * np.sum(radii[~intersects_arr] ** 2)

    total_area += process_arcs(adjacency_arr, neighbor_count, ubp, counts, locations, radii, tol)

    return np.asarray(total_area).item()


############################################################################################################################################

class XLayersPDB_H:  # Please put the class name same as the function name
    def __init__(self, x=0.1, E=10.0, fname='./Data/1ubq.pdb', mpar={
        'Model': {'Layers': ['top', 'bottom'], 'd': [0.0, 1.0], 'rho': [0.0, 0.333], 'mu': [0.0, 0.0],
                  'sig': [0.0, 3.0]}},
                 dz=0.5, fslice=0.5, rrf=True, fix_sig=True, qoff=0.0, yscale=1, bkg=0.0,
                 theta=0.0, phi=0.0, protins=0.0, cov=0.1):
        """
        Calculates X-ray reflectivity from a system of multiple layers using Parratt formalism

        x      	 : array of wave-vector transfer along z-direction
        E      	 : Energy of x-rays in inverse units of x
        dz   	 : The thickness (Angstrom) of each layer for applying Parratt formalism
        fslice   : The thickness (Angstrom) of each layer of protein electron density
        rrf    	 : True for Frensnel normalized reflectivity and False for just reflectivity
        qoff   	 : q-offset to correct the zero q of the instrument
        yscale   : a scale factor for R or R/Rf
        bkg      : Incoherent background
        theta/phi: Euler angles of protein
        protins  : insertion of ED profile in Angs,
        cov      : coverage of ED profile in superposition with layers, number between 0 and 1.
        mpar  	 : Dictionary of Phases where,
                   Layers: Layer description,
                   d: thickness of each layer in Angs,
                   rho:Electron density of each layer in el/Angs^3,
                   mu: Absorption coefficient of each layer in 1/cm,
                   sig: roughness of interface separating each layer in Angs,
                   The upper and lower thickness should be always  fixed. The roughness of the topmost layer should be always kept 0.
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
        self.choices = {'rrf': [True, False], 'fix_sig': [True, False]}
        self.filepaths = {'fname': self.fname}

        self.__d__ = {}
        self.__rho__ = {}
        self.__mu__ = {}
        self.__sig__ = {}
        self.__fit__ = False
        self.__mkeys__ = list(self.__mpar__.keys())
        self.__fix_sig_changed__ = 0
        self.init_params()
        self.output_params = {'scaler_parameters': {}}
        xcoord, ycoord, zcoord, electrons, radius, mass = pdbread(self.fname)
        self.__protvol__ = self.protvolcalc(tuple(xcoord), tuple(ycoord), tuple(zcoord), tuple(radius), fslice)
        """self.__protpos__, self.__protrho__, self.__layarea__ = self.ED_profrot(tuple(xcoord), tuple(ycoord),
                                                                   tuple(zcoord), tuple(radius),
                                                                   tuple(electrons), self.theta, self.phi,
                                                                   self.fslice, self.__protvol__)"""

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
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
                        # if key!='sig':
                        #     self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=1e-10,max=np.inf,expr=None,brute_step=0.1)
                        # else:
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0, max=np.inf, expr=None, brute_step=0.05)

    @lru_cache(maxsize=16)
    def protvolcalc(self, xcoord, ycoord, zcoord, radius, fslice):
        xcoord = np.array(xcoord)
        ycoord = np.array(ycoord)
        zcoord = np.array(zcoord)
        radius = np.array(radius)

        xtop = np.max(xcoord + radius) + fslice
        xbot = np.min(xcoord - radius) - fslice
        ytop = np.max(ycoord + radius) + fslice
        ybot = np.min(ycoord - radius) - fslice
        ztop = np.max(zcoord + radius) + fslice
        zbot = np.min(zcoord - radius) - fslice

        edges = np.array([xtop, xbot, ytop, ybot, ztop, zbot])

        """Calculation of grid for volume contribution"""

        xgridnum = np.round((xtop - xbot) / fslice)
        ygridnum = np.round((ytop - ybot) / fslice)
        zgridnum = np.round((ztop - zbot) / fslice)

        xslice = (xtop - xbot) / xgridnum
        yslice = (ytop - ybot) / ygridnum
        zslice = (ztop - zbot) / zgridnum
        volslice = xslice * yslice * zslice

        slices = np.array([xslice, yslice, zslice, volslice])

        x_gridpos = xbot + xslice * (np.arange(1, xgridnum + 1) - 0.5)
        y_gridpos = ybot + yslice * (np.arange(1, ygridnum + 1) - 0.5)
        z_gridpos = zbot + zslice * (np.arange(1, zgridnum + 1) - 0.5)

        Vol_grid = np.zeros([int(xgridnum), int(ygridnum), int(zgridnum)], dtype=bool)
        Vol_grid = Vol_gridcalc(xcoord, ycoord, zcoord, radius, slices, x_gridpos, y_gridpos, z_gridpos, edges,
                                Vol_grid)
        protvol = volslice * np.sum(np.count_nonzero(Vol_grid))

        return protvol

    @lru_cache(maxsize=16)
    def ED_profrot(self, xcoord, ycoord, zcoord, radius, electrons, theta, phi, protins, d, fslice, protvol):
        """Move protein to center of mass"""

        coord = np.transpose(np.array([xcoord, ycoord, zcoord]))
        radius = np.array(radius)
        electrons = np.array(electrons)
        """CoM = np.sum(mass[:, np.newaxis]*coord, axis=0)/np.sum(mass)

        coordCoM = coord - CoM"""

        """Rotate coordinates"""

        rot = R.from_euler('zx', [[phi, theta]], degrees=True)

        rotcoord = rot.apply(coord)
        """Replace coord with coordCoM when done or make it an option"""

        """Calculation of edges of box that enclose protein structure"""

        xtop = np.max(rotcoord[:, 0] + radius) + fslice
        xbot = np.min(rotcoord[:, 0] - radius) - fslice
        ytop = np.max(rotcoord[:, 1] + radius) + fslice
        ybot = np.min(rotcoord[:, 1] - radius) - fslice
        ztop = np.max(rotcoord[:, 2] + radius) + fslice
        zbot = np.min(rotcoord[:, 2] - radius) - fslice

        edges = np.array([xtop, xbot, ytop, ybot, ztop, zbot])

        """Calculation of grid for volume contribution"""

        """xgridnum = np.round((xtop - xbot) / fslice)
        ygridnum = np.round((ytop - ybot) / fslice)
        zgridnum = np.round((ztop - zbot) / fslice)"""
        """xgridnum = int(np.ceil((xtop - xbot) / fslice))
        ygridnum = int(np.ceil((ytop - ybot) / fslice))"""
        zgridnum = int(np.ceil((ztop - zbot) / fslice))

        """xslice = (xtop - xbot) / xgridnum
        yslice = (ytop - ybot) / ygridnum
        zslice = (ztop - zbot) / zgridnum"""
        xslice = fslice
        yslice = fslice
        zslice = fslice
        volslice = xslice * yslice * zslice

        """slices = np.array([xslice, yslice, zslice, volslice])"""

        """x_gridpos = xbot + xslice * (np.arange(1, xgridnum + 1) - 0.5)
        y_gridpos = ybot + yslice * (np.arange(1, ygridnum + 1) - 0.5)
        z_gridpos = zbot + zslice * (np.arange(1, zgridnum + 1) - 0.5)"""

        """x and y grids are defined with respect to x,y = 0,0 so that 0,0
        is always on the edge of the grid."""

        """x_gridpos = xslice*(xbot//xslice + np.arange(1, xgridnum + 1) - 0.5)
        y_gridpos = yslice*(ybot//yslice + np.arange(1, ygridnum + 1) - 0.5)"""

        """z grid is defined with respect to z = 0 in protein-membrane system.
        Need to align protein with its insertion depth so grid encapsulates
        region where protein is found. Grid is then defined to the nearest
        fslice position so absolute 0 in the protein-membrane system would lie
        on the edge of an extended grid."""

        ztop_grid = - np.sum(d[:-1]) + protins
        zbot_grid = zbot - ztop + ztop_grid
        rotcoord[:, 2] += -ztop + ztop_grid
        z_gridpos = zslice * (zbot_grid // zslice + np.arange(1, zgridnum + 1) - 0.5)

        ElecFracsum = np.zeros(len(z_gridpos))
        Volfrac = np.zeros(len(z_gridpos))
        [SphereElec, Volfrac] = SphereElecFrac(rotcoord[:, 2], radius, electrons, z_gridpos, zslice, ElecFracsum,
                                               Volfrac)

        xyarea = greens_grid_n(rotcoord[:, :2], np.array(radius))
        area_foot = xyarea * fslice

        protpos = np.flipud(z_gridpos)
        protrho = np.flipud(SphereElec) / area_foot
        layarea = 1 - np.flipud(Volfrac) / xyarea

        return protpos, protrho, layarea

    @lru_cache(maxsize=16)
    def calcProfComb(self, d, rho, mu, sig, protins, cov, phase, dz, protpos, protrho, layarea):
        """
        Calculates the electron and absorption density superposition of user-provided
        files and the electron density profile in the uploaded file
        """
        d = np.array(d)
        rho = np.array(rho)
        mu = np.array(mu)
        sig = np.array(sig)
        n = len(d)
        protins = np.array(protins)
        cov = np.array(cov)
        protpos = np.array(protpos)
        protrho = np.array(protrho)
        layarea = np.array(layarea)
        protmu = np.zeros(protpos.shape)

        """zprot = -(protpos - protpos[0] - np.sum(d[:-1]) + protins)"""
        zprot = -protpos
        dprot = np.diff(-protpos)
        """protpos = np.cumsum(np.insert(-dprot, 0, -np.sum(d[:-1]) + protins))"""
        membpos = -1 * np.cumsum(d)
        dprot = np.append(dprot, -protpos[-1] + protpos[-2])

        """Need to combine two arrays to create one array of thicknesses and associated
        electron density along with sig and mu arrays"""

        protmembpos = np.flip(np.unique(np.concatenate((membpos[:-1], protpos))), -1)
        protmembpos = np.append(protmembpos, protmembpos[-1] - dprot[-1])
        m = len(protmembpos)

        protmembrho, protmembmu, protmembsig = protmembcomb(protmembpos, membpos, d, rho, mu, sig,
                                                            protpos, protrho, layarea, cov)

        zprotmemb = -protmembpos
        """dprotmemb = np.append(np.diff(-1 * protmembpos), d[-1])"""
        protmembrho = np.insert(protmembrho, 0, rho[0])
        protmembmu = np.insert(protmembmu, 0, mu[0])
        protmembsig = np.insert(protmembsig, 0, sig[0])

        maxsig = max(np.abs(np.max(protmembsig[1:])), 3)
        """Nlayers = int((protmembpos[0] - protmembpos[-1] + 10 * maxsig) / dz)
        halfstep = (np.sum(d[:-1]) + 10 * maxsig) / 2 / Nlayers"""
        Flayers = (protmembpos[0] - protmembpos[-1] + 10 * maxsig) / dz
        Nlayers = int(np.ceil(Flayers))
        halfstep = (np.sum(d[:-1]) + 10 * maxsig) / 2 / Flayers
        __z__ = np.arange(-protmembpos[0] - 5 * maxsig + halfstep, -protmembpos[-1] + 5 * maxsig - halfstep, dz)

        """__z__ = np.linspace(-protmembpos[0] - 5*maxsig + halfstep, -protmembpos[-1] + 5*maxsig - halfstep,
                            Nlayers)"""
        __d__ = np.diff(__z__)
        __d__ = np.append(__d__, [__d__[-1]])
        __rho__ = self.sldCalFun(tuple(-protmembpos), tuple(protmembrho), tuple(protmembsig), tuple(__z__))
        __mu__ = self.sldCalFun(tuple(-protmembpos), tuple(protmembmu), tuple(protmembsig), tuple(__z__))

        return n, __z__, __d__, __rho__, __mu__, zprot, protmu, zprotmemb, protmembrho, protmembmu

    @lru_cache(maxsize=10)
    def sldCalFun(self, pos, y, sigma, x):
        """wholesld = []"""
        pos = np.array(pos)
        y = np.array(y)
        x = np.array(x)
        sigma = np.array(sigma)

        """posterm = np.tile(np.append(pos, 2 * pos[-1] - pos[-2]), [len(x), 1])
        sigmaterm = np.tile(np.reciprocal(np.append(sigma[1:], sigma[-1])),[len(x), 1])
        yterm = np.tile(np.append(y[1:], y[-1]) - y, [len(x), 1])
        xterm = np.transpose(np.tile(x, [len(y), 1]))

        sldvec = (special.erf((xterm - posterm) * sigmaterm / math.sqrt(2)) * yterm)
        wholesld = (np.sum(sldvec[:, :-1], axis=1) + y[0] + y[-1]) / 2
        wholesld[wholesld < 0] = 0
        wholesld = tuple(wholesld)"""

        """wholesld = []
        for j in range(len(x)):
            sld = 0
            for i in range(len(pos) - 1):
                sld += math.erf((x[j] - pos[i]) / sigma[i + 1] / math.sqrt(2)) * (y[i + 1] - y[i])
            wholesld.append(max((sld + y[0] + y[-1]) / 2, 0))
            sldvec = (special.erf((x[j] - np.append(pos, 2 * pos[-1] - pos[-2])) *
                                  np.reciprocal(np.append(sigma[1:], sigma[-1])) /
                                  math.sqrt(2)) * (np.append(y[1:], y[-1]) - y))
            sld = np.sum(sldvec[:-1])"""

        wholesld = tuple(sldCalFun_numba(pos, y, sigma, x))
        return wholesld

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

    @lru_cache(maxsize=10)
    def py_parratt(self, x, lam, d, rho, mu):
        return parratt_numba(np.array(x), lam, np.array(d), np.array(rho), np.array(mu))

    # return parratt(np.array(x), lam, np.array(d), np.array(rho), np.array(mu))

    def update_parameters(self):
        for mkey in self.__mpar__.keys():
            # for key in self.__mpar__[mkey].keys():
            Nlayers = len(self.__mpar__[mkey]['d'])
            self.__d__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'd', i)].value for i in range(Nlayers)])
            self.__rho__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'rho', i)].value for i in range(Nlayers)])
            self.__mu__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'mu', i)].value for i in range(Nlayers)])
            sig = np.array([self.params['__%s_%s_%03d' % (mkey, 'sig', i)].value for i in range(Nlayers)])
            sig = np.where(sig < 1e-5, 1e-5, sig)
            self.__sig__[mkey] = tuple(sig)

    def y(self):
        """
        Define the function in terms of x to return some value
        """
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
                    for i in range(2, Nlayers):
                        self.params['__%s_%s_%03d' % (mkey, 'sig', i)].expr = '__%s_%s_%03d' % (mkey, 'sig', 1)
                    self.__fix_sig_changed__ = 1
                if not self.fix_sig and self.__fix_sig_changed__ > 0:
                    for i in range(2, Nlayers):
                        self.params['__%s_%s_%03d' % (mkey, 'sig', i)].expr = None
                    self.__fix_sig_changed__ = 0

        self.update_parameters()

        xcoord, ycoord, zcoord, electrons, radius, mass = pdbread(self.fname)
        self.__protvol__ = self.protvolcalc(tuple(xcoord), tuple(ycoord), tuple(zcoord), tuple(radius), self.fslice)
        self.__protpos__, self.__protrho__, self.__layarea__ = self.ED_profrot(tuple(xcoord), tuple(ycoord),
                                                                               tuple(zcoord), tuple(radius),
                                                                               tuple(electrons), self.theta, self.phi,
                                                                               self.protins, self.__d__[mkey],
                                                                               self.fslice, self.__protvol__)

        mkey = list(self.__mpar__.keys())[0]
        n, z, d, rho, mu, zprot, protmu, zprotmemb, protmembrho, protmembmu = self.calcProfComb(self.__d__[mkey],
                                                                                                self.__rho__[mkey],
                                                                                                self.__mu__[mkey],
                                                                                                self.__sig__[mkey],
                                                                                                self.protins, self.cov,
                                                                                                mkey, self.dz,
                                                                                                tuple(self.__protpos__),
                                                                                                tuple(self.__protrho__),
                                                                                                tuple(self.__layarea__))

        if not self.__fit__:
            tz, trho, tmu = self.stepFun(z[0], z[-1], self.__d__[mkey], self.__rho__[mkey], self.__mu__[mkey])
            pz, prho, pmu = self.protstepFun(zprot[0], zprot[-1], tuple(zprot + (zprot[-1] - zprot[-2])),
                                             tuple(self.__protrho__), tuple(protmu))
            ptz, ptrho, ptmu = self.protstepFun(z[0], z[-1], tuple(zprotmemb), tuple(protmembrho), tuple(protmembmu))
            self.output_params['%s_EDP' % self.__mkeys__[0]] = {'x': z, 'y': rho,
                                                                'names': ['z (Angs)', 'rho (el/Angs^3)'],
                                                                'plotType': 'step'}
            self.output_params['%s_dEDP' % self.__mkeys__[0]] = {'x': z[:-1], 'y': np.diff(rho) / self.dz,
                                                                 'names': ['z (Angs)', 'drho/dz (el/Angs^4)'],
                                                                 'plotType': 'step'}
            self.output_params['%s_step_layEDP' % self.__mkeys__[0]] = {'x': tz, 'y': trho,
                                                                        'names': ['z (Angs)', 'rho (el/Angs^3)'],
                                                                        'plotType': 'step'}
            self.output_params['%s_step_protEDP' % self.__mkeys__[0]] = {'x': pz, 'y': prho,
                                                                         'names': ['z (Angs)', 'rho (el/Angs^3)'],
                                                                         'plotType': 'step'}
            self.output_params['%s_step_EDP' % self.__mkeys__[0]] = {'x': ptz, 'y': ptrho,
                                                                     'names': ['z (Angs)', 'rho (el/Angs^3)'],
                                                                     'plotType': 'step'}
            self.output_params['%s_ADP' % self.__mkeys__[0]] = {'x': z, 'y': mu, 'names': ['z (Angs)', 'mu (1/cm)'],
                                                                'plotType': 'step'}
            self.output_params['%s_dADP' % self.__mkeys__[0]] = {'x': z[:-1], 'y': np.diff(mu) / self.dz,
                                                                 'names': ['z (Angs)', 'dmu/dz (1/cm^2)'],
                                                                 'plotType': 'step'}
            self.output_params['%s_step_ADP' % self.__mkeys__[0]] = {'x': tz, 'y': tmu,
                                                                     'names': ['z (Angs)', 'mu (1/cm)'],
                                                                     'plotType': 'step'}
        refq, r2 = self.py_parratt(tuple(x), lam, tuple(d), tuple(rho), tuple(mu))
        if self.rrf:
            rhos = (self.params['__%s_rho_000' % (mkey)].value, self.params['__%s_rho_%03d' % (mkey, n - 1)].value)
            mus = (0, 0)
            ref, r2 = self.py_parratt(tuple(x - self.qoff), lam, (0.0, 1.0), rhos, mus)
            refq = refq / ref
        return refq * self.yscale + self.bkg


if __name__ == '__main__':
    x = np.linspace(0.001, 1.0, 200)
    fun = XLayersPDB_H(x=x)
    print(fun.y())