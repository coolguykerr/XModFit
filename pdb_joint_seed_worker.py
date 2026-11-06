"""
Parallel worker for JOINT-fit XLayersPDB MCMC seeding (both-shared orientation).

The both-shared joint seeding scans a single shared (theta, phi) over the orientation
grid; at each cell it profiles out ALL the remaining master parameters -- including
any SHARED nuisance such as the insertion depth ``protins`` -- with a single
``least_squares`` over the FULL concatenated joint residual, so only orientations
consistent with the shared parameters survive. The cells are independent and each
re-profiles the shared params jointly, so distributing cells across processes yields
the SAME chi^2 map as the serial in-``Minimizer.emcee`` joint scan -- just faster.

This mirrors ``pdb_seed_worker`` (single model) but ships ALL per-dataset models +
data and reconstructs ``JointFit.residual``'s PLAIN (no-covariance) branch, reusing
the picklable machinery from ``pdb_joint_logprob`` (init_worker + per-curve
max-normalised residual). Covariance/GLS joint seeding stays serial (the whitened
residual needs the shipped Cholesky), matching the parallel-joint-sampling gating.
"""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# reuse the joint model-shipping + per-curve normalised residual
from pdb_joint_logprob import init_worker, _scale_residual_norm, _W
from pdb_joint_logprob import prepare_datas as _prepare   # (x,y,e[,imask[,yfit_masked]])

_G = {}  # per-worker: the scan objective + profiled start values


class JointSeedScan:
    """Picklable joint cost-scan objective. Reconstructs JointFit.residual (no-cov)
    over the master params, fixing the shared (theta, phi) per cell and profiling the
    rest. Models/data are shipped separately via ``init_worker`` (dropped on pickle)."""

    def __init__(self, var_names, param_map, fit_scale, master_params, priors,
                 theta_key='theta', phi_key='phi', models=None, datas=None):
        self.var_names = list(var_names)
        self.param_map = {k: list(v) for k, v in param_map.items()}
        self.fit_scale = fit_scale
        self.master_params = master_params           # lmfit Parameters (picklable)
        self.priors = dict(priors) if priors else {}
        self.theta_key = theta_key
        self.phi_key = phi_key
        # profiled = every varying master param except the fixed shared theta/phi
        self.prof_names = [k for k in self.var_names if k not in (theta_key, phi_key)]
        self._models = models
        self._datas = None
        self._fitsel = None
        if models is not None:
            self._datas, self._fitsel = _prepare(datas)
            init_worker(models, datas)

    def __getstate__(self):
        st = self.__dict__.copy()
        st['_models'] = None
        st['_datas'] = None
        st['_fitsel'] = None
        return st

    def _resolve(self):
        if self._models is not None:
            return self._models, self._datas, self._fitsel
        return _W['models'], _W['datas'], _W.get('fitsel') or [None] * len(_W['datas'])

    def start_vals(self):
        return np.array([self.master_params[k].value for k in self.prof_names], dtype=float)

    def residual(self, fvars, theta, phi):
        p = self.master_params
        for name, val in zip(self.prof_names, fvars):
            p[name].value = val
        p[self.theta_key].value = float(theta)
        p[self.phi_key].value = float(phi)
        p.update_constraints()
        models, datas, fitsel = self._resolve()
        # push master -> per-objective (mirror JointFit.push_params)
        for mkey, targets in self.param_map.items():
            v = p[mkey].value
            for (idx, lkey) in targets:
                mdl = models[idx]
                mdl.__dict__[lkey] = v
                lp = mdl.params[lkey]
                if lp.expr:
                    lp.expr = None
                lp.value = v
        parts = []
        for mdl, (x, y, e), sel in zip(models, datas, fitsel):
            fit = np.asarray(mdl.y()).ravel()
            if sel is not None:
                fit = fit[sel]
            parts.append(_scale_residual_norm(y, fit, e, self.fit_scale))
        resid = np.concatenate(parts)
        if self.priors:
            extra = [(p[k].value - mu) / sig for k, (mu, sig) in self.priors.items()]
            resid = np.concatenate([resid, np.asarray(extra, dtype=float)])
        return resid


def _init(scan, models, datas):
    init_worker(models, datas)              # applies imask (prepare_datas)
    _G['scan'] = scan
    _G['start'] = scan.start_vals()


def cell_cost(cell):
    """cell = (thetameshind, phimeshind, theta, phi) -> (ti, pi, cost=chi^2/2)."""
    ti, pi, theta, phi = cell
    scan = _G['scan']
    try:
        ret = least_squares(scan.residual, _G['start'], args=(float(theta), float(phi)))
        cost = float(ret.cost)
        if not np.isfinite(cost):
            cost = np.inf
    except Exception:
        cost = np.inf
    return (ti, pi, cost)


def map_cell_costs(cells, scan, models, datas, workers):
    """Return [(ti, pi, cost), ...] aligned to ``cells``. Spawn Pool when workers>1.

    ``datas`` entries are ``(x, y, e)`` or ``(x, y, e, imask, yfit_masked)``; see
    ``_prepare``."""
    if workers is None or workers <= 1:
        _init(scan, models, datas)
        return [cell_cost(c) for c in cells]
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    chunk = max(1, len(cells) // (workers * 4))
    with ctx.Pool(workers, initializer=_init, initargs=(scan, models, datas)) as pool:
        return pool.map(cell_cost, cells, chunksize=chunk)
