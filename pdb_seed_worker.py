"""
Parallel worker for XLayersPDB MCMC seeding (Phase-1 orientation chi^2 grid scan).

Each grid cell fixes (theta, phi) and profiles out the remaining ("nuisance")
parameters with an unconstrained scipy least_squares, returning the least_squares
"cost" (0.5 * sum(residual**2)). This reproduces the serial seeding objective
exactly (verified: cost matches chi-square/2 of the known least-squares minima).

Because the scan is deterministic and the cells are independent, distributing
them across processes yields a bit-identical chi^2 map. Used only when the caller
requests workers > 1; the serial path in the caller is left untouched.

Design notes:
  * Only the model (Functions.XRR.XLayersPDB instance) is shipped to workers -- it
    pickles cheaply (~25 KB) and, unlike the Fit object, carries no Qt signal.
  * A spawn-context Pool re-imports the launching module as __mp_main__, so the
    XModFit GUI (guarded by `if __name__=='__main__':`) does NOT start in workers.
  * The residual mirrors Fit.residual (Fit_Routines.py) for a single dataset.
"""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

# Make the repo root (this file's directory) importable in spawned workers so the
# pickled model's class (Functions.XRR.XLayersPDB) can be unpickled there.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pdb_logprob import apply_mask   # shared with the sampling objective (A1.3/A1.18)

# Per-process worker state, populated by _init().
_G = {}


def _scale_residual(y, fit, err, fit_scale):
    """Mirror Fit.residual's single-(non-dict)-dataset residual for each scale."""
    if fit_scale == 'Log':
        return (np.log(y) - np.log(fit)) * y / err
    elif fit_scale == 'Log w/o error':
        return np.log10(y) - np.log10(fit)
    elif fit_scale == 'Linear w/o error':
        return y - fit
    else:  # 'Linear'
        return (y - fit) / err


def _init(func, x, y, err, fit_scale, nuisance_names, start_vals, mask=None,
          yfit_masked=False):
    # The frozen window mask (Fit.imask_for), applied in Fit.residual's order:
    # evaluate on the whole window x[imin:imax+1], THEN select the active points of
    # fit, y and err. A __masked_x__ model (yfit_masked) is handed only the active
    # points instead, so its output is already active-only and is not masked again.
    x, y, err, fitsel = apply_mask(x, y, err, mask, yfit_masked)
    func.x = x                     # evaluate() slices func.x to the fit range
    func.__fit__ = True             # matches Fit.residual; skips output_params work
    _G['func'] = func
    _G['y'] = y
    _G['err'] = err
    _G['fitsel'] = fitsel
    _G['fit_scale'] = fit_scale
    _G['nuisance_names'] = list(nuisance_names)
    _G['start_vals'] = np.asarray(start_vals, dtype=float)


def _cell_residual(fvars, theta, phi):
    func = _G['func']
    p = func.params
    # Set nuisance params (both the lmfit param and the attribute, as Fit.residual does,
    # since y() reads qoff/protins/cov/... as attributes).
    for name, val in zip(_G['nuisance_names'], fvars):
        setattr(func, name, val)
        p[name].value = val
    setattr(func, 'theta', float(theta)); p['theta'].value = float(theta)
    setattr(func, 'phi', float(phi));     p['phi'].value = float(phi)
    p.update_constraints()
    fit = np.asarray(func.y()).ravel()
    if _G.get('fitsel') is not None:
        fit = fit[_G['fitsel']]
    return _scale_residual(_G['y'], fit, _G['err'], _G['fit_scale'])


def cell_cost(cell):
    """cell = (thetameshind, phimeshind, theta, phi) -> (ti, pi, cost)."""
    ti, pi, theta, phi = cell
    try:
        ret = least_squares(_cell_residual, _G['start_vals'], args=(float(theta), float(phi)))
        cost = float(ret.cost)
        if not np.isfinite(cost):
            cost = np.inf
    except Exception:
        # Serial code would crash here; returning inf makes the cell contribute
        # zero likelihood instead of aborting the whole seed. Rare in practice.
        cost = np.inf
    return (ti, pi, cost)


def map_cell_costs(cells, func, x, y, err, fit_scale, nuisance_names, start_vals, workers,
                   mask=None, yfit_masked=False):
    """Return a list of (ti, pi, cost). Uses a spawn Pool when workers > 1.

    x/y/err are the [imin, imax+1] window. ``mask`` is the window-restricted frozen
    column mask (``Fit.imask_for()`` / a joint objective's ``imask``) or None;
    ``yfit_masked`` says the model is a ``__masked_x__`` one that is evaluated on
    the active points only. mask=None is the unmasked scan, unchanged."""
    if workers is None or workers <= 1:
        _init(func, x, y, err, fit_scale, nuisance_names, start_vals, mask, yfit_masked)
        return [cell_cost(c) for c in cells]
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    chunk = max(1, len(cells) // (workers * 4))
    with ctx.Pool(workers, initializer=_init,
                  initargs=(func, x, y, err, fit_scale, nuisance_names, start_vals,
                            mask, yfit_masked)) as pool:
        return pool.map(cell_cost, cells, chunksize=chunk)
