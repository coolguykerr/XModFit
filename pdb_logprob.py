"""
Picklable log-posterior for XLayersPDB* MCMC sampling, enabling multi-core emcee.

The stock objective (Minimizer._lnprob -> Fit.residual) is bound to a Fit QObject,
which cannot be pickled, so multiprocessing.Pool.map crashes at cores>1. This module
provides a self-contained, picklable objective that ships only the model (which
pickles cheaply) plus the sliced data, and reproduces _lnprob's vector-residual
branch exactly:

    lnprob = -0.5 * sum(resid**2)                       (weighted residual)
    lnprob = -0.5 * sum((resid/exp(lnsigma))**2 + c)    (unweighted + __lnsigma)

with an out-of-bounds prior of -inf and non-finite -> -1e100, matching
lmfit/minimizer.py `_lnprob`. No sin(theta) prior is applied (that branch of
_lnprob only fires for scalar objectives and is dead for the XRR residual).

The model is placed in each worker's globals once via `init_worker` (a Pool
initializer), so the per-evaluation objective stays tiny and is not re-shipped
each step. For serial/testing use, a model may be embedded directly.
"""
import os
import sys
import numpy as np

# Make the repo root importable in spawned workers so the pickled model's class
# (Functions.XRR.XLayersPDB) can be unpickled.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Per-process worker state (populated by init_worker in Pool workers).
_W = {}


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


def apply_mask(x, y, err, mask=None, yfit_masked=False):
    """Apply the frozen window mask (Fit.imask_for) in Fit.residual's order.

    x/y/err are the [imin, imax+1] window. Returns (x_eval, y, err, fitsel): the
    model is evaluated on x_eval (the whole window, or only the active points for
    a __masked_x__ model, ``yfit_masked``), y/err are the active points, and
    fitsel is the selection still to apply to the model output (None when the
    model was already given only the active points, or when there is no mask).
    mask=None returns the arrays unchanged."""
    x, y, err = np.asarray(x), np.asarray(y), np.asarray(err)
    if mask is None:
        return x, y, err, None
    mask = np.asarray(mask, bool)[:len(y)]
    y, err = y[mask], err[mask]
    if yfit_masked:
        return x[mask[:len(x)]], y, err, None
    return x, y, err, mask


def init_worker(model, x, y, err, mask=None, yfit_masked=False):
    """Pool initializer: stash the model + sliced data in this worker's globals.
    ``mask``/``yfit_masked``: see apply_mask (A1.18)."""
    x, y, err, fitsel = apply_mask(x, y, err, mask, yfit_masked)
    model.x = x                 # evaluate() slices func.x to the fit range
    model.__fit__ = True        # matches Fit.residual; skips output_params work
    _W['model'] = model
    _W['x'] = x
    _W['y'] = y
    _W['err'] = err
    _W['fitsel'] = fitsel


class PdbLogProb:
    """Picklable emcee log-probability callable.

    Holds only lightweight config; the (heavier) model + data live in worker
    globals set by init_worker. For serial use, pass model/x/y/err directly.
    """

    def __init__(self, var_names, bounds, fit_scale, is_weighted,
                 has_lnsigma=False, lnsigma_name='__lnsigma',
                 model=None, x=None, y=None, err=None, mask=None, yfit_masked=False):
        self.var_names = list(var_names)
        self.bounds = np.asarray(bounds, dtype=float)
        self.fit_scale = fit_scale
        self.is_weighted = bool(is_weighted)
        self.has_lnsigma = bool(has_lnsigma)
        self.lnsigma_name = lnsigma_name
        # Optional embedded model/data (serial or unit-test use). Excluded from
        # pickling so pool workers use their own init_worker-provided copy.
        self._model = model
        self._x = None if x is None else np.asarray(x)
        self._y = None if y is None else np.asarray(y)
        self._err = None if err is None else np.asarray(err)
        self._fitsel = None
        if x is not None and y is not None and err is not None:
            self._x, self._y, self._err, self._fitsel = apply_mask(
                x, y, err, mask, yfit_masked)
        if self._model is not None:
            self._model.x = self._x
            self._model.__fit__ = True

    def __getstate__(self):
        # Do not pickle the embedded model/data; workers resolve from globals.
        state = self.__dict__.copy()
        state['_model'] = None
        state['_x'] = state['_y'] = state['_err'] = None
        state['_fitsel'] = None
        return state

    def _resolve(self):
        if self._model is not None:
            return self._model, self._x, self._y, self._err, self._fitsel
        return _W['model'], _W['x'], _W['y'], _W['err'], _W.get('fitsel')

    def __call__(self, theta):
        theta = np.asarray(theta)
        # Uniform prior: out of bounds -> -inf (compare theta to bounds, not params).
        if np.any(theta > self.bounds[:, 1]) or np.any(theta < self.bounds[:, 0]):
            return -np.inf
        model, x, y, err, fitsel = self._resolve()
        p = model.params
        for name, val in zip(self.var_names, theta):
            setattr(model, name, val)   # y() reads qoff/protins/cov/theta/phi as attrs
            p[name].value = val
        p.update_constraints()
        fit = np.asarray(model.y()).ravel()
        if fitsel is not None:
            fit = fit[fitsel]
        resid = np.asarray(_scale_residual(y, fit, err, self.fit_scale)).ravel()
        if self.has_lnsigma and not self.is_weighted:
            ls = p[self.lnsigma_name].value
            c = np.log(2 * np.pi) + 2 * ls
            val = -0.5 * np.sum((resid / np.exp(ls)) ** 2 + c)
        else:
            val = -0.5 * np.sum(resid * resid)
        if not np.isfinite(val):
            return -1.e100
        return float(val)
