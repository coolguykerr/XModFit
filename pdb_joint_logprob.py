"""
Picklable log-posterior for PARALLEL joint-fit MCMC (no-covariance branch only).

The joint objective (JointFit.residual) is bound to a QObject spanning multiple
models, so multiprocessing.Pool cannot pickle it. This module provides a
self-contained, picklable objective that ships the per-dataset models + data and
reproduces JointFit.residual's PLAIN (no-covariance) branch exactly:

  * push_params: fan each master value out to every (obj, local_key) endpoint,
    writing both the scalar attribute and the lmfit param (clearing dynamic expr),
    mirroring Joint_Fit_Routines.push_params.
  * per-curve max-normalisation then (data-fit)/err (or Log/w-o-error variants),
    concatenated across datasets -- byte-identical to Joint_Fit_Routines.residual.
  * optional Gaussian-prior terms appended: (value - mu)/sigma.
  * reduction: lnprob = -0.5 * sum(resid**2)  (or the __lnsigma variant).

The COVARIANCE/GLS branch is intentionally NOT handled here -- covariance joint
MCMC stays serial until the whitened residual is validated bit-for-bit against a
covariance dataset. JointFit.set_emcee_minimizer only wires this objective when
_use_cov is False.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_W = {}  # per-worker state


def _scale_residual_norm(y, fit, err, fit_scale):
    """Per-curve max-normalised residual, matching Joint_Fit_Routines.residual."""
    maxd = np.max(y)
    y = y / maxd
    err = err / maxd
    fit = fit / maxd
    if fit_scale == 'Log':
        return (np.log(y) - np.log(fit)) * y / err
    elif fit_scale == 'Log w/o error':
        return np.log10(y) - np.log10(fit)
    elif fit_scale == 'Linear w/o error':
        return y - fit
    else:  # 'Linear'
        return (y - fit) / err


def prepare_datas(datas):
    """Split ``datas`` entries ``(x, y, e[, imask[, yfit_masked]])`` -- x/y/e the
    [imin, imax+1] window, imask the objective's frozen window mask -- into the
    ``(x, y, e)`` the models are evaluated and compared on, plus the per-dataset
    fit-side selection. JointFit.residual's order: evaluate on the window, then
    select the active points of fit, y and err, BEFORE the max-normalisation. A
    ``__masked_x__`` model (yfit_masked) gets only the active x, so its output is
    already active-only and is not selected again. imask None: unchanged."""
    plain, fitsel = [], []
    for d in datas:
        x, y, e = np.asarray(d[0]), np.asarray(d[1]), np.asarray(d[2])
        m = d[3] if len(d) > 3 else None
        ym = bool(d[4]) if len(d) > 4 else False
        sel = None
        if m is not None:
            m = np.asarray(m, bool)[:len(y)]
            y, e = y[m], e[m]
            if ym:
                x = x[m[:len(x)]]
            else:
                sel = m
        plain.append((x, y, e))
        fitsel.append(sel)
    return plain, fitsel


def init_worker(models, datas):
    """Pool initializer: stash per-dataset models + sliced (x, y, yerr) in globals.

    models : list of model instances (e.g. XLayersPDB), one per dataset.
    datas  : list of (x_slice, y_slice, yerr_slice) already sliced to [imin, imax],
             optionally extended to (..., imask, yfit_masked); see prepare_datas.
    """
    plain, fitsel = prepare_datas(datas)
    for mdl, (x, y, e) in zip(models, plain):
        mdl.x = x
        mdl.__fit__ = True
    _W['models'] = models
    _W['datas'] = plain
    _W['fitsel'] = fitsel


class JointLogProb:
    """Picklable emcee log-probability for the no-covariance joint objective."""

    def __init__(self, var_names, bounds, fit_scale, is_weighted, param_map, priors,
                 master_params, has_lnsigma=False, lnsigma_name='__lnsigma',
                 models=None, datas=None):
        self.var_names = list(var_names)
        self.bounds = np.asarray(bounds, dtype=float)
        self.fit_scale = fit_scale
        self.is_weighted = bool(is_weighted)
        # param_map: {master_key: [(obj_idx, local_key), ...]}
        self.param_map = {k: list(v) for k, v in param_map.items()}
        self.priors = dict(priors) if priors else {}
        self.master_params = master_params        # lmfit Parameters (picklable)
        self.has_lnsigma = bool(has_lnsigma)
        self.lnsigma_name = lnsigma_name
        # embedded models/data for serial/unit-test use (excluded from pickling)
        self._models = models
        self._datas = None
        self._fitsel = None
        if models is not None:
            self._datas, self._fitsel = prepare_datas(datas)
            init_worker(models, datas)            # also warms serial use via _W

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

    def __call__(self, theta):
        theta = np.asarray(theta)
        if np.any(theta > self.bounds[:, 1]) or np.any(theta < self.bounds[:, 0]):
            return -np.inf
        models, datas, fitsel = self._resolve()
        p = self.master_params
        for name, val in zip(self.var_names, theta):
            p[name].value = val
        p.update_constraints()
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
        # evaluate + concatenate per-curve normalised residuals
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
        if self.has_lnsigma and not self.is_weighted:
            ls = p[self.lnsigma_name].value
            c = np.log(2 * np.pi) + 2 * ls
            val = -0.5 * np.sum((resid / np.exp(ls)) ** 2 + c)
        else:
            val = -0.5 * np.sum(resid * resid)
        if not np.isfinite(val):
            return -1.e100
        return float(val)
