"""
joint_fit_stats.py -- WS2 joint-fit reporting: snapshot, per-dataset split, exact
Gaussian likelihood and objective labels (spec handoff/20-spec-ws2-reporting.md §3).

Qt-free and model-free after ``snapshot``: everything downstream of the snapshot reads
only the ``JointSnapshot`` (spec §3.1). ``snapshot`` itself evaluates the live
``JointFit`` once (one ``evaluate()`` at θ̂ serves the fitted residual, the raw parts and,
in model mode, the model blocks), inside ``JointFit.quiet()``, so it neither counts
``Niter`` nor emits ``functionCalled`` and leaves every model as it was.

Model evaluations (one evaluation = every dataset's model once): ``snapshot`` 1;
``exact_objective`` 1 per call; ``fisher_shares`` nvarys; ``optimum_distance`` 2·nvarys
(only when C depends on the model). ``diagnostics`` costs those two plus 1 (the final
evaluation at ``result.params`` that leaves the model outputs at the best fit) with
``JointFit.report_diagnostics = 'full'`` (the default), and none with ``'cheap'`` or
after a fit stopped by the user. Report text is ASCII; Unicode appears only in docstrings
and comments.

The split (§3.2): with L = chol(C(θ̂)) over the concatenated used rows in fit order,
``z = L⁻¹ r``. The sequential term of dataset i is ``‖z_i‖²`` (χ² of i given the
datasets before it; they sum exactly to the data part of the objective), the marginal
term is ``r_iᵀ (L Lᵀ)_ii⁻¹ r_i`` (i scored under its own block). The log-determinant
splits the same way row by row (§3.5): ``ln det C = 2 Σ ln L_jj``.

This module must stay out of ``Functions/<category>/`` (CLAUDE.md rule 1).
"""
import contextlib
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import covariance_core as cc

LN2PI = float(np.log(2.0 * np.pi))

_DATA_CLAUSE = '; covariance built from the measured data; approximate'
_FROZEN_CLAUSE = '; sampled with C frozen at the prior fit, not the exact likelihood'
LABEL_MCMC = 'at posterior medians'
LABEL_ABORTED = 'aborted fit; not the minimised value'


# ----------------------------------------------------------------- records --
@dataclass(frozen=True)
class SnapDataset:
    name: str
    x: np.ndarray                    # active rows
    y: np.ndarray
    yerr: np.ndarray
    active_index: np.ndarray
    index_hash: str
    window: Tuple[float, float]
    rows: Tuple[int, int]            # half-open, in the residual (priors excluded)
    used: np.ndarray                 # bool over active rows (the covariance mask)
    r: np.ndarray                    # unwhitened residual on used rows
    f: np.ndarray                    # model on used rows
    fit_intensity: Optional[np.ndarray]   # C the fit was scored under (cov path)
    intensity: Optional[np.ndarray]       # C at θ̂ (cov path)
    cov: Optional[cc.DatasetCovariance]   # as used, on used rows
    cov_source: Optional[str]
    cov_hash: Optional[str]
    model_class: str
    model_files: dict


@dataclass(frozen=True)
class JointSnapshot:
    datasets: tuple
    fit_scale: str
    cov_in_use: bool
    cov_mode: Optional[str]
    z: np.ndarray                    # data rows whitened under C(θ̂)
    z_fit: np.ndarray                # data rows of the residual the fit minimised
    logdiag: np.ndarray              # ln diag(L_exact) (no cov: ln σ); empty unweighted
    prior: np.ndarray
    jitter: float
    method: str
    nvarys: int
    ndata: int
    chisqr: float
    params: Dict[str, float]
    vary: Dict[str, bool]
    stderr: Dict[str, Optional[float]]
    covar: Optional[np.ndarray]
    var_names: tuple
    param_map: dict
    warnings: tuple
    label: Optional[str]
    rescale: Optional[dict] = None   # Task 5 defines its keys
    notes: tuple = ()                # model_blocks substitution notes (model mode)


@dataclass
class DatasetTerm:
    name: str
    n_points: int
    sequential: float
    marginal: float
    chi2_per_point: Optional[float]
    logdet_seq: Optional[float]
    logdet_marg: Optional[float]
    nll2_seq: Optional[float]
    nll2_marg: Optional[float]
    window: Tuple[float, float]
    index_hash: str
    scale: str
    cov: bool
    cov_mode: Optional[str]
    rescale_s: Optional[float] = None


# ----------------------------------------------------------------- helpers --
def _hash(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    return h.hexdigest()


def _cov_hash(cov):
    """sha1 over every term's kind, scope, labels, cov and weights, and sigma_meas
    (never the intensities)."""
    if cov is None:
        return None
    h = hashlib.sha1()
    for t in cov.terms:
        h.update(repr((t.kind, t.scope, None if t.labels is None else list(t.labels)))
                 .encode('utf-8'))
        h.update(np.ascontiguousarray(np.asarray(t.cov, float)).tobytes())
        if t.weights is not None:
            h.update(np.ascontiguousarray(np.asarray(t.weights, float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(cov.sigma_meas, float)).tobytes())
    return h.hexdigest()


def _cov_path(jf, fit_scale):
    return bool(jf._use_cov) and fit_scale in ('Linear', 'Log')


def _weighted(fit_scale):
    return 'w/o' not in fit_scale


def _jitter_of(C, fwarn):
    if any('jitter' in w for w in fwarn):
        import joint_covariance as jcov
        return float(jcov.cholesky_with_jitter(C)[1])
    return 0.0


def _used_masks(jf, cov_path):
    """Per objective, the bool over its active rows that the residual keeps."""
    out, off = [], 0
    for k in range(len(jf.objectives)):
        n = len(jf.active_index(k))
        if cov_path:
            out.append(np.asarray(jf._cov_mask[off:off + n], bool).copy())
        else:
            out.append(np.ones(n, bool))
        off += n
    return out


# ---------------------------------------------------------------- raw parts --
def _evaluate_at(jf, params):
    """One quiet ``evaluate()`` at ``params``: ``(yfits, yfit_masked flags)``."""
    with jf.quiet():
        jf.push_params(params)
        yfits = jf.evaluate()
        flags = [bool(o.get("yfit_masked")) for o in jf.objectives]
    return yfits, flags


def raw_parts(jf, params, evaluated=None):
    """Per dataset ``(r_i, f_i)`` on its used rows at ``params``, unwhitened, assembled
    as ``JointFit.residual`` assembles them (slice, then imask -- honouring
    ``yfit_masked`` -- then the dataset's ``_cov_mask`` segment on the covariance path),
    evaluated inside ``quiet()`` (one model evaluation), or from ``evaluated`` =
    ``_evaluate_at(jf, params)`` (none). ``r`` is d - f (Linear), ln d - ln f (Log), and
    for the unweighted scales the residual rows themselves."""
    scale = jf._fit_scale
    cov_path = _cov_path(jf, scale)
    used = _used_masks(jf, cov_path)
    yfits, flags = _evaluate_at(jf, params) if evaluated is None else evaluated
    out = []
    for k, (obj, fit) in enumerate(zip(jf.objectives, yfits)):
        i0, i1 = obj["imin"], obj["imax"] + 1
        data = np.asarray(obj["y"], float)[i0:i1]
        fit = np.asarray(fit, float)
        im = obj.get("imask")
        if im is not None:
            data = data[im]
            if not flags[k]:
                fit = fit[im]
        u = used[k]
        data, fit = data[u], fit[u]          # drop masked rows BEFORE any log
        if scale == 'Log':
            r = np.log(data) - np.log(fit)
        elif scale == 'Log w/o error':
            r = np.log10(data) - np.log10(fit)
        elif scale == 'Linear w/o error':
            r = (data - fit) / np.max(data) if len(data) else data - fit
        else:
            r = data - fit
        out.append((r, fit))
    return out


# ----------------------------------------------------------------- snapshot --
def snapshot(jf):
    """The immutable record of a finished joint fit (spec §3.1); see module doc."""
    res = jf.result
    if res is None:
        raise RuntimeError('snapshot needs a finished fit (jf.result is None)')
    params = res.params
    scale = jf._fit_scale
    if scale is None:
        raise RuntimeError('snapshot needs a perform_fit (no fit scale recorded)')
    cscale = 'Log' if scale == 'Log' else 'Linear'
    cov_path = _cov_path(jf, scale)
    info = jf.cov_info or {}
    cov_mode = info.get('mode', jf.covariance_mode) if cov_path else None

    # (1) the residual the fit minimised, quietly, at the reported parameters; one
    # evaluation serves it, the raw parts and the model blocks
    with jf.quiet():
        jf.push_params(params)
        yfits = jf.evaluate()
        flags = [bool(o.get("yfit_masked")) for o in jf.objectives]
        full = np.asarray(jf._assemble_residual(params, scale, yfits), float)
    evaluated = (yfits, flags)
    prior = np.asarray(jf.prior_rows(params), float)
    z_fit = full[:len(full) - len(prior)].copy()

    parts = raw_parts(jf, params, evaluated)
    ranges = jf.row_ranges()
    used = _used_masks(jf, cov_path)

    fit_blocks = exact_blocks = None
    block_notes = ()
    if cov_path:
        fit_blocks = list(jf._last_blocks)
        exact_blocks = fit_blocks
        if cov_mode == 'model':
            exact_blocks, _notes = jf.model_blocks(params, scale, evaluated)
            block_notes = tuple(str(n) for n in (_notes or ()))
    needs = cov_path and cov_mode == 'model' and cc.needs_intensity(fit_blocks, cscale)
    # opt-in error rescaling (§6.3): the residual the fit minimised was r_i / s_i whitened by
    # the unscaled L, i.e. whitened by chol(S C S) = S L; z and logdiag follow suit
    s_use = getattr(jf, 'rescale_s', None)
    s_use = None if s_use is None else [float(v) for v in s_use]
    if s_use is not None and not _weighted(scale):
        s_use = None
    sizes = [int(b - a) for (a, b) in ranges]
    row_s = None if s_use is None else np.repeat(np.asarray(s_use, float), sizes)

    # (2) z and logdiag under C(θ̂)
    if not cov_path:
        z = z_fit
        jitter = 0.0
        if _weighted(scale):
            sig = []
            for k, obj in enumerate(jf.objectives):
                idx = jf.active_index(k)
                e = np.asarray(obj["yerr"], float)[idx]
                if scale == 'Log':
                    e = e / np.asarray(obj["y"], float)[idx]
                sig.append(e)
            logdiag = np.log(np.concatenate(sig)) if sig else np.zeros(0)
            if row_s is not None and len(logdiag) == len(row_s):
                logdiag = logdiag + np.log(row_s)       # sigma -> sigma * s_i
        else:
            logdiag = np.zeros(0)
    elif needs:
        # the one case whose C(θ̂) differs from the C the fit used: factorise afresh
        C, _aw = cc.assemble(exact_blocks, cscale)
        L, fwarn = cc.factorize(C)
        jitter = _jitter_of(C, fwarn)
        rcat = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
        z = cc.whiten(L, rcat if row_s is None else rcat / row_s)
        logdiag = np.log(np.diag(L))
        if row_s is not None:
            logdiag = logdiag + np.log(row_s)
    else:
        z = z_fit
        logdiag = np.log(np.diag(jf._L))
        if row_s is not None:
            logdiag = logdiag + np.log(row_s)
        jitter = float(jf._jitter)

    datasets = []
    for k, obj in enumerate(jf.objectives):
        idx = np.asarray(jf.active_index(k))
        func = obj["func"]
        xv = np.asarray(obj["x"], float)
        r, f = parts[k]
        if cov_path:
            cov = fit_blocks[k].cov
            fit_int = np.asarray(fit_blocks[k].intensity, float).copy()
            inten = np.asarray(exact_blocks[k].intensity, float).copy()
        else:
            cov, fit_int, inten = None, None, None
        datasets.append(SnapDataset(
            name=obj["name"], x=xv[idx].copy(), y=np.asarray(obj["y"], float)[idx].copy(),
            yerr=np.asarray(obj["yerr"], float)[idx].copy(), active_index=idx.copy(),
            index_hash=_hash(idx),
            window=(float(xv[obj["imin"]]), float(xv[obj["imax"]])),
            rows=(int(ranges[k][0]), int(ranges[k][1])), used=used[k],
            r=np.asarray(r, float).copy(), f=np.asarray(f, float).copy(),
            fit_intensity=fit_int, intensity=inten, cov=cov,
            cov_source=None if cov is None else (cov.source or None),
            cov_hash=_cov_hash(cov), model_class=type(func).__name__,
            model_files=dict(getattr(func, 'filepaths', {}) or {})))

    label = None
    if jf._fit_method == 'emcee':
        label = LABEL_MCMC
        if info.get('mcmc_frozen') == 'model':
            label += _FROZEN_CLAUSE
    elif info.get('aborted'):
        label = LABEL_ABORTED

    rescale = None
    if s_use is not None:
        info = getattr(jf, 'rescale_info', None) or {}
        rescale = {'mode': jf.rescale_errors, 'inflate_only': bool(jf.rescale_inflate_only),
                   's': tuple(s_use), 'nu': tuple(float(v) for v in info.get('nu', ())),
                   'rounds': int(info.get('rounds', 0)), 'converged': bool(info.get('converged', False))}
    covar = getattr(res, 'covar', None)
    chisqr = getattr(res, 'chisqr', None)
    if chisqr is None:
        # an aborted minimisation carries no statistics: the recomputed objective at
        # the reported parameters stands in (and is labelled as such above)
        chisqr = float(np.sum(z_fit ** 2) + np.sum(prior ** 2))
    chisqr = float(chisqr)
    return JointSnapshot(
        datasets=tuple(datasets), fit_scale=scale, cov_in_use=cov_path, cov_mode=cov_mode,
        z=np.asarray(z, float).copy(), z_fit=z_fit, logdiag=np.asarray(logdiag, float),
        prior=prior, jitter=float(jitter), method=str(jf._fit_method),
        nvarys=int(getattr(res, 'nvarys', 0) or 0), ndata=int(getattr(res, 'ndata', len(full))),
        chisqr=chisqr,
        params={k: float(p.value) for k, p in params.items()},
        vary={k: bool(p.vary) for k, p in params.items()},
        stderr={k: (None if p.stderr is None else float(p.stderr)) for k, p in params.items()},
        covar=None if covar is None else np.array(covar, float),
        var_names=tuple(getattr(res, 'var_names', None) or ()),
        param_map={k: tuple((int(i), str(lk)) for (i, lk) in v) for k, v in jf.param_map.items()},
        warnings=tuple(str(w) for w in jf.warnings), label=label, rescale=rescale,
        notes=block_notes)


# --------------------------------------------------------------- rebuild L --
def _snap_blocks(snap, which='exact'):
    out = []
    for d in snap.datasets:
        inten = d.intensity if which == 'exact' else d.fit_intensity
        out.append(cc.Block(d.cov, d.y[d.used], d.yerr[d.used], inten, name=d.name))
    return out


def rebuild_L(snap, which='exact'):
    """The Cholesky factor of C over the used rows, rebuilt from the snapshot:
    ``which='exact'`` at the fitted model (``intensity``), ``'fit'`` the one the fit
    was scored under (``fit_intensity``). ``None`` off the covariance path."""
    if not snap.cov_in_use:
        return None
    if which not in ('exact', 'fit'):
        raise ValueError("which must be 'exact' or 'fit', got %r" % (which,))
    scale = 'Log' if snap.fit_scale == 'Log' else 'Linear'
    return cc.factorize(cc.assemble(_snap_blocks(snap, which), scale)[0])[0]


# -------------------------------------------------------------------- split --
def _block_of(L, a, b):
    """(L Lᵀ)[a:b, a:b]: rows a:b of a lower factor reach only columns < b."""
    Li = L[a:b, :b]
    return Li @ Li.T


def _s_of(snap):
    rs = snap.rescale
    if not rs or rs.get('s') is None:
        return None
    return [float(v) for v in rs['s']]


def _exact_CL(snap, cache=None):
    """``(C, L)``: the exact C over the used rows and its Cholesky factor (equal to
    ``rebuild_L(snap)``), assembled and factorised once per ``cache`` dict."""
    if cache is not None and 'L' in cache:
        return cache['C'], cache['L']
    C = cc.assemble(_snap_blocks(snap, 'exact'), _cscale(snap))[0]
    L = cc.factorize(C)[0]
    if cache is not None:
        cache['C'], cache['L'] = C, L
    return C, L


def _L_of(snap, L=None, cache=None):
    if L is not None:
        return L
    return _exact_CL(snap, cache)[1]


def _block_chos(snap, L=None, cache=None):
    """Per dataset the ``cho_factor`` of its exact marginal block ``(L Lᵀ)_ii`` (None for a
    dataset with no rows), factorised once per ``cache`` dict (covariance path only)."""
    if cache is not None and 'chos' in cache and L is None:
        return cache['chos']
    Lx = _L_of(snap, L, cache)
    out = []
    for d in snap.datasets:
        a, b = d.rows
        out.append(None if b <= a else cho_factor(_block_of(Lx, a, b), lower=True))
    if cache is not None and L is None:
        cache['chos'] = out
    return out


def _marginal_raw(snap, L=None, cache=None):
    """Per dataset ``(m_i, ln det (C)_ii)`` of the UNSCALED exact marginal block, both from
    one Cholesky factor (log det = 2 Σ ln diag; spec §3.5) -- covariance path only."""
    if cache is not None and 'mraw' in cache and L is None:
        return cache['mraw']
    out = []
    for d, cf in zip(snap.datasets, _block_chos(snap, L, cache)):
        if cf is None:
            out.append((0.0, 0.0))
            continue
        r = d.r
        out.append((float(r @ cho_solve(cf, r)),
                    float(2.0 * np.sum(np.log(np.diag(cf[0]))))))
    if cache is not None and L is None:
        cache['mraw'] = out
    return out


def marginal_terms(snap, L=None, scaled=True, cache=None):
    """``m_i = r_iᵀ (L Lᵀ)_ii⁻¹ r_i`` with the exact L; equals the sequential term off
    the covariance path. For a rescaled snapshot the C in use is S C S, whose i-th
    diagonal block is s_i² (C)_ii, so ``scaled=True`` (the term of the objective that was
    minimised) is the unscaled term over s_i²; ``scaled=False`` undoes the rescaling
    exactly (what ``rescale_factors`` needs). ``cache`` (a dict) shares the factorisations
    with the other functions of one diagnostics pass."""
    s = _s_of(snap)
    if not snap.cov_in_use:
        out = [float(np.sum(snap.z[a:b] ** 2)) for (a, b) in (d.rows for d in snap.datasets)]
        if s is not None and not scaled:
            out = [m * si ** 2 for m, si in zip(out, s)]
        return out
    out = []
    for k, (m, _ld) in enumerate(_marginal_raw(snap, L, cache)):
        out.append(m / s[k] ** 2 if (s is not None and scaled) else m)
    return out


def rescale_factors(snap, inflate_only=False, notes=None):
    """``(s, nu)`` for the opt-in per-dataset error rescaling (spec §6.3):
    ``nu_i = n_i − p n_i / N`` (so ``Σ nu_i = N − p``) and ``s_i = √(m_i / nu_i)`` with
    ``m_i`` the marginal term of the UNSCALED objective (the current ``s`` undone), or
    ``max(1, s_i)`` when ``inflate_only``. A dataset with ``nu_i <= 0``, or whose factor
    is not finite or ``<= 1e-8`` (e.g. a model that fits it exactly, m_i = 0: dividing by
    s_i would put NaN in the residual), keeps ``s_i = 1`` and a note is appended to
    ``notes`` (when given)."""
    m = marginal_terms(snap, scaled=False)
    n = [int(d.rows[1] - d.rows[0]) for d in snap.datasets]
    N = float(sum(n))
    p = float(snap.nvarys)
    s, nu = [], []
    for d, mi, ni in zip(snap.datasets, m, n):
        v = ni - p * ni / N if N > 0 else 0.0
        nu.append(float(v))
        if v <= 0 or not np.isfinite(mi):
            s.append(1.0)
            if notes is not None:
                notes.append('%s: no degrees of freedom left for an error scale; s = 1' % d.name)
            continue
        si = float(np.sqrt(mi / v)) if mi >= 0 else float('nan')
        if not np.isfinite(si) or si <= 1e-8:
            s.append(1.0)
            if notes is not None:
                notes.append('%s: error scale sqrt(chi2_red) = %.3g is not usable (the model '
                             'fits this dataset exactly or chi2 is not finite); s = 1'
                             % (d.name, si))
            continue
        s.append(max(1.0, si) if inflate_only else si)
    return s, nu


def objective_label(fit_scale, cov_in_use, cov_mode, rescaled=None):
    """The §3.4 name of the objective and whether it is a (Gaussian) likelihood."""
    if fit_scale == 'Log w/o error':
        return 'unweighted sum of squares of log10 y', False
    if fit_scale == 'Linear w/o error':
        return 'unweighted sum of squares of y / max(y), per dataset', False
    log = fit_scale == 'Log'
    # ASCII spellings throughout (report text must print to a cp1252 console)
    if cov_in_use:
        lab = ('GLS chi2 of ln y (Log, covariance %s)' if log
               else 'GLS chi2 (Linear, covariance %s)') % cov_mode
        if cov_mode == 'data':
            lab += _DATA_CLAUSE
    else:
        lab = ('chi2 of ln y (Log, sigma_ln = sigma/y)' if log
               else 'chi2 (Linear, weighted by the error column)')
    if rescaled:
        s = rescaled.get('s') if isinstance(rescaled, dict) else None
        if isinstance(s, dict):
            s_txt = ', '.join('%s %.4g' % (k, v) for k, v in s.items())
        elif s is not None:
            s_txt = ', '.join('%.4g' % v for v in np.atleast_1d(s))
        else:
            s_txt = '...'
        lab = ('rescaled ' + lab + ' (errors rescaled per dataset by sqrt(chi2_red), s = %s; '
               'error scales fitted per dataset; approximate)' % s_txt)
        return lab, False        # errors fitted per dataset: not a known-error likelihood
    return lab, True


def per_dataset_objective(snap, cache=None):
    """The per-dataset split of the objective (§3.2, §3.3, §3.5).
    Returns ``(list[DatasetTerm], extras)``. With a ``cache`` dict the result (and the
    factorisations) are computed once per diagnostics pass."""
    if cache is not None and 'terms' in cache:
        return cache['terms']
    weighted = _weighted(snap.fit_scale)
    marg = marginal_terms(snap, cache=cache)
    mraw = _marginal_raw(snap, cache=cache) if snap.cov_in_use else None
    s_all = _s_of(snap)
    terms = []
    for d, m in zip(snap.datasets, marg):
        a, b = d.rows
        n = int(b - a)
        s = float(np.sum(snap.z[a:b] ** 2)) if n else 0.0
        m = float(m) if n else 0.0
        lds = ldm = nls = nlm = None
        if weighted:
            lds = float(2.0 * np.sum(snap.logdiag[a:b])) if n else 0.0
            if mraw is None or not n:
                ldm = lds
            else:
                ldm = mraw[len(terms)][1]          # 2 Σ ln diag chol((C)_ii)
                if s_all is not None:
                    ldm += 2.0 * n * np.log(s_all[len(terms)])     # (S C S)_ii = s_i² C_ii
            nls = s + lds + n * LN2PI
            nlm = m + ldm + n * LN2PI
        terms.append(DatasetTerm(
            name=d.name, n_points=n, sequential=s, marginal=m,
            chi2_per_point=(m / n) if n else None, logdet_seq=lds, logdet_marg=ldm,
            nll2_seq=nls, nll2_marg=nlm, window=d.window, index_hash=d.index_hash,
            scale=snap.fit_scale, cov=d.cov is not None, cov_mode=snap.cov_mode,
            rescale_s=None if s_all is None else s_all[len(terms)]))
    prior = float(np.sum(snap.prior ** 2))
    label, lik = objective_label(snap.fit_scale, snap.cov_in_use, snap.cov_mode, snap.rescale)
    fit_gap = None
    if snap.cov_in_use and snap.cov_mode == 'model':
        scale = 'Log' if snap.fit_scale == 'Log' else 'Linear'
        if cc.needs_intensity(_snap_blocks(snap, 'fit'), scale):
            fit_gap = sum(t.sequential for t in terms) + prior - snap.chisqr
    extras = {
        'prior': prior,
        'coupling': sum(t.sequential for t in terms) - sum(t.marginal for t in terms),
        'order': [d.name for d in snap.datasets],
        'objective_label': label,
        'likelihood': lik,
        'nll2': sum(t.nll2_seq for t in terms) if weighted else None,
        'fit_gap': fit_gap,
    }
    if cache is not None:
        cache['terms'] = (terms, extras)
    return terms, extras


# -------------------------------------------------------------- comparability --
class NotComparable(ValueError):
    """Two fits do not share a likelihood surface; the message names the first
    differing field (spec §4)."""


def comparability_key(snap):
    """The plain-dict key two fits must agree on before they are compared (spec §4):
    fit scale; per dataset, in order, name, hash of (x, y, yerr) over the active points,
    window, active-index hash and the covariance as used (None, or ``(cov_source,
    cov_hash, cov_mode)``; ``cov_hash`` never covers the intensities, so model-mode fits
    from different starting points share a key); the error-rescaling option (None, or
    ``(mode, inflate_only)``). Model class, priors, method and parameter state are
    deliberately absent."""
    ds = []
    for d in snap.datasets:
        ds.append({
            'name': d.name,
            'data_hash': _hash(d.x, d.y, d.yerr),
            'window': (float(d.window[0]), float(d.window[1])),
            'index_hash': d.index_hash,
            'cov': None if d.cov is None else (d.cov_source, d.cov_hash, snap.cov_mode),
        })
    rs = snap.rescale
    rescale = None if not rs else (rs.get('mode'), bool(rs.get('inflate_only', False)))
    return {'fit_scale': snap.fit_scale, 'datasets': ds, 'rescale': rescale}


def _fmt_window(w):
    return '%.3f:%.3f' % (w[0], w[1])


def _first_difference(ka, kb):
    """A message naming the first differing field of two keys, or None."""
    if ka['fit_scale'] != kb['fit_scale']:
        return 'fit scale %s vs %s' % (ka['fit_scale'], kb['fit_scale'])
    na, nb = [d['name'] for d in ka['datasets']], [d['name'] for d in kb['datasets']]
    if len(na) != len(nb):
        return 'dataset count %d vs %d' % (len(na), len(nb))
    if na != nb:
        if sorted(na) == sorted(nb):
            return 'dataset order %s vs %s' % (','.join(na), ','.join(nb))
        i = next(j for j in range(len(na)) if na[j] != nb[j])
        return 'dataset %d name %s vs %s' % (i + 1, na[i], nb[i])
    for i, (da, db) in enumerate(zip(ka['datasets'], kb['datasets'])):
        tag = 'dataset %d (%s)' % (i + 1, da['name'])
        if da['window'] != db['window']:
            return '%s window %s vs %s' % (tag, _fmt_window(da['window']),
                                            _fmt_window(db['window']))
        if da['index_hash'] != db['index_hash']:
            return '%s active index (mask) differs' % tag
        if da['data_hash'] != db['data_hash']:
            return '%s data (x, y, yerr) differs' % tag
        ca, cb = da['cov'], db['cov']
        if ca != cb:
            if ca is None or cb is None:
                return '%s covariance %s vs %s' % (
                    tag, 'none' if ca is None else 'attached', 'none' if cb is None else 'attached')
            if ca[0] != cb[0]:
                return '%s covariance source %s vs %s' % (tag, ca[0], cb[0])
            if ca[1] != cb[1]:
                return '%s covariance content differs (terms, weights or sigma_meas)' % tag
            return '%s covariance mode %s vs %s' % (tag, ca[2], cb[2])
    if ka['rescale'] != kb['rescale']:
        return 'error rescale %s vs %s' % (ka['rescale'], kb['rescale'])
    return None


def check_comparable(a, b):
    """Raise ``NotComparable`` naming the first differing field; None when the two
    snapshots are comparable (spec §4, §4.1)."""
    msg = _first_difference(comparability_key(a), comparability_key(b))
    if msg is not None:
        raise NotComparable(msg)


# --------------------------------------------------------------- comparison --
@dataclass
class Comparison:
    """The calibrated comparison of two fits of the same data (spec §5). Positive
    ``sum_d`` prefers B. ``coupled`` and ``rescaled`` are rendering aids beyond the
    brief's field list."""
    names: list
    d: list
    sum_d: float
    sd_d: Optional[float]
    N: int
    z: Optional[float]
    preferred: Optional[str]
    sign_count: Tuple[int, int]
    delta_k: Optional[int]
    nested: Optional[str]
    lr_p: Optional[float]
    worst_dataset: str
    objective_label: str
    likelihood: bool
    delta_chi2: float
    delta_logdet: float
    prior_difference: float
    model_differences: list
    notes: list
    coupled: bool = False
    rescaled: bool = False


NO_Z_NOTE = 'no calibrated z: too few datasets to estimate the spread'
COUPLING_SENTENCE = ('datasets treated as independent after conditioning on shared '
                     'error terms in the listed order')
LR_CAVEAT = 'the model is likely misspecified; leave-one-out is the honest check'


def _endpoint_sets(snap):
    return {m: frozenset(v) for m, v in snap.param_map.items()}


def _free_endpoints(snap):
    out = set()
    for m, ends in _endpoint_sets(snap).items():
        if snap.vary.get(m, False):
            out |= ends
    return out


def _coarsens(a, b):
    """True when B is a restriction of A: every B master's endpoint set is a union of
    A masters' endpoint sets, every endpoint free in B is free in A, and per dataset
    the model class and model files agree."""
    for da, db in zip(a.datasets, b.datasets):
        if da.model_class != db.model_class or da.model_files != db.model_files:
            return False
    ea = _endpoint_sets(a)
    for mb, sb in _endpoint_sets(b).items():
        union = set()
        for ma, e in ea.items():
            if e <= sb:
                union |= e
                # B fixed here: every A master it absorbs that A holds fixed must sit at
                # B's value, else B is not a restriction of A
                if not b.vary.get(mb, False) and not a.vary.get(ma, False):
                    va, vb = a.params.get(ma), b.params.get(mb)
                    if va is None or vb is None or abs(va - vb) > 1e-12 * max(1.0, abs(vb)):
                        return False
        if union != sb:
            return False
    return _free_endpoints(b) <= _free_endpoints(a)


def nesting(a, b):
    """``'B in A'`` when B is a restriction of A (see ``_coarsens``), ``'A in B'``
    symmetrically, else None. Identical parameterisations (both directions) are not
    nested."""
    if len(a.datasets) != len(b.datasets):
        return None
    ab, ba = _coarsens(a, b), _coarsens(b, a)
    if ab and ba:
        return None
    if ab:
        return 'B in A'
    if ba:
        return 'A in B'
    return None


def _model_differences(a, b):
    groups = {}
    for da, db in zip(a.datasets, b.datasets):
        if da.model_class != db.model_class:
            groups.setdefault('different model: %s vs %s' % (da.model_class, db.model_class),
                              []).append(da.name)
        elif da.model_files != db.model_files:
            groups.setdefault('different model files (%s)' % da.model_class, []).append(da.name)
    out = []
    for msg, names in groups.items():
        where = ', '.join(names) if len(names) <= 3 else '%d datasets' % len(names)
        out.append('%s (%s)' % (msg, where))
    return out


def compare_fits(a, b):
    """Compare two fits of the same data from the exact per-dataset terms (spec §5).
    Raises ``NotComparable`` when the comparability key differs."""
    from scipy.stats import chi2
    check_comparable(a, b)
    ta, ea = per_dataset_objective(a)
    tb, eb = per_dataset_objective(b)
    weighted = _weighted(a.fit_scale)
    if weighted:
        d = [0.5 * (x.nll2_seq - y.nll2_seq) for x, y in zip(ta, tb)]
        dchi = float(sum(x.sequential - y.sequential for x, y in zip(ta, tb)))
        dld = float(sum(x.logdet_seq - y.logdet_seq for x, y in zip(ta, tb)))
    else:
        d = [x.sequential - y.sequential for x, y in zip(ta, tb)]
        dchi, dld = float(sum(d)), 0.0
    d = [float(v) for v in d]
    N = len(d)
    sum_d = float(sum(d))
    sd = float(np.std(d, ddof=1)) if N >= 2 else None
    # what each side's numbers are (posterior medians, an aborted fit, ...)
    notes = ['%s: %s' % (side, s.label) for side, s in (('A', a), ('B', b)) if s.label]
    z = None
    if N < 4:
        notes.append(NO_Z_NOTE)
    elif sd is not None and sd > 0:
        z = float(abs(sum_d) / (sd * np.sqrt(N)))
    else:
        notes.append('no calibrated z: the per-dataset differences have zero spread')
    preferred = 'B' if sum_d > 0 else ('A' if sum_d < 0 else None)
    if preferred is None:
        cnt = 0
    else:
        sgn = 1.0 if preferred == 'B' else -1.0
        cnt = int(sum(1 for v in d if sgn * v > 0))
    names = [t.name for t in ta]
    worst = names[int(np.argmax(np.abs(d)))] if d else ''
    nest = nesting(a, b)
    rescaled = a.rescale is not None
    likelihood = bool(ea['likelihood'])
    delta_k = lr_p = None
    stat = 0.0
    if nest == 'B in A':
        delta_k = int(a.nvarys - b.nvarys)
        stat = -2.0 * sum_d
    elif nest == 'A in B':
        delta_k = int(b.nvarys - a.nvarys)
        stat = 2.0 * sum_d
    if nest and delta_k is not None and delta_k > 0 and likelihood and not rescaled:
        lr_p = float(chi2.sf(max(0.0, stat), delta_k))
    elif nest and rescaled:
        notes.append('no likelihood-ratio p-value: error scales are fitted per dataset '
                     '(rescaled), so the objective is not an exact likelihood')

    def _coupled(e, t):
        return abs(e['coupling']) > 1e-9 * max(1.0, sum(abs(x.sequential) for x in t))

    return Comparison(
        names=names, d=d, sum_d=sum_d, sd_d=sd, N=N, z=z, preferred=preferred,
        sign_count=(cnt, N), delta_k=delta_k, nested=nest, lr_p=lr_p, worst_dataset=worst,
        objective_label=ea['objective_label'], likelihood=likelihood, delta_chi2=dchi,
        delta_logdet=dld, prior_difference=float(ea['prior'] - eb['prior']),
        model_differences=_model_differences(a, b), notes=notes,
        coupled=bool(_coupled(ea, ta) or _coupled(eb, tb)), rescaled=rescaled)


def information_criteria(snap, cache=None):
    """Known-error AIC / BIC on the §3 objective (spec §8). Weighted: ``Φ + 2k`` and
    ``Φ + k ln n`` with ``Φ = nll2 + prior``; unweighted: the ``n ln(RSS/n)`` form."""
    terms, ex = per_dataset_objective(snap, cache)
    n = int(sum(t.n_points for t in terms))
    k = int(snap.nvarys)
    out = {'n': n}
    if ex['nll2'] is not None:
        if snap.rescale is not None:
            k += len(snap.datasets)
            out['note'] = 'conditional on the fitted s'
        phi = ex['nll2'] + ex['prior']
        out.update(aic=float(phi + 2 * k), bic=float(phi + k * np.log(n)), k=k,
                   form='known errors')
    else:
        rss = sum(t.sequential for t in terms) + ex['prior']
        base = n * np.log(rss / n) if n and rss > 0 else float('-inf')
        out.update(aic=float(base + 2 * k), bic=float(base + k * np.log(max(n, 1))), k=k,
                   form='unknown error scale')
    return out


def format_comparison(c, ic_a=None, ic_b=None):
    """Plain-text rendering of a ``Comparison`` (spec §5, §8)."""
    lines = []
    dform = 'd_i = 1/2 (nll2_A - nll2_B)' if c.likelihood else 'd_i = s^A - s^B'
    if c.z is not None:
        head = 'Delta = %+.4g +/- %.3g (z = %.2f)' % (c.sum_d, c.sd_d * np.sqrt(c.N), c.z)
    else:
        head = 'Comparison'     # the bare Δ lives only in the detail line (spec §5)
    lines.append(head + '   [Sum d, %s; positive prefers B; %s]' % (dform, c.objective_label))
    if c.preferred is None:
        lines.append('no fit preferred (%d datasets)' % c.N)
    else:
        lines.append('%s preferred on %d/%d datasets' % ((c.preferred,) + tuple(c.sign_count)))
    if c.coupled:
        lines.append(COUPLING_SENTENCE)
    for n in c.notes:
        lines.append(n)
    if not c.likelihood:
        lines.append('difference in sum of squares; not a likelihood')
    lines.append('per-dataset d (positive favours B):')
    sn = short_names(c.names)
    w = _name_width(sn.values())
    for nm, v in zip(c.names, c.d):
        lines.append('  %-*s %+.4g' % (w, sn[nm], v))
    lines.append('largest |d|: %s' % sn.get(c.worst_dataset, c.worst_dataset))
    if c.likelihood:
        det = 'Delta(-logL) = %+.6g (not a confidence measure)' % c.sum_d
    else:
        det = ('difference in sum of squares = %+.6g; not a likelihood '
               '(not a confidence measure)' % c.sum_d)
    if c.delta_logdet != 0:
        det += ' (Delta chi2/2 = %+.6g, Delta log det/2 = %+.6g)' % (0.5 * c.delta_chi2,
                                                                     0.5 * c.delta_logdet)
    lines.append(det)
    for m in c.model_differences:
        lines.append(m)
    if c.nested:
        s = 'nested: %s, Delta k = %d' % (c.nested, c.delta_k)
        if c.lr_p is not None:
            s += ', likelihood-ratio p = %.3g; %s' % (c.lr_p, LR_CAVEAT)
        lines.append(s)
    else:
        lines.append('not nested; no likelihood-ratio p-value')
    lines.append('prior terms difference (A - B) = %+.6g (not in Sum d)' % c.prior_difference)
    if ic_a is not None or ic_b is not None:
        lines.append('information criteria (assume a correct model; read with caution)')
        for lab, ic in (('A', ic_a), ('B', ic_b)):
            if ic is not None:
                lines.append('  %s: AIC %.6g  BIC %.6g  (k = %d, n = %d, %s%s)' % (
                    lab, ic['aic'], ic['bic'], ic['k'], ic['n'], ic['form'],
                    '; ' + ic['note'] if ic.get('note') else ''))
        if ic_a is not None and ic_b is not None:
            lines.append('  Delta AIC (A - B) = %+.6g  Delta BIC (A - B) = %+.6g' % (
                ic_a['aic'] - ic_b['aic'], ic_a['bic'] - ic_b['bic']))
    return '\n'.join(lines)



# ------------------------------------------------- joint vs individual (§6.1) --
@contextlib.contextmanager
def _saved_model_state(func, keys):
    """Save a model's ``x``, ``__fit__`` and, for each of ``keys``, its scalar attribute and
    ``params`` entry (value and expr); restore them on exit, also on an exception. The same
    save/restore as ``JointFit.quiet`` applies to one model."""
    _missing = object()
    x = getattr(func, 'x', _missing)
    fitflag = getattr(func, '__fit__', _missing)
    pv = {}
    for k in keys:
        if k in func.params and k not in pv:
            lp = func.params[k]
            pv[k] = (func.__dict__.get(k, _missing), lp.value, lp.expr)
    try:
        yield func
    finally:
        if x is not _missing:
            func.x = x
        if fitflag is not _missing:
            func.__fit__ = fitflag
        for k, (attr, val, expr) in pv.items():
            if attr is _missing:
                func.__dict__.pop(k, None)
            else:
                func.__dict__[k] = attr
            lp = func.params[k]
            lp.value = val
            lp.expr = expr


def _usable_intensity(inten, y, scale):
    """The rule of ``JointFit._substitute_unusable`` / ``_build_chol``: the data stand in
    where the model intensity is non-finite (or <= 0 in Log)."""
    inten = np.asarray(inten, float)
    bad = ~np.isfinite(inten)
    if scale == 'Log':
        with np.errstate(invalid='ignore'):
            bad = bad | (inten <= 0)
    return np.where(bad, np.asarray(y, float), inten)


def _residual_of(fit_scale, data, fit):
    """The raw residual ``raw_parts`` forms, for an arbitrary model ``fit``."""
    if fit_scale == 'Log':
        return np.log(data) - np.log(fit)
    if fit_scale == 'Log w/o error':
        return np.log10(data) - np.log10(fit)
    if fit_scale == 'Linear w/o error':
        return (data - fit) / np.max(data) if len(data) else data - fit
    return data - fit


def _quad_logdet(C, r):
    """``(r C⁻¹ r, ln det C)`` from the Cholesky factor (log det from its diagonal)."""
    C = 0.5 * (np.asarray(C, float) + np.asarray(C, float).T)
    try:
        Lc = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        Lc = cc.factorize(C)[0]
    z = cc.whiten(Lc, r)
    return float(z @ z), float(2.0 * np.sum(np.log(np.diag(Lc))))


def _individual_scale(fit):
    ua = getattr(getattr(fit, 'fitter', None), 'userargs', None)
    if ua is None or len(ua) == 0:
        return None
    return ua[0]


def _individual_model(fit, d):
    """The individual model at its fitted parameters on dataset ``d``'s active x, quietly.
    The ``Fit`` is left exactly as it was."""
    func = fit.func
    params = fit.result.params
    with _saved_model_state(func, list(params.keys())):
        func.__fit__ = True
        for k in params.keys():
            if k in func.params:
                v = params[k].value
                func.__dict__[k] = v
                func.params[k].value = v
                func.params[k].expr = None
        func.x = d.x
        f = np.asarray(func.y(), float)
    return f


def _mad_flags(deltas, available):
    """``delta > median + 3·1.4826·MAD and delta > 2`` over the available rows."""
    v = np.array([deltas[i] for i in available], float)
    flags = [False] * len(deltas)
    if len(v) == 0:
        return flags
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    thr = med + 3.0 * 1.4826 * mad
    for i in available:
        flags[i] = bool(deltas[i] > thr and deltas[i] > 2.0)
    return flags


def individual_terms(jf, snap, individual_fits, cache=None):
    """Joint vs individual fits per dataset (spec §6.1). ``individual_fits`` maps dataset
    name -> ``Fit_Routines.Fit`` (``fit.func``, ``fit.result.params``, and its fit scale in
    ``fit.fitter.userargs[0]``). Each side is scored on the joint fit's used rows under its
    own C_ii (the joint side under the snapshot's exact marginal block; the individual side,
    in model mode when C depends on the model, under C built from the individual model's
    intensity). An error rescaling of the joint fit is ignored on both sides (the
    individual fit was never rescaled). ``delta = joint_nll2 - indiv_nll2``, in -2 ln L
    units. Returns one dict per dataset: name, n, joint_m, indiv_m, joint_nll2, indiv_nll2,
    delta, flagged, unavailable. ``jf`` is not read (the snapshot is self-contained)."""
    fits = individual_fits or {}
    scale = snap.fit_scale
    cscale = 'Log' if scale == 'Log' else 'Linear'
    weighted = _weighted(scale)
    L = _L_of(snap, None, cache) if snap.cov_in_use else None
    model_mode = snap.cov_in_use and snap.cov_mode == 'model'
    rows = []
    for d in snap.datasets:
        row = {'name': d.name, 'n': int(np.sum(d.used)), 'joint_m': None, 'indiv_m': None,
               'joint_nll2': None, 'indiv_nll2': None, 'delta': None, 'flagged': False,
               'unavailable': None}
        rows.append(row)
        fit = fits.get(d.name)
        if fit is None:
            row['unavailable'] = 'no individual fit supplied for this dataset'
            continue
        if getattr(fit, 'result', None) is None:
            row['unavailable'] = 'the individual fit has no result'
            continue
        sc = _individual_scale(fit)
        if sc is None:
            row['unavailable'] = 'the individual fit scale is unknown (no fitter.userargs)'
            continue
        if sc != scale:
            row['unavailable'] = ('the individual fit scale %s differs from the joint fit '
                                  'scale %s' % (sc, scale))
            continue
        n = row['n']
        if n == 0:
            row['unavailable'] = 'no points left in the joint fit for this dataset'
            continue
        try:
            f_all = _individual_model(fit, d)
        except Exception as e:                    # noqa: BLE001 -- degrade, report
            row['unavailable'] = 'the individual model failed: %s: %s' % (type(e).__name__, e)
            continue
        if f_all.shape != d.x.shape:
            row['unavailable'] = ('the individual model returned %d points for %d'
                                  % (f_all.size, d.x.size))
            continue
        u = d.used
        y, f = d.y[u], f_all[u]
        with np.errstate(invalid='ignore', divide='ignore'):
            r_i = _residual_of(scale, y, f)
        if not np.all(np.isfinite(r_i)):
            row['unavailable'] = 'the individual model is not finite on the joint fit\'s used rows'
            continue
        a, b = d.rows
        if snap.cov_in_use:
            C_j = _block_of(L, a, b)
            C_i = C_j
            if model_mode:
                blk = cc.Block(d.cov, y, d.yerr[u], _usable_intensity(f, y, cscale), name=d.name)
                if cc.needs_intensity([blk], cscale):
                    C_i = cc.assemble([blk], cscale)[0]
        elif weighted:
            sig = d.yerr[u] / y if scale == 'Log' else d.yerr[u]
            C_j = C_i = np.diag(sig ** 2)
        else:
            C_j = C_i = None
        if C_j is None:                              # unweighted: a sum of squares
            mj, mi = float(np.sum(d.r ** 2)), float(np.sum(r_i ** 2))
            lj = li = 0.0
            const = 0.0
        else:
            mj, lj = _quad_logdet(C_j, d.r)
            mi, li = _quad_logdet(C_i, r_i)
            const = n * LN2PI
        row.update(joint_m=mj, indiv_m=mi, joint_nll2=mj + lj + const,
                   indiv_nll2=mi + li + const)
        row['delta'] = row['joint_nll2'] - row['indiv_nll2']
    avail = [i for i, r in enumerate(rows) if r['unavailable'] is None]
    flags = _mad_flags([r['delta'] for r in rows], avail) if avail else []
    for i in avail:
        rows[i]['flagged'] = flags[i]
    return rows


def _fmt_indiv_calibrated(rows, likelihood):
    """The §5 calibrated comparison over ``d_i = ½ delta_i`` (positive: the individual fit
    is better than the shared one), small-N rule included."""
    d = [0.5 * r['delta'] for r in rows if r['unavailable'] is None]
    N = len(d)
    if N == 0:
        return ['nothing to compare: no individual fit is available']
    sum_d = float(sum(d))
    form = 'd_i = 1/2 delta_i' if likelihood else 'd_i = 1/2 (s^joint - s^individual)'
    cnt = int(sum(1 for v in d if v > 0))
    if N >= 2:
        sd = float(np.std(d, ddof=1))
        spread = sd * np.sqrt(N)
        head = 'Sum d = %+.4g +/- %.3g' % (sum_d, spread)
        z = None
        note = None
        if N < 4:
            note = NO_Z_NOTE
        elif sd > 0:
            z = abs(sum_d) / spread
        else:
            note = 'no calibrated z: the per-dataset differences have zero spread'
        if z is not None:
            head += ' (z = %.2f)' % z
    else:
        head = 'Sum d = %+.4g' % sum_d
        note = NO_Z_NOTE
    out = ['%s   [%s; positive favours the individual fits]' % (head, form)]
    if note:
        out.append(note)
    out.append('the individual fits are better on %d/%d datasets' % (cnt, N))
    if not likelihood:
        out.append('difference in sum of squares; not a likelihood')
    return out


def format_joint_vs_individuals(jf, individual_fits, snap=None):
    """The full joint-vs-individual table and its calibrated comparison (spec §6.1)."""
    if snap is None:
        snap = getattr(jf, 'last_snapshot', None)
    if snap is None:
        raise RuntimeError('format_joint_vs_individuals needs a finished joint fit (no snapshot)')
    rows = individual_terms(jf, snap, individual_fits)
    label, lik = objective_label(snap.fit_scale, snap.cov_in_use, snap.cov_mode, None)
    out = ['joint vs individual fits (marginal terms)%s'
           % (' (unrescaled)' if snap.rescale is not None else ''),
          'objective: %s' % label,
          'each side scored on the joint fit\'s used rows under its own C_ii; '
          'delta = joint - individual in %s' % ('-2 ln L units' if lik or _weighted(snap.fit_scale)
                                                 else 'sum of squares')]
    sn = short_names([r['name'] for r in rows])
    w = _name_width(sn.values())
    out.append('  %-*s %-5s %12s %12s %12s' % (w, 'dataset', 'n', 'joint chi2/n', 'indiv chi2/n',
                                               'delta'))
    for r in rows:
        if r['unavailable'] is not None:
            out.append('  %-*s %-5d unavailable: %s' % (w, sn[r['name']], r['n'], r['unavailable']))
            continue
        n = r['n']
        out.append('  %-*s %-5d %12.4g %12.4g %+12.4g%s' % (
            w, sn[r['name']], n, r['joint_m'] / n, r['indiv_m'] / n, r['delta'],
            '  FLAGGED: the shared value fits this dataset worse than the others'
            if r['flagged'] else ''))
    out.extend(_fmt_indiv_calibrated(rows, lik))
    return '\n'.join(out)


# --------------------------- Fisher shares, tension, exact optimum (§6.2, §6.4) --
_DEG_UNAVAILABLE = 'no parameter covariance (the Hessian failed)'
_DEG_MCMC = 'no parameter covariance (MCMC result)'
COUPLED_NOTE = 'the joint information is smaller than the sum: shared errors are counted once'
MEAN_INFO_NOTE = ('mean information only; the information the covariance carries about the '
                  'parameters is omitted')
TENSION_HEAD = 'model-reduction tension'
SKIP_USER = 'skipped: fit stopped by the user'
SKIP_CHEAP = "skipped (report_diagnostics='cheap')"


class DiagnosticSkipped(Exception):
    """A diagnostics section that did not run (the user stopped the fit, or the cheap
    report mode); ``diagnostics`` records it as ``{'skipped': <reason>}``."""


def _check_abort(jf):
    if getattr(jf, 'fit_abort', False):
        raise DiagnosticSkipped(SKIP_USER)


def _check_live(jf, snap):
    """The functions that evaluate the live ``jf`` need ``snap`` to describe its current
    state: it must be ``jf.last_snapshot`` and the row ranges must still agree."""
    if snap is not getattr(jf, 'last_snapshot', None):
        raise ValueError('the snapshot is not this JointFit\'s last fit (jf.last_snapshot); '
                         'the live models and masks may belong to another fit')
    try:
        ranges = [tuple(int(v) for v in r) for r in jf.row_ranges()]
    except Exception as e:                        # noqa: BLE001 -- say why, plainly
        raise ValueError('the JointFit row ranges are unavailable (%s)' % e)
    if ranges != [tuple(d.rows) for d in snap.datasets]:
        raise ValueError('the snapshot row ranges %s differ from the JointFit\'s %s (datasets, '
                         'windows or masks changed since the fit)'
                         % ([tuple(d.rows) for d in snap.datasets], ranges))


def _cscale(snap):
    return 'Log' if snap.fit_scale == 'Log' else 'Linear'


def _model_needs(snap):
    """True when C depends on the model at the fitted point (model mode, an intensity-
    dependent term at this scale): the one case whose C moves with θ."""
    return bool(snap.cov_in_use and snap.cov_mode == 'model'
                and cc.needs_intensity(_snap_blocks(snap, 'fit'), _cscale(snap)))


def _shared_masters(snap):
    """Free masters whose ``param_map`` entry spans more than one dataset."""
    return [k for k in snap.var_names
            if len({int(i) for (i, _lk) in snap.param_map.get(k, ())}) > 1]


def _step_for(snap, name, step):
    """Finite-difference step: ``step`` may be a number (every parameter) or a dict; the
    default is ``1e-6·max(|v|, stderr or 1, 1e-8)``."""
    if isinstance(step, dict) and name in step:
        return float(step[name])
    if step is not None and not isinstance(step, dict):
        return float(step)
    se = snap.stderr.get(name)
    return 1e-6 * max(abs(snap.params[name]), float(se) if se else 1.0, 1e-8)


def _params_at(jf, snap, values):
    """A copy of ``jf.result.params`` with the free masters at the snapshot values, then
    ``values`` (name -> value) on top; expressions follow through lmfit."""
    p = jf.result.params.copy()
    for k in snap.var_names:
        p[k].value = snap.params[k]
    for k, v in values.items():
        p[k].value = float(v)
    return p


def _row_s(snap):
    s = _s_of(snap)
    if s is None:
        return None
    return np.repeat(np.asarray(s, float), [int(d.rows[1] - d.rows[0]) for d in snap.datasets])


def _marginal_cov(snap, k, L=None):
    """C_ii of dataset ``k`` as the fit weighted it: the exact marginal block (times s_i²
    when rescaled) on the covariance path; ``diag((yerr·s_i)²)`` in Linear and
    ``diag((yerr·s_i/y)²)`` in Log off it; the identity for the unweighted scales."""
    d = snap.datasets[k]
    s = _s_of(snap)
    si = 1.0 if s is None else s[k]
    a, b = d.rows
    if snap.cov_in_use:
        if L is None:
            L = rebuild_L(snap)
        return _block_of(L, a, b) * si ** 2
    n = int(b - a)
    if not _weighted(snap.fit_scale):
        return np.eye(n)
    u = d.used
    sig = d.yerr[u] / d.y[u] if snap.fit_scale == 'Log' else d.yerr[u]
    return np.diag((sig * si) ** 2)


def _is_coupled(snap, cache=None):
    """True when C has a non-zero entry outside the per-dataset diagonal blocks."""
    if not snap.cov_in_use or len(snap.datasets) < 2:
        return False
    C = _exact_CL(snap, cache)[0]
    for d in snap.datasets:
        a, b = d.rows
        if b > a and (np.any(C[a:b, :a] != 0.0) or np.any(C[a:b, b:] != 0.0)):
            return True
    return False


def fisher_shares(jf, snap, step=None, cache=None):
    """What each dataset tells you about the shared parameters (spec §6.2).

    Forward-difference Jacobian of every dataset's unwhitened residual (used rows) w.r.t.
    every free master: nvarys quiet model evaluations (none without a shared free master);
    per dataset the marginal information ``I_i = J_iᵀ C_ii⁻¹ J_i`` (``_marginal_cov``) and,
    for each shared master p, the share ``I_i[p,p] / Σ_j I_j[p,p]``. Returns ``{'params',
    'shares', 'information' (I_i[p,p]), 'matrix' (I_i over var_names), 'var_names',
    'coupled'}`` plus ``'mean_information_only': True`` when C depends on the model (the
    covariance's own information term is omitted) and ``'note'`` for an unweighted scale.
    Every model and ``jf.result`` are left as they were. ``snap`` must be
    ``jf.last_snapshot`` (ValueError otherwise); ``jf.fit_abort`` stops the loop
    (``DiagnosticSkipped``)."""
    names = list(snap.var_names)
    shared = _shared_masters(snap)
    out = {'params': shared, 'var_names': names, 'coupled': _is_coupled(snap, cache),
           'shares': {d.name: {} for d in snap.datasets},
           'information': {d.name: {} for d in snap.datasets}, 'matrix': {}}
    if _model_needs(snap):
        out['mean_information_only'] = True
    if not _weighted(snap.fit_scale):
        out['note'] = 'unweighted fit scale: shares of the sum-of-squares curvature, not a likelihood'
    if not shared:
        return out
    _check_live(jf, snap)
    r0 = [np.asarray(d.r, float) for d in snap.datasets]
    J = [np.zeros((len(r), len(names))) for r in r0]
    with jf.quiet():
        for j, name in enumerate(names):
            _check_abort(jf)
            h = _step_for(snap, name, step)
            parts = raw_parts(jf, _params_at(jf, snap, {name: snap.params[name] + h}))
            for k, (r, _f) in enumerate(parts):
                J[k][:, j] = (np.asarray(r, float) - r0[k]) / h
    chos = _block_chos(snap, cache=cache) if snap.cov_in_use else None
    s_all = _s_of(snap)
    tot = {p: 0.0 for p in shared}
    for k, d in enumerate(snap.datasets):
        Jk = J[k]
        if Jk.shape[0] == 0:
            I = np.zeros((len(names), len(names)))
        else:
            if chos is not None:      # (S C S)_ii = s_i² (C)_ii: one factor per block
                si = 1.0 if s_all is None else s_all[k]
                I = (Jk.T @ cho_solve(chos[k], Jk)) / si ** 2
            else:
                C = _marginal_cov(snap, k)
                I = Jk.T @ cho_solve(cho_factor(C, lower=True), Jk)
            I = 0.5 * (I + I.T)
        out['matrix'][d.name] = I
        for p in shared:
            v = float(I[names.index(p), names.index(p)])
            out['information'][d.name][p] = v
            tot[p] += v
    for d in snap.datasets:
        for p in shared:
            out['shares'][d.name][p] = (out['information'][d.name][p] / tot[p]
                                        if tot[p] > 0 else None)
    return out


def _tension_groups(snap):
    """The multiplicative components as ``(dataset label, term, G over the whole fit,
    Σ_t, labels, scope)``. Shared terms are grouped as ``assemble`` groups them (one set
    of random variables over the datasets that carry identical copies)."""
    blocks = _snap_blocks(snap, 'exact')
    scale = _cscale(snap)
    n = int(snap.datasets[-1].rows[1]) if snap.datasets else 0
    groups, shared = [], {}
    for k, (d, b) in enumerate(zip(snap.datasets, blocks)):
        if d.cov is None:
            continue
        a, e = d.rows
        if e <= a:                       # no rows left: the component is not informed
            continue
        for t in d.cov.terms:
            if t.kind != 'multiplicative':
                continue
            if t.scope == 'shared':
                shared.setdefault(t.name, []).append((k, t))
                continue
            G = np.zeros((n, np.asarray(t.cov).shape[0]))
            G[a:e] = cc._loading(t, b, scale)
            groups.append((d.name, t.name, G, np.asarray(t.cov, float), t.labels, 'dataset'))
    for name, copies in shared.items():
        same = all(cc._same_term(x[1], y[1]) for i, x in enumerate(copies)
                   for y in copies[i + 1:])
        for grp in ([copies] if same else [[c] for c in copies]):
            grp = [(k, t) for (k, t) in grp
                   if snap.datasets[k].rows[1] > snap.datasets[k].rows[0]]
            if not grp:
                continue
            t0 = grp[0][1]
            G = np.zeros((n, np.asarray(t0.cov).shape[0]))
            for k, t in grp:
                a, e = snap.datasets[k].rows
                G[a:e] = cc._loading(t, blocks[k], scale)
            groups.append((', '.join(snap.datasets[k].name for k, _ in grp), name, G,
                           np.asarray(t0.cov, float), t0.labels, 'shared'))
    return groups


def tension(snap, cache=None):
    """Model-reduction tension (spec §6.4 item 5, spec 17 §4.6): for every multiplicative
    component, ``eps = Σ_t Gᵀ C⁻¹ r`` over the whole fit (C the snapshot's exact C, G the
    term's loading at the stored intensity; a rescaled fit uses r_i / s_i, i.e. the
    covariance S C S with loading S G) and ``pull = eps_k / √Σ_kk``; components with
    ``Σ_kk = 0`` (an anchor) are omitted. ``None`` off the covariance path or when there is
    no multiplicative term. It is a diagnostic of the model against the reduction, never a
    fitted scale factor."""
    if not snap.cov_in_use:
        return None
    groups = _tension_groups(snap)
    if not groups:
        return None
    L = _L_of(snap, None, cache)
    r = np.concatenate([d.r for d in snap.datasets]) if snap.datasets else np.zeros(0)
    rs = _row_s(snap)
    if rs is not None:
        r = r / rs
    w = cho_solve((L, True), r)
    rows = []
    for ds, term, G, S, labels, scope in groups:
        eps = S @ (G.T @ w)
        for c in range(len(eps)):
            if not S[c, c] > 0:
                continue
            lab = str(labels[c]) if labels is not None and c < len(labels) else str(c)
            rows.append({'dataset': ds, 'term': term, 'label': lab, 'scope': scope,
                         'eps': float(eps[c]), 'pull': float(eps[c] / np.sqrt(S[c, c]))})
    return rows


_tension = tension           # diagnostics() has a ``tension`` flag that shadows it


def exact_objective(jf, snap, params):
    """Φ(θ) = −2 ln L(θ) + Σ prior_rows(θ)² at ``params`` (spec §3.5, §6.4 item 6): the raw
    residual from ``raw_parts``; C(θ) from ``jf.model_blocks`` when it depends on the
    model, else the snapshot's fixed blocks; ``2 Σ ln diag chol(C)`` for the log det; the
    snapshot's rescale factors applied as the fit applied them (r_i / s_i before whitening,
    ``ln s_i`` per row in the log det). An unweighted scale returns its sum of squares.
    Quiet; nothing is left changed; one model evaluation (shared by the residual and C(θ)).
    At the snapshot's parameters it equals ``nll2 + prior`` of ``per_dataset_objective``.
    ``snap`` must be ``jf.last_snapshot`` (ValueError otherwise)."""
    _check_live(jf, snap)
    scale = snap.fit_scale
    ev = _evaluate_at(jf, params)
    parts = raw_parts(jf, params, ev)
    blocks = None
    if snap.cov_in_use:
        if _model_needs(snap):
            blocks = jf.model_blocks(params, scale, ev)[0]
        else:
            blocks = _snap_blocks(snap, 'exact')
    prior = np.asarray(jf.prior_rows(params), float)
    pr = float(np.sum(prior ** 2))
    r = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
    if not _weighted(scale):
        return float(np.sum(r ** 2)) + pr
    rs = _row_s(snap)
    if blocks is not None:
        Lc = cc.factorize(cc.assemble(blocks, _cscale(snap))[0])[0]
        z = cc.whiten(Lc, r if rs is None else r / rs)
        logdet = 2.0 * float(np.sum(np.log(np.diag(Lc))))
    else:
        sig = np.concatenate([(d.yerr[d.used] / d.y[d.used]) if scale == 'Log'
                              else d.yerr[d.used] for d in snap.datasets])
        z = r / sig if rs is None else r / (sig * rs)
        logdet = 2.0 * float(np.sum(np.log(sig)))
    if rs is not None:
        logdet += 2.0 * float(np.sum(np.log(rs)))
    return float(z @ z) + logdet + len(r) * LN2PI + pr


OPT_COND_TOL = 1e-6     # correlation eigenvalues below this x the largest: direction dropped
OPT_MIN_RATIO = 0.25    # the checked step must achieve this share of the predicted decrease
OPT_NEGLIGIBLE = 0.01   # |change in -2 ln L| below this: already at the optimum


def optimum_distance(jf, snap, step=None, cache=None):
    """How far the fitted point is from the exact likelihood's optimum (spec §6.4 item 6).
    ``None`` unless C depends on the model (model mode, intensity-dependent term), and
    ``None`` for an emcee snapshot (posterior medians, not the fixed point). With
    ``snap.covar`` missing: ``{'unavailable': ...}``. Else the central-difference gradient
    g of ``exact_objective`` over ``snap.var_names`` and the Newton step
    ``δ = −½·covar·g`` (Φ's Hessian is about 2·covar⁻¹: on the covariance path lmfit runs
    with ``scale_covar=False``, so covar = (Jᵀ C⁻¹ J)⁻¹), taken in stderr units over the
    correlation eigen-directions above ``OPT_COND_TOL`` of the largest (near-null directions,
    where gradient noise is amplified without limit, are left out and counted in
    ``dropped``). The step is then checked with one evaluation at θ̂ + δ (clipped to the
    bounds): ``reliable`` only when it lowers Φ by at least ``OPT_MIN_RATIO`` of the
    predicted ``pred_change`` = −¼ g̃ᵀR g̃ (or both are below ``OPT_NEGLIGIBLE``). Returns
    ``{'delta', 'in_stderr', 'worst': (name, δ/stderr), 'fit_gap', 'pred_change',
    'actual_change', 'reliable', 'reason', 'dropped', 'n_directions', 'clipped'}``. Costs
    2·nvarys + 1 quiet model evaluations (one per ``exact_objective``) and as many
    factorisations; leaves every model and ``jf.result`` as
    they were. ``snap`` must be ``jf.last_snapshot`` (ValueError otherwise); ``jf.fit_abort``
    stops the loop (``DiagnosticSkipped``)."""
    if snap.method == 'emcee' or not _model_needs(snap):
        # an MCMC snapshot reports posterior medians, not the reweighting fixed point
        return None
    names = list(snap.var_names)
    cov = snap.covar
    if (cov is None or np.shape(cov) != (len(names), len(names))
            or not np.all(np.isfinite(cov))):
        return {'unavailable': _DEG_UNAVAILABLE}
    _check_live(jf, snap)
    g = np.zeros(len(names))
    with jf.quiet():
        for j, name in enumerate(names):
            _check_abort(jf)
            h = _step_for(snap, name, step)
            v = snap.params[name]
            fp = exact_objective(jf, snap, _params_at(jf, snap, {name: v + h}))
            fm = exact_objective(jf, snap, _params_at(jf, snap, {name: v - h}))
            g[j] = (fp - fm) / (2.0 * h)
    cov = np.asarray(cov, float)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if np.any(sd <= 0):
        return {'unavailable': 'parameter covariance is not positive definite (singular '
                               'Hessian: some parameters are exactly degenerate or unconstrained)'}
    # Newton step in stderr units on the well-determined directions only: a direction whose
    # correlation eigenvalue is below OPT_COND_TOL of the largest is barely constrained by
    # the data, and the finite-difference gradient's noise along it is amplified by 1/lambda
    corr = cov / np.outer(sd, sd)
    lam, V = np.linalg.eigh(0.5 * (corr + corr.T))
    keep = lam > OPT_COND_TOL * float(lam.max())
    proj = V.T @ (sd * g)
    delta = sd * (-0.5 * (V[:, keep] @ (lam[keep] * proj[keep])))
    pred = -0.25 * float(np.sum(lam[keep] * proj[keep] ** 2))
    dd, ins = {}, {}
    for j, name in enumerate(names):
        dd[name] = float(delta[j])
        se = snap.stderr.get(name)
        ins[name] = float(delta[j] / se) if se else None
    avail = [(n, v) for n, v in ins.items() if v is not None]
    worst = max(avail, key=lambda t: abs(t[1])) if avail else None
    # check the step with one evaluation (clipped to the bounds): the estimate is used only
    # when it actually lowers -2 ln L by a fair share of what the quadratic model predicts
    _, ex = per_dataset_objective(snap, cache)
    phi0 = ex['nll2'] + ex['prior']
    trial, clipped = {}, []
    for name in names:
        v = snap.params[name] + dd[name]
        p = jf.result.params[name]
        lo = p.min if p.min is not None and np.isfinite(p.min) else None
        hi = p.max if p.max is not None and np.isfinite(p.max) else None
        if lo is not None and v < lo:
            v = lo
        if hi is not None and v > hi:
            v = hi
        if v != snap.params[name] + dd[name]:
            clipped.append(name)
        trial[name] = v
    _check_abort(jf)
    with jf.quiet():
        actual = exact_objective(jf, snap, _params_at(jf, snap, trial)) - phi0
    if abs(pred) < OPT_NEGLIGIBLE and abs(actual) < OPT_NEGLIGIBLE:
        reliable, reason = True, None
    elif pred < 0 and actual < 0 and actual / pred >= OPT_MIN_RATIO:
        reliable, reason = True, None
    else:
        reliable = False
        reason = ('a step to the estimate changes -2 ln L by %+.4g against a predicted %+.4g'
                  % (actual, pred))
    return {'delta': dd, 'in_stderr': ins, 'worst': worst, 'fit_gap': ex['fit_gap'],
            'pred_change': pred, 'actual_change': float(actual), 'reliable': reliable,
            'reason': reason, 'dropped': int(np.sum(~keep)), 'n_directions': len(names),
            'clipped': clipped}


# ------------------------------------------------------------- diagnostics --


def _fmt_coef(c):
    return '%.2g' % abs(c)


def _degeneracy(snap, rho_tol=0.9):
    """Correlation pairs above ``rho_tol`` and the softest direction (spec §6.4 item 3).
    Reads only ``snap.covar``; coefficients are in stderr units because the correlation
    matrix is the stderr-normalised covariance."""
    cov = snap.covar
    names = list(snap.var_names)
    if cov is None and snap.method == 'emcee':
        return {'unavailable': _DEG_MCMC}      # lmfit's emcee result carries no covar
    if cov is None or len(names) == 0 or np.shape(cov) != (len(names), len(names)):
        return {'unavailable': _DEG_UNAVAILABLE}
    cov = np.asarray(cov, float)
    if not np.all(np.isfinite(cov)):
        return {'unavailable': _DEG_UNAVAILABLE}
    if np.any(np.diag(cov) <= 0):
        # an exactly singular Hessian leaves numerical junk (negative variances); a
        # correlation read from it would be spurious, so say what it means instead
        return {'unavailable': 'parameter covariance is not positive definite (singular '
                               'Hessian: some parameters are exactly degenerate or unconstrained)'}
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > rho_tol:
                pairs.append((names[i], names[j], float(corr[i, j])))
    pairs.sort(key=lambda t: -abs(t[2]))
    softest = []
    if len(names) >= 2:
        w, v = np.linalg.eigh(corr)
        vec = v[:, int(np.argmax(w))]
        vec = vec / vec[int(np.argmax(np.abs(vec)))]      # largest |coef| is +1
        order = np.argsort(-np.abs(vec), kind='stable')
        softest = [(names[i], float(vec[i])) for i in order if abs(vec[i]) >= 0.1]
    return {'pairs': pairs, 'softest': softest}


def _section(diag, name, fn):
    try:
        diag[name] = fn()
    except DiagnosticSkipped as e:
        diag[name] = {'skipped': str(e)}
    except Exception as e:                       # noqa: BLE001 -- degrade, report
        diag[name] = None
        diag['unavailable'].append((name, '%s: %s' % (type(e).__name__, e)))


def _restore_outputs(jf):
    """One quiet evaluation at ``jf.result.params``: ``quiet()`` restores the models' x and
    parameters but not what ``y()`` computed (outputs, caches), so after the diagnostics'
    finite differences this leaves those at the best fit, as the snapshot did."""
    with jf.quiet():
        jf.push_params(jf.result.params)
        jf.evaluate()


def diagnostics(jf, snap, individual_fits=None, *, qrange_tol=0.10, hetero_ratio=3.0,
                rho_tol=0.9, fisher=True, tension=True, optimum=True, mode=None):
    """End-of-fit diagnostics of a joint fit (spec §6, §6.3 "Always", §6.4 items 1-4).
    Every section is computed on its own; one that raises is listed in ``unavailable``
    as ``(section, '<ExcType>: <msg>')`` and never propagates. ``fisher`` (§6.2; nvarys
    model evaluations), ``tension`` (§6.4 item 5; ``None`` without a multiplicative term;
    no evaluation) and ``optimum_distance`` (§6.4 item 6; 2·nvarys model evaluations when C
    depends on the model) are skipped, ``None``, when their flag is False.

    ``mode`` (default ``jf.report_diagnostics``, itself ``'full'``): ``'cheap'`` runs no
    model evaluation -- the two model-evaluating sections become ``{'skipped': ...}`` --
    and neither does a fit stopped by the user (``jf.fit_abort``, also checked inside the
    finite-difference loops). When a model-evaluating section ran, one final quiet
    evaluation at ``jf.result.params`` leaves the model outputs at the best fit. One
    ``cache`` serves every section: C is assembled and factorised once, each marginal
    block once, and the per-dataset split is computed once."""
    if mode is None:
        mode = getattr(jf, 'report_diagnostics', 'full') or 'full'
    if mode not in ('full', 'cheap'):
        raise ValueError("report_diagnostics must be 'full' or 'cheap', got %r" % (mode,))
    d = {'unavailable': []}
    cache = {}
    ran = []                     # model-evaluating sections that ran (or started)

    def terms():
        return per_dataset_objective(snap, cache)

    def objective():
        ts, ex = terms()
        coup = abs(ex['coupling']) > 1e-9 * (1.0 + abs(sum(t.sequential for t in ts)))
        o = {'label': ex['objective_label'], 'likelihood': ex['likelihood'],
             'mode': snap.cov_mode if snap.cov_in_use else None, 'nll2': ex['nll2'],
             'notes': list(snap.notes), 'fit_label': snap.label}
        if coup:
            o['order'] = list(ex['order'])
        if snap.cov_in_use and snap.cov_mode == 'model':
            o['fit_gap'] = ex['fit_gap']
        return o

    def chi2n():
        """chi2 per point per dataset. A rescaled fit shows the UNRESCALED values: under
        its own s_i the term is ~1 by construction and would hide misfit (spec 6.3)."""
        ts, _ = terms()
        if snap.rescale is None:
            return [t.chi2_per_point for t in ts], False
        m = marginal_terms(snap, scaled=False, cache=cache)
        return [(mi / t.n_points) if t.n_points else None for mi, t in zip(m, ts)], True

    def quality():
        ts, _ = terms()
        vals, unresc = chi2n()
        rows = [{'name': t.name, 'n': t.n_points, 'chi2_per_point': v,
                 'unrescaled': unresc} for t, v in zip(ts, vals)]
        if individual_fits is not None:
            try:
                for row, it in zip(rows, individual_terms(jf, snap, individual_fits, cache)):
                    ok = it['unavailable'] is None
                    row['indiv_chi2_per_point'] = (it['indiv_m'] / it['n']) if ok and it['n'] else None
                    row['delta'] = it['delta']
                    row['flagged'] = it['flagged']
                    row['indiv_unavailable'] = it['unavailable']
            except Exception as e:               # noqa: BLE001 -- degrade, report
                d['unavailable'].append(('individual_fits', '%s: %s' % (type(e).__name__, e)))
        return rows

    def heterogeneity():
        v = [c for c in chi2n()[0] if c is not None]
        if len(v) < 2:
            return {'min': (v[0] if v else None), 'max': (v[0] if v else None),
                    'ratio': None, 'flagged': False}
        lo, hi = float(min(v)), float(max(v))
        ratio = hi / lo if lo > 0 else float('inf')
        return {'min': lo, 'max': hi, 'ratio': ratio, 'flagged': bool(ratio > hetero_ratio)}

    def qrange():
        qm = [float(x.window[1]) for x in snap.datasets]
        spread = (max(qm) / min(qm) - 1.0) if qm and min(qm) > 0 else None
        return {'qmax': qm, 'spread': spread,
                'flagged': bool(spread is not None and spread > qrange_tol),
                'mode': 'native windows'}

    def shared_warnings():
        return [w for w in snap.warnings if w.startswith('shared ')]

    def evaluating(fn, applies):
        """A model-evaluating section: skipped in the cheap mode or after a stopped fit
        (when it would evaluate at all), else run and remembered for the final restore."""
        def run():
            if applies():
                if mode == 'cheap':
                    raise DiagnosticSkipped(SKIP_CHEAP)
                _check_abort(jf)
                ran.append(1)
            return fn()
        return run

    def fisher_applies():
        return bool(_shared_masters(snap))

    def optimum_applies():
        if snap.method == 'emcee' or not _model_needs(snap):
            return False
        cov = snap.covar
        n = len(snap.var_names)
        return cov is not None and np.shape(cov) == (n, n) and bool(np.all(np.isfinite(cov)))

    _section(d, 'objective', objective)
    _section(d, 'quality', quality)
    if individual_fits is not None:
        ran.append(1)            # individual models were evaluated (possibly shared objects)
    _section(d, 'heterogeneity', heterogeneity)
    _section(d, 'qrange', qrange)
    _section(d, 'shared_warnings', shared_warnings)
    _section(d, 'degeneracy', lambda: _degeneracy(snap, rho_tol))
    _section(d, 'information_criteria', lambda: information_criteria(snap, cache))
    d['rescale'] = dict(snap.rescale) if snap.rescale is not None else None
    for name, flag, fn in (
            ('fisher', fisher, evaluating(lambda: fisher_shares(jf, snap, cache=cache),
                                          fisher_applies)),
            ('tension', tension, lambda: _tension(snap, cache)),
            ('optimum_distance', optimum,
             evaluating(lambda: optimum_distance(jf, snap, cache=cache), optimum_applies))):
        if flag:
            _section(d, name, fn)
        else:
            d[name] = None
    if ran and getattr(jf, 'result', None) is not None:
        try:
            _restore_outputs(jf)
        except Exception as e:                   # noqa: BLE001 -- degrade, report
            d['unavailable'].append(('restore', 'model outputs not re-evaluated at the best '
                                                'fit: %s: %s' % (type(e).__name__, e)))
    d['thresholds'] = {'qrange_tol': qrange_tol, 'hetero_ratio': hetero_ratio, 'rho_tol': rho_tol}
    d['mode'] = mode
    return d


SOFTEST_MAX_TERMS = 6      # the report lists this many softest-direction terms; data keeps all
DEGENERATE_MAX_PAIRS = 10  # ... and this many degenerate pairs (largest |rho| first)
FISHER_TABLE_MAX = 6       # more shared parameters than this: one line per parameter


def _softest_text(softest, max_terms=None):
    """``max_terms`` caps the rendered terms (largest |coef| first, as ``_degeneracy``
    orders them) and appends how many smaller ones were left out."""
    shown = softest if max_terms is None else softest[:max_terms]
    parts = []
    for k, (nm, c) in enumerate(shown):
        if k == 0:
            parts.append(nm if c > 0 else '-%s' % nm)
        else:
            parts.append('%s %s*%s' % ('-' if c < 0 else '+', _fmt_coef(c), nm))
    rest = len(softest) - len(shown)
    if rest > 0:
        parts.append('+ %d smaller term%s (|coef| <= %s)'
                     % (rest, '' if rest == 1 else 's', _fmt_coef(abs(softest[len(shown)][1]))))
    return ' '.join(parts)


def short_names(names):
    """Display labels for dataset names: a name carrying a file path (the GUI's
    ``'<i><>C:/dir/file.rrf'``) is shown by its file name (``'<i><>file.rrf'``). Names
    whose short form would collide with another keep their full form. Rendering only;
    every key and record keeps the full name."""
    names = list(names)
    short = []
    for nm in names:
        s = str(nm)
        head, sep, path = s.rpartition('<>')
        base = path.replace('\\', '/').rsplit('/', 1)[-1] or path
        short.append(head + sep + base)
    counts = {}
    for s in short:
        counts[s] = counts.get(s, 0) + 1
    return {nm: (s if counts[s] == 1 else str(nm)) for nm, s in zip(names, short)}


def _name_width(labels, floor=16):
    return max([floor] + [len(s) for s in labels])


def _fmt_objective(diag, th, L):
    o = diag.get('objective')
    if not o:
        return
    L.append('  objective: %s' % o['label'])
    if o.get('fit_label'):
        L.append('  fit state: %s' % o['fit_label'])
    if o.get('order'):
        sn = short_names(o['order'])
        L.append('  dataset order (sequential split): %s' % ', '.join(sn[n] for n in o['order']))
    if o.get('nll2') is not None:
        L.append('  -2 ln L = %.6g' % o['nll2'])
    if o.get('fit_gap') is not None:
        L.append('  fit_gap (exact chi2 at the fitted point - result.chisqr) = %+.4g'
                 % o['fit_gap'])
    for n in o.get('notes') or ():
        L.append('  note: %s' % n)


def _fmt_quality(diag, th, L):
    q = diag.get('quality')
    if not q:
        return
    L.append('  per-dataset quality (chi2 per point%s, each dataset scored under its own block):'
             % (' (unrescaled)' if any(r.get('unrescaled') for r in q) else ''))
    sn = short_names([r['name'] for r in q])
    w = _name_width(sn.values())
    for r in q:
        line = '    %-*s n = %-5d %s' % (w, sn[r['name']], r['n'], 'no points'
               if r['chi2_per_point'] is None or not r['n']
               else 'chi2/n = %.4g' % r['chi2_per_point'])
        if 'delta' in r:
            if r.get('indiv_unavailable'):
                line += '  indiv: unavailable (%s)' % r['indiv_unavailable']
            elif r['delta'] is not None:
                line += '  indiv chi2/n = %.4g  delta = %+.4g%s' % (
                    r['indiv_chi2_per_point'], r['delta'],
                    '  FLAGGED (shared value fits this dataset worse than the others)'
                    if r['flagged'] else '')
        L.append(line)
    if any('delta' in r for r in q):
        L.append('  indiv: the individual fit re-scored on the joint fit\'s used rows under its own '
                 'C_ii%s; delta = joint - indiv in -2 ln L units'
                 % (' (unrescaled)' if any(r.get('unrescaled') for r in q) else ''))


def _fmt_heterogeneity(diag, th, L):
    h = diag.get('heterogeneity')
    if h and h.get('ratio') is not None:
        L.append('  chi2/n spread: min %.4g, max %.4g, max/min = %.3g'
                 % (h['min'], h['max'], h['ratio']))
        if h['flagged']:
            L.append('  error bars or model adequacy differ between datasets '
                     '(max/min chi2/n > %g)' % th.get('hetero_ratio', 3.0))


def _fmt_qrange(diag, th, L):
    g = diag.get('qrange')
    if g and g.get('spread') is not None:
        L.append('  q-range (%s): qmax = %s, spread %.3g' % (
            g['mode'], ', '.join('%.4g' % v for v in g['qmax']), g['spread']))
        if g['flagged']:
            L.append('  q-ranges differ between datasets (%s; spread > %g%%)' % (
                g['mode'], 100 * th.get('qrange_tol', 0.1)))


def _fmt_shared(diag, th, L):
    for w in diag.get('shared_warnings') or ():
        L.append('  %s' % w)


def _fmt_degeneracy(diag, th, L):
    dg = diag.get('degeneracy')
    if not dg:
        return
    if 'unavailable' in dg:
        L.append('  degeneracy: %s' % dg['unavailable'])
        return
    if dg['pairs']:
        pairs = dg['pairs']
        for a, b, r in pairs[:DEGENERATE_MAX_PAIRS]:
            L.append('  degenerate pair: %s, %s (rho = %+.3f)' % (a, b, r))
        rest = pairs[DEGENERATE_MAX_PAIRS:]
        if rest:
            L.append('  ... and %d more pair%s with |rho| above %g (%d of all %d at |rho| >= 0.999)'
                     % (len(rest), '' if len(rest) == 1 else 's', th.get('rho_tol', 0.9),
                        sum(abs(t[2]) >= 0.999 for t in pairs), len(pairs)))
    else:
        L.append('  no parameter pair with |rho| above the threshold')
    if dg['softest']:
        L.append('  softest direction: %s (coefficients in stderr units)'
                 % _softest_text(dg['softest'], SOFTEST_MAX_TERMS))


def _fmt_fisher(diag, th, L):
    f = diag.get('fisher')
    if not f:
        return
    if 'skipped' in f:
        L.append('  information on the shared parameters: %s' % f['skipped'])
        return
    if not f['params']:
        L.append('  information on the shared parameters: no shared free parameter')
        return
    ps = f['params']
    L.append('  information on the shared parameters (Fisher, marginal; share per dataset):')
    sn = short_names(list(f['shares']))
    label = {p: p.lstrip('_') or p for p in ps}
    if len(ps) <= FISHER_TABLE_MAX:
        w = _name_width(sn.values())
        cw = max([12] + [len(label[p]) for p in ps])
        L.append('    %-*s %s' % (w, 'dataset', ' '.join('%*s' % (cw, label[p]) for p in ps)))
        for name, sh in f['shares'].items():
            L.append('    %-*s %s' % (w, sn[name], ' '.join(
                '%*s' % (cw, 'n/a' if sh.get(p) is None else '%.3g' % sh[p]) for p in ps)))
    else:
        # many shared parameters: one line each, listing only the datasets that inform it
        pw = max(len(label[p]) for p in ps)
        for p in ps:
            got = [(sn[nm], sh.get(p)) for nm, sh in f['shares'].items()]
            na = [d for d, v in got if v is None]
            parts = ['%s %.3g' % (d, v) for d, v in got if v is not None and v > 0]
            if na:
                parts.append('n/a: %s' % ', '.join(na))
            L.append('    %-*s  %s' % (pw, label[p], ', '.join(parts) if parts
                                       else 'no information'))
    if f.get('coupled'):
        L.append('    %s' % COUPLED_NOTE)
    if f.get('mean_information_only'):
        L.append('    %s' % MEAN_INFO_NOTE)
    if f.get('note'):
        L.append('    %s' % f['note'])


def _fmt_tension(diag, th, L):
    t = diag.get('tension')
    if not t:
        return
    L.append('  %s (per multiplicative error component: eps = Sum G^T C^-1 r, the shift the '
             'fit leaned on, and its pull eps/sqrt(Sigma_kk)):' % TENSION_HEAD)
    def _who(r):
        if r.get('scope') == 'shared' and ', ' in r['dataset']:
            return 'shared (%d datasets)' % len(r['dataset'].split(', '))
        return sn.get(r['dataset'], r['dataset'])
    sn = short_names([r['dataset'] for r in t if not (r.get('scope') == 'shared'
                                                     and ', ' in r['dataset'])])
    w = _name_width([_who(r) for r in t])
    for r in t:
        L.append('    %-*s %-12s %-10s eps = %+.4g  pull = %+.3g' % (
            w, _who(r), r['term'], r['label'], r['eps'], r['pull']))


def _fmt_optimum(diag, th, L):
    o = diag.get('optimum_distance')
    if not o:
        return
    if 'unavailable' in o:
        L.append('  distance to the exact-likelihood optimum: unavailable (%s)' % o['unavailable'])
        return
    if 'skipped' in o:
        L.append('  distance to the exact-likelihood optimum: %s' % o['skipped'])
        return
    w = o.get('worst')
    gap = o.get('fit_gap')
    poor = ''
    if o.get('dropped'):
        poor = ('; %d of %d parameter directions are too poorly determined to use'
                % (o['dropped'], o['n_directions']))
    if o.get('reliable') is False:
        L.append('  distance to the exact-likelihood optimum: not estimable from this fit (%s%s); '
                 'fit gap Delta chi2 = %s' % (o['reason'], poor,
                                             'n/a' if gap is None else '%+.4g' % gap))
        return
    line = ('  the fitted point is the reweighting fixed point; the exact-likelihood optimum is '
            'estimated %s stderr away (%s); fit gap Delta chi2 = %s' % (
                'n/a' if w is None else '%.3g' % abs(w[1]), 'n/a' if w is None else w[0],
                'n/a' if gap is None else '%+.4g' % gap))
    if o.get('pred_change') is not None:
        line += ('; -2 ln L would change by %+.4g (checked: %+.4g)'
                 % (o['pred_change'], o['actual_change']))
    line += poor
    if o.get('clipped'):
        line += '; step clipped at bounds for %d parameter%s' % (
            len(o['clipped']), '' if len(o['clipped']) == 1 else 's')
    if w is not None and abs(w[1]) > 0.3:
        line += '; the WS6 exact optimiser should settle it'
    L.append(line)


def _fmt_ic(diag, th, L):
    ic = diag.get('information_criteria')
    if ic:
        L.append('  information criteria (assume a correct model; read with caution): '
                 'AIC %.6g, BIC %.6g (k = %d, n = %d, %s%s)' % (
                     ic['aic'], ic['bic'], ic['k'], ic['n'], ic['form'],
                     '; ' + ic['note'] if ic.get('note') else ''))


def _fmt_rescale(diag, th, L):
    r = diag.get('rescale')
    if r:
        L.append('  errors rescaled per dataset by sqrt(chi2_red): s = %s (%d rounds, %s)' % (
            ', '.join('%.4g' % v for v in r['s']), r['rounds'],
            'converged' if r['converged'] else 'not converged'))


def format_diagnostics(diag):
    """Plain-text rendering of ``diagnostics`` (starts ``Joint-fit diagnostics:``). Each
    section renders in its own try: one that fails prints an 'unavailable' line."""
    th = diag.get('thresholds', {})
    L = ['Joint-fit diagnostics:']
    for name, fn in (('objective', _fmt_objective), ('quality', _fmt_quality),
                     ('heterogeneity', _fmt_heterogeneity), ('qrange', _fmt_qrange),
                     ('shared_warnings', _fmt_shared), ('degeneracy', _fmt_degeneracy),
                     ('fisher', _fmt_fisher), ('tension', _fmt_tension),
                     ('optimum_distance', _fmt_optimum),
                     ('information_criteria', _fmt_ic), ('rescale', _fmt_rescale)):
        part = []
        try:
            fn(diag, th, part)
            L.extend(part)
        except Exception as e:                   # noqa: BLE001 -- degrade, report
            L.append('  diagnostics unavailable (%s, rendering): %s: %s'
                     % (name, type(e).__name__, e))
    for sec, why in diag.get('unavailable') or ():
        L.append('  diagnostics unavailable (%s): %s' % (sec, why))
    return '\n'.join(L)


# ------------------------------------------------------------------------ LOO --
@dataclass
class LOOResult:
    """Leave-one-out refits (spec 7). ``estimates[m][i]`` is the value of shared master m
    in the refit that omitted dataset ``omitted[i]`` (None: m has no endpoint left, or that
    refit failed); ``influence`` is the shift from the full fit over the full-fit stderr
    (raw shift, ``units[m] == 'raw'``, when the stderr is unknown)."""
    omitted: list
    estimates: dict
    spread: dict
    influence: dict
    units: dict
    failures: list
    notes: list


def _loo_shared_masters(jf):
    """Varying master parameters that tie two or more endpoints."""
    return [k for k, t in jf.param_map.items()
            if len(t) >= 2 and k in jf.master_params and jf.master_params[k].vary]


def _loo_copy(jf, k, start, scale, method, windows, mode, fit_kws):
    """A fresh JointFit without dataset k (its covariance block is never added, which is
    exact marginalisation: shared terms among the remaining datasets stay shared), fitted
    from ``start`` on the full fit's windows (``imin``/``imax`` indices, not x values) and
    column masks, with its priors, error rescaling, covariance mode and fit keywords.
    Everything it touches is a deep copy; ``jf`` is not modified. The copy's report runs
    the cheap diagnostics (no model evaluation; the text is not used)."""
    import copy
    new = type(jf)()
    keep = [i for i in range(len(jf.objectives)) if i != k]
    remap = {old: j for j, old in enumerate(keep)}
    new.objectives = [copy.deepcopy(jf.objectives[i]) for i in keep]
    mp = {}
    for m, targets in jf.param_map.items():
        t = [(remap[i], lk) for (i, lk) in targets if i in remap]
        if t:
            mp[m] = t
    new.param_map = mp
    params = copy.deepcopy(jf.master_params)
    for m in list(params):
        if m not in mp:
            del params[m]
    for m in params:
        if m in start and not params[m].expr:
            params[m].value = start[m]
    new.master_params = params
    new.shared = list(jf.shared)
    new.covariance_mode = mode
    new.rescale_errors = jf.rescale_errors
    new.rescale_inflate_only = jf.rescale_inflate_only
    new._priors = {m: v for m, v in jf._priors.items() if m in mp}
    new.report_diagnostics = 'cheap'
    new._window_override = {o['name']: windows[o['name']] for o in new.objectives}
    first = new.objectives[0]
    x0 = np.asarray(first['x'], float)
    lo, hi = windows[first['name']]
    new.perform_fit(float(x0[lo]), float(x0[hi]), fit_scale=scale, fit_method=method,
                    covariance_mode=mode, **fit_kws)
    return new


def leave_one_out(jf, workers=1, fit_kws=None):
    """Refit N times, each omitting one dataset, warm-started from the full fit, on deep
    copies; ``jf`` is never mutated. See ``LOOResult``. The refits reuse the full fit's
    windows (as indices), column masks, priors, error rescaling, covariance mode and
    fit keywords (``maxiter`` and the solver keywords recorded by ``perform_fit``);
    ``fit_kws`` overrides those. ``workers`` is accepted for forward compatibility; the
    refits run serially (a note says so when workers > 1)."""
    n = len(jf.objectives)
    if n < 2:
        raise ValueError('leave-one-out needs at least 2 datasets')
    if jf._fit_scale is None:
        raise ValueError('leave-one-out needs a fitted JointFit (call perform_fit first)')
    notes = []
    if workers and workers > 1:
        notes.append('workers=%d requested; the refits ran serially' % workers)
    rec = getattr(jf, '_fit_args', None) or {}
    scale, method = jf._fit_scale, jf._fit_method
    kws = dict(rec.get('fit_kws') or {})
    if 'maxiter' in rec:
        kws['maxiter'] = rec['maxiter']
    kws.update(fit_kws or {})
    mode = rec.get('covariance_mode', jf.covariance_mode)
    if method == 'emcee':
        method = 'leastsq'
        notes.append('the full fit is an emcee run; the refits are least squares from its '
                     'posterior medians')
    full = jf.result.params if jf.result is not None else jf.master_params
    start = {m: full[m].value for m in jf.master_params if m in full}
    windows = dict(rec.get('windows') or {})
    for o in jf.objectives:
        windows.setdefault(o['name'], (int(o['imin']), int(o['imax'])))
    shared = _loo_shared_masters(jf)
    names = [o['name'] for o in jf.objectives]
    est = {m: [] for m in shared}
    failures = []
    for k in range(n):
        try:
            new = _loo_copy(jf, k, start, scale, method, windows, mode, kws)
            if new.result is None or new.result.params is None:
                raise RuntimeError('refit produced no result')
            row = {m: (float(new.result.params[m].value) if m in new.result.params else None)
                   for m in shared}
            if not getattr(new.result, 'success', True):
                notes.append('refit omitting %s did not report success' % names[k])
        except Exception as e:                     # noqa: BLE001 -- a failed refit is a row
            failures.append((names[k], '%s: %s' % (type(e).__name__, e)))
            row = {m: None for m in shared}
        for m in shared:                           # one entry per omission, always
            est[m].append(row[m])
    spread, influence, units = {}, {}, {}
    any_raw = False
    for m in shared:
        vals = [v for v in est[m] if v is not None]
        spread[m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        v0, se = float(full[m].value), full[m].stderr
        raw = se is None or not np.isfinite(se) or se <= 0
        units[m] = 'raw' if raw else 'stderr'
        any_raw |= raw
        influence[m] = [None if v is None else (v - v0 if raw else (v - v0) / se)
                        for v in est[m]]
    if any_raw:
        notes.append('no parameter stderr in the full fit for some masters; their influence '
                     'is the raw shift (units raw)')
    return LOOResult(omitted=names, estimates=est, spread=spread, influence=influence,
                     units=units, failures=failures, notes=notes)


def format_loo(res):
    """Text for a ``LOOResult`` (ascii only)."""
    L = ['Leave-one-out refits (%d datasets, each omitted in turn):' % len(res.omitted)]
    sn = short_names(res.omitted)
    w = _name_width(sn.values(), floor=12)
    if not res.estimates:
        L.append('  no shared varying parameter to compare')
    for m, ests in res.estimates.items():
        unit = res.units[m]
        L.append('  %s: spread (SD of the estimates) = %.6g' % (m, res.spread[m]))
        for nm, e, inf in zip(res.omitted, ests, res.influence[m]):
            if e is None:
                L.append('    without %-*s no estimate' % (w, sn[nm]))
            else:
                L.append('    without %-*s %.6g   influence %+.3g %s' % (
                    w, sn[nm], e, inf, 'stderr' if unit == 'stderr' else '(raw shift)'))
    for nm, why in res.failures:
        L.append('  refit without %s failed: %s' % (sn.get(nm, nm), why))
    for nt in res.notes:
        L.append('  note: %s' % nt)
    return '\n'.join(L)
