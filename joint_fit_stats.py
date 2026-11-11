"""
joint_fit_stats.py -- WS2 joint-fit reporting: snapshot, per-dataset split, exact
Gaussian likelihood and objective labels (spec handoff/20-spec-ws2-reporting.md §3).

Qt-free and model-free after ``snapshot``: everything downstream of the snapshot reads
only the ``JointSnapshot`` (spec §3.1). ``snapshot`` itself evaluates the live
``JointFit`` once, inside ``JointFit.quiet()``, so it neither counts ``Niter`` nor emits
``functionCalled`` and leaves every model as it was.

The split (§3.2): with L = chol(C(θ̂)) over the concatenated used rows in fit order,
``z = L⁻¹ r``. The sequential term of dataset i is ``‖z_i‖²`` (χ² of i given the
datasets before it; they sum exactly to the data part of the objective), the marginal
term is ``r_iᵀ (L Lᵀ)_ii⁻¹ r_i`` (i scored under its own block). The log-determinant
splits the same way row by row (§3.5): ``ln det C = 2 Σ ln L_jj``.

This module must stay out of ``Functions/<category>/`` (CLAUDE.md rule 1).
"""
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
def raw_parts(jf, params):
    """Per dataset ``(r_i, f_i)`` on its used rows at ``params``, unwhitened, assembled
    as ``JointFit.residual`` assembles them (slice, then imask -- honouring
    ``yfit_masked`` -- then the dataset's ``_cov_mask`` segment on the covariance path),
    evaluated inside ``quiet()``. ``r`` is d - f (Linear), ln d - ln f (Log), and for
    the unweighted scales the residual rows themselves."""
    scale = jf._fit_scale
    cov_path = _cov_path(jf, scale)
    used = _used_masks(jf, cov_path)
    with jf.quiet():
        jf.push_params(params)
        yfits = jf.evaluate()
        flags = [bool(o.get("yfit_masked")) for o in jf.objectives]
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

    # (1) the residual the fit minimised, quietly, at the reported parameters
    with jf.quiet():
        full = np.asarray(jf.residual(params, scale), float)
    prior = np.asarray(jf.prior_rows(params), float)
    z_fit = full[:len(full) - len(prior)].copy()

    parts = raw_parts(jf, params)
    ranges = jf.row_ranges()
    used = _used_masks(jf, cov_path)

    fit_blocks = exact_blocks = None
    if cov_path:
        fit_blocks = list(jf._last_blocks)
        exact_blocks = fit_blocks
        if cov_mode == 'model':
            exact_blocks, _notes = jf.model_blocks(params, scale)
    needs = cov_path and cov_mode == 'model' and cc.needs_intensity(fit_blocks, cscale)

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
        else:
            logdiag = np.zeros(0)
    elif needs:
        # the one case whose C(θ̂) differs from the C the fit used: factorise afresh
        C, _aw = cc.assemble(exact_blocks, cscale)
        L, fwarn = cc.factorize(C)
        jitter = _jitter_of(C, fwarn)
        rcat = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
        z = cc.whiten(L, rcat)
        logdiag = np.log(np.diag(L))
    else:
        z = z_fit
        logdiag = np.log(np.diag(jf._L))
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
        warnings=tuple(str(w) for w in jf.warnings), label=label, rescale=None)


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


def marginal_terms(snap, L=None):
    """``m_i = r_iᵀ (L Lᵀ)_ii⁻¹ r_i`` with the exact L; equals the sequential term off
    the covariance path. (Task 5's rescale divides r_i by s_i here.)"""
    if not snap.cov_in_use:
        return [float(np.sum(snap.z[a:b] ** 2)) for (a, b) in (d.rows for d in snap.datasets)]
    if L is None:
        L = rebuild_L(snap)
    out = []
    for d in snap.datasets:
        a, b = d.rows
        if b <= a:
            out.append(0.0)
            continue
        r = d.r                               # rescale hook (Task 5): r / s_i
        out.append(float(r @ cho_solve(cho_factor(_block_of(L, a, b), lower=True), r)))
    return out


def objective_label(fit_scale, cov_in_use, cov_mode, rescaled=None):
    """The §3.4 name of the objective and whether it is a (Gaussian) likelihood."""
    if fit_scale == 'Log w/o error':
        return 'unweighted sum of squares of log10 y', False
    if fit_scale == 'Linear w/o error':
        return 'unweighted sum of squares of y / max(y), per dataset', False
    log = fit_scale == 'Log'
    if cov_in_use:
        lab = ('GLS χ² of ln y (Log, covariance %s)' if log
               else 'GLS χ² (Linear, covariance %s)') % cov_mode
        if cov_mode == 'data':
            lab += _DATA_CLAUSE
    else:
        lab = 'χ² of ln y (Log, σ_ln = σ/y)' if log else 'χ² (Linear, weighted by the error column)'
    if rescaled:
        s = rescaled.get('s') if isinstance(rescaled, dict) else None
        if isinstance(s, dict):
            s_txt = ', '.join('%s %.4g' % (k, v) for k, v in s.items())
        elif s is not None:
            s_txt = ', '.join('%.4g' % v for v in np.atleast_1d(s))
        else:
            s_txt = '…'
        lab = ('rescaled ' + lab + ' (errors rescaled per dataset by √χ²_red, s = %s; '
               'error scales fitted per dataset; approximate)' % s_txt)
    return lab, True


def per_dataset_objective(snap):
    """The per-dataset split of the objective (§3.2, §3.3, §3.5).
    Returns ``(list[DatasetTerm], extras)``."""
    weighted = _weighted(snap.fit_scale)
    L = rebuild_L(snap) if snap.cov_in_use else None
    marg = marginal_terms(snap, L)
    terms = []
    for d, m in zip(snap.datasets, marg):
        a, b = d.rows
        n = int(b - a)
        s = float(np.sum(snap.z[a:b] ** 2)) if n else 0.0
        m = float(m) if n else 0.0
        lds = ldm = nls = nlm = None
        if weighted:
            lds = float(2.0 * np.sum(snap.logdiag[a:b])) if n else 0.0
            if L is None or not n:
                ldm = lds
            else:
                sign, ldm = np.linalg.slogdet(_block_of(L, a, b))
                ldm = float(ldm) if sign > 0 else float('nan')
            nls = s + lds + n * LN2PI
            nlm = m + ldm + n * LN2PI
        terms.append(DatasetTerm(
            name=d.name, n_points=n, sequential=s, marginal=m,
            chi2_per_point=(m / n) if n else None, logdet_seq=lds, logdet_marg=ldm,
            nll2_seq=nls, nll2_marg=nlm, window=d.window, index_hash=d.index_hash,
            scale=snap.fit_scale, cov=d.cov is not None, cov_mode=snap.cov_mode,
            rescale_s=None))
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
    sa = list(_endpoint_sets(a).values())
    for sb in _endpoint_sets(b).values():
        union = set()
        for e in sa:
            if e <= sb:
                union |= e
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
    notes = []
    z = None
    if N < 4:
        notes.append(NO_Z_NOTE)
    elif sd is not None and sd > 0:
        z = float(abs(sum_d) / (sd * np.sqrt(N)))
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

    def _coupled(e, t):
        return abs(e['coupling']) > 1e-9 * max(1.0, sum(abs(x.sequential) for x in t))

    return Comparison(
        names=names, d=d, sum_d=sum_d, sd_d=sd, N=N, z=z, preferred=preferred,
        sign_count=(cnt, N), delta_k=delta_k, nested=nest, lr_p=lr_p, worst_dataset=worst,
        objective_label=ea['objective_label'], likelihood=likelihood, delta_chi2=dchi,
        delta_logdet=dld, prior_difference=float(ea['prior'] - eb['prior']),
        model_differences=_model_differences(a, b), notes=notes,
        coupled=bool(_coupled(ea, ta) or _coupled(eb, tb)), rescaled=rescaled)


def information_criteria(snap):
    """Known-error AIC / BIC on the §3 objective (spec §8). Weighted: ``Φ + 2k`` and
    ``Φ + k ln n`` with ``Φ = nll2 + prior``; unweighted: the ``n ln(RSS/n)`` form."""
    terms, ex = per_dataset_objective(snap)
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
    if c.z is not None:
        head = 'Δ = %+.4g ± %.3g (z = %.2f)' % (c.sum_d, c.sd_d * np.sqrt(c.N), c.z)
    else:
        head = 'Δ = %+.4g' % c.sum_d
    lines.append(head + '   [Σd, d_i = ½(nll2_A − nll2_B); positive prefers B; %s]'
                 % c.objective_label)
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
    for nm, v in zip(c.names, c.d):
        lines.append('  %-16s %+.4g' % (nm, v))
    lines.append('largest |d|: %s' % c.worst_dataset)
    det = 'Δ(−logL) = %+.6g (not a confidence measure)' % c.sum_d
    if c.delta_logdet != 0:
        det += ' (Δχ²/2 = %+.6g, Δ log det/2 = %+.6g)' % (0.5 * c.delta_chi2,
                                                          0.5 * c.delta_logdet)
    lines.append(det)
    for m in c.model_differences:
        lines.append(m)
    if c.nested:
        s = 'nested: %s, Δk = %d' % (c.nested, c.delta_k)
        if c.lr_p is not None:
            s += ', likelihood-ratio p = %.3g; %s' % (c.lr_p, LR_CAVEAT)
        lines.append(s)
    else:
        lines.append('not nested; no likelihood-ratio p-value')
    lines.append('prior terms difference (A − B) = %+.6g (not in Σd)' % c.prior_difference)
    if ic_a is not None or ic_b is not None:
        lines.append('information criteria (assume a correct model; read with caution)')
        for lab, ic in (('A', ic_a), ('B', ic_b)):
            if ic is not None:
                lines.append('  %s: AIC %.6g  BIC %.6g  (k = %d, n = %d, %s%s)' % (
                    lab, ic['aic'], ic['bic'], ic['k'], ic['n'], ic['form'],
                    '; ' + ic['note'] if ic.get('note') else ''))
        if ic_a is not None and ic_b is not None:
            lines.append('  ΔAIC (A − B) = %+.6g  ΔBIC (A − B) = %+.6g' % (
                ic_a['aic'] - ic_b['aic'], ic_a['bic'] - ic_b['bic']))
    return '\n'.join(lines)
