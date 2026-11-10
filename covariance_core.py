"""
covariance_core.py -- data model and loaders for general correlated-error fitting.

Qt-free; imports neither PyQt5 nor xmodfit.  Lives at the repo root, never in a
Functions/<category>/ folder (that folder is a model namespace).  See
handoff/17-spec-general-covariance.md, sections 4.2, 6.1 and 7.1.

A dataset's covariance is measurement error plus a list of terms:
    C = diag(sigma_meas**2) + sum_terms C_term
This module holds the objects, the loader registry, and assembly / factorisation /
whitening / outer iteration (spec 4.2-4.5).  It imports joint_covariance (for
cholesky_with_jitter); joint_covariance must never import this module (import cycle).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.linalg import solve_triangular

from joint_covariance import cholesky_with_jitter

KINDS = ('multiplicative', 'additive', 'dense')
SCOPES = ('dataset', 'shared')
SYM_TOL = 1e-12          # max|cov - cov.T| <= SYM_TOL * max|cov|
PSD_TOL = 1e-10          # smallest eigenvalue >= -PSD_TOL * largest eigenvalue


class CovarianceError(ValueError):
    """Every load-time refusal; the message names the rule (and the first bad index)."""


@dataclass
class CovTerm:
    name: str
    kind: str                                   # 'multiplicative' | 'additive' | 'dense'
    cov: np.ndarray                             # K x K, or N x N for dense
    weights: Optional[np.ndarray] = None        # N x K (not for dense)
    labels: Optional[List[str]] = None
    scope: str = 'dataset'                      # 'dataset' | 'shared'
    provenance: dict = field(default_factory=dict)


def _first_bad(bad):
    """First index of a boolean array (tuple for 2-D) as a printable string."""
    i = np.argwhere(bad)[0]
    return str(int(i[0])) if i.size == 1 else str(tuple(int(k) for k in i))


@dataclass
class DatasetCovariance:
    x: np.ndarray
    sigma_meas: np.ndarray
    terms: List[CovTerm]
    y: Optional[np.ndarray] = None              # v1: the y the sidecar was built for
    replaces_y: Optional[np.ndarray] = None     # legacy formats only
    source: str = ''
    loader: str = ''
    data_file: Optional[str] = None

    @property
    def n(self):
        return len(self.x)

    def subset(self, index):
        """Restrict to the points in `index`: rows of x, y, replaces_y, sigma_meas and
        every W; rows and columns of dense cov.  A K x K cov is untouched."""
        index = np.asarray(index)
        terms = []
        for t in self.terms:
            if t.kind == 'dense':
                cov = np.array(t.cov)[np.ix_(index, index)]
            else:
                cov = np.array(t.cov)
            w = None if t.weights is None else np.array(t.weights)[index]
            terms.append(CovTerm(t.name, t.kind, cov, w,
                                 None if t.labels is None else list(t.labels),
                                 t.scope, dict(t.provenance)))
        pick = lambda a: None if a is None else np.array(a)[index]
        return DatasetCovariance(np.array(self.x)[index], np.array(self.sigma_meas)[index],
                                 terms, pick(self.y), pick(self.replaces_y),
                                 self.source, self.loader, self.data_file)

    def match_problems(self, x, y=None, rtol=1e-4):
        """Reasons this covariance does not belong to (x, y); [] means it matches.
        y is compared only when this covariance carries one."""
        problems = []
        pairs = [('x', self.x, x)]
        if self.y is not None and y is not None:
            pairs.append(('y', self.y, y))
        for name, mine, theirs in pairs:
            mine, theirs = np.asarray(mine, float), np.asarray(theirs, float)
            if mine.shape != theirs.shape:
                problems.append('%s length %d does not match the covariance length %d'
                                % (name, theirs.size, mine.size))
                continue
            bad = ~np.isclose(theirs, mine, rtol=rtol, atol=0.0)
            if bad.any():
                i = int(np.argmax(bad))
                problems.append(
                    '%s mismatch at index %d (data %.6g, covariance %.6g; %d of %d points '
                    'differ by more than rtol %g)' % (name, i, theirs[i], mine[i],
                                                      int(bad.sum()), bad.size, rtol))
        return problems

    def y_for_fit(self, y_dataset):
        return self.replaces_y if self.replaces_y is not None else y_dataset

    def without_terms(self):
        return DatasetCovariance(self.x, self.sigma_meas, [], self.y, self.replaces_y,
                                 self.source, self.loader, self.data_file)


# ----------------------------------------------------------------- registry --
_LOADERS: Dict[str, tuple] = {}


def register_loader(name: str, accepts: Callable[[str], bool],
                    load: Callable[[str, Optional[str]], DatasetCovariance]):
    """Register a loader.  load_covariance tries them in registration order and uses the
    first whose accepts(path) is True.  Re-registering a name replaces it in place."""
    _LOADERS[name] = (accepts, load)


def load_covariance(path, raw_path=None):
    """Load a covariance sidecar through the registry; CovarianceError if unrecognised."""
    path = str(path)
    if not os.path.isfile(path):
        raise CovarianceError('covariance file not found: %s' % path)
    for name, (accepts, load) in list(_LOADERS.items()):
        try:
            ok = accepts(path)
        except Exception:
            ok = False
        if ok:
            return load(path, raw_path)
    raise CovarianceError('unrecognised covariance file format (no registered loader accepts it): %s'
                          % path)


# --------------------------------------------------------------- v1 JSON ------
def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (ValueError, UnicodeDecodeError) as e:
        raise CovarianceError('covariance file is not valid JSON (%s): %s' % (e, path))


def _v1_accepts(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
    except (ValueError, UnicodeDecodeError, OSError):
        return False
    # A JSON object without a format key is claimed too, so the loader can refuse it by name.
    return isinstance(doc, dict) and doc.get('format', 'xmodfit-covariance') == 'xmodfit-covariance'


def _vec(doc, key, n=None):
    if key not in doc:
        raise CovarianceError('missing required field %s' % key)
    try:
        a = np.asarray(doc[key], dtype=float)
    except (ValueError, TypeError):
        raise CovarianceError('%s is not a numeric list' % key)
    if a.ndim != 1:
        raise CovarianceError('%s must be a 1-D list' % key)
    if n is not None and a.size != n:
        raise CovarianceError('%s has length %d, expected N = %d (the length of x)'
                              % (key, a.size, n))
    return a


def _check_cov(label, cov, size):
    if cov.shape != (size, size):
        raise CovarianceError('%s: cov must be %d x %d, got %s' % (label, size, size, cov.shape))
    if not np.all(np.isfinite(cov)):
        raise CovarianceError('%s: cov is not finite (first bad index %s)'
                              % (label, _first_bad(~np.isfinite(cov))))
    scale = float(np.max(np.abs(cov))) if cov.size else 0.0
    asym = np.abs(cov - cov.T) > SYM_TOL * scale
    if asym.any():
        raise CovarianceError('%s: cov is not symmetric (first bad index %s)'
                              % (label, _first_bad(asym)))
    ev = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    if ev.size and ev[0] < -PSD_TOL * max(float(ev[-1]), 0.0):
        raise CovarianceError('%s: cov is not positive semi-definite (smallest eigenvalue %.3g, '
                              'largest %.3g)' % (label, ev[0], ev[-1]))


def _v1_term(raw, n, i):
    label = 'term %d' % i
    if not isinstance(raw, dict):
        raise CovarianceError('%s must be an object' % label)
    name = raw.get('name')
    if not isinstance(name, str) or not name:
        raise CovarianceError('%s: name is required' % label)
    label = 'term %r' % name
    kind = raw.get('kind')
    if kind not in KINDS:
        raise CovarianceError('%s: kind %r must be one of %s' % (label, kind, ', '.join(KINDS)))
    scope = raw.get('scope', 'dataset')
    if scope not in SCOPES:
        raise CovarianceError('%s: scope %r must be one of %s' % (label, scope, ', '.join(SCOPES)))
    try:
        cov = np.asarray(raw['cov'], dtype=float)
    except KeyError:
        raise CovarianceError('%s: cov is required' % label)
    except (ValueError, TypeError):
        raise CovarianceError('%s: cov is not a numeric matrix' % label)
    if kind == 'dense':
        if raw.get('weights') is not None:
            raise CovarianceError('%s: a dense term has no weights (cov is the N x N '
                                  'covariance itself)' % label)
        _check_cov(label, cov, n)
        return CovTerm(name, kind, cov, None, None, scope, dict(raw.get('provenance') or {}))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise CovarianceError('%s: cov must be a square K x K matrix, got shape %s'
                              % (label, cov.shape))
    k = cov.shape[0]
    if raw.get('weights') is None:
        raise CovarianceError('%s: weights (N x K) are required for kind %r' % (label, kind))
    try:
        w = np.asarray(raw['weights'], dtype=float)
    except (ValueError, TypeError):
        raise CovarianceError('%s: weights is not a numeric matrix' % label)
    if w.shape != (n, k):
        raise CovarianceError('%s: weights must be N x K = %d x %d, got %s'
                              % (label, n, k, w.shape))
    if not np.all(np.isfinite(w)):
        raise CovarianceError('%s: weights not finite (first bad index %s)'
                              % (label, _first_bad(~np.isfinite(w))))
    _check_cov(label, cov, k)
    labels = raw.get('labels')
    if labels is not None:
        labels = [str(s) for s in labels]
        if len(labels) != k:
            raise CovarianceError('%s: labels has %d entries, expected K = %d'
                                  % (label, len(labels), k))
    return CovTerm(name, kind, cov, w, labels, scope, dict(raw.get('provenance') or {}))


def _refuse_duplicate_names(terms):
    """Term names are unique within one covariance: two shared terms of one name would
    be one set of random variables counted once (final review T2)."""
    seen = set()
    for t in terms:
        if t.name in seen:
            raise CovarianceError('term %r: duplicate term name (each term of a covariance '
                                  'needs its own name)' % t.name)
        seen.add(t.name)


def _v1_load(path, raw_path=None):
    doc = _read_json(path)
    if not isinstance(doc, dict) or doc.get('format') != 'xmodfit-covariance':
        raise CovarianceError('format key is required and must be "xmodfit-covariance"')
    if doc.get('version') != 1:
        raise CovarianceError('unsupported version %r (this build reads version 1)'
                              % (doc.get('version'),))
    x = _vec(doc, 'x')
    n = x.size
    if not np.all(np.isfinite(x)):
        raise CovarianceError('x is not finite (first bad index %s)'
                              % _first_bad(~np.isfinite(x)))
    sigma = _vec(doc, 'sigma_meas', n)
    bad = ~(np.isfinite(sigma) & (sigma > 0))
    if bad.any():
        raise CovarianceError('sigma_meas must be finite and > 0 (first bad index %s, value %r)'
                              % (_first_bad(bad), float(sigma[np.argmax(bad)])))
    y = _vec(doc, 'y', n) if doc.get('y') is not None else None
    terms = [_v1_term(t, n, i) for i, t in enumerate(doc.get('terms') or [])]
    _refuse_duplicate_names(terms)
    return DatasetCovariance(x, sigma, terms, y, None, str(path), 'xmodfit-covariance-v1',
                             doc.get('data_file'))


register_loader('xmodfit-covariance-v1', _v1_accepts, _v1_load)


# ------------------------------------------------------------------ assembly --
@dataclass
class Block:
    """One dataset's active points as they enter the residual."""
    cov: Optional[DatasetCovariance]      # already restricted to the active points
    y: np.ndarray                         # data in the residual
    yerr: np.ndarray                      # used only when cov is None
    intensity: np.ndarray                 # D: the data ("data" mode) or the model ("model")
    name: str = ''                        # dataset name, used in warnings


def needs_intensity(blocks, scale):
    """True when any term's contribution depends on the intensity D (spec 4.4, 4.5)."""
    for b in blocks:
        if b.cov is None:
            continue
        for t in b.cov.terms:
            if scale == 'Linear' and t.kind == 'multiplicative':
                return True
            if scale == 'Log' and t.kind in ('additive', 'dense'):
                return True
    return False


def _loading(term, block, scale):
    """N x K matrix G with the term's contribution equal to G Sigma G^T (not for dense)."""
    w = np.asarray(term.weights, float)
    d = np.asarray(block.intensity, float)
    if scale == 'Linear':
        return d[:, None] * w if term.kind == 'multiplicative' else w
    return w if term.kind == 'multiplicative' else w / d[:, None]


def _describe(bi, block):
    src = os.path.basename(block.cov.source) if block.cov is not None and block.cov.source else ''
    if block.name:
        return '%s (%s)' % (block.name, src) if src else block.name
    return '%d (%s)' % (bi, src) if src else str(bi)


def _same_term(a, b):
    # labels are compared only when both copies carry real ones: a legacy dict without
    # SF_LIST has labels None and matches any labels of the same K (final review M6)
    la, lb = a.labels or None, b.labels or None
    if a.kind != b.kind or (la is not None and lb is not None and la != lb):
        return False
    ca, cb = np.asarray(a.cov), np.asarray(b.cov)
    return ca.shape == cb.shape and bool(np.allclose(ca, cb, rtol=1e-12, atol=0.0))


def assemble(blocks, scale):
    """C over the concatenation of `blocks` (in order) for scale 'Linear' or 'Log'.
    Returns (C, warnings).  Shared terms of the same name are one set of random variables
    and couple the blocks that carry them; if their labels or cov differ, every copy is
    made private and one warning names the term and the datasets."""
    if scale not in ('Linear', 'Log'):
        raise ValueError("scale must be 'Linear' or 'Log', got %r" % (scale,))
    sizes = [len(b.y) for b in blocks]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    n = int(offs[-1])
    C = np.zeros((n, n))
    warnings = []
    shared = {}                                    # name -> [(block index, term)]
    for bi, b in enumerate(blocks):
        sl = slice(offs[bi], offs[bi + 1])
        y = np.asarray(b.y, float)
        if b.cov is None:
            sig = np.asarray(b.yerr, float)
        else:
            sig = np.asarray(b.cov.sigma_meas, float)
        meas = sig ** 2 if scale == 'Linear' else (sig / y) ** 2
        C[sl, sl][np.diag_indices(sizes[bi])] += meas
        if b.cov is None:
            continue
        e = 1.0 / np.asarray(b.intensity, float)
        names = [t.name for t in b.cov.terms if t.scope == 'shared' and t.kind != 'dense']
        dup = sorted(set(nm for nm in names if names.count(nm) > 1))
        if dup:
            raise CovarianceError('term %r: dataset %s has two shared terms of this name; '
                                  'they would be counted once' % (dup[0], _describe(bi, b)))
        for t in b.cov.terms:
            if t.kind == 'dense':
                cd = np.asarray(t.cov, float)
                C[sl, sl] += cd if scale == 'Linear' else e[:, None] * cd * e[None, :]
            elif t.scope == 'shared':
                shared.setdefault(t.name, []).append((bi, t))
            else:
                g = _loading(t, b, scale)
                C[sl, sl] += g @ np.asarray(t.cov, float) @ g.T
    for name, copies in shared.items():
        # pairwise: with labels None matching anything, "same as the first" is not transitive
        if len(copies) > 1 and not all(_same_term(a[1], b[1]) for i, a in enumerate(copies)
                                       for b in copies[i + 1:]):
            warnings.append('shared term %r differs between datasets %s (labels or covariance); '
                            'treated as private to each dataset'
                            % (name, ', '.join(_describe(bi, blocks[bi]) for bi, _ in copies)))
            copies = [[c] for c in copies]
        else:
            copies = [copies]
        for group in copies:
            k = np.asarray(group[0][1].cov).shape[0]
            G = np.zeros((n, k))
            for bi, t in group:
                G[offs[bi]:offs[bi + 1]] = _loading(t, blocks[bi], scale)
            C += G @ np.asarray(group[0][1].cov, float) @ G.T
    return 0.5 * (C + C.T), warnings


# ------------------------------------------------- factorise, whiten, iterate --
def factorize(C):
    """Lower Cholesky factor of C and a list of warnings.  Plain Cholesky first; if that
    fails, joint_covariance.cholesky_with_jitter, reporting the jitter used or that the
    eigenvalues had to be floored."""
    C = 0.5 * (np.asarray(C, float) + np.asarray(C, float).T)
    try:
        return np.linalg.cholesky(C), []
    except np.linalg.LinAlgError:
        pass
    L, jitter = cholesky_with_jitter(C)
    target = C + jitter * np.eye(C.shape[0])
    if np.linalg.norm(L @ L.T - target) > 1e-8 * np.linalg.norm(C):
        return L, ['covariance matrix is not positive definite: eigenvalues floored']
    return L, ['covariance matrix is not positive definite: added diagonal jitter %.3g' % jitter]


def whiten(L, r):
    return solve_triangular(L, r, lower=True, check_finite=False)


def whiten_jac(L, J):
    return solve_triangular(L, J, lower=True, check_finite=False)


def _moved(old, new, tol):
    """True if any varying parameter changed by >= tol x the newer stderr."""
    for name, p in new.params.items():
        if not getattr(p, 'vary', False):
            continue
        q = old.params[name]
        dv = abs(p.value - q.value)
        se = getattr(p, 'stderr', None)
        if se:
            if dv >= tol * se:
                return True
        elif dv >= 1e-6 * max(abs(p.value), abs(q.value), 1e-300):
            return True
    return False


def outer_iterate(fit_once, rebuild, max_rounds=5, tol=0.1):
    """Model-based covariance iteration (spec 4.4).  fit_once() fits from the current
    parameters; rebuild(result) re-assembles C from the model at result.params.  Round 0 is
    the data-based fit; up to max_rounds model rounds follow.  Returns
    (last_result, model_rounds, converged)."""
    result = fit_once()
    for rounds in range(1, max_rounds + 1):
        rebuild(result)
        new = fit_once()
        moved = _moved(result, new, tol)
        result = new
        if not moved:
            return result, rounds, True
    return result, max_rounds, False


# ------------------------------------------------------------------- report --
def chi2_red_band(nfree, coverage=0.99):
    """Central `coverage` interval of chi2(nfree)/nfree, or None when nfree < 1."""
    if not nfree or nfree < 1:
        return None
    from scipy.stats import chi2
    a = 0.5 * (1.0 - coverage)
    return chi2.ppf(a, nfree) / nfree, chi2.ppf(1.0 - a, nfree) / nfree


def _term_k(t):
    return 'N=%d (dense)' % np.asarray(t.cov).shape[0] if t.kind == 'dense' \
        else str(np.asarray(t.cov).shape[0])


def report_block(covs, info, warnings, result):
    """The report's 'Covariance:' block (spec 6.6) and the chi2_red line (spec 6.2).

    covs: [(dataset name, DatasetCovariance | None)]; info: mode, scale, rounds,
    converged, factor_warnings (and optionally jitter); warnings: covariance
    downgrades to list; result: the lmfit result (redchi, nfree)."""
    info = info or {}
    out = ['Covariance:']
    for name, cov in covs:
        if cov is None:
            out.append('  %s: none (diagonal errors from the dataset)' % name)
            continue
        out.append('  %s: %s (loader %s)' % (name, cov.source or '<in memory>',
                                             cov.loader or '?'))
        if not cov.terms:
            out.append('    no terms (measurement errors only)')
        for t in cov.terms:
            out.append('    term %s: kind=%s scope=%s K=%s' % (t.name, t.kind, t.scope,
                                                              _term_k(t)))
    out.append('  mode: %s' % info.get('mode', '?'))
    out.append('  scale: %s' % info.get('scale', '?'))
    rounds = int(info.get('rounds', 0) or 0)
    if info.get('mcmc_frozen'):
        out.append('  model rounds: n/a (MCMC: C frozen for the run at the %s)'
                   % ('best-fit model' if info['mcmc_frozen'] == 'model' else 'data'))
    elif info.get('mode') == 'model' and info.get('aborted'):
        out.append('  model rounds: %d (aborted by the user, not converged)' % rounds)
    elif info.get('mode') == 'model' and rounds:
        out.append('  model rounds: %d (%s)' % (rounds, 'converged' if info.get('converged')
                                                else 'round limit reached, not converged'))
    elif info.get('mode') == 'model':
        out.append('  model rounds: 0 (no term depends on the intensity at this scale)')
    else:
        out.append('  model rounds: 0 (data-based covariance)')
    fw = list(info.get('factor_warnings') or [])
    out.append('  jitter/floor: %s' % ('; '.join(fw) if fw else 'none'))
    for w in warnings or []:
        out.append('  downgrade: %s' % w)
    redchi = getattr(result, 'redchi', None)
    nfree = getattr(result, 'nfree', None)
    out.append('  objective: whitened, r_w = L^-1 r with C = L L^T; Chi-Sqr, Red_Chi_Sqr and '
               'the residuals shown are of r_w')
    if redchi is not None:
        out.append('  chi2_red = %.6g (errors are absolute, not rescaled by chi2_red)'
                   % redchi)
        band = chi2_red_band(nfree)
        if band is not None and redchi > band[1]:
            out.append('  WARNING: chi2_red is above the central 99%% of chi2(nfree)/nfree '
                       '[%.4g, %.4g] (nfree = %d): the misfit exceeds what the stated errors '
                       'allow, and the error bars cover noise only'
                       % (band[0], band[1], nfree))
        elif band is not None and redchi < band[0]:
            out.append('  WARNING: chi2_red is below the central 99%% of chi2(nfree)/nfree '
                       '[%.4g, %.4g] (nfree = %d): the stated errors are larger than this '
                       'misfit suggests, so the error bars may be too wide'
                       % (band[0], band[1], nfree))
    return '\n'.join(out)


# ------------------------------------------------------ legacy bilayer formats --
import joint_covariance as _jc


def from_legacy_meta(meta, x, source, loader):
    """DatasetCovariance from today's cov dict (y_corr, sigma_meas, k_index, Sigma_logc and
    optionally v + var_fwhm); spec 7.2.  x comes from the caller (the dict has none).

    attenuator_cal: multiplicative, shared, one-hot W from k_index, labels = SF_LIST
    strings (None when the meta carries no SF_LIST, so it shares with any labelled copy
    of the same Sigma_logc).  beam_fwhm (only when v is
    present and var_fwhm != 0): multiplicative, shared, W = v / y_corr, cov = [[var_fwhm]],
    so that in data mode (intensity = y_corr) it equals var_fwhm * v v^T exactly."""
    y_corr = np.asarray(meta['y_corr'], float)
    sigma = np.asarray(meta['sigma_meas'], float)
    sig_c = np.asarray(meta['Sigma_logc'], float)
    k_index = np.asarray(meta['k_index']).astype(int)
    kk = sig_c.shape[0]
    n = y_corr.size
    w = np.zeros((n, kk))
    w[np.arange(n), k_index] = 1.0
    sf = meta.get('SF_LIST')
    labels = [str(float(s)) for s in sf] if sf is not None else None
    terms = [CovTerm('attenuator_cal', 'multiplicative', sig_c, w, labels, 'shared',
                     {'origin': 'legacy scale-factor correction'})]
    v = meta.get('v')
    var_fwhm = float(meta.get('var_fwhm') or 0.0)
    if v is not None and var_fwhm:
        v = np.asarray(v, float)
        bad = (v != 0) & ~(np.isfinite(y_corr) & (y_corr != 0))
        if bad.any():
            raise CovarianceError('beam_fwhm: y_corr is zero or not finite where the footprint '
                                  'sensitivity v is nonzero (first bad index %s); the term is '
                                  'defined relative to y_corr' % _first_bad(bad))
        wv = np.zeros(n)
        nz = v != 0
        wv[nz] = v[nz] / y_corr[nz]                # 0 where v == 0, whatever y_corr is
        terms.append(CovTerm('beam_fwhm', 'multiplicative', np.array([[var_fwhm]]),
                             wv[:, None], None, 'shared',
                             {'origin': 'legacy shared-beam footprint'}))
    return DatasetCovariance(np.asarray(x, float), sigma, terms, None, y_corr,
                             str(source), loader)


def _is_bundle(path):
    try:
        with np.load(path, allow_pickle=True) as z:
            return any('__' in k for k in z.keys())
    except Exception:
        return False


def _npz_accepts(path):
    # a bundle is claimed too so _npz_load can refuse it by name (use load_bundle)
    return path.lower().endswith('.npz')


def _npz_load(path, raw_path=None):
    if _is_bundle(path):
        raise CovarianceError('%s is a multi-dataset bundle; use load_bundle(path)' % path)
    try:
        meta = _jc.load_cov_npz(path)
    except (KeyError, ValueError) as e:
        raise CovarianceError('%s: %s' % (path, e))
    return from_legacy_meta(meta, meta['x'], path, 'legacy-npz')


def _cf_accepts(path):
    return path.lower().endswith('_c_factors.txt')


def _cf_load(path, raw_path=None):
    if raw_path is None:
        raise CovarianceError('%s: a *_c_factors.txt sidecar needs raw_path, the raw 4-column '
                              'data file (x, y, err, sf), to recover y_corr / sigma_meas / '
                              'k_index' % path)
    try:
        meta = _jc.load_dataset_cov(path, raw_path=raw_path)
        x = _jc.load_rrf(raw_path)['x']
    except (KeyError, ValueError, OSError) as e:
        raise CovarianceError('%s: %s' % (path, e))
    return from_legacy_meta(meta, x, path, 'legacy-c-factors')


def load_bundle(path, datasets=None):
    """The multi-dataset xrr_corrected_bundle npz as {dataset name: DatasetCovariance}.
    Different var_fwhm between datasets is refused (CovarianceError)."""
    try:
        metas, _ = _jc.load_bundle_npz(str(path), datasets)
    except (KeyError, ValueError) as e:
        raise CovarianceError('%s: %s' % (path, e))
    return {nm: from_legacy_meta(m, m['x'], path, 'legacy-bundle') for nm, m in metas.items()}


register_loader('legacy-npz', _npz_accepts, _npz_load)
register_loader('legacy-c-factors', _cf_accepts, _cf_load)
