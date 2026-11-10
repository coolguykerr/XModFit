"""
covariance_core.py -- data model and loaders for general correlated-error fitting.

Qt-free; imports neither PyQt5 nor xmodfit.  Lives at the repo root, never in a
Functions/<category>/ folder (that folder is a model namespace).  See
handoff/17-spec-general-covariance.md, sections 4.2, 6.1 and 7.1.

A dataset's covariance is measurement error plus a list of terms:
    C = diag(sigma_meas**2) + sum_terms C_term
This module holds the objects and the loader registry.  Assembly and whitening arrive
in a later task.  joint_covariance.py must never import this module (import cycle).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

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
    return DatasetCovariance(x, sigma, terms, y, None, str(path), 'xmodfit-covariance-v1',
                             doc.get('data_file'))


register_loader('xmodfit-covariance-v1', _v1_accepts, _v1_load)
