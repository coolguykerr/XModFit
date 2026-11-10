"""
Joint_Fit_Routines.py  --  the XModJointFit joint-fit engine ("Option B")
=========================================================================

A generalized joint / global objective for XModFit, modelled on refnx's
``GlobalObjective`` pattern.  It sits *alongside* the stock single-dataset
``Fit`` (in ``Fit_Routines.py``) and is only used when the user explicitly
requests a joint fit; a 1-dataset, no-covariance ``JointFit`` reduces exactly
to the stock ``Fit`` behaviour (regression-tested).

Two orthogonal mechanisms (see the implementation plan, sections 3 / 5 / 5a):

  * **Axis 1 -- parameter sharing.**  Structural parameters (e.g. theta / phi /
    protins for XLayersPDB, or any XLayers scalar) can be tied across datasets.
    A shared parameter keeps a BARE global name in ``master_params``; every
    per-dataset parameter is namespaced ``d{i}_<name>`` (scalars) or
    ``__d{i}_<mkey>_<pkey>_<num>`` (multipar).  Sharing is expressed with lmfit
    ``expr`` / a single shared key, so it round-trips through the ``.par`` format
    unchanged and needs no minimizer change.

  * **Axis 2 -- cross-dataset error covariance.**  When a covariance is attached,
    the residual concatenates the *raw* (data - fit) in absolute units and
    returns the whitened residual ``L^{-1} (d - f)`` where ``L = chol(C_full)``.
    Under ``leastsq`` / ``least_squares`` this makes the minimized quantity
    ``(d-f)^T C^{-1} (d-f)`` = generalized least squares, and lmfit's
    Jacobian-based ``.covar`` then gives correct GLS parameter uncertainties --
    NO MCMC required.  Under emcee (``is_weighted=True``) the same whitened
    residual is exactly the multivariate-normal log-likelihood (the det(C) term
    is parameter-independent and drops out).

The two are kept composable per plan section 5a: the residual branches ONLY on
"is a covariance attached?", never on "are parameters shared?".  So a future
combined fit (shared params AND cross-dataset covariance) is a no-rewrite
addition.

The class is head-less testable: PyQt5 is imported if available, otherwise a
no-op shim stands in for QObject / pyqtSignal so the numerical core runs under
plain Python for validation.
"""

import numpy as np
from lmfit import fit_report, Minimizer, Parameters
from Fit_Routines import _window_indices   # one definition of the x window

try:
    from scipy.linalg import solve_triangular as _solve_tri
except Exception:                                        # pragma: no cover
    _solve_tri = None

import joint_covariance as jcov
import covariance_core as cc

# --------------------------------------------------------------------------- #
#  Optional-PyQt shim (so the engine runs head-less for validation)
# --------------------------------------------------------------------------- #
try:
    from PyQt5.QtCore import pyqtSignal, QObject
    _HAVE_QT = True
except Exception:                                        # pragma: no cover
    _HAVE_QT = False

    class QObject:                                       # minimal stand-in
        def __init__(self, parent=None):
            pass

    class _DummySignal:
        def connect(self, *a, **k):
            pass

        def disconnect(self, *a, **k):
            pass

        def emit(self, *a, **k):
            pass

    def pyqtSignal(*a, **k):
        return _DummySignal()


# --------------------------------------------------------------------------- #
#  JointFit
# --------------------------------------------------------------------------- #
class ShareMismatchError(ValueError):
    """The endpoints of one share-group disagree on ``min``, ``max`` or ``vary``.

    A shared parameter is ONE master parameter, so it can only have one set of
    bounds and one vary flag. Different start *values* are normal (each dataset
    seeded from its own individual fit) and only warn; different bounds or vary
    are a real conflict and refuse unless ``on_mismatch='first'``."""


class _RunWarning(str):
    """A ``JointFit.warnings`` entry added by ``perform_fit`` (run-time), as
    opposed to one added by ``build_master_params`` (build-time). The subclass is
    the bookkeeping: the next ``perform_fit`` drops exactly these entries."""


class _Downgrade(_RunWarning):
    """A run-time warning recording a covariance downgrade (spec 6.5): the fit
    continued in a weaker mode. Any of these puts ``DEGRADED FIT:`` on the
    report's first line."""


def _fmt_num(v):
    return '%.10g' % v


def chisqr_on(func, x, y, yerr, idx, masked_x=False, imin=None, imax=None):
    """Plain chi-square Sum(((y-f)/yerr)^2) of ``func`` on the points ``idx``.

    ``idx`` are integer indices into the full ``x``/``y``/``yerr`` (see
    JointFit.active_index). The model is evaluated on the objective's whole
    window x[imin:imax+1], as JointFit.residual does, and ``idx`` is then
    selected. Pass the objective's ``imin``/``imax``; without them the window
    falls back to x[idx.min():idx.max()+1], which differs when the mask drops
    points at a window edge (only matters for a model that is not pointwise
    in x). With ``masked_x`` (a ``__masked_x__`` model; the objective's
    ``yfit_masked``) the model is evaluated on the active points only, so its
    output is already selected and must not be masked again.
    Returns (chisqr, len(idx)).
    """
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return 0.0, 0
    x, y, yerr = np.asarray(x), np.asarray(y), np.asarray(yerr)
    func.__fit__ = False
    if masked_x:
        func.x = x[idx]
        f = np.asarray(func.y(), dtype=float)
    else:
        lo = int(idx.min()) if imin is None else int(imin)
        hi = int(idx.max()) if imax is None else int(imax)
        func.x = x[lo:hi + 1]
        f = np.asarray(func.y(), dtype=float)[idx - lo]
    e = np.where(np.abs(yerr[idx]) < 1e-30, 1.0, yerr[idx])
    return float(np.sum(((y[idx] - f) / e) ** 2)), int(len(idx))


class JointFit(QObject):
    # same signature as Fit.functionCalled so the GUI callbacks are reusable
    functionCalled = pyqtSignal(dict, int, np.ndarray, str)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.objectives = []          # list of dicts (see add_dataset)
        self.master_params = Parameters()
        self.param_map = {}           # master_key -> [(obj_idx, local_key), ...]
        self.shared = []              # bare param names tied across datasets
        self.fitter = None
        self.result = None
        self.Niter = 0
        self.fit_abort = False
        self._current = 0             # index of the "current" objective (viewer)
        # covariance state: each objective's "cov" is a DatasetCovariance or None
        # (_use_cov / _use_footprint are read-only properties derived from them)
        self._L = None
        self._cov_mask = None
        self._jitter = 0.0
        self._cov_scale = None        # fit scale _L was assembled for
        self._cov_warnings = []       # assemble/factorize warnings of this fit
        self._cov_factor_warnings = []  # factorize warnings of the last build
        self._cov_info_notes = []     # non-downgrade notes (points dropped)
        self._cov_ready = False
        self.covariance_mode = 'model'   # 'model' | 'data' (spec 4.4)
        self.cov_info = {}
        # optional per-parameter Gaussian priors (soft constraints):
        #   {master_key: (mu, sigma)} -> append (value - mu)/sigma to the residual.
        self._priors = {}
        # human-readable notes about the fit set-up (see _add_run_warning)
        self.warnings = []

    # ----------------------------------------------------------------- #
    #  dataset registration
    # ----------------------------------------------------------------- #
    def add_dataset(self, name, funcClass=None, x=None, y=None, yerr=None,
                    func=None, cov=None, mask=None):
        """Register one dataset as a joint-fit objective.

        Parameters
        ----------
        name : str          dataset key (unique).
        funcClass : type    model class (e.g. XLayers); ignored if ``func`` given.
        func : instance      a pre-built model instance (alternative to funcClass).
        x, y, yerr : arrays  the data.  A legacy covariance (``replaces_y``)
                             OVERRIDES y / yerr with its ``y_corr`` / ``sigma_meas``.
        cov : covariance_core.DatasetCovariance, a legacy dict, or None.
                             A legacy dict (keys y_corr, sigma_meas, k_index,
                             SF_LIST, Sigma_logc[, v, var_fwhm], as returned by
                             ``joint_covariance.load_dataset_cov``) is converted
                             with ``covariance_core.from_legacy_meta``. A
                             covariance whose x (or v1 y) does not match the data
                             is dropped with a warning (spec 6.5, fit-time row).
                             Attaching a cov on ANY dataset turns on the joint
                             covariance-whitening residual for the whole fit.
        """
        if func is None:
            if funcClass is None:
                raise ValueError("add_dataset needs funcClass or func")
            func = funcClass()
        x = np.asarray(x, dtype=float)
        func.x = x
        if not getattr(func, "params", None):
            func.init_params()

        cov_notes = []
        if cov is not None and not isinstance(cov, cc.DatasetCovariance):
            cov = cc.from_legacy_meta(cov, x, source='<dict>', loader='legacy-dict')
        if cov is not None:
            probs = cov.match_problems(x, y)
            if probs:
                cov_notes.append(_Downgrade(
                    "%s: covariance %s does not match the data (%s); fitted without it"
                    % (name, cov.source or '<in memory>', '; '.join(probs))))
                cov = None
        y_use = np.zeros_like(x) if y is None else np.asarray(y, dtype=float)
        yerr_use = np.ones_like(x) if yerr is None else np.asarray(yerr, dtype=float)
        if cov is not None and cov.replaces_y is not None:
            # legacy: the corrected data with the measurement-only error is the
            # dataset (also its no-covariance fallback, spec 6.5)
            y_use = np.asarray(cov.y_for_fit(y_use), dtype=float)
            yerr_use = np.asarray(cov.sigma_meas, dtype=float)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            # Assert rather than trust: the host builds masks from a DataFrame
            # column while _primary_curve registers x from the FIRST sub_key only.
            # Those are row-aligned today, and this is what makes a future
            # violation fail here at registration instead of silently
            # misaligning the covariance whitening later.
            if len(mask) != len(x):
                raise ValueError("mask for %r has %d points but x has %d"
                                 % (name, len(mask), len(x)))
        obj = {
            "name": name,
            "func": func,
            "x": x,
            "y": y_use,
            "yerr": yerr_use,
            "imin": 0,
            "imax": len(x) - 1,
            "imask": None,      # window-restricted mask, frozen by _set_ranges
            "yfit_masked": False,  # did evaluate() already apply imask to yfit?
            "mask": mask,       # full-length eligibility mask, or None
            "cov": cov,         # covariance_core.DatasetCovariance or None
            "cov_notes": cov_notes,   # downgrades decided here (re-added per fit)
        }
        self.objectives.append(obj)
        return obj

    def remove_dataset(self, name_or_index):
        """Remove one registered objective (by name or index).

        The caller must re-run ``build_master_params`` afterwards, because the
        per-dataset master keys are position-based (``d{i}_...``) and renumber
        when an earlier objective is dropped.  Covariance/Cholesky state is
        invalidated so the next ``perform_fit`` rebuilds it.  Also clears any
        Gaussian priors whose (now-stale) master keys may no longer exist.
        """
        idx = self._resolve_idx(name_or_index)
        del self.objectives[idx]
        if self._current >= len(self.objectives):
            self._current = max(0, len(self.objectives) - 1)
        # covariance is a property of the concatenation, which just changed
        self._L = None
        self._cov_ready = False
        # priors are keyed on master names that build_master_params will rebuild
        self._priors = {}

    def clear(self):
        """Drop all objectives and reset covariance/param state (fresh JointFit)."""
        self.objectives = []
        self.master_params = Parameters()
        self.param_map = {}
        self.shared = []
        self._current = 0
        self._L = None
        self._cov_mask = None
        self._jitter = 0.0
        self._cov_scale = None
        self._cov_ready = False
        self.cov_info = {}
        self._priors = {}
        self.warnings = []

    # ---- warnings bookkeeping ---------------------------------------- #
    # ``self.warnings`` holds build-time entries (reset by build_master_params)
    # and run-time entries (added by perform_fit). A perform_fit call replaces
    # only the run-time entries of its previous call, so build-time warnings
    # reach the report and neither kind piles up across re-fits.
    def _reset_run_warnings(self):
        self.warnings = [w for w in self.warnings if not isinstance(w, _RunWarning)]

    def _add_run_warning(self, msg):
        """Append a run-time warning (call from inside perform_fit)."""
        self.warnings.append(_RunWarning(msg))

    def _check_share_group(self, mkey, members, on_mismatch):
        """Compare every endpoint of one share-group to the first (01 6.8).
        Returns the group's warnings; mutates nothing.

        value differs        -> one build-time warning for the group.
        min/max/vary differs -> ShareMismatchError, or a warning if
                                ``on_mismatch == 'first'``.
        Bounds compare with ``==`` after ``float()``, so ``inf == inf``."""
        names = self.objective_names
        ps = [(names[idx], lk, self.objectives[idx]["func"].params[lk])
              for (idx, lk) in members]
        d0, lk0, p0 = ps[0]
        out = []
        if any(float(p.value) != float(p0.value) for (_, _, p) in ps[1:]):
            out.append(
                "shared %s: endpoints start at different values (%s); using %s=%s"
                % (mkey, ', '.join('%s=%s' % (d, _fmt_num(p.value)) for (d, _, p) in ps),
                   d0, _fmt_num(p0.value)))

        def bounds(p):
            return (float(p.min), float(p.max), bool(p.vary))
        if all(bounds(p) == bounds(p0) for (_, _, p) in ps[1:]):
            return out
        listing = ', '.join('(%s %r: min=%s, max=%s, vary=%s)'
                            % (d, lk, _fmt_num(bounds(p)[0]), _fmt_num(bounds(p)[1]),
                               bounds(p)[2])
                            for (d, lk, p) in ps)
        if on_mismatch == 'raise':
            raise ShareMismatchError(
                "shared %r (local %r): endpoints disagree on min/max/vary: %s. "
                "Make them agree, or apply the first endpoint's bounds/vary to all "
                "members (on_mismatch='first'; the GUI offers this as a Yes/No "
                "prompt)." % (mkey, lk0, listing))
        out.append(
            "shared %s: endpoints disagree on min/max/vary %s; bounds unified: "
            "every endpoint now uses the first endpoint's (%s) min/max/vary"
            % (mkey, listing, d0))
        return out

    def _unify_share_group(self, members):
        """Copy the first endpoint's min/max/vary onto every other endpoint's
        local Parameter (``on_mismatch='first'``). Without this, push_params
        writes the master value into a narrower local Parameter and lmfit's
        value setter clips it silently, so that dataset is evaluated at a
        different value from the one the master reports (final review I1)."""
        idx0, lk0 = members[0]
        p0 = self.objectives[idx0]["func"].params[lk0]
        for (idx, lk) in members[1:]:
            p = self.objectives[idx]["func"].params[lk]
            p.set(min=p0.min, max=p0.max, vary=p0.vary)

    @property
    def _use_cov(self):
        """True when any objective has a covariance attached (read-only; callers
        outside this file read it: lmfit/minimizer.py, xmodfit.py)."""
        return any(o.get("cov") is not None for o in self.objectives)

    @property
    def _use_footprint(self):
        """True when any attached covariance carries a ``beam_fwhm`` term."""
        return any(o.get("cov") is not None
                   and any(t.name == 'beam_fwhm' for t in o["cov"].terms)
                   for o in self.objectives)

    # ----------------------------------------------------------------- #
    #  parameter namespacing / master params
    # ----------------------------------------------------------------- #
    def _master_key(self, idx, local_key):
        """Map a per-objective local parameter name to its master name.

        Shared params keep a BARE global name; individuals are namespaced.
        Multipar keys ``__<mkey>_<pkey>_<num>`` become ``__d{i}_<mkey>_<pkey>_<num>``
        (pkey/num are underscore-free, so a downstream ``rsplit('_',2)`` recovers
        the parts even with a '_'-containing namespaced mkey -- this is the
        ``loadParameters`` fix at xmodfit.py lines ~3246/3254).
        """
        if local_key in self.shared:
            return local_key
        if local_key.startswith("__"):
            body = local_key[2:]
            mkey, pkey, num = body.rsplit("_", 2)
            return "__d%d_%s_%s_%s" % (idx, mkey, pkey, num)
        return "d%d_%s" % (idx, local_key)

    # ---- dataset-reference resolution (name or index -> index) ------- #
    def _resolve_idx(self, ref):
        if isinstance(ref, str):
            return self.objective_names.index(ref)
        return int(ref)

    # ---- share-group builders (convenience) -------------------------- #
    def share_all(self, local_key):
        """A share-group tying ``local_key`` across EVERY dataset that has it.

        Returns a group dict with ``bare=True`` so the master parameter keeps an
        un-namespaced global name -- this preserves the emcee angle gate
        (``var_names=='theta'``) for a shared XLayersPDB orientation (Axis-1)."""
        members = [(i, local_key) for i, o in enumerate(self.objectives)
                   if local_key in o["func"].params]
        if not members:
            raise ValueError(f"no dataset exposes parameter '{local_key}'")
        return {"members": members, "bare": True}

    def share_within(self, local_key, dataset_refs):
        """A share-group tying ``local_key`` across a SUBSET of datasets.

        ``dataset_refs`` is a list of dataset names or indices (e.g. all datasets
        on one wafer sharing a substrate parameter). Use this when the SAME local
        key names the parameter in every member (true for scalars, and for a
        layer row that sits at the same index in every member)."""
        members = [(self._resolve_idx(r), local_key) for r in dataset_refs]
        return {"members": members}

    def share_endpoints(self, endpoints):
        """A share-group over explicit ``(dataset_ref, local_key)`` endpoints.

        REQUIRED when a physically-shared parameter has DIFFERENT local keys in
        different datasets -- e.g. the SiO2 layer is ``__Model_d_005`` in a
        bilayer run but ``__Model_d_001`` in a bare-buffer run, yet both name the
        one oxide thickness on the same wafer.  ``push_params`` already routes one
        master value to heterogeneous local keys, so this Just Works."""
        return {"members": [(self._resolve_idx(r), lk) for (r, lk) in endpoints]}

    def build_master_params(self, shared=None, groups=None, on_mismatch='raise'):
        """Build ``master_params`` + ``param_map`` from the registered objectives.

        Two ways to declare sharing (they may be combined):

        * ``shared`` -- iterable of bare local names tied across ALL datasets
          (legacy convenience; each becomes a ``share_all`` group).  Default
          None.  ``shared=None, groups=None`` -> share nothing (pure Axis-2 /
          independent structural params).

        * ``groups`` -- list of share-group specs for ARBITRARY-SUBSET sharing
          (the substrate-per-wafer / bilayer-per-replicate case).  Each spec is a
          dict ``{'members': [(idx, local_key), ...], 'bare': bool, 'name': str}``
          -- build them with ``share_all`` / ``share_within`` / ``share_endpoints``.
          Every endpoint in a spec is tied to ONE master parameter; endpoints not
          claimed by any group become per-dataset (namespaced) parameters.

        This is the "per-parameter list of which datasets share it" scheme
        (equality via a shared lmfit parameter): substrate-across-a-wafer-subset,
        bilayer-across-a-replicate-subset, and per-dataset yscale/bkg can coexist
        in one fit.  It is Axis-1 (parameter sharing); it composes freely with the
        Axis-2 covariance because the residual branches only on "covariance
        attached?", never on "params shared?" (plan section 5a).

        The master parameter for a group inherits value/vary/bounds/brute_step
        from its FIRST endpoint's local parameter. The endpoints are checked
        first (``_check_share_group``): different start values add a warning to
        ``self.warnings``; different ``min``/``max``/``vary`` raise
        ``ShareMismatchError`` unless ``on_mismatch='first'``, which copies the
        first endpoint's min/max/vary onto every other endpoint's local
        Parameter (so push_params cannot clip a later endpoint) and warns.
        ``self.warnings`` is reset on every call.
        """
        if on_mismatch not in ('raise', 'first'):
            raise ValueError("on_mismatch must be 'raise' or 'first', got %r"
                             % (on_mismatch,))
        specs = []
        for nm in (shared or []):
            specs.append(self.share_all(nm))
        for g in (groups or []):
            specs.append(dict(g))

        # 0) resolve + check every group BEFORE mutating any state, so a
        #    refused build (ShareMismatchError) leaves the object as it was
        resolved, warnings, used_names = [], [], set()
        for gid, spec in enumerate(specs):
            members = [(self._resolve_idx(i) if not isinstance(i, int) else i, lk)
                       for (i, lk) in spec["members"]]
            if not members:
                continue
            mkey = spec.get("name") or self._group_master_key(
                gid, members[0][1], bare=spec.get("bare", False), used=used_names)
            used_names.add(mkey)
            warnings += self._check_share_group(mkey, members, on_mismatch)
            resolved.append((mkey, members))

        # every group passed the check: under 'first', unify the endpoints'
        # bounds/vary on the local Parameters so push_params cannot clip
        if on_mismatch == 'first':
            for (_mk, members) in resolved:
                self._unify_share_group(members)

        self.warnings = warnings
        self.shared = [nm for nm in (shared or [])]
        self.master_params = Parameters()
        self.param_map = {}

        # 1) claimed endpoints -> group master keys
        claimed = {}                       # (idx, lkey) -> master key
        for mkey, members in resolved:
            first_idx, first_lk = members[0]
            p0 = self.objectives[first_idx]["func"].params[first_lk]
            self.master_params.add(mkey, value=p0.value, vary=p0.vary,
                                   min=p0.min, max=p0.max, expr=None,
                                   brute_step=p0.brute_step)
            self.param_map[mkey] = []
            for (idx, lk) in members:
                if (idx, lk) in claimed:
                    raise ValueError(
                        f"endpoint ({idx},{lk}) is in more than one share-group")
                claimed[(idx, lk)] = mkey
                self.param_map[mkey].append((idx, lk))

        # 2) everything else -> individual (namespaced) master keys
        for idx, obj in enumerate(self.objectives):
            for lkey in obj["func"].params.keys():
                if (idx, lkey) in claimed:
                    continue
                mkey = self._master_key(idx, lkey)   # namespaced d{i}_ / __d{i}_
                p = obj["func"].params[lkey]
                if mkey not in self.master_params:
                    self.master_params.add(mkey, value=p.value, vary=p.vary,
                                           min=p.min, max=p.max, expr=None,
                                           brute_step=p.brute_step)
                    self.param_map[mkey] = []
                self.param_map[mkey].append((idx, lkey))
        return self.master_params

    def _group_master_key(self, gid, local_key, bare=False, used=None):
        """Master key for a share-group.

        ``bare`` (all-datasets share) keeps the local name un-namespaced so the
        emcee angle gate keeps firing (``theta``/``phi``).  A subset share is
        group-namespaced (``g{gid}_<name>`` / ``__g{gid}_<mkey>_<pkey>_<num>``);
        the ``g{gid}_`` prefix contains an underscore, which the loadParameters
        ``rsplit('_',2)`` fix (xmodfit.py ~3246/3254) already tolerates -- exactly
        like the per-dataset ``d{i}_`` scheme."""
        used = used or set()
        if local_key.startswith("__"):
            mkey, pkey, num = local_key[2:].rsplit("_", 2)
            cand = ("__%s_%s_%s" % (mkey, pkey, num) if bare
                    else "__g%d_%s_%s_%s" % (gid, mkey, pkey, num))
        else:
            cand = local_key if bare else "g%d_%s" % (gid, local_key)
        # guarantee uniqueness even if two bare groups collide (shouldn't happen)
        base, n = cand, 1
        while cand in used:
            cand = "%s_g%d" % (base, gid + n); n += 1
        return cand

    def set_shared_expr(self, master_key, expr):
        """Tie one master parameter to an expression of others (Axis 1 helper).
        Thin wrapper so callers/GUI don't touch lmfit directly."""
        self.master_params[master_key].set(expr=expr)

    def set_priors(self, priors):
        """Attach optional per-parameter Gaussian priors (soft constraints).

        ``priors`` is ``{master_key: (mu, sigma)}``.  Each entry appends one extra
        residual element ``(params[master_key].value - mu) / sigma`` to the
        (whitened, if a covariance is attached) residual vector, so least-squares
        minimizes ... + Σ_k ((v_k - mu_k)/sigma_k)^2 -- a Gaussian log-prior /
        Tikhonov term.  Composes with the Axis-2 covariance path and TRF: it is
        appended AFTER the whitening, because the prior is on a parameter, not on
        a data point, so it must not be mixed into the data covariance C.

        Use it to inform a low-leverage parameter (e.g. the buried Si/SiO2
        roughness, which the full cross-dataset covariance otherwise pulls to its
        bound) from an individual/diagonal fit value ± its stderr, WITHOUT hard-
        fixing it -- the parameter still floats and its posterior uncertainty is
        propagated.  Each prior adds 1 to ``ndata``/dof, exactly like an extra
        measurement, which is the correct accounting for a soft constraint.
        """
        bad = [k for k in priors if k not in self.master_params]
        if bad:
            raise KeyError(f"set_priors: unknown master keys {bad}")
        self._priors = {k: (float(mu), float(sig)) for k, (mu, sig) in priors.items()}

    # ----------------------------------------------------------------- #
    #  push master -> per-objective, evaluate
    # ----------------------------------------------------------------- #
    def push_params(self, params):
        """Write master parameter values into every objective's model.

        Mirrors Fit.residual's dual update: scalar attributes (read directly in
        ``XLayers.y()`` as ``self.qoff`` etc.) AND the lmfit Parameters (read by
        ``update_parameters()`` for multipar).  Local dynamic exprs are cleared
        so the master value is authoritative (share-nothing GLS); Axis-1 sharing
        is expressed at master level instead.
        """
        for mkey, targets in self.param_map.items():
            v = params[mkey].value
            for (idx, lkey) in targets:
                func = self.objectives[idx]["func"]
                func.__dict__[lkey] = v            # scalar attribute path
                lp = func.params[lkey]
                if lp.expr:
                    lp.expr = None
                lp.value = v                       # multipar path (update_parameters)

    def evaluate(self):
        """Evaluate every objective's model on its [imin, imax] slice.

        A model declaring ``__masked_x__ = True`` receives only the ACTIVE points
        (see Fit.evaluate for the safety taxonomy). ``obj["yfit_masked"]`` records
        whether that happened, so residual() knows not to mask the result twice.
        """
        yfits = []
        for obj in self.objectives:
            func = obj["func"]
            func.__fit__ = True
            seg = obj["x"][obj["imin"]:obj["imax"] + 1]
            im = obj.get("imask")
            if im is not None and bool(getattr(func, "__masked_x__", False)):
                seg = seg[im[:len(seg)]]
                obj["yfit_masked"] = True
            else:
                obj["yfit_masked"] = False
            func.x = seg
            yfits.append(np.asarray(func.y(), dtype=float))
        return yfits

    # ----------------------------------------------------------------- #
    #  covariance / Cholesky over the concatenation
    # ----------------------------------------------------------------- #
    def _build_chol(self, fit_scale='Linear', model=False):
        """Assemble C over the concatenated active points of all objectives (in
        residual order) through covariance_core and cache its Cholesky factor.

        Per objective: slice -> imask (``cov.subset`` of the active index) -> one
        ``Block`` (no covariance: diag(yerr^2), or its log equivalent). The finite
        mask over the concatenation (finite y, sigma > 0, finite intensity; in Log
        also y > 0 and intensity > 0) is ``_cov_mask``, frozen on the first build
        of a fit; C is assembled on the masked rows only. ``model=False`` uses the
        data as the intensity D (the data-based C, round 0); ``model=True`` the
        current model (the caller pushes the parameters first).

        MASK COMPOSITION ORDER, and it matters: (1) slice [imin,imax], (2) apply
        this objective's imask, (3) concatenate, (4) apply _cov_mask. _cov_mask is
        DEFINED against the output of (3), so imask must be folded in here AND in
        residual()'s assembly, or the whitening rows silently misalign with the
        data rows -- no exception, no warning, just a plausible wrong answer."""
        scale = 'Log' if fit_scale == 'Log' else 'Linear'
        yfits = self.evaluate() if model else None
        blocks, ycat, scat, icat = [], [], [], []
        for k, obj in enumerate(self.objectives):
            idx = self.active_index(k)
            y = np.asarray(obj["y"], dtype=float)[idx]
            yerr = np.asarray(obj["yerr"], dtype=float)[idx]
            cov = obj["cov"]
            cs = None if cov is None else cov.subset(idx)
            if model:
                f = np.asarray(yfits[k], dtype=float)
                im = obj.get("imask")
                if im is not None and not obj.get("yfit_masked"):
                    f = f[im]
                inten = f
            else:
                inten = y
            blocks.append(cc.Block(cs, y, yerr, inten, name=obj["name"]))
            ycat.append(y)
            scat.append(yerr if cs is None else np.asarray(cs.sigma_meas, dtype=float))
            icat.append(inten)
        ycat, scat, icat = (np.concatenate(a) for a in (ycat, scat, icat))

        if model and self._cov_mask is not None and len(self._cov_mask) == len(ycat):
            # the mask is frozen once per fit (rule 4): a model intensity that is
            # unusable on an active row falls back to the data there, with a note
            m = self._cov_mask
            bad = m & ~np.isfinite(icat)
            if scale == 'Log':
                bad |= m & (icat <= 0)
            if bad.any():
                self._cov_warnings.append(
                    'model intensity not usable at %d active points (non-finite%s); '
                    'the data are used there' % (int(bad.sum()),
                                                 ' or <= 0' if scale == 'Log' else ''))
                off = 0
                for b in blocks:
                    nb = len(b.y)
                    sub = bad[off:off + nb]
                    b.intensity = np.where(sub, b.y, b.intensity)
                    off += nb
        else:
            m = np.isfinite(ycat) & np.isfinite(scat) & (scat > 0) & np.isfinite(icat)
            if scale == 'Log':
                with np.errstate(invalid='ignore'):
                    m &= (ycat > 0) & (icat > 0)
            self._cov_mask = m
            off, dropped = 0, []
            for b in blocks:
                nb = len(b.y)
                nd = int(nb - m[off:off + nb].sum())
                if nd:
                    dropped.append('%s: %d' % (b.name, nd))
                off += nb
            if dropped:
                self._cov_info_notes.append(
                    'covariance fit (%s): points excluded (non-finite, sigma <= 0%s): %s'
                    % (scale, ' or y <= 0' if scale == 'Log' else '', ', '.join(dropped)))

        masked, off = [], 0
        for b in blocks:
            nb = len(b.y)
            keep = np.flatnonzero(m[off:off + nb])
            off += nb
            masked.append(cc.Block(None if b.cov is None else b.cov.subset(keep),
                                   b.y[keep], b.yerr[keep], np.asarray(b.intensity)[keep],
                                   name=b.name))
        C, awarn = cc.assemble(masked, scale)
        self._L, fwarn = cc.factorize(C)
        self._jitter = 0.0
        if any('jitter' in w for w in fwarn):
            self._jitter = float(jcov.cholesky_with_jitter(C)[1])
        for w in awarn + fwarn:
            if w not in self._cov_warnings:
                self._cov_warnings.append(w)
        self._cov_factor_warnings = list(fwarn)
        self._cov_scale = fit_scale
        self._cov_ready = True

    # ----------------------------------------------------------------- #
    #  residual
    # ----------------------------------------------------------------- #
    def residual(self, params, fit_scale):
        self.push_params(params)
        yfits = self.evaluate()

        if self._use_cov and fit_scale in ('Linear', 'Log'):
            # ---- Axis-2 / combined: whitened residual over concatenation ----
            # absolute units (no max-normalisation): whitened (d - f) is
            # scale-invariant. 'w/o error' scales take the no-covariance branch.
            d, f = [], []
            for obj, fit in zip(self.objectives, yfits):
                i0, i1 = obj["imin"], obj["imax"] + 1
                data = obj["y"][i0:i1]
                im = obj.get("imask")
                fit = np.asarray(fit)
                if im is None:
                    d.append(data); f.append(fit)
                else:                            # see _build_chol on the order
                    if not obj.get("yfit_masked"):
                        fit = fit[im]            # else evaluate() already masked it
                    d.append(data[im]); f.append(fit)
            d, f = np.concatenate(d), np.concatenate(f)
            if self._L is None or self._cov_scale != fit_scale:
                self._build_chol(fit_scale)
            if len(d) != len(self._cov_mask):
                raise RuntimeError(
                    "joint residual length %d != covariance mask length %d: the "
                    "column mask was applied inconsistently between _build_chol "
                    "and residual" % (len(d), len(self._cov_mask)))
            # drop the masked rows BEFORE any log (Log: y <= 0 rows are masked)
            d, f = d[self._cov_mask], f[self._cov_mask]
            r = d - f if fit_scale == 'Linear' else np.log(d) - np.log(f)
            # L is lower-triangular (Cholesky): O(N^2) forward-substitution, NOT
            # a general O(N^3) solve -- this matters a lot at N~10^3 over many
            # least-squares iterations.
            y = cc.whiten(self._L, r)
        else:
            # ---- no covariance: byte-identical to Fit per-curve behaviour ----
            y = []
            for obj, fit in zip(self.objectives, yfits):
                i0, i1 = obj["imin"], obj["imax"] + 1
                data = obj["y"][i0:i1]
                err = obj["yerr"][i0:i1]
                im = obj.get("imask")
                if im is not None:
                    # before maxd: an excluded hot point must not rescale the
                    # residual vector, and before any log10 of the data
                    data, err = data[im], err[im]
                    if not obj.get("yfit_masked"):
                        fit = np.asarray(fit)[im]
                maxd = np.max(data)
                data = data / maxd
                err = err / maxd
                fit = fit / maxd
                if fit_scale == "Log":
                    y.append((np.log(data) - np.log(fit)) * data / err)
                elif fit_scale == "Log w/o error":
                    y.append(np.log10(data) - np.log10(fit))
                elif fit_scale == "Linear w/o error":
                    y.append(data - fit)
                else:
                    y.append((data - fit) / err)
            y = np.concatenate(y)

        # ---- optional Gaussian priors: append (value - mu)/sigma terms -------
        # A prior constrains a PARAMETER, not a data point, so it is appended
        # after any whitening (it must not enter the data covariance C).
        if self._priors:
            extra = [(params[k].value - mu) / sig
                     for k, (mu, sig) in self._priors.items()]
            y = np.concatenate([y, np.asarray(extra, dtype=float)])

        self.Niter += 1
        self.functionCalled.emit(params, self.Niter, y, fit_scale)
        return y

    def callback(self, params, iterations, residual, fit_scale):
        return True if self.fit_abort else None

    # ----------------------------------------------------------------- #
    #  fit driver
    # ----------------------------------------------------------------- #
    def active_index(self, i):
        """Integer indices (into objective ``i``'s full x) of the points actually
        fitted: arange(imin, imax+1) filtered by the frozen window mask. Mirrors
        Fit.active_index. imask is frozen by _set_ranges inside perform_fit, so
        before a fit this is the whole dataset (imin=0, imax=len-1, imask=None)."""
        obj = self.objectives[i]
        idx = np.arange(obj["imin"], obj["imax"] + 1)
        im = obj.get("imask")
        return idx if im is None else idx[im[:len(idx)]]

    def _set_ranges(self, xmin, xmax, ranges=None):
        """Set each objective's [imin, imax] fit window.

        Without ``ranges`` every objective uses the scalar (xmin, xmax) -- the
        original behavior. With ``ranges`` (a dict keyed by objective NAME, or
        by index, mapping to a ``(lo, hi)`` pair) each objective uses its own
        bounds; a missing entry, a ``None`` pair, or a ``None`` within a pair
        falls back to the scalar. Empty slices are avoided by clamping to the
        full data extent.
        """
        for idx, obj in enumerate(self.objectives):
            x = obj["x"]
            lo, hi = xmin, xmax
            if ranges:
                pair = ranges.get(obj["name"], ranges.get(idx, None))
                if pair is not None:
                    plo, phi = pair
                    if plo is not None:
                        lo = plo
                    if phi is not None:
                        hi = phi
            # shared with Fit.perform_fit: the old 0.99*lo factor moved a
            # NEGATIVE bound the wrong way and disagreed with the scalar branch
            obj["imin"], obj["imax"] = _window_indices(x, lo, hi)
            m = obj.get("mask")
            obj["imask"] = (None if m is None
                            else np.asarray(m, bool)[obj["imin"]:obj["imax"] + 1])

    # ----------------------------------------------------------------- #
    #  per-dataset orientation seeding (theta/phi not both-shared)
    # ----------------------------------------------------------------- #
    def orientation_layout(self, params=None):
        """Describe the theta/phi columns of a fit for orientation seeding/wrapping.

        Uses ``param_map`` to find, per dataset, the master key driving its ``theta``
        and ``phi`` (bare-shared or ``d{i}_theta``/``d{i}_phi``) and that key's
        column in the varying-parameter order. Returns ``None`` if no theta/phi
        varies, else a dict with:
          ``col``          master_key -> column index (varying params only)
          ``datasets``     per dataset: {'theta': mkey|None, 'phi': mkey|None}
          ``both_shared``  one theta AND one phi, each shared across ALL datasets
          ``neither_shared`` every dataset has its OWN (per-dataset) theta AND phi
        """
        p = params if params is not None else self.master_params
        var_names = [k for k in p if p[k].vary]
        col = {k: i for i, k in enumerate(var_names)}
        loc2master = {}
        for mkey, targets in self.param_map.items():
            for (idx, lk) in targets:
                loc2master[(idx, lk)] = mkey
        K = len(self.objectives)
        datasets = []
        any_angle = False
        for i in range(K):
            tk = loc2master.get((i, 'theta'))
            pk = loc2master.get((i, 'phi'))
            tk = tk if tk in col else None
            pk = pk if pk in col else None
            if tk is not None or pk is not None:
                any_angle = True
            datasets.append({'theta': tk, 'phi': pk})
        if not any_angle:
            return None
        tkeys = set(d['theta'] for d in datasets if d['theta'] is not None)
        pkeys = set(d['phi'] for d in datasets if d['phi'] is not None)
        both_shared = (len(tkeys) == 1 and len(pkeys) == 1
                       and all(d['theta'] is not None and d['phi'] is not None for d in datasets)
                       and len(self.param_map.get(next(iter(tkeys)), [])) == K
                       and len(self.param_map.get(next(iter(pkeys)), [])) == K)
        neither_shared = (all(d['theta'] is not None and d['phi'] is not None for d in datasets)
                          and all(len(self.param_map.get(d['theta'], [])) == 1 and
                                  len(self.param_map.get(d['phi'], [])) == 1 for d in datasets))
        return {'col': col, 'datasets': datasets, 'loc2master': loc2master,
                'both_shared': both_shared, 'neither_shared': neither_shared}

    def _build_orientation_seed(self, emcee_params, fit_scale, workers, temperature, rng,
                                both_shared_mode='joint'):
        """Precompute (p0, angleindex, anglepairs) from per-dataset single-model
        orientation landscapes. Each dataset's (theta,phi) is scanned on its OWN model
        with the shared params PINNED at the joint best-fit, so only orientation basins
        consistent with the shared structure survive; the scans reuse the single-model
        grid scan (parallel). Handles neither-shared (per-dataset pairs) and mixed
        (marginal + conditional). If both-shared and ``both_shared_mode=='individual'``,
        the single shared (theta,phi) is seeded from the SUM of the per-dataset maps.
        Returns None when both-shared and the caller wants the JOINT landscape."""
        from lmfit import minimizer as _mz
        import pdb_seed_worker as _psw

        lay = self.orientation_layout(emcee_params)
        if lay is None:
            return None
        if lay['both_shared'] and both_shared_mode != 'individual':
            return None
        col = lay['col']
        loc2master = lay['loc2master']
        ds = lay['datasets']
        var_names = [k for k in emcee_params if emcee_params[k].vary]
        nvar = len(var_names)
        nwalkers = int(self._emcee_nwalkers)

        # pin the shared structure: push the joint best-fit into every model, so the
        # per-dataset scans hold shared params fixed (only local nuisances profiled).
        self.push_params(emcee_params)

        base = np.array([emcee_params[k].value for k in var_names], dtype=float)
        p0 = (1.0 + rng.randn(nwalkers, nvar) * 1e-4) * base
        angleindex = np.zeros(nvar)
        anglepairs = []
        for d in ds:
            if d['theta'] is not None:
                angleindex[col[d['theta']]] = 1
            if d['phi'] is not None:
                angleindex[col[d['phi']]] = 2
            anglepairs.append((col[d['theta']] if d['theta'] is not None else None,
                               col[d['phi']] if d['phi'] is not None else None))

        thetameshrange, phires, phimeshrange, cells = _mz.build_orientation_grid(thetares=10)

        # per-dataset cost maps M_i (single-model scans, shared structure pinned)
        Mi, ndata_i, ndof_i = {}, {}, {}
        for i, d in enumerate(ds):
            if d['theta'] is None and d['phi'] is None:
                continue
            func = self.objectives[i]["func"]
            imin, imax = self.objectives[i]["imin"], self.objectives[i]["imax"]
            xs = np.asarray(self.objectives[i]["x"])[imin:imax + 1]
            ys = np.asarray(self.objectives[i]["y"])[imin:imax + 1]
            es = np.asarray(self.objectives[i]["yerr"])[imin:imax + 1]
            nuis = [lk for lk in func.params.keys()
                    if lk not in ('theta', 'phi')
                    and loc2master.get((i, lk)) in col
                    and len(self.param_map.get(loc2master.get((i, lk)), [])) == 1]
            start = [func.params[lk].value for lk in nuis]
            # A1.3: the frozen imask (set by _set_ranges), applied after evaluating on
            # the window. yfit_masked is derived as evaluate() would set it -- no
            # evaluate() has run since _set_ranges, so the stored flag may be stale.
            im = self.objectives[i].get("imask")
            ym = im is not None and bool(getattr(func, "__masked_x__", False))
            costs = _psw.map_cell_costs(cells, func, xs, ys, es, fit_scale, nuis, start, workers,
                                        mask=im, yfit_masked=ym)
            Mi[i] = np.array([c[2] for c in costs], dtype=float)
            ndata_i[i] = len(xs) if im is None else int(np.count_nonzero(im[:len(xs)]))
            ndof_i[i] = len(nuis) + 2

        # both-shared 'individual' mode: ONE shared (theta,phi) seeded from the SUM of
        # the per-dataset landscapes (shared params pinned above).
        if lay['both_shared']:
            tcol = col[ds[0]['theta']]; pcol = col[ds[0]['phi']]
            combined = np.sum([Mi[i] for i in Mi], axis=0)
            th, ph = _mz.seed_thetaphi_from_costmap(
                cells, phires, 10, combined,
                ndata=sum(ndata_i.values()), ndof=sum(ndof_i.values()),
                nwalkers=nwalkers, rng=rng, temperature=temperature)
            p0[:, tcol] = th; p0[:, pcol] = ph
            return p0, angleindex, anglepairs

        theta_members, phi_members = {}, {}
        for i, d in enumerate(ds):
            if d['theta'] is not None:
                theta_members.setdefault(col[d['theta']], []).append(i)
            if d['phi'] is not None:
                phi_members.setdefault(col[d['phi']], []).append(i)
        done_t, done_p = set(), set()

        # (1) per-dataset pairs: theta col AND phi col each used by exactly one dataset
        for i, d in enumerate(ds):
            tk, pk = d['theta'], d['phi']
            if tk is None or pk is None:
                continue
            tcol, pcol = col[tk], col[pk]
            if len(theta_members[tcol]) == 1 and len(phi_members[pcol]) == 1:
                th, ph = _mz.seed_thetaphi_from_costmap(
                    cells, phires, 10, Mi[i], ndata_i[i], ndof_i[i],
                    nwalkers, rng, temperature)
                p0[:, tcol] = th; p0[:, pcol] = ph
                done_t.add(tcol); done_p.add(pcol)

        # (2) shared theta + per-dataset phi (mixed): marginal theta + conditional phi_i
        for tcol, members in theta_members.items():
            if tcol in done_t or len(members) < 2:
                continue
            self._seed_shared_axis(
                p0, members, ds, col, Mi, ndata_i, ndof_i, thetameshrange, phires,
                phimeshrange, cells, nwalkers, rng, temperature, shared='theta',
                shared_col=tcol, done_other=done_p)
            done_t.add(tcol)

        # (3) shared phi + per-dataset theta (mixed): marginal phi + conditional theta_i
        for pcol, members in phi_members.items():
            if pcol in done_p or len(members) < 2:
                continue
            self._seed_shared_axis(
                p0, members, ds, col, Mi, ndata_i, ndof_i, thetameshrange, phires,
                phimeshrange, cells, nwalkers, rng, temperature, shared='phi',
                shared_col=pcol, done_other=done_t)
            done_p.add(pcol)

        return p0, angleindex, anglepairs

    def _resample_regular(self, chi2, thetameshrange, phires, phimeshrange, cells, nphi=72):
        """Resample a ragged (theta,phi) cost map onto a regular nt x nphi grid by
        nearest cell, so marginalizing over either axis is a simple array op."""
        nt = len(thetameshrange)
        cellcost = {}
        for (ti, pi, th, ph), c in zip(cells, chi2):
            cellcost[(ti, pi)] = c
        R = np.full((nt, nphi), np.inf)
        for t in range(nt):
            npc = len(phimeshrange[t])
            for j in range(nphi):
                phi = j * 360.0 / nphi
                pj = int(round(phi / phires[t])) % npc if npc > 0 else 0
                R[t, j] = cellcost.get((t, pj), np.inf)
        return R

    def _seed_shared_axis(self, p0, members, ds, col, Mi, ndata_i, ndof_i,
                          thetameshrange, phires, phimeshrange, cells, nwalkers, rng,
                          temperature, shared, shared_col, done_other, nphi=72):
        """Seed a SHARED angle (marginalized over the other, summed across members)
        and each member's per-dataset OTHER angle conditional on the shared draw."""
        from lmfit import minimizer as _mz
        nt = len(thetameshrange)
        # regular maps per member: rows=theta, cols=phi
        Rs = {i: self._resample_regular(Mi[i], thetameshrange, phires, phimeshrange, cells, nphi)
              for i in members}
        thetavals = np.asarray(thetameshrange, dtype=float)
        phivals = np.arange(nphi) * 360.0 / nphi
        thetares = thetavals[1] - thetavals[0] if nt > 1 else 10.0
        # solid-angle weight per theta ring; uniform over phi
        lo = np.clip(thetavals - thetares / 2, 0, 180) * np.pi / 180
        hi = np.clip(thetavals + thetares / 2, 0, 180) * np.pi / 180
        ring_area = np.clip(np.cos(lo) - np.cos(hi), 1e-12, None)
        ndata_sum = sum(ndata_i[i] for i in members)
        ndof_sum = sum(ndof_i[i] for i in members)

        if shared == 'theta':
            c_shared = np.sum([np.min(Rs[i], axis=1) for i in members], axis=0)   # (nt,)
            area = ring_area
            svals, sres = thetavals, thetares
        else:  # shared == 'phi'
            c_shared = np.sum([np.min(Rs[i], axis=0) for i in members], axis=0)   # (nphi,)
            area = np.ones(nphi)
            svals, sres = phivals, 360.0 / nphi

        chimin = np.min(c_shared) / max(ndata_sum - ndof_sum, 1)
        probs = _mz._seed_cell_probs(c_shared, area, chimin, temperature=temperature)
        sidx = _mz._inverse_cdf_draw(probs, nwalkers, rng)
        sdraw = svals[sidx] + (rng.random_sample(nwalkers) - 0.5) * sres
        sdraw = np.clip(sdraw, 0, 180) if shared == 'theta' else (sdraw % 360.0)
        p0[:, shared_col] = sdraw

        # conditional OTHER angle per member, given each walker's shared bin
        for i in members:
            R = Rs[i]
            other_key = 'phi' if shared == 'theta' else 'theta'
            ocol = col[ds[i][other_key]]
            ovals = phivals if shared == 'theta' else thetavals
            ores = (360.0 / nphi) if shared == 'theta' else thetares
            cm = np.min(R) / max(ndata_i[i] - ndof_i[i], 1)
            odraw = np.empty(nwalkers)
            for w in range(nwalkers):
                row = R[sidx[w], :] if shared == 'theta' else R[:, sidx[w]]
                pw = np.exp(-0.5 * (row - np.min(row)) / (max(temperature, 1e-6) * max(cm, 1e-12)))
                s = pw.sum()
                pw = pw / s if s > 0 else np.ones_like(row) / len(row)
                j = int(_mz._inverse_cdf_draw(pw, 1, rng)[0])
                odraw[w] = ovals[j] + (rng.random_sample() - 0.5) * ores
            odraw = (odraw % 360.0) if shared == 'theta' else np.clip(odraw, 0, 180)
            p0[:, ocol] = odraw
            done_other.add(ocol)

    def _attach_joint_parallel(self, fitter, fit_scale, is_weighted, emcee_cores):
        """Attach a picklable ``JointLogProb`` + spawn-Pool initializer so the emcee
        SAMPLING runs across ``emcee_cores`` processes (NO-COVARIANCE branch only).

        Shared by every parallel joint-MCMC path -- the orientation-seeded ones
        (both-shared / per-dataset / mixed, which sample via the angle-aware
        ``CustomSampler.custom_run_mcmc`` over the pool) and the generic non-PDB path
        (which steps the ``EnsembleSampler`` manually). ``Minimizer.emcee`` builds the
        pool from ``fitter._ext_obj_init`` and swaps in ``fitter._ext_objective`` when
        ``workers > 1``. Returns True on success.

        Covariance/GLS joint MCMC stays serial: the whitened residual is not in
        ``JointLogProb`` yet, so this is gated on ``not self._use_cov``."""
        if not (isinstance(emcee_cores, int) and emcee_cores > 1) or self._use_cov:
            return False
        try:
            import pdb_joint_logprob as _pjl
            var_names = [k for k in self.emcee_params if self.emcee_params[k].vary]
            bounds = [(self.emcee_params[k].min, self.emcee_params[k].max) for k in var_names]
            models = [obj["func"] for obj in self.objectives]
            # A1.18: + the frozen imask; the model is evaluated on the window and the
            # active points selected before the per-dataset max-normalisation
            datas = [(obj["x"][obj["imin"]:obj["imax"] + 1],
                      obj["y"][obj["imin"]:obj["imax"] + 1],
                      obj["yerr"][obj["imin"]:obj["imax"] + 1],
                      obj.get("imask"),
                      obj.get("imask") is not None
                      and bool(getattr(obj["func"], "__masked_x__", False)))
                     for obj in self.objectives]
            priors = {k: v for k, v in self._priors.items()} if self._priors else {}
            obj = _pjl.JointLogProb(var_names, bounds, fit_scale, is_weighted,
                                    self.param_map, priors, self.emcee_params.copy(),
                                    has_lnsigma=('__lnsigma' in self.emcee_params))
            fitter._ext_objective = obj
            fitter._ext_obj_init = (_pjl.init_worker, (models, datas))
            fitter._ext_workers = int(emcee_cores)
            return True
        except Exception:
            import traceback; traceback.print_exc()
            return False

    def set_emcee_minimizer(self, fit_scale='Linear', emcee_walkers=100, emcee_steps=100,
                            emcee_cores=1, emcee_burn=0, reuse_sampler=False, emcee_thin=1,
                            backend=None, seed_temperature=1.0, seed_mode='joint'):
        """Build a Minimizer configured for emcee on the JOINT objective.

        Samples the global master_params via the whitened/concatenated joint residual.
        funcname is 'JointFit'. When the SHARED master params include varying theta &
        phi (an XLayersPDB joint with one shared orientation), orientation seeding is
        enabled (``fitter._orient_seed``): Minimizer.emcee runs the (theta,phi) grid
        scan over the concatenated joint residual and uses the angle-aware CustomSampler
        (so theta/phi wrap -- else phi random-walks unbounded). That path is SERIAL.
        Otherwise the generic sampler path is used; the no-covariance branch may run
        parallel via a picklable JointLogProb."""
        self.emcee_params = self.result.params.copy()
        if 'w/o' in fit_scale:
            self.emcee_params.add('__lnsigma', value=-1.0, vary=True,
                                  min=np.log(0.001), max=np.log(2.0))
            is_weighted = False
        else:
            is_weighted = True

        # Orientation seeding/wrapping. Classify the theta/phi sharing layout.
        _pdb = any('XLayersPDB' in type(o["func"]).__name__ for o in self.objectives)
        lay = self.orientation_layout(self.emcee_params)
        both_shared = lay is not None and lay['both_shared']

        workers = 1
        fitter = Minimizer(self.residual, self.emcee_params, fcn_args=(fit_scale,),
                           iter_cb=self.callback, nan_policy='propagate',
                           burn=emcee_burn, steps=emcee_steps, thin=emcee_thin,
                           is_weighted=is_weighted, backend=backend, fiterr=0,
                           funcname='JointFit', nwalkers=emcee_walkers, workers=workers,
                           reuse_sampler=reuse_sampler, run_mcmc_kwargs={'store': True},
                           seed_temperature=seed_temperature)

        if both_shared:
            fitter._orient_seed = True
            # Parallel SAMPLING (no-cov): attach the picklable joint objective so the
            # angle-aware CustomSampler evaluates the joint residual across the pool
            # while wrapping theta/phi in the main process. Applies to both seed_modes.
            _par = self._attach_joint_parallel(fitter, fit_scale, is_weighted, emcee_cores)
            if seed_mode == 'individual':
                # seed the shared (theta,phi) from the SUM of per-dataset single-model
                # landscapes (shared params pinned at the joint best-fit); parallel scans.
                try:
                    self._emcee_nwalkers = emcee_walkers
                    seed = self._build_orientation_seed(self.emcee_params, fit_scale,
                                                        emcee_cores, seed_temperature,
                                                        np.random.RandomState(),
                                                        both_shared_mode='individual')
                except Exception:
                    import traceback; traceback.print_exc(); seed = None
                if seed is not None:
                    p0, angleindex, anglepairs = seed
                    fitter._angleindex = angleindex
                    fitter._anglepairs = anglepairs
                    fitter._seed_pos = p0
                    print('Joint XLayersPDB both-shared: seeding from INDIVIDUAL per-dataset '
                          'landscapes (shared params pinned); sampling is %s.'
                          % ('parallel across %d cores' % emcee_cores if _par else 'serial'))
                    return fitter
                # fall through to the joint landscape on failure
            # 'joint' (default): the in-emcee single-pair joint scan. Cells are
            # independent and each re-profiles the shared params jointly, so the scan
            # parallelizes exactly (pdb_joint_seed_worker) while SAMPLING stays serial
            # (joint objective unpicklable). `_seed_workers` decouples the two.
            if isinstance(emcee_cores, int) and emcee_cores > 1:
                if not self._use_cov:
                    fitter._seed_workers = int(emcee_cores)
                    print('Joint XLayersPDB both-shared: JOINT-landscape grid scan runs across '
                          '%d cores; sampling is %s.'
                          % (emcee_cores, 'parallel across the same cores' if _par else 'serial'))
                else:
                    print('Joint covariance fit: orientation scan + sampling serial for now '
                          '(parallel joint-covariance seeding is a follow-up).')
            return fitter

        # Per-dataset orientations (neither-shared): precompute p0 + angleindex from
        # per-dataset single-model scans (shared structure pinned at the joint best-fit).
        # Seeding scans parallelize across cores; sampling is serial (joint objective
        # is unpicklable), same as the both-shared case.
        if lay is not None and not both_shared:
            try:
                self._emcee_nwalkers = emcee_walkers
                seed = self._build_orientation_seed(self.emcee_params, fit_scale,
                                                    emcee_cores, seed_temperature,
                                                    np.random.RandomState())
            except Exception:
                import traceback; traceback.print_exc(); seed = None
            if seed is not None:
                p0, angleindex, anglepairs = seed
                fitter._orient_seed = True
                fitter._angleindex = angleindex
                fitter._anglepairs = anglepairs
                fitter._seed_pos = p0
                # Parallel SAMPLING (no-cov): same picklable joint objective; the
                # per-dataset angle pairing wraps in the main process each step.
                _par = self._attach_joint_parallel(fitter, fit_scale, is_weighted, emcee_cores)
                if isinstance(emcee_cores, int) and emcee_cores > 1:
                    print('Joint XLayersPDB per-dataset orientations: seeding scans parallel; '
                          'sampling is %s.'
                          % ('parallel across %d cores' % emcee_cores if _par else 'serial'))
                return fitter
            if _pdb:
                print('Joint fit: orientation seeding could not be built (unexpected); '
                      'using the generic path (no seeding/wrapping).')

        # Generic (non-orientation) parallel joint sampling: only the no-covariance
        # branch is supported for now (the whitened GLS residual needs bit-for-bit
        # validation against a covariance dataset first). The generic EnsembleSampler
        # steps manually over the pool (no angle wrapping needed here).
        self._attach_joint_parallel(fitter, fit_scale, is_weighted, emcee_cores)
        return fitter

    def perform_fit(self, xmin, xmax, fit_scale="Linear", fit_method="leastsq",
                    maxiter=1000, emcee_walkers=100, emcee_steps=100, emcee_burn=0,
                    emcee_cores=1, reuse_sampler=False, emcee_thin=1, backend=None,
                    ranges=None, seed_temperature=1.0, seed_mode='joint',
                    covariance_mode=None, **fit_kws):
        """Run a joint fit. Least-squares family as before; emcee samples the
        joint objective. The no-covariance branch samples in PARALLEL across
        ``emcee_cores`` when >1 (a picklable JointLogProb attached by
        set_emcee_minimizer); the covariance/GLS branch stays serial for now.

        With a covariance attached and a weighted scale ('Linear' / 'Log'):
        least squares runs with ``scale_covar=False`` (absolute errors, spec
        6.2); ``covariance_mode`` ('model' | 'data'; default
        ``self.covariance_mode``) selects the intensity in C. In 'model' mode,
        when a term depends on the intensity at this scale, the fit is wrapped
        in ``covariance_core.outer_iterate`` (round 0 data-based, then C from
        the model at the latest result, up to 5 rounds). emcee freezes C before
        sampling: at ``self.result.params`` in 'model' mode, else from the data.
        ``self.cov_info`` records mode, scale, rounds, converged, factor_warnings.

        Extra keyword args (e.g. ``ftol``/``xtol``/``gtol`` for the
        ``least_squares`` TRF backend) are forwarded to ``Minimizer.minimize``
        and thence to the underlying scipy solver. Backward-compatible: with no
        extra kwargs the behaviour is unchanged."""
        mode = self.covariance_mode if covariance_mode is None else covariance_mode
        if mode not in ('model', 'data'):
            raise ValueError("covariance_mode must be 'model' or 'data', got %r" % (mode,))
        self.Niter = 0
        self.fit_abort = False
        self._reset_run_warnings()          # build-time warnings stay
        for obj in self.objectives:         # add_dataset downgrades, every fit
            for w in obj.get("cov_notes") or []:
                self.warnings.append(_Downgrade(w))
        cov_on = self._use_cov and fit_scale in ('Linear', 'Log')
        if self._use_cov and not cov_on:
            self.warnings.append(_Downgrade(
                "covariance ignored: fit scale %r is unweighted, so the fit runs "
                "exactly as without a covariance (use 'Linear' or 'Log')" % (fit_scale,)))
        self.cov_info = {}
        self._cov_warnings = []
        self._cov_info_notes = []
        if (len(self.objectives) >= 2 and not cov_on
                and fit_scale == 'Linear w/o error'):
            # 01 6.7: per-dataset 1/max(y) weighting is silent otherwise
            self._add_run_warning(
                "fit scale 'Linear w/o error' divides each dataset by its own "
                "maximum, so datasets are weighted by 1/max(y); use 'Linear' "
                "(error-weighted) for a comparable joint objective")
        self._set_ranges(xmin, xmax, ranges=ranges)
        self._L = None                      # force Cholesky rebuild for new range
        self._cov_mask = None               # refrozen by the first build below
        self._cov_ready = False
        needs, frozen = False, 'data'
        if cov_on:
            self._build_chol(fit_scale)     # data-based C (round 0); freezes the mask
            needs = mode == 'model' and cc.needs_intensity(
                [cc.Block(o["cov"], o["y"], o["yerr"], o["y"]) for o in self.objectives],
                'Log' if fit_scale == 'Log' else 'Linear')
            if fit_method == "emcee" and needs:
                if self.result is None:
                    self.warnings.append(_Downgrade(
                        "model-based covariance needs a prior fit to freeze C at; "
                        "none exists, so MCMC uses the data-based covariance"))
                else:
                    self.push_params(self.result.params)
                    self._build_chol(fit_scale, model=True)   # frozen for the run
                    frozen = 'model'
            if fit_method == "emcee" and isinstance(emcee_cores, int) and emcee_cores > 1:
                self.warnings.append(_Downgrade(
                    "parallel MCMC requested with a covariance: sampling runs serial"))

        if fit_method == "emcee":
            self.fitter = self.set_emcee_minimizer(
                fit_scale=fit_scale, emcee_walkers=emcee_walkers, emcee_steps=emcee_steps,
                emcee_cores=emcee_cores, emcee_burn=emcee_burn, reuse_sampler=reuse_sampler,
                emcee_thin=emcee_thin, backend=backend, seed_temperature=seed_temperature,
                seed_mode=seed_mode)
            # Parallel only for the no-covariance branch (set_emcee_minimizer attaches a
            # picklable objective there); covariance/GLS joint MCMC stays serial for now.
            _workers = emcee_cores if (isinstance(emcee_cores, int) and emcee_cores > 1
                                       and not self._use_cov
                                       and getattr(self.fitter, '_ext_objective', None) is not None) else 1
            self.result = self.fitter.minimize(
                method='emcee', params=self.emcee_params, burn=0, steps=emcee_steps,
                thin=emcee_thin, is_weighted=('w/o' not in fit_scale), nwalkers=emcee_walkers,
                workers=_workers, reuse_sampler=reuse_sampler)
            if cov_on:
                self._finish_cov(mode, fit_scale, 0, True, mcmc_frozen=frozen)
            return self._report(), 'None'

        # with a covariance the errors are absolute (spec 6.2): no chi2_red rescale
        _sc = {'scale_covar': False} if cov_on else {}
        if fit_method == "leastsq":
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", max_nfev=maxiter, **_sc)
        elif fit_method == "least_squares":
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", max_nfev=maxiter, **_sc)
        elif fit_method == "differential_evolution":
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", calc_covar=True,
                                    max_nfev=maxiter, popsize=300,
                                    updating="immediate", **_sc)
        else:
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", max_nfev=maxiter, **_sc)

        if needs:
            # model-based C by outer iteration (spec 4.4); L frozen within a round
            state = {'params': self.master_params, 'last': None}

            def fit_once():
                if self.fit_abort and state['last'] is not None:
                    return state['last']             # aborted: stop iterating
                res = self.fitter.minimize(method=fit_method, params=state['params'],
                                           **fit_kws)
                state['params'], state['last'] = res.params, res
                return res

            def rebuild(res):
                self.push_params(res.params)
                self._build_chol(fit_scale, model=True)

            self.result, rounds, converged = cc.outer_iterate(fit_once, rebuild)
            if not converged:
                self.warnings.append(_Downgrade(
                    "model-based covariance did not converge in %d rounds (some varying "
                    "parameter still moved >= 0.1 stderr); the last result is kept"
                    % rounds))
            self._finish_cov(mode, fit_scale, rounds, converged)
            return self._report(), self.result.message

        self.result = self.fitter.minimize(method=fit_method,
                                           params=self.master_params,
                                           **fit_kws)
        if cov_on:
            self._finish_cov(mode, fit_scale, 0, True)
        return self._report(), self.result.message

    def _finish_cov(self, mode, fit_scale, rounds, converged, mcmc_frozen=None):
        """Record ``cov_info`` and turn this fit's assembly/factorisation warnings
        into run warnings (downgrades) and its point-exclusion notes into plain
        run warnings."""
        self.cov_info = {'mode': mode, 'scale': fit_scale, 'rounds': int(rounds),
                         'converged': bool(converged),
                         'factor_warnings': list(self._cov_factor_warnings),
                         'jitter': float(self._jitter)}
        if mcmc_frozen:
            self.cov_info['mcmc_frozen'] = mcmc_frozen
        for w in self._cov_warnings:
            self.warnings.append(_Downgrade(w))
        for w in self._cov_info_notes:
            self.warnings.append(_RunWarning(w))

    def _report(self):
        """lmfit's fit_report plus each objective's derived quantities.

        A model may expose ``derived_report(result, keymap, label)``, reporting a
        FUNCTION of its fitted parameters with the error propagated through
        ``result.covar`` -- XLayersPDB_XPR's top-referenced insertion, whose
        relation to the fitted (centroid-referenced) one depends on theta and phi.
        The keymap is what makes it work here: master parameters are namespaced
        per dataset unless shared, so the model cannot find 'protins' by name.
        It is read off ``param_map`` (which master key drives each dataset's local
        key), so share-groups -- bare, named or ``g{n}_`` subset groups from
        ``JointFitSpec.compile`` -- resolve as well as the legacy ``shared`` list.

        One block per objective, but a SHARED insertion+orientation gives every
        dataset the same three master parameters and hence the same block, so
        duplicates are collapsed.

        Any ``self.warnings`` (build-time and run-time) are appended as a
        ``Joint-fit warnings:`` block. With a covariance in use (``cov_info``
        set) ``covariance_core.report_block`` adds the ``Covariance:`` block,
        and any covariance downgrade puts ``DEGRADED FIT: <reasons>`` on the
        first line. With no covariance attached the text is unchanged.
        """
        rep = fit_report(self.result)
        downgrades = [w for w in self.warnings if isinstance(w, _Downgrade)]
        for obj in self.objectives:          # add_dataset notes are always reported
            for w in obj.get("cov_notes") or []:
                if w not in downgrades:
                    downgrades.append(w)
        blocks, seen = [], set()
        keymaps = self._derived_keymaps()
        for idx, obj in enumerate(self.objectives):
            func = obj.get('func')
            derive = getattr(func, 'derived_report', None)
            if derive is None:
                continue
            keymap = keymaps[idx]
            key = tuple(sorted(keymap.values()))
            if key in seen:
                continue
            try:
                txt = derive(self.result, keymap=keymap,
                             label='(%s)' % self.objective_names[idx]) or ''
            except Exception:
                txt = ''
            if txt:
                seen.add(key)
                blocks.append(txt)
        if blocks:
            rep = rep + '\n\n' + '\n\n'.join(blocks)
        if self.cov_info:
            rep = rep + '\n\n' + cc.report_block(
                [(o["name"], o["cov"]) for o in self.objectives], self.cov_info,
                downgrades, self.result)
        warns = list(self.warnings) + [w for w in downgrades if w not in self.warnings]
        if warns:
            rep = rep + '\n\nJoint-fit warnings:\n' + ''.join(
                '  - %s\n' % w for w in warns)
        if downgrades:
            rep = 'DEGRADED FIT: %s\n\n' % '; '.join(downgrades) + rep
        return rep

    def _derived_keymaps(self, keys=('protins', 'theta', 'phi')):
        """Per objective, {local key: master key driving it} for ``keys``.

        A reverse lookup of ``param_map``, so a key tied through a share-group
        maps to that group's master key (audit N2: ``_master_key`` alone knows
        only the legacy ``shared`` list, and sent a spec-shared pose to
        non-existent ``d{i}_theta`` names). Keys absent from ``param_map`` fall
        back to ``_master_key``."""
        loc2master = {}
        for mkey, targets in self.param_map.items():
            for (i, lk) in targets:
                loc2master[(i, lk)] = mkey
        return [{k: loc2master.get((idx, k), self._master_key(idx, k)) for k in keys}
                for idx in range(len(self.objectives))]

    # ----------------------------------------------------------------- #
    #  viewer / GUI duck-typing contract
    # ----------------------------------------------------------------- #
    @property
    def func(self):
        """Currently-selected objective's model instance (for NGLViewerDialog
        duck-typing: it reads ``parent.fit.func``)."""
        if not self.objectives:
            return None
        return self.objectives[self._current]["func"]

    @property
    def fit_params(self):
        """The CURRENT objective's local lmfit ``Parameters`` (master/detail view).

        The stock param tables (``update_sfit_parameters`` /
        ``update_mfit_parameters_new`` / ``update_fixed_parameters``) read
        ``parent.fit.fit_params`` and ``parent.fit.params['__mpar__']`` with bare
        local keys (``qoff``, ``__Model_d_001``).  Returning the current
        objective's own ``params`` lets those tables render / edit ONE dataset of
        a joint fit with no change -- the "detail" pane of the master/detail GUI.
        Edits land in that objective's model; ``build_master_params`` reads the
        edited values as the joint-fit starting point (hold persistent per-dataset
        model instances so edits survive a re-``compile``)."""
        f = self.func
        return f.params if f is not None else Parameters()

    @property
    def params(self):
        """The CURRENT objective model's ``__dict__`` — matches stock
        ``Fit.params = self.func.__dict__``. Holds scalar attributes plus the
        ``'__mpar__'`` and ``'choices'`` entries that the stock parameter tables
        AND ``update_plot`` read as ``self.fit.params[...]``. (``fit_params``,
        above, is the lmfit ``Parameters`` object — a different thing.) Returning
        the lmfit object here instead caused ``self.fit.params['choices']`` to
        ``KeyError`` under a joint fit."""
        f = self.func
        return f.__dict__ if f is not None else {}

    @property
    def objective_names(self):
        return [o["name"] for o in self.objectives]

    def set_current(self, index_or_name):
        if isinstance(index_or_name, str):
            index_or_name = self.objective_names.index(index_or_name)
        self._current = int(index_or_name)

    # convenience for regression tests
    def chisqr(self):
        return None if self.result is None else self.result.chisqr
