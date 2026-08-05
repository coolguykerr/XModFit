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

try:
    from scipy.linalg import solve_triangular as _solve_tri
except Exception:                                        # pragma: no cover
    _solve_tri = None

import joint_covariance as jcov

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
        # covariance state
        self._use_cov = False
        self._L = None
        self._cov_mask = None
        self._Sigma_logc = None       # shared K x K
        self._var_fwhm = 0.0          # shared beam FWHM variance (footprint term)
        self._use_footprint = False
        self._cov_ready = False
        # optional per-parameter Gaussian priors (soft constraints):
        #   {master_key: (mu, sigma)} -> append (value - mu)/sigma to the residual.
        self._priors = {}

    # ----------------------------------------------------------------- #
    #  dataset registration
    # ----------------------------------------------------------------- #
    def add_dataset(self, name, funcClass=None, x=None, y=None, yerr=None,
                    func=None, cov=None):
        """Register one dataset as a joint-fit objective.

        Parameters
        ----------
        name : str          dataset key (unique).
        funcClass : type    model class (e.g. XLayers); ignored if ``func`` given.
        func : instance      a pre-built model instance (alternative to funcClass).
        x, y, yerr : arrays  the data.  If ``cov`` is supplied, its ``y_corr`` /
                             ``sigma_meas`` OVERRIDE y / yerr for the fit.
        cov : dict or None   per-dataset covariance metadata as returned by
                             ``joint_covariance.build_dataset_cov_meta`` (keys
                             y_corr, sigma_meas, k_index, SF_LIST, Sigma_logc).
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

        if cov is not None:
            y_use = np.asarray(cov["y_corr"], dtype=float)
            yerr_use = np.asarray(cov["sigma_meas"], dtype=float)
            self._register_shared_sigma(cov["Sigma_logc"])
            self._use_cov = True
            # optional shared-beam footprint term (v + var_fwhm); see
            # joint_covariance.build_full_covariance. var_fwhm is a single shared
            # scalar (one beam), so it couples across datasets like Sigma_logc.
            if cov.get("v", None) is not None and cov.get("var_fwhm", 0.0):
                self._register_shared_var_fwhm(float(cov["var_fwhm"]))
                self._use_footprint = True
        else:
            y_use = np.zeros_like(x) if y is None else np.asarray(y, dtype=float)
            yerr_use = np.ones_like(x) if yerr is None else np.asarray(yerr, dtype=float)

        obj = {
            "name": name,
            "func": func,
            "x": x,
            "y": y_use,
            "yerr": yerr_use,
            "imin": 0,
            "imax": len(x) - 1,
            "cov": cov,
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
        self._use_cov = False
        self._L = None
        self._cov_mask = None
        self._Sigma_logc = None
        self._var_fwhm = 0.0
        self._use_footprint = False
        self._cov_ready = False
        self._priors = {}

    def _register_shared_sigma(self, Sigma_logc):
        Sigma_logc = np.asarray(Sigma_logc, dtype=float)
        if self._Sigma_logc is None:
            self._Sigma_logc = Sigma_logc
        else:
            if not np.allclose(self._Sigma_logc, Sigma_logc):
                raise ValueError(
                    "attached covariances use different Sigma_logc matrices; "
                    "cross-dataset correlation requires a single shared "
                    "correction-factor covariance (same attenuator calibration)."
                )

    def _register_shared_var_fwhm(self, var_fwhm):
        if not self._var_fwhm:
            self._var_fwhm = float(var_fwhm)
        elif not np.isclose(self._var_fwhm, float(var_fwhm)):
            raise ValueError(
                "attached covariances use different var_fwhm; the shared-beam "
                "footprint term assumes ONE beam FWHM across all datasets.")

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

    def build_master_params(self, shared=None, groups=None):
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
        from its FIRST endpoint's local parameter (shared parameters should agree;
        the first is authoritative).
        """
        specs = []
        for nm in (shared or []):
            specs.append(self.share_all(nm))
        for g in (groups or []):
            specs.append(dict(g))

        self.shared = [nm for nm in (shared or [])]
        self.master_params = Parameters()
        self.param_map = {}

        # 1) claimed endpoints -> group master keys
        claimed = {}                       # (idx, lkey) -> master key
        used_names = set()
        for gid, spec in enumerate(specs):
            members = [(self._resolve_idx(i) if not isinstance(i, int) else i, lk)
                       for (i, lk) in spec["members"]]
            if not members:
                continue
            first_idx, first_lk = members[0]
            mkey = spec.get("name") or self._group_master_key(
                gid, first_lk, bare=spec.get("bare", False), used=used_names)
            used_names.add(mkey)
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
        """Evaluate every objective's model on its [imin, imax] slice."""
        yfits = []
        for obj in self.objectives:
            func = obj["func"]
            func.__fit__ = True
            func.x = obj["x"][obj["imin"]:obj["imax"] + 1]
            yfits.append(np.asarray(func.y(), dtype=float))
        return yfits

    # ----------------------------------------------------------------- #
    #  covariance / Cholesky over the concatenation
    # ----------------------------------------------------------------- #
    def _build_chol(self):
        """Assemble C_full over the concatenated [imin,imax] slices of all
        objectives (in residual order) and cache its Cholesky factor."""
        yc, sm, ki, vv = [], [], [], []
        for obj in self.objectives:
            i0, i1 = obj["imin"], obj["imax"] + 1
            cov = obj["cov"]
            yc.append(cov["y_corr"][i0:i1])
            sm.append(cov["sigma_meas"][i0:i1])
            ki.append(np.asarray(cov["k_index"])[i0:i1])
            if self._use_footprint:
                # missing v on one dataset -> zero sensitivity (no beam coupling
                # for that dataset), consistent with build_full_covariance's rank-1
                vseg = cov.get("v", None)
                vv.append(np.asarray(vseg, dtype=float)[i0:i1] if vseg is not None
                          else np.zeros(i1 - i0))
        yc = np.concatenate(yc)
        sm = np.concatenate(sm)
        ki = np.concatenate(ki).astype(int)
        v = np.concatenate(vv) if self._use_footprint else None

        m = np.isfinite(yc) & np.isfinite(sm) & (sm > 0)
        if v is not None:
            m &= np.isfinite(v)
        self._cov_mask = m
        C = jcov.build_full_covariance(
            yc[m], sm[m], ki[m], self._Sigma_logc,
            v=(v[m] if v is not None else None),
            var_fwhm=(self._var_fwhm if self._use_footprint else 0.0))
        self._L, self._jitter = jcov.cholesky_with_jitter(C)
        self._cov_ready = True

    # ----------------------------------------------------------------- #
    #  residual
    # ----------------------------------------------------------------- #
    def residual(self, params, fit_scale):
        self.push_params(params)
        yfits = self.evaluate()

        if self._use_cov:
            # ---- Axis-2 / combined: whitened residual over concatenation ----
            r = []
            for obj, fit in zip(self.objectives, yfits):
                i0, i1 = obj["imin"], obj["imax"] + 1
                data = obj["y"][i0:i1]           # y_corr (absolute units)
                r.append(data - fit)
            r = np.concatenate(r)
            if self._L is None:
                self._build_chol()
            r = r[self._cov_mask]
            # L is lower-triangular (Cholesky): O(N^2) forward-substitution, NOT
            # a general O(N^3) solve -- this matters a lot at N~10^3 over many
            # least-squares iterations.
            if _solve_tri is not None:
                y = _solve_tri(self._L, r, lower=True, check_finite=False)
            else:
                y = np.linalg.solve(self._L, r)
        else:
            # ---- no covariance: byte-identical to Fit per-curve behaviour ----
            y = []
            for obj, fit in zip(self.objectives, yfits):
                i0, i1 = obj["imin"], obj["imax"] + 1
                data = obj["y"][i0:i1]
                err = obj["yerr"][i0:i1]
                maxd = np.max(data)
                data = data / maxd
                err = err / maxd
                fit = fit / maxd
                if fit_scale == "Log":
                    y.append((np.log10(data) - np.log10(fit)) * data / err)
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
    def _set_ranges(self, xmin, xmax):
        for obj in self.objectives:
            x = obj["x"]
            obj["imin"] = int(np.where(x >= 0.99 * xmin)[0][0])
            obj["imax"] = int(np.where(x <= xmax)[0][-1])

    def perform_fit(self, xmin, xmax, fit_scale="Linear", fit_method="leastsq",
                    maxiter=1000, **fit_kws):
        """Run a joint least-squares fit.  (emcee is Axis-1 only; not wired here.)

        Extra keyword args (e.g. ``ftol``/``xtol``/``gtol`` for the
        ``least_squares`` TRF backend) are forwarded to ``Minimizer.minimize``
        and thence to the underlying scipy solver. Backward-compatible: with no
        extra kwargs the behaviour is unchanged."""
        self.Niter = 0
        self.fit_abort = False
        self._set_ranges(xmin, xmax)
        self._L = None                      # force Cholesky rebuild for new range
        self._cov_ready = False

        if fit_method == "leastsq":
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", max_nfev=maxiter)
        elif fit_method == "least_squares":
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", max_nfev=maxiter)
        elif fit_method == "differential_evolution":
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", calc_covar=True,
                                    max_nfev=maxiter, popsize=300,
                                    updating="immediate")
        else:
            self.fitter = Minimizer(self.residual, self.master_params,
                                    fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy="raise", max_nfev=maxiter)

        self.result = self.fitter.minimize(method=fit_method,
                                           params=self.master_params,
                                           **fit_kws)
        return fit_report(self.result), self.result.message

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
