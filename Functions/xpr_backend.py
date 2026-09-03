"""
xpr_backend.py -- process-wide juliacall session driving the XPR Julia reflectivity
model (smoothed / autodifferentiable XLayersPDB).

XPR (Elias Boshara, ``XPR-main``) is a Julia implementation of the n-box + PDB-protein
specular reflectivity model in which every step of the forward model is continuously
differentiable, so ForwardDiff gives exact gradients and Hessians of the
log-likelihood.  This module is the thin Python side of it:

  * lazily starts ONE Julia session per process and activates the XPR project;
  * owns the (module-level, not per-instance) handle, so XModFit model objects that
    reference it stay picklable -- required by the parallel emcee/seeding paths,
    which ship the model to ``multiprocessing`` workers;
  * fingerprints ``XPR.load_data`` calls so the expensive preallocation is redone
    only when the data, the PDB, or a sizing bound actually changes;
  * guards the preallocated slab budget so an out-of-range proposal raises a Python
    error (or triggers a reload) instead of segfaulting the host process.

This module lives in ``Functions/`` and NOT in a category folder such as
``Functions/XRR/``: XModFit lists every ``.py`` in a category folder as a fittable
model class named after the file, so a helper there breaks the function list.
See ``Functions/pdb_cache.py`` for the same rule.

Julia project resolution order:
  1. ``$XPR_JULIA_PROJECT``
  2. ``<repo>/julia/XPR``                (vendored, version-pinned)
  3. ``<repo>/../XPR-main/XPR``          (a sibling working copy)
"""

import os
import sys
import threading

import numpy as np

try:
    from Functions.pdb_cache import pdbread
except Exception:  # running with Functions/ on sys.path
    from pdb_cache import pdbread


# Full Julia parameter vector is 8 + 2*num_boxes long, in this order.
IDX_QOFF, IDX_THETA, IDX_PHI, IDX_PROTINS, IDX_COV = 0, 1, 2, 3, 4
IDX_SIG, IDX_YSCALE, IDX_BKG = 5, 6, 7
IDX_BOXES = 8

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

_lock = threading.RLock()
_jl = None            # juliacall.Main once started
_convert = None       # juliacall.convert once started
_loaded = None        # fingerprint of the last successful XPR.load_data
_alloc = None         # sizing bounds baked into that load (see _Alloc)


class XPRUnavailable(RuntimeError):
    """Julia / juliacall / the XPR project could not be brought up."""


def julia_project_dir():
    """Absolute path of the XPR Julia project, or raise XPRUnavailable."""
    candidates = []
    env = os.environ.get('XPR_JULIA_PROJECT')
    if env:
        candidates.append(env)
    candidates.append(os.path.join(_REPO, 'julia', 'XPR'))
    candidates.append(os.path.join(os.path.dirname(_REPO), 'XPR-main', 'XPR'))
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'src', 'XPR.jl')):
            return os.path.abspath(c)
    raise XPRUnavailable(
        'Could not find the XPR Julia project (looked for src/XPR.jl under: %s). '
        'Set $XPR_JULIA_PROJECT to point at it.' % ', '.join(candidates))


def default_juliapkg_project():
    """Where we ask juliapkg to keep the Julia runtime, absent an explicit setting.

    Mirrors juliapkg's own no-virtualenv-no-conda fallback: ``<depot>/environments/
    pyjuliapkg``, honouring ``$JULIA_DEPOT_PATH``. One shared path for every Python
    environment on the machine.
    """
    depot = os.environ.get('JULIA_DEPOT_PATH')
    if depot:
        depot = os.path.abspath(depot.split(';' if os.name == 'nt' else ':')[0])
    else:
        depot = os.path.join(os.path.expanduser('~'), '.julia')
    return os.path.join(depot, 'environments', 'pyjuliapkg')


def _juliapkg_runtime_installed():
    """True if a Julia executable already sits in the configured juliapkg project."""
    proj = os.environ.get('PYTHON_JULIAPKG_PROJECT') or default_juliapkg_project()
    exe = 'julia.exe' if os.name == 'nt' else 'julia'
    return os.path.isfile(os.path.join(proj, 'pyjuliapkg', 'install', 'bin', exe))


def _forget_juliapkg_modules():
    """Drop juliacall/juliapkg from sys.modules so a re-import re-reads the config.

    juliapkg snapshots its configuration into a module-level ``STATE`` at import
    time (``state.reset_state()`` runs once, at the bottom of the module), so an
    environment variable set afterwards is simply not seen.
    """
    for name in [k for k in list(sys.modules)
                 if k == 'juliacall' or k.startswith('juliacall.')
                 or k == 'juliapkg' or k.startswith('juliapkg.')]:
        del sys.modules[name]


def _import_juliacall():
    from juliacall import Main as jl, convert as jlconvert
    return jl, jlconvert


def _juliacall_unavailable(exc):
    return XPRUnavailable(
        'Could not start juliacall: %s\n'
        '  * not installed?   python -m pip install -r requirements_xpr.txt\n'
        '  * the first run downloads a private Julia runtime (~1 GB installed) into\n'
        '    %s; override with $PYTHON_JULIAPKG_PROJECT.\n'
        '  * an SSL/certificate error here means juliapkg cannot reach\n'
        '    julialang-s3.julialang.org for its version check. That is retried\n'
        '    automatically with PYTHON_JULIAPKG_OFFLINE=yes when a runtime is\n'
        '    already installed -- so seeing this means there is not one yet.\n'
        'See julia/README.md.'
        % (exc, os.environ.get('PYTHON_JULIAPKG_PROJECT')
           or default_juliapkg_project()))


def get_jl():
    """Start (once) and return the Julia ``Main`` module with XPR loaded."""
    global _jl
    if _jl is not None:
        return _jl
    with _lock:
        if _jl is not None:
            return _jl
        # Both must be set BEFORE juliacall is imported.
        #
        # HANDLE_SIGNALS: without it Julia installs its own SIGINT handler and a
        # Ctrl-C (or a Qt-thread signal) can abort the process rather than raising
        # into Python.
        #
        # AUTOLOAD_IPYTHON_EXTENSION: juliacall auto-loads an IPython extension
        # whenever ``IPython`` merely happens to be in sys.modules with a live
        # shell -- which it is inside XModFit, via pyqtgraph. That extension
        # rewires Julia's stdin/stdout onto IPython's, and doing that inside the Qt
        # process kills it stone dead: the interpreter exits, silently and with
        # status 0, the instant juliacall is imported. Nothing here wants the
        # %julia magics, so turn it off.
        os.environ.setdefault('PYTHON_JULIACALL_HANDLE_SIGNALS', 'yes')
        os.environ.setdefault('PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION', 'no')
        # Pin the juliapkg project to one shared location unless the user has said
        # otherwise. juliapkg's own default keys on $CONDA_PREFIX, so every ACTIVATED
        # conda environment would get a private ~1 GB Julia install and its own
        # precompile -- and, worse, running the same interpreter by absolute path
        # (no activation) resolves somewhere else entirely, so the two disagree about
        # whether anything is installed at all. An explicit $PYTHON_JULIAPKG_PROJECT,
        # or -X juliapkg-project (which juliapkg checks first), still wins.
        os.environ.setdefault('PYTHON_JULIAPKG_PROJECT', default_juliapkg_project())
        try:
            jl, _jlconvert = _import_juliacall()
        except Exception as e:
            # juliapkg pings julialang-s3 for the version list on EVERY startup, so a
            # broken SSL stack (or no network) stops a perfectly good existing install
            # from being used. If the runtime is already there, retry once offline.
            # This runs before libjulia is loaded -- juliacall fails inside
            # juliapkg.executable() -- so re-importing is safe.
            if (_juliapkg_runtime_installed()
                    and 'PYTHON_JULIAPKG_OFFLINE' not in os.environ):
                os.environ['PYTHON_JULIAPKG_OFFLINE'] = 'yes'
                _forget_juliapkg_modules()
                try:
                    jl, _jlconvert = _import_juliacall()
                except Exception:
                    del os.environ['PYTHON_JULIAPKG_OFFLINE']
                    raise _juliacall_unavailable(e)
            else:
                raise _juliacall_unavailable(e)
        global _convert
        _convert = _jlconvert
        proj = julia_project_dir()
        try:
            # The project juliapkg set up for us, holding PythonCall and its
            # dependency graph. Remember it BEFORE activating ours.
            juliapkg_project = str(jl.seval('Base.active_project()') or '')
            # Fast path: swap the active project with Base and load. Importing Pkg
            # and running activate+instantiate costs ~4.4 s of the ~16 s it takes a
            # worker process to become useful, and Pkg.instantiate additionally takes
            # depot locks -- so with a 16-way seed Pool they serialise against each
            # other. Everything it would do is already done after the first run, so
            # only fall back to it if the cheap load actually fails.
            jl.seval('Base.set_active_project(raw"%s")'
                     % os.path.join(proj, 'Project.toml'))
            # Keep juliapkg's project on LOAD_PATH as a FALLBACK (appended, so the
            # XPR manifest still wins every lookup). Without it, loading XPR pulls in
            # StaticArraysCore via ForwardDiff, which fires the package-extension
            # callback for StructUtils -- already loaded from juliapkg's project via
            # PythonCall -> JSON. Julia then tries to load
            # StructUtilsStaticArraysCoreExt, cannot find it in the now-active XPR
            # project, and dumps a 26-frame "required but does not seem to be
            # installed" error. Harmless to the model, but it fires in every process
            # that starts Julia -- so with a multi-core emcee or seed scan you get one
            # per worker, which looks like something is badly wrong.
            if juliapkg_project:
                jl.seval('let p = raw"%s"; p in LOAD_PATH || push!(LOAD_PATH, p); end'
                         % juliapkg_project)
            try:
                jl.seval('using XPR')
            except Exception:
                # First ever run, or the manifest changed: resolve properly, once.
                jl.seval('using Pkg')
                jl.Pkg.activate(proj)
                jl.Pkg.instantiate()
                jl.seval('using XPR')
        except Exception as e:
            raise XPRUnavailable('Failed to activate/instantiate the XPR project at '
                                 '%s: %s' % (proj, e))
        _jl = jl
        return _jl


def julia_available():
    """True if a Julia session can be started, without raising."""
    try:
        get_jl()
        return True
    except XPRUnavailable:
        return False


class _Alloc(object):
    """The sizing bounds a given ``load_data`` was preallocated against.

    XPR sizes ``max_ed_slabs``/``max_refl_slabs`` from these; exceeding them writes
    past the end of the preallocated buffers, which segfaults the process instead of
    raising.  We keep them here so ``check_headroom`` can catch it in Python first.
    """

    __slots__ = ('max_box_length', 'max_sig', 'slab_headroom', 'num_boxes',
                 'protlen', 'normalized', 'p_lo', 'p_hi')

    def __init__(self, max_box_length, max_sig, slab_headroom,
                 num_boxes=0, protlen=0.0):
        self.normalized = False
        self.p_lo = None
        self.p_hi = None
        self.max_box_length = float(max_box_length)
        self.max_sig = float(max_sig)
        self.slab_headroom = float(slab_headroom)
        self.num_boxes = int(num_boxes)
        # sq_max_l + 2*max_r: the z extent XPR budgets for the protein, matching
        # the expression in prealloc().
        self.protlen = float(protlen)

    def covers(self, other):
        return (self.max_box_length >= other.max_box_length - 1e-9
                and self.max_sig >= other.max_sig - 1e-9
                and self.slab_headroom >= other.slab_headroom - 1e-9)


def _arr_key(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    return (a.shape, a.tobytes())


def load(x, num_boxes, pdbfile=None, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
         E=10.0, rrf=True, max_box_length=30.0, max_sig=5.0, slab_headroom=0.0,
         data_R=None, data_err=None, electron_scale=1.0,
         param_lower=None, param_upper=None, force=False):
    """Load data + geometry into the Julia globals, skipping no-op reloads.

    ``electron_scale`` multiplies the per-atom electron counts. A CONSTANT, folded
    into the loaded array, so it is invisible to the parameters and to ForwardDiff.
    ``XLayersPDB_XPR`` uses it to repair a dimensional mismatch in XPR's fast
    volume path -- see ``XLayersPDB_XPR.cov_scale``.

    ``param_lower``/``param_upper`` (each 8+2n long) reparameterise the model onto
    the unit cube: every entry point then TAKES a vector in [0,1] and returns
    derivatives with respect to it, because XPR applies the mapping inside its own
    AD tape. Intended for the bounded global search, where a raw-unit condition
    number around 1e13 makes any solver crawl. Pass neither for physical units.

    ``data_R``/``data_err`` only feed XPR's own log-likelihood.  XModFit computes its
    own residual from ``y()``, so the defaults (zeros / ones) are fine for stage-1
    forward evaluation; pass the real arrays when you want XPR's ``logL``/gradient
    with respect to the actual data (stages 2-3).

    Returns the number of q points loaded.
    """
    global _loaded, _alloc
    jl = get_jl()

    x = np.ascontiguousarray(np.asarray(x, dtype=float).ravel())
    if x.size == 0:
        raise ValueError('XPR: empty q array')
    data_R = np.zeros_like(x) if data_R is None else \
        np.ascontiguousarray(np.asarray(data_R, dtype=float).ravel())
    data_err = np.ones_like(x) if data_err is None else \
        np.ascontiguousarray(np.asarray(data_err, dtype=float).ravel())
    if len(data_R) != len(x) or len(data_err) != len(x):
        raise ValueError('XPR: data_R/data_err length must match q')

    include_protein = bool(pdbfile) and os.path.isfile(pdbfile)
    want = _Alloc(max_box_length, max_sig, slab_headroom)

    fp = (_arr_key(x), _arr_key(data_R), _arr_key(data_err), int(num_boxes),
          include_protein, os.path.abspath(pdbfile) if include_protein else None,
          _pdb_mtime(pdbfile) if include_protein else None,
          str(vol_calc), float(rho_top), float(rho_bottom), float(E), bool(rrf),
          float(electron_scale), _arr_key(param_lower) if param_lower is not None else None,
          _arr_key(param_upper) if param_upper is not None else None)

    with _lock:
        if (not force and _loaded is not None and _loaded == fp
                and _alloc is not None and _alloc.covers(want)):
            return len(x)

        # Keep any headroom a previous load already paid for, so a fit that grows
        # the bounds once does not shrink back and re-trigger reloads.
        if _loaded == fp and _alloc is not None:
            want = _Alloc(max(want.max_box_length, _alloc.max_box_length),
                          max(want.max_sig, _alloc.max_sig),
                          max(want.slab_headroom, _alloc.slab_headroom))
        want.num_boxes = int(num_boxes)

        # Hand Julia real Julia Arrays rather than PyArray views, matching upstream
        # XPRsim: XPR keeps the data for the life of the load, so it must not alias
        # numpy buffers that Python is free to free or mutate afterwards.
        def _jarr(a):
            return _convert(jl.Array, np.ascontiguousarray(np.asarray(a, dtype=float)))

        if (param_lower is None) != (param_upper is None):
            raise ValueError('param_lower and param_upper must be given together')
        kwargs = dict(vol_calc=str(vol_calc), rho_top=float(rho_top),
                      rho_bottom=float(rho_bottom), xray_energy=float(E),
                      fres=bool(rrf), max_box_length=want.max_box_length,
                      max_sig=want.max_sig, slab_headroom=want.slab_headroom)
        if include_protein:
            xc, yc, zc, electrons, radii, _mass = pdbread(pdbfile)
            coord = np.transpose(np.array([xc, yc, zc]))
            kwargs['coordinates_py'] = _jarr(coord)
            kwargs['electrons_py'] = _jarr(np.asarray(electrons) * float(electron_scale))
            kwargs['radii_py'] = _jarr(radii)
            span = coord.max(axis=0) - coord.min(axis=0)
            want.protlen = float(np.sqrt(np.sum(span ** 2)) + 2.0 * np.max(radii))

        # fit_sig/fit_yscl/fit_bkg only select whether XPR reads slots 6/7/8 of the
        # parameter vector or substitutes its own defaults. XModFit always supplies
        # them and lets lmfit decide what varies, so they are unconditionally True.
        if param_lower is not None:
            lo = np.asarray(param_lower, dtype=float).ravel()
            hi = np.asarray(param_upper, dtype=float).ravel()
            want.normalized = True
            want.p_lo, want.p_hi = lo.copy(), hi.copy()
            kwargs['param_lower'] = _jarr(lo)
            kwargs['param_upper'] = _jarr(hi)

        jl.load_data(_jarr(x), _jarr(data_R), _jarr(data_err), int(num_boxes),
                     include_protein, True, True, True, **kwargs)
        _loaded = fp
        _alloc = want
        return len(x)


def _pdb_mtime(fname):
    try:
        return os.path.getmtime(fname)
    except OSError:
        return None


def loaded_alloc():
    """The sizing bounds of the current load, or None."""
    return _alloc


def is_normalized():
    """True if the current load reparameterised the model onto the unit cube."""
    return bool(_alloc is not None and _alloc.normalized)


def to_physical(u):
    """Unit-cube vector -> physical parameter vector (identity if not normalised)."""
    if _alloc is None or not _alloc.normalized:
        return np.asarray(u, dtype=float)
    return _alloc.p_lo + (_alloc.p_hi - _alloc.p_lo) * np.asarray(u, dtype=float)


def to_normalized(p):
    """Physical parameter vector -> unit cube (identity if not normalised)."""
    if _alloc is None or not _alloc.normalized:
        return np.asarray(p, dtype=float)
    return (np.asarray(p, dtype=float) - _alloc.p_lo) / (_alloc.p_hi - _alloc.p_lo)


def alloc_bounds_for(param_lower, param_upper, num_boxes, protins_idx=IDX_PROTINS):
    """Slab-budget sizing derived from the parameter BOUNDS rather than the current
    values, so a bounded search never triggers a reallocation mid-run."""
    lo = np.asarray(param_lower, dtype=float)
    hi = np.asarray(param_upper, dtype=float)
    d_hi = hi[IDX_BOXES:IDX_BOXES + num_boxes]
    max_box_length = float(max(np.max(d_hi) if num_boxes else 0.0, 1.0))
    max_sig = float(max(hi[IDX_SIG], 1.0))
    overhang = max(0.0, -float(lo[protins_idx])) + max(
        0.0, float(hi[protins_idx]) - max_box_length * max(num_boxes, 1))
    return max_box_length, max_sig, float(max(overhang, 10.0))


def check_headroom(full_params, num_boxes, include_protein):
    """Raise ValueError if ``full_params`` would overrun XPR's preallocated slabs.

    XPR budgets its slab arrays for ``max_box_length`` per box, ``max_sig``
    roughness and ``slab_headroom`` of slack; going past any of them is a segfault,
    not an exception, so this has to be checked before every call.
    """
    if _alloc is None:
        return
    # the guard reasons in physical units whatever the caller passes
    p = to_physical(full_params)
    d = p[IDX_BOXES:IDX_BOXES + num_boxes]
    bad = []
    if num_boxes and float(np.max(d)) > _alloc.max_box_length:
        bad.append('box length %.4g > allocated max_box_length %.4g'
                   % (float(np.max(d)), _alloc.max_box_length))
    if float(p[IDX_SIG]) > _alloc.max_sig:
        bad.append('sig %.4g > allocated max_sig %.4g'
                   % (float(p[IDX_SIG]), _alloc.max_sig))
    if include_protein:
        # XPR lays the profile on a spacez grid running from the lower of
        # (protein bottom, -sum(d)) to the higher of (protein top, 0); the protein
        # sits with its top at protins - sum(d). Both a NEGATIVE protins (protein
        # dropped below the stack) and a protins larger than the protein itself
        # (protein lifted clear above it) stretch that span past what prealloc
        # budgeted, and overrunning it is a segfault rather than an error.
        ins = float(p[IDX_PROTINS])
        sumd = float(np.sum(d))
        need = max(ins - sumd, 0.0) + max(sumd + _alloc.protlen - ins, sumd)
        budget = (_alloc.protlen + _alloc.max_box_length * _alloc.num_boxes
                  + _alloc.slab_headroom)
        if need > budget:
            bad.append('protins %.4g gives a %.4g A profile, allocated %.4g A '
                       '(raise slab_headroom)' % (ins, need, budget))
    if bad:
        raise ValueError(
            'XPR parameters exceed the preallocated slab budget (%s). Widen the '
            'parameter bounds used for allocation, or lower the parameters -- '
            'calling Julia here would segfault.' % '; '.join(bad))


def sim(full_params):
    """(logL, R) for the full 8+2n parameter vector against the loaded data."""
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    logL, R = jl.XPR_sim_ref(p)
    return float(logL), np.array(R, dtype=float)


def grad(full_params):
    """d(logL)/d(params), full 8+2n vector (ForwardDiff).

    Exact for ``XLayersPDB_XPR``: that model passes its fitted parameters through
    untouched, so nothing the gradient needs sits outside Julia's tape. Its covered
    fraction is derived from ``cov`` afterwards rather than folded into it, which is
    what keeps this true.
    """
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    return np.array(jl.XPR_grad(p), dtype=float)


def hess(full_params):
    """d2(logL)/d(params)2, (8+2n) x (8+2n) (ForwardDiff)."""
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    return np.array(jl.XPR_hess(p), dtype=float)


def edp(full_params):
    """(z, rho) of the smeared electron-density profile for these parameters.

    z is returned in XModFit's convention -- 0 at the top of the box stack,
    increasing INTO the film/subphase -- i.e. negated from XPR's own axis.
    """
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    zmid, rho = jl.XPR_edp(p)
    return -np.array(zmid, dtype=float), np.array(rho, dtype=float)


def jac(full_params, active=None):
    """dR_i/dp_j for the selected parameters, shape (len(q), len(active)).

    ``active`` is a sequence of 0-based indices into the 8+2n vector; ForwardDiff
    seeds only those, so the cost tracks the number of VARYING parameters rather
    than the full vector. This is the Jacobian of the reflectivity, which is what a
    least-squares fit needs -- ``grad``/``hess`` are of the scalar log-likelihood.
    """
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    if active is None:
        active = np.arange(len(p))
    a = np.ascontiguousarray(np.asarray(active, dtype=np.int64).ravel() + 1)
    return np.array(jl.XPR_jac(p, a), dtype=float).reshape(-1, len(a))


def sim_edp(full_params):
    """(logL, R, z, rho) from ONE model evaluation.

    ``sim`` followed by ``edp`` runs the whole forward model twice -- protein
    rotation, slice areas and per-slab electron densities included. Anything that
    wants both (i.e. every ``y()`` outside a fit) should use this instead: it is
    ~1.9x faster than the pair.
    """
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    logL, R, zmid, rho = jl.XPR_sim_edp(p)
    return (float(logL), np.array(R, dtype=float),
            -np.array(zmid, dtype=float), np.array(rho, dtype=float))


def reset():
    """Drop the load fingerprint so the next ``load`` really reloads."""
    global _loaded, _alloc
    with _lock:
        _loaded = None
        _alloc = None
