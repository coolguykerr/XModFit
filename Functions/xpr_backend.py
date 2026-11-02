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
import warnings

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
    """The sizing arguments a given ``load_data`` was preallocated against.

    Since XPR's 2026-09-09 revision the slab COUNT is a load-time constant and the
    SPACING varies with the parameters, so these no longer guard a buffer overrun --
    the fixed-count design cannot overrun one.  They matter for two other reasons:

    * **``max_box_length`` and ``spacez`` change the ANSWER.**  ``cells_per_box =
      ceil(max_box_length/spacez)``, so together they set the in-box mesh
      ``d_b/cells_per_box``.  Measured on the 1697-atom test protein: raising
      ``max_box_length`` from 30 to 400 moves logL by 0.19, and halving ``spacez``
      moves it by 0.76.  They must therefore be PINNED for the life of a fit -- a
      reload that changes either one steps the objective mid-descent.  Contrast
      ``d_protein_bounds`` and ``sig_max``, which change the slab count (hence the
      cost) but NOT the answer to four decimals, because they only extend
      constant-density far field.
    * **``d_protein_bounds`` is enforced by Julia with a hard ``error()``.**
      ``check_headroom`` mirrors that test so we fail in Python with a useful
      message instead of unwinding a Julia exception through juliacall mid-fit.
    """

    __slots__ = ('max_box_length', 'sig_max', 'spacez', 'min_box_length_sum',
                 'd_protein_bounds', 'num_boxes', 'prot_extent',
                 'normalized', 'p_lo', 'p_hi')

    def __init__(self, max_box_length, sig_max, spacez=0.5,
                 min_box_length_sum=0.0, d_protein_bounds=(-30.0, 30.0),
                 num_boxes=0, prot_extent=0.0):
        self.normalized = False
        self.p_lo = None
        self.p_hi = None
        self.max_box_length = float(max_box_length)
        self.sig_max = float(sig_max)
        self.spacez = float(spacez)
        self.min_box_length_sum = float(min_box_length_sum)
        self.d_protein_bounds = (float(d_protein_bounds[0]),
                                 float(d_protein_bounds[1]))
        self.num_boxes = int(num_boxes)
        # max_i(|r_i| + rad_i) about the mass centroid -- the rotation-INVARIANT
        # bounding radius, matching `prot_extent` in prealloc(). Upstream's
        # out-of-bounds check uses this, not the oriented extent, so ours must too.
        self.prot_extent = float(prot_extent)

    # Every sizing argument now participates in the geometry, so a load is reusable
    # only if they all match. There is no "covers" relation to exploit any more:
    # a LARGER max_box_length is not a superset, it is a different mesh.
    def matches(self, other):
        return (abs(self.max_box_length - other.max_box_length) < 1e-9
                and abs(self.sig_max - other.sig_max) < 1e-9
                and abs(self.spacez - other.spacez) < 1e-12
                and abs(self.min_box_length_sum - other.min_box_length_sum) < 1e-9
                and abs(self.d_protein_bounds[0] - other.d_protein_bounds[0]) < 1e-9
                and abs(self.d_protein_bounds[1] - other.d_protein_bounds[1]) < 1e-9)

    def top_reach(self):
        """Grid ceiling above z=0, in Angstroms (``top_cells * spacez``)."""
        import math
        dmax = self.d_protein_bounds[1]
        n = math.ceil(max(0.0, dmax + self.prot_extent - self.min_box_length_sum)
                      / self.spacez)
        return n * self.spacez

    def bottom_reach(self):
        """Grid floor below the bottom box interface (``bottom_cells * spacez``)."""
        import math
        dmin = self.d_protein_bounds[0]
        n = math.ceil(max(0.0, self.prot_extent - dmin) / self.spacez)
        return n * self.spacez


def _arr_key(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    return (a.shape, a.tobytes())


def load(x, num_boxes, pdbfile=None, vol_calc='approx', rho_top=0.0, rho_bottom=0.334,
         E=10.0, rrf=True, max_box_length=30.0, sig_max=5.0, spacez=0.5,
         min_box_length_sum=0.0, d_protein_bounds=(-30.0, 30.0),
         data_R=None, data_err=None, electron_scale=1.0,
         param_lower=None, param_upper=None, force=False):
    """Load data + geometry into the Julia globals, skipping no-op reloads.

    THE SIZING ARGUMENTS ARE PART OF THE MODEL, not just an allocation hint.  Since
    XPR's 2026-09-09 revision the slab count is fixed at load time and the spacing
    varies with the parameters, so:

    * ``max_box_length`` and ``spacez`` set the in-box mesh (``cells_per_box =
      ceil(max_box_length/spacez)``, spacing ``d_b/cells_per_box``) and the mesh
      everywhere outside the boxes (rigidly ``spacez``).  **Both change the
      answer.**  Derive them ONCE from the parameter bounds and hold them fixed for
      the life of a fit; ratcheting them from live values steps the objective
      mid-descent.
    * ``d_protein_bounds`` and ``min_box_length_sum`` size the grid window the
      protein may move in.  Julia enforces the window with a hard ``error()``, so
      they must bracket the fitted ``protins`` -- :func:`check_headroom` mirrors the
      test.  They change the slab count and hence the cost, but not the answer.
    * ``sig_max`` sizes the Gaussian tails (``10*sig_max`` each end).  Cost only,
      provided it bounds the fitted ``sig``; below it the profile is clipped.

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
    want = _Alloc(max_box_length, sig_max, spacez, min_box_length_sum,
                  d_protein_bounds, num_boxes=num_boxes)

    # The sizing arguments are in the FINGERPRINT, not compared with a "covers"
    # relation: max_box_length and spacez change the answer, so a load with
    # different ones is a different model and must not be silently reused. There is
    # also nothing left to ratchet -- the old code grew the slab budget to avoid a
    # segfault, and the fixed-count design has removed that failure mode.
    fp = (_arr_key(x), _arr_key(data_R), _arr_key(data_err), int(num_boxes),
          include_protein, os.path.abspath(pdbfile) if include_protein else None,
          _pdb_mtime(pdbfile) if include_protein else None,
          str(vol_calc), float(rho_top), float(rho_bottom), float(E), bool(rrf),
          float(electron_scale), float(max_box_length), float(sig_max),
          float(spacez), float(min_box_length_sum),
          (float(d_protein_bounds[0]), float(d_protein_bounds[1])),
          _arr_key(param_lower) if param_lower is not None else None,
          _arr_key(param_upper) if param_upper is not None else None)

    with _lock:
        if not force and _loaded is not None and _loaded == fp and _alloc is not None:
            return len(x)

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
                      sig_max=want.sig_max, spacez=want.spacez,
                      min_box_length_sum=want.min_box_length_sum,
                      d_protein_bounds=want.d_protein_bounds)
        if include_protein:
            xc, yc, zc, electrons, radii, mass = pdbread(pdbfile)
            coord = np.transpose(np.array([xc, yc, zc]))
            kwargs['coordinates_py'] = _jarr(coord)
            kwargs['electrons_py'] = _jarr(np.asarray(electrons) * float(electron_scale))
            kwargs['radii_py'] = _jarr(radii)
            # XPR re-centres the coordinates on the MASS centroid at load time and
            # measures protins from there, so the masses are part of the geometry
            # now, not bookkeeping. electron_scale must not touch them.
            kwargs['masses_py'] = _jarr(mass)
            # prot_extent as prealloc() computes it: max_i(|r_i| + rad_i) about the
            # MASS centroid, rotation invariant. This is the radius upstream's
            # out-of-bounds error() tests against, so check_headroom must use the
            # same one -- the oriented extent (pdb_geometry.com_extent) is smaller
            # and would let a call through that Julia then rejects.
            com = (coord * np.asarray(mass)[:, None]).sum(axis=0) / np.sum(mass)
            want.prot_extent = float(np.max(
                np.linalg.norm(coord - com, axis=1) + np.asarray(radii)))

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


def slab_count():
    """(M, B): physical slab boundaries and smearing nodes for the current load.

    Both are load-time constants now, and ``eval_smeared!`` is O(B^2), so this is
    what the sizing arguments actually cost. Generous ``protins``/``sig`` bounds are
    the usual reason B is larger than it needs to be.
    """
    jl = get_jl()
    sd = 'XPR.GLOBAL_SD[].'
    try:
        tc = int(jl.seval(sd + 'top_cells'))
        bc = int(jl.seval(sd + 'bottom_cells'))
        nt = int(jl.seval(sd + 'tail_cells'))
        bx = int(np.sum(np.array(jl.seval(sd + 'box_cells'))))
    except Exception:
        return None, None
    M = 2 + tc + bx + bc
    return M, (M - 1) + 2 * nt


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


def alloc_bounds_for(param_lower, param_upper, num_boxes, protins_idx=IDX_PROTINS,
                     spacez=0.5, box_ref_length=30.0):
    """The sizing arguments :func:`load` needs, for a given parameter box.

    Returns ``dict(max_box_length=, sig_max=, spacez=, min_box_length_sum=,
    d_protein_bounds=)``, ready to splat into :func:`load`.

    **The split between what comes from the bounds and what does not is deliberate**,
    because the five arguments are not the same kind of thing:

    * ``max_box_length`` and ``spacez`` set the mesh -- ``cells_per_box =
      ceil(max_box_length/spacez)`` and in-box spacing ``d_b/cells_per_box`` -- so
      they set THE ANSWER.  They come from ``box_ref_length`` and ``spacez``, NOT
      from the parameter bounds: two people fitting the same data with different
      ``d`` bounds must get the same chi2.  ``box_ref_length`` is the box length at
      which the in-box mesh equals ``spacez``; upstream's default of 30 A is kept so
      our numbers reproduce ``demo.ipynb``.
    * ``d_protein_bounds`` IS taken from the ``protins`` bounds, because Julia
      enforces it with a hard ``error("protein out of bounds")``.  If the declared
      window were narrower than the fitted bounds, a proposal legal to lmfit could
      raise inside Julia and abort the fit.  It changes the slab count, hence cost,
      but not the answer.
    * ``sig_max`` comes from the ``sig`` upper bound for the same reason in
      miniature: below the actual ``sig`` the Gaussian tail is clipped.  Cost only.
    * ``min_box_length_sum`` is the sum of the box LOWER bounds, which upstream
      subtracts from the top-region cell count.  Pure speed, cannot affect the
      answer.

    A box longer than ``box_ref_length`` is not an error -- the mesh simply coarsens
    to ``d_b*spacez/box_ref_length`` -- but it is no longer the spacing that was
    asked for, so :func:`check_headroom` warns.
    """
    lo = np.asarray(param_lower, dtype=float)
    hi = np.asarray(param_upper, dtype=float)
    d_lo = lo[IDX_BOXES:IDX_BOXES + num_boxes]
    return dict(
        max_box_length=float(max(box_ref_length, 1.0)),
        sig_max=float(max(hi[IDX_SIG], 1.0)),
        spacez=float(spacez),
        min_box_length_sum=float(max(np.sum(np.maximum(d_lo, 0.0))
                                     if num_boxes else 0.0, 0.0)),
        d_protein_bounds=(float(lo[protins_idx]), float(hi[protins_idx])),
    )


def maxh(theta, phi):
    """``(maxh, mindepth, atom)`` for the loaded structure at this orientation.

    ``maxh`` is the distance from the protein's centre of mass to the topmost point
    of the topmost atomic sphere along the rotated z axis, and ``mindepth`` the same
    downwards.  ``maxh`` is exactly the offset between XPR's two definitions of the
    insertion parameter::

        protins_top = protins + maxh(theta, phi)

    where ``protins`` (what the model fits, and what XPR uses) places the CENTRE OF
    MASS above the bottom of the box stack, and ``protins_top`` (the old definition)
    places the TOP OF THE PROTEIN there.  ``atom`` is the 1-based index of the atom
    attaining the maximum; ``maxh`` is a maximum over atoms, so it is only piecewise
    smooth in (theta, phi) and a derivative needs to know which term is active --
    which is exactly why the offset was taken OUT of the forward model.

    Cheap: no reflectivity is computed. Requires a prior :func:`load` with a PDB.
    """
    jl = get_jl()
    h, dn, k = jl.XPR_maxh(float(theta), float(phi))
    return float(h), float(dn), int(k)


_warned = set()


def _warn_once(key, msg):
    """One warning per condition per process -- these fire inside the residual."""
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def check_headroom(full_params, num_boxes, include_protein,
                   prot_up=None, prot_down=None):
    """Raise ValueError if ``full_params`` would trip XPR's out-of-bounds ``error()``.

    This no longer guards a buffer overrun -- since the 2026-09-09 revision the slab
    count is a load-time constant, so no proposal can overrun an array and the
    segfault class is gone.  What it guards is the check upstream added on
    2026-09-10 in ``XPR!``::

        z_top = d_protein + prot_extent - sum_d
        z_bot = d_protein - prot_extent - sum_d
        if z_top > top_cells*spacez || z_bot < -sum_d - bottom_cells*spacez
            error("protein out of bounds")

    That is the right call by upstream -- the previous behaviour was to clip the
    protein silently and return a smooth, wrong likelihood, which an MCMC sampler
    accepts happily.  But a Julia exception unwinding through juliacall in the middle
    of an lmfit descent or an emcee chain aborts the run, so we test it first and
    raise a Python error naming the numbers.

    Mirrored EXACTLY, including upstream's use of the rotation-invariant
    ``prot_extent`` rather than the oriented extent: using the (smaller) oriented
    extent here would let a call through that Julia then rejects.  ``prot_up`` and
    ``prot_down`` are accepted for backwards compatibility and ignored.

    Only the protein window RAISES, because only it is fatal.  A box longer than
    ``max_box_length``, or a ``sig`` above ``sig_max``, degrades gracefully in Julia
    (a coarser in-box mesh; a clipped Gaussian tail), so those warn once per
    condition instead -- raising would break a legitimate fit whose film is simply
    thicker than the reference length.
    """
    if _alloc is None:
        return
    # the guard reasons in physical units whatever the caller passes
    p = to_physical(full_params)
    d = p[IDX_BOXES:IDX_BOXES + num_boxes]
    bad = []
    if num_boxes and float(np.max(d)) > _alloc.max_box_length + 1e-9:
        _warn_once('box_length',
                   'XPR: box length %.4g exceeds the mesh reference length %.4g, so '
                   'the in-box mesh coarsens to %.4g A against the requested %.4g. '
                   'Not an error, but raise box_ref_length if you want the finer '
                   'mesh -- and note that changes chi2.'
                   % (float(np.max(d)), _alloc.max_box_length,
                      float(np.max(d)) * _alloc.spacez / _alloc.max_box_length,
                      _alloc.spacez))
    if float(p[IDX_SIG]) > _alloc.sig_max + 1e-9:
        _warn_once('sig_max',
                   'XPR: sig %.4g exceeds sig_max %.4g, so the Gaussian tail is '
                   'clipped at %.4g A instead of %.4g. Raise the sig upper bound.'
                   % (float(p[IDX_SIG]), _alloc.sig_max, 10.0 * _alloc.sig_max,
                      10.0 * float(p[IDX_SIG])))
    if include_protein:
        ins = float(p[IDX_PROTINS])
        sumd = float(np.sum(d))
        ext = _alloc.prot_extent
        z_top = ins + ext - sumd
        z_bot = ins - ext - sumd
        top = _alloc.top_reach()
        bot = -sumd - _alloc.bottom_reach()
        if z_top > top:
            bad.append('protins %.4g puts the protein top at z=%.4g, above the grid '
                       'ceiling z=%.4g (widen d_protein_bounds[1] beyond %.4g)'
                       % (ins, z_top, top, _alloc.d_protein_bounds[1]))
        if z_bot < bot:
            bad.append('protins %.4g puts the protein bottom at z=%.4g, below the '
                       'grid floor z=%.4g (lower d_protein_bounds[0] below %.4g)'
                       % (ins, z_bot, bot, _alloc.d_protein_bounds[0]))
    if bad:
        raise ValueError(
            'XPR parameters are outside the geometry declared at load time (%s). '
            'Fix the fitted bounds, or reload with wider sizing arguments.'
            % '; '.join(bad))


def sim(full_params):
    """(logL, R) for the full 8+2n parameter vector against the loaded data."""
    return sim_ref(full_params)[:2]


def sim_ref(full_params):
    """(logL, R, protins_top) -- XPR's own three-value return.

    ``protins_top`` is ``protins + maxh(theta, phi)``: the insertion measured to the
    TOP of the protein, which is how the parameter was defined before XPR moved it
    to the centre of mass. XPR computes it as a by-product of the rotation, so it is
    free here; :func:`maxh` gets the same number without the model evaluation.
    """
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    logL, R, prot_h = jl.XPR_sim_ref(p)
    return float(logL), np.array(R, dtype=float), float(prot_h)


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


def _to_xmodfit_z(zmid, rho):
    """XPR's lattice -> XModFit's z convention, ascending.

    XPR measures z upwards from the top interface; XModFit's other XRR models put 0
    at the top of the box stack and increase INTO the film, so the axis is negated.
    XPR's lattice now runs bottom-to-top (it was top-to-bottom before the rigid-
    lattice rewrite), so negating leaves z DESCENDING -- reverse it, since a
    ``plotType='line'`` output and the ``np.diff`` that forms dEDP both assume the
    points arrive in order.
    """
    z = -np.array(zmid, dtype=float)
    rho = np.array(rho, dtype=float)
    if z.size > 1 and z[0] > z[-1]:
        z, rho = z[::-1], rho[::-1]
    return z, rho


def edp(full_params):
    """(z, rho) of the smeared electron-density profile for these parameters.

    z is returned in XModFit's convention -- 0 at the top of the box stack,
    increasing INTO the film/subphase -- i.e. negated from XPR's own axis.
    """
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    zmid, rho = jl.XPR_edp(p)
    return _to_xmodfit_z(zmid, rho)


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
    """(logL, R, protins_top, z, rho) from ONE model evaluation.

    ``sim`` followed by ``edp`` runs the whole forward model twice -- protein
    rotation, slice areas and per-slab electron densities included. Anything that
    wants both (i.e. every ``y()`` outside a fit) should use this instead: it is
    ~1.9x faster than the pair.

    ``protins_top`` is the top-referenced insertion, as in :func:`sim_ref`.
    """
    jl = get_jl()
    p = np.ascontiguousarray(np.asarray(full_params, dtype=float).ravel())
    logL, R, prot_h, zmid, rho = jl.XPR_sim_edp(p)
    z, rho = _to_xmodfit_z(zmid, rho)
    return float(logL), np.array(R, dtype=float), float(prot_h), z, rho


def reset():
    """Drop the load fingerprint so the next ``load`` really reloads."""
    global _loaded, _alloc
    with _lock:
        _loaded = None
        _alloc = None


# --------------------------------------------------------------- Ipopt solver ---
# The fast global search needs a bounded local solver called a few hundred times
# per run. It lives HERE rather than in the fastscan package for one structural
# reason: get_jl() is the only place that sets
# PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=no before importing juliacall, and
# importing juliacall with a live IPython shell in sys.modules kills the process
# silently with status 0 -- which XModFit always has, because create_consoleDock
# installs one. Keeping the include and the solve wrapper next to get_jl() makes
# "everything goes through get_jl()" true by locality instead of by discipline.

_solvers_loaded = False


def solvers_file():
    """Path to the Julia Ipopt wrapper: ``$XPR_SOLVERS_JL``, else ``fastscan/solvers.jl``."""
    env = os.environ.get('XPR_SOLVERS_JL')
    if env:
        return os.path.abspath(env)
    return os.path.join(_REPO, 'fastscan', 'solvers.jl')


def include_solvers(path=None):
    """``jl.include`` the Ipopt wrapper. Idempotent; safe to call on every run.

    Raises :class:`XPRUnavailable` with an actionable message rather than letting a
    Julia exception unwind through juliacall. ``Ipopt`` is a declared dependency of
    the vendored ``julia/XPR`` project, but the first ``using Ipopt`` in a cold
    depot precompiles, which can take a minute -- so call this up front behind a
    progress label, never lazily from inside a solve loop.
    """
    global _solvers_loaded
    with _lock:
        jl = get_jl()
        if _solvers_loaded and bool(jl.seval('isdefined(Main, :XPR_SOLVERS_V1)')):
            return True
        f = os.path.abspath(path or solvers_file())
        if not os.path.isfile(f):
            raise XPRUnavailable(
                'the Julia Ipopt wrapper is missing: %s\n'
                'Set $XPR_SOLVERS_JL or restore fastscan/solvers.jl.' % f)
        try:
            jl.include(f.replace(os.sep, '/'))
        except Exception as ex:                       # noqa: BLE001 -- juliacall
            raise XPRUnavailable(
                'could not load the Ipopt solver from %s: %s\n'
                'Ipopt is a dependency of the XPR Julia project; if this is a fresh '
                'depot it may need to precompile once (Pkg.instantiate).'
                % (f, str(ex)[:400]))
        if not bool(jl.seval('isdefined(Main, :XPR_SOLVERS_V1)')):
            raise XPRUnavailable('%s loaded but did not define XPR_SOLVERS_V1' % f)
        _solvers_loaded = True
        return True


def solvers_available():
    """Whether :func:`include_solvers` succeeds, without raising."""
    try:
        return bool(include_solvers())
    except Exception:                                 # noqa: BLE001
        return False


# Ipopt.jl ApplicationReturnStatus. Worth naming rather than comparing integers:
# testing the wrong code here is silent, and the two budget-exhaustion codes are
# exactly the ones that invalidate a ranking.
IPOPT_STATUS = {
    0: 'Solve_Succeeded', 1: 'Solved_To_Acceptable_Level',
    2: 'Infeasible_Problem_Detected', 3: 'Search_Direction_Becomes_Too_Small',
    4: 'Diverging_Iterates', 5: 'User_Requested_Stop', 6: 'Feasible_Point_Found',
    -1: 'Maximum_Iterations_Exceeded', -2: 'Restoration_Failed',
    -3: 'Error_In_Step_Computation', -4: 'Maximum_CpuTime_Exceeded',
    -5: 'Maximum_WallTime_Exceeded', -10: 'Not_Enough_Degrees_Of_Freedom',
    -11: 'Invalid_Problem_Definition', -12: 'Invalid_Option',
    -13: 'Invalid_Number_Detected', -100: 'Unrecoverable_Exception',
    -101: 'NonIpopt_Exception_Thrown', -102: 'Insufficient_Memory',
    -199: 'Internal_Error',
}
IPOPT_CONVERGED = (0, 1)          # Solve_Succeeded, Solved_To_Acceptable_Level
IPOPT_ITER_LIMIT = -1
IPOPT_CPU_LIMIT = (-4, -5)        # CPU and wall clock


def solve(u0, exact_hess=False, max_iter=500, tol=1e-8, max_cpu=60.0,
          print_level=0, keep_best=True):
    """One bounded Ipopt solve of ``-logL`` over the unit cube.

    ``u0`` is a normalised 8+2n vector. Returns a dict::

        u, obj            the point RETURNED and its objective
        u_end, obj_end    where Ipopt actually stopped
        obj_in            the objective at u0
        status,status_name, converged, iter_limited, cpu_limited, regressed
        nf, ng, nh        evaluation counts

    Requires a prior :func:`load` that passed ``param_lower``/``param_upper``: the
    [0,1]^n box is baked into the Julia problem, so solving against a
    physical-units load would optimise nonsense. Asserted, not assumed, because it
    is otherwise silent.

    **``keep_best`` is not a convenience.** Ipopt is an interior-point method and
    gives no monotone-descent guarantee; when it is truncated by ``max_iter`` or
    ``max_cpu_time`` it returns whichever iterate it stopped on, which can be far
    worse than where it started. Measured on a real dataset: a refine from
    ``-logL`` 4335 returned 115881. Since a two-stage search RANKS these endpoints,
    accepting such a point silently corrupts the ranking -- so by default the
    better of {start, endpoint} is returned and ``regressed`` records that it
    happened.

    Note ``converged`` is nearly always False for the deliberately sloppy screening
    stage; that is the design. It matters for the convergence stage, where "no
    endpoint converged" is a warning the user needs to see.
    """
    if not is_normalized():
        raise XPRUnavailable(
            'solve() needs a unit-cube load: call load(..., param_lower=, '
            'param_upper=) first. The active load is in physical units, and '
            'solving over [0,1] against it would optimise the wrong thing.')
    include_solvers()
    jl = get_jl()
    u = np.ascontiguousarray(np.clip(np.asarray(u0, dtype=float).ravel(), 0.0, 1.0))
    alloc = loaded_alloc()
    n = len(alloc.p_lo) if (alloc is not None and alloc.p_lo is not None) else len(u)
    if len(u) != n:
        raise ValueError('solve() got a length-%d vector, the load expects %d'
                         % (len(u), n))
    with _lock:
        fn = jl.xpr_ipopt_newton if exact_hess else jl.xpr_ipopt_bfgs
        r = fn(u, max_cpu=float(max_cpu), max_iter=int(max_iter),
               tol=float(tol), print_level=int(print_level))
        u_end = np.array(r.minimum, dtype=float)
        obj_end = float(r.obj)
        nf, ng, nh = int(r.nf), int(r.ng), int(r.nh)
        status = int(r.status)
        obj_in = -float(jl.XPR_sim_ref(u)[0]) if keep_best else float('nan')

    regressed = bool(keep_best and np.isfinite(obj_in)
                     and (not np.isfinite(obj_end) or obj_end > obj_in))
    u_out, obj_out = (u.copy(), obj_in) if regressed else (u_end, obj_end)
    return dict(u=u_out, obj=obj_out, u_end=u_end, obj_end=obj_end, obj_in=obj_in,
                status=status, status_name=IPOPT_STATUS.get(status, str(status)),
                converged=(status in IPOPT_CONVERGED),
                iter_limited=(status == IPOPT_ITER_LIMIT),
                cpu_limited=(status in IPOPT_CPU_LIMIT),
                regressed=regressed, nf=nf, ng=ng, nh=nh)
