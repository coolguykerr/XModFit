"""The search itself: the scan box, the start allocation, and stages 1 and 2.

Imports numpy, scipy.stats.qmc and Functions.xpr_backend. **No Qt and no lmfit at
module level**, so this runs in a ``spawn`` child and under the headless validator
on exactly the code path the GUI uses.

The algorithm, and why each number is what it is:

* **Stage 1** draws N1 scrambled Sobol points over the pose block and overwrites the
  membrane block with a screened homometric candidate, cycled across starts. Every
  coordinate the user is varying stays free in the solve. Each start is solved
  sloppily (``max_iter=30, tol=1e-3``), which is ~10x cheaper than converging it.
* **Stage 2** converges the top K by depth. **K is the safety parameter, not the
  stage-1 tolerance**: the cheap ranking correlates with the converged ordering at
  only rho ~ 0.20, so the scheme works because the true basin lands SOMEWHERE in the
  top K, not because the ranking is good. If basins are being missed, raise K before
  tightening stage 1.
* **Certification** is by the exact Hessian. Without it "converged" is not a claim:
  quasi-Newton solvers report success at saddles (scipy L-BFGS-B scored 0 of 16 MIN
  on this problem) and 19.5-26.7% of converged endpoints are saddles.

Six attempts to improve seed QUALITY were all null (pose-conditioned membranes,
Patterson-derived coverage and insertion, native 5-D Sobol, tempered cost-map
draws, cost-map basin polish). The two things that worked were having a plausible
membrane at all, and buying more starts. So: if this needs to be better, add starts.
Do not refine them.
"""
import os
import time

import numpy as np

try:
    from Functions import xpr_backend as xb
except ImportError:                       # running with Functions/ on sys.path
    import xpr_backend as xb

from . import modes as M
from . import seeds as S


class ScanCancelled(RuntimeError):
    """The caller's ``cancelled()`` returned True. Carries whatever was finished."""


# --------------------------------------------------------------- the scan box ---

def column_names(model, num_boxes):
    """lmfit parameter name for each column of XPR's 8+2n vector."""
    mkey = model.__mkeys__[0]
    names = [''] * (8 + 2 * num_boxes)
    for nm, col in model._SCALAR_COLUMNS.items():
        names[col] = nm
    for i in range(num_boxes):
        names[xb.IDX_BOXES + i] = '__%s_d_%03d' % (mkey, i + 1)
        names[xb.IDX_BOXES + num_boxes + i] = '__%s_rho_%03d' % (mkey, i + 1)
    return tuple(names)


def membrane_columns(model, num_boxes):
    """Columns of the membrane block, in ``[sig, d..., rho...]`` seed order."""
    return ([xb.IDX_SIG]
            + [xb.IDX_BOXES + i for i in range(num_boxes)]
            + [xb.IDX_BOXES + num_boxes + i for i in range(num_boxes)])


def scan_box(model, num_boxes, overrides=None):
    """``(lo, hi, free, names)``: the 8+2n search box, the free columns, the names.

    This is deliberately NOT the lmfit bound set, and the distinction matters:

    * ``theta`` and ``phi`` are **unbounded in lmfit on purpose** -- finite bounds
      act as reflecting walls at the poles and the phi wrap during sampling, so the
      model folds them internally and canonicalises after a fit. Here they get the
      fundamental domain [0,180] x [0,360], which is LOSSLESS for optimisation
      (every orientation is representable exactly once) but is a *scan artefact*.
      It must never be written back into the parameter table.
    * ``cov`` has no useful universal upper bound, so the ceiling is derived from
      the structure via ``cov_scale()`` -- 120% of full coverage -- rather than
      guessed.
    * box ``rho`` is capped at the **physicality** limit (0.62 el/A^3), not the
      model class's ``RHO_MAX`` of 5.0, which exists to allow gold.
    * a coordinate the user has pinned (``vary=False``) gets **degenerate bounds**
      rather than being removed, so the Ipopt problem keeps a constant dimension of
      8+2n. XPR rejects ``hi <= lo``, so the gap is strictly positive and scaled to
      the magnitude of the value.

    ``overrides`` is ``{lmfit_name: (lo, hi)}`` from the UI's editable box table.

    **The ``free`` tuple must be used to restrict the Hessian before classifying.**
    Pinned columns carry a ``(hi-lo) ~ 1e-9`` factor through the chain rule, so
    their Hessian rows are numerically zero and including them drags every endpoint
    to DEGENERATE.
    """
    lo, hi = model._param_box(num_boxes)      # fills protins, sig and the box d only
    names = column_names(model, num_boxes)
    P = model.params

    def put(col, a, b):
        lo[col], hi[col] = float(a), float(b)

    def from_bounds(name, col, fb_lo, fb_hi):
        v = float(P[name].value)
        a, b = float(P[name].min), float(P[name].max)
        if not np.isfinite(a):
            a = min(fb_lo, v)
        if not np.isfinite(b):
            b = max(fb_hi, v)
        put(col, min(a, v), max(b, v))

    from_bounds('qoff', xb.IDX_QOFF, -1e-3, 1e-3)
    put(xb.IDX_THETA, 0.0, 180.0)
    put(xb.IDX_PHI, 0.0, 360.0)

    cs = float(model.cov_scale())
    if not np.isfinite(cs) or cs <= 0:
        raise ValueError(
            'cov_scale() returned %r, so the coverage ceiling cannot be derived '
            "from the structure. This happens under cov_norm='xpr'; set an explicit "
            'upper bound on cov, or switch cov_norm to "protvol".' % cs)
    put(xb.IDX_COV, 0.0, max(1.2 * cs, 1.2 * float(P['cov'].value)))

    for i in range(num_boxes):
        nm = names[xb.IDX_BOXES + num_boxes + i]
        v = float(P[nm].value)
        put(xb.IDX_BOXES + num_boxes + i,
            min(float(model.__rho__[model.__mkeys__[0]][0]), v),
            max(S.RHO_PHYS_MAX, v))

    for nm, (a, b) in (overrides or {}).items():
        if nm in names:
            put(names.index(nm), a, b)

    # pin whatever the user is not varying, INCLUDING yscale/bkg by default
    free = []
    for col, nm in enumerate(names):
        if not nm or nm not in P:
            v = 0.0
            put(col, v, v + 1e-9)
            continue
        if bool(P[nm].vary):
            free.append(col)
        else:
            v = float(P[nm].value)
            put(col, v, v + max(1e-9, 1e-12 * abs(v)))

    bad = [names[c] or c for c in range(len(lo)) if not (hi[c] > lo[c])]
    if bad:
        raise ValueError('the search box is empty for %s (XPR requires hi > lo)'
                         % ', '.join(str(b) for b in bad))
    return lo, hi, tuple(free), names


def check_expr(model):
    """Refuse a scan when any varying parameter is driven by an ``expr``.

    ``parameter_columns`` has no chain rule for a constrained parameter, so the
    analytic derivatives would silently be wrong. ``Fit.can_jac`` refuses the same
    case for the same reason.
    """
    bad = [n for n, p in model.params.items()
           if getattr(p, 'expr', None) and getattr(p, 'vary', False)]
    return bad


# ------------------------------------------------------------------- the load ---

def load_for_scan(model, q, y, err, box=None):
    """Put the backend on the unit cube against the user's actual data.

    Returns ``(lo, hi, free, names)``.

    The sizing arguments are computed with **exactly** the call
    ``XLayersPDB_XPR._prepare`` makes, and that is not incidental: since XPR's
    fixed-slab-count revision ``cells_per_box = ceil(max_box_length/spacez)``, so
    ``spacez`` and ``box_ref_length`` set the in-box mesh and therefore CHANGE THE
    ANSWER. Reproducing the call is what makes the scan's chi2 the same number the
    parameter table shows.
    """
    model.update_parameters()
    mkey = model.__mkeys__[0]
    d, rho = model.__d__[mkey], model.__rho__[mkey]
    num_boxes = len(d) - 2
    if num_boxes < 1:
        raise ValueError('the layer table needs at least 3 rows (top medium, one '
                         'box, bulk); it has %d' % len(d))

    include_protein = (bool(model.use_protein) and bool(model.fname)
                       and os.path.isfile(model.fname))
    # XModFit eval()s Fixed-Parameter cells, so a choices value that happens to be a
    # Python name arrives as that object rather than as the string.
    vol_calc = getattr(model.vol_calc, '__name__', None) or str(model.vol_calc)

    lo, hi, free, names = box if box is not None else scan_box(model, num_boxes)
    sizing = xb.alloc_bounds_for(*model._param_box(num_boxes),
                                 num_boxes=num_boxes, spacez=float(model.spacez),
                                 box_ref_length=float(model.box_ref_length))
    xb.load(np.asarray(q, dtype=float), num_boxes,
            pdbfile=model.fname if include_protein else None,
            vol_calc=vol_calc, rho_top=float(rho[0]), rho_bottom=float(rho[-1]),
            E=float(model.E), rrf=bool(model.rrf),
            electron_scale=(model.electron_scale(vol_calc) if include_protein
                            else 1.0),
            data_R=np.asarray(y, dtype=float), data_err=np.asarray(err, dtype=float),
            param_lower=lo, param_upper=hi, **sizing)
    return lo, hi, free, names


def active_arrays(fit, xmin=None, xmax=None):
    """``(q, y, err)`` exactly as ``Fit.perform_fit`` would see them.

    Reuses ``Fit_Routines._window_indices`` -- the same function ``perform_fit``
    calls -- rather than re-deriving the window, so the scan and a subsequent fit
    cannot disagree about which points are in.

    Deliberately does NOT call ``_freeze_imask()``: freezing the mask belongs to
    ``perform_fit`` (one freeze per fit, or lmfit's leastsq gets a varying residual
    length), and the scan must not mutate the ``Fit``.

    Imported locally: ``Fit_Routines`` imports PyQt5 at module level and this module
    must import in a process with no Qt.
    """
    from Fit_Routines import _window_indices

    x = np.asarray(fit.x, dtype=float)
    if isinstance(fit.x, dict):
        raise ValueError('the fast scan needs a single-column dataset: the '
                         'multi-column residual applies a per-curve '
                         'max-normalisation, which breaks the identity between the '
                         "scan's objective and the fit's chi2")
    imin, imax = _window_indices(x, xmin, xmax)
    idx = np.arange(imin, imax + 1)
    mask = getattr(fit, 'mask', None)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)[imin:imax + 1]
        idx = idx[m[:len(idx)]]
    return x[idx], np.asarray(fit.y, dtype=float)[idx], \
        np.asarray(fit.yerr, dtype=float)[idx]


# ----------------------------------------------------------------- the starts ---

def sobol_starts(n1, free, memb_cols, seeds_u, u0, seed=2024, band_stratified=True):
    """The stage-1 start matrix, shape ``(n1, len(u0))`` in the unit cube.

    Sobol over the free coordinates, then the membrane block is overwritten by a
    screened candidate cycled across starts. That ordering is the point: the
    measured gain from seeding is DIMENSION REDUCTION, so the membrane values
    replace the quasi-random ones rather than perturbing them.

    ``band_stratified`` forces half the starts into each hemisphere of ``cos
    theta``. The admissible set is typically dominated by the OPPOSITE band and the
    bands are antipodal, so an allocation that guarantees coverage of both targets
    exactly the measured failure. (The related cost-map machinery failed as a
    *ranker* -- a tempered draw put 91% of starts in the wrong band -- but was never
    tested as a coverage guarantee, which is a different and easier job.)
    """
    from scipy.stats import qmc

    free = list(free)
    starts = np.tile(np.asarray(u0, dtype=float), (n1, 1))
    if free:
        sob = qmc.Sobol(d=len(free), scramble=True, seed=seed).random(n1)
        starts[:, free] = sob
        if band_stratified and xb.IDX_THETA in free:
            j = free.index(xb.IDX_THETA)
            # u[theta] < 0.5  <=>  theta < 90  <=>  cos(theta) > 0  <=> band 'up'
            starts[:, xb.IDX_THETA] = 0.5 * (sob[:, j] + (np.arange(n1) % 2))
    if seeds_u:
        cols = list(memb_cols)
        for i in range(n1):
            cand = np.asarray(seeds_u[i % len(seeds_u)], dtype=float).ravel()
            if len(cand) == len(cols):
                starts[i, cols] = cand
    return np.clip(starts, 0.0, 1.0)


# -------------------------------------------------------------------- the scan ---

class ScanResult(object):
    """Raw output of :func:`scan`, before it becomes a ``ModeSet``.

    Mirrors the field names of the study's ``endpoints_*.npz`` archives so those
    archives can be replayed through the same ranking code -- they are the
    regression target for this feature.
    """

    def __init__(self):
        self.stage1_us = None
        self.stage1_fs = None
        self.stage1_starts = None
        self.order = None
        self.stage2_us = None
        self.stage2_fs = None
        self.stage2_cls = None
        self.endpoints = []          # list of dict, one per stage-2 endpoint
        self.timings = {}
        self.stage2_meta = []
        self.warnings = []
        self.cancelled = False
        self.n_cpu_limited = 0
        self.n_converged = 0
        self.n_regressed = 0
        self.n1_ok = 0
        self.box = None


def _noop_progress(*a, **k):
    return None


def _never_cancelled():
    return False


def scan(model, q, y, err, n1=64, k=6, seed=2024, membrane_seeds=None,
         box=None, iter1=30, tol1=1e-3, max_cpu1=15.0,
         iter2=300, tol2=1e-8, max_cpu2=60.0,
         band_stratified=True, certify=True, gauss_newton=True,
         progress=None, cancelled=None):
    """Run the two-stage search. Returns a :class:`ScanResult`.

    ``progress(stage, i, n, best)`` and ``cancelled() -> bool`` default to no-ops,
    so the validator drives exactly the code path the GUI drives.

    On cancellation this returns PARTIAL results rather than raising: both loops
    have deterministic counts, which is what makes a partial result coherent.
    """
    progress = progress or _noop_progress
    cancelled = cancelled or _never_cancelled
    res = ScanResult()
    t00 = time.time()

    bad = check_expr(model)
    if bad:
        raise ValueError('the fast scan cannot handle constrained parameters: %s '
                         'have an expr. Clear the expr or pin them.' % ', '.join(bad))

    mkey = model.__mkeys__[0]
    num_boxes = len(model.__mpar__[mkey]['d']) - 2

    t = time.time()
    lo, hi, free, names = load_for_scan(model, q, y, err, box=box)
    res.box = (lo, hi, free, names)
    res.timings['load'] = time.time() - t

    memb_cols = membrane_columns(model, num_boxes)
    seeds_u = []
    if membrane_seeds:
        seeds_u = S.to_unit_cube(membrane_seeds, lo[memb_cols], hi[memb_cols])

    u_here = xb.to_normalized(
        model._full_params(model.__d__[mkey], model.__rho__[mkey], num_boxes))
    u0 = np.clip(u_here, 0.0, 1.0)

    starts = sobol_starts(n1, free, memb_cols, seeds_u, u0, seed=seed,
                          band_stratified=band_stratified)
    res.stage1_starts = starts

    # One throwaway exact-Hessian call up front. It is ~32 s of JIT the first time
    # in a process, and paying it here -- behind its own progress label -- keeps it
    # out of the middle of the run where it looks like a hang.
    if certify:
        t = time.time()
        progress('prepare', 0, 1, None)
        try:
            xb.hess(u0)
        except Exception as ex:                        # noqa: BLE001
            res.warnings.append('exact Hessian unavailable (%s); endpoints will not '
                                'be certified' % str(ex)[:120])
            certify = False
        res.timings['hess_jit'] = time.time() - t

    # ------------------------------------------------------------- stage 1
    t = time.time()
    u1 = np.full((n1, len(u0)), np.nan)
    f1 = np.full(n1, np.inf)
    for i in range(n1):
        if cancelled():
            res.cancelled = True
            break
        try:
            r = xb.solve(starts[i], max_iter=iter1, tol=tol1, max_cpu=max_cpu1)
            u1[i] = r['u']
            f1[i] = r['obj']
            res.n_cpu_limited += int(bool(r['cpu_limited']))
        except Exception as ex:                        # noqa: BLE001
            res.warnings.append('stage-1 start %d failed: %s' % (i, str(ex)[:100]))
        progress('stage1', i + 1, n1, float(np.nanmin(f1)) if np.isfinite(f1).any()
                 else None)
    res.stage1_us, res.stage1_fs = u1, f1
    res.timings['stage1'] = time.time() - t
    n_ok = int(np.isfinite(f1).sum())

    # ------------------------------------------------------------- stage 2
    order = np.argsort(np.where(np.isfinite(f1), f1, np.inf))[:max(k, 0)]
    order = np.array([i for i in order if np.isfinite(f1[i])], dtype=int)
    res.order = order
    t = time.time()
    u2, f2, meta2 = [], [], []
    for j, i in enumerate(order):
        if cancelled():
            res.cancelled = True
            break
        try:
            r = xb.solve(u1[i], max_iter=iter2, tol=tol2, max_cpu=max_cpu2)
            u2.append(r['u'])
            f2.append(r['obj'])
            meta2.append(r)
            res.n_cpu_limited += int(bool(r['cpu_limited']))
            res.n_converged += int(bool(r['converged']))
            if r['regressed']:
                # xb.solve already handed back the better point, so the RANKING is
                # safe; but this endpoint is the stage-1 point, not a stationary
                # point, so it must not be presented as converged.
                res.n_regressed += 1
                res.warnings.append(
                    'endpoint %d: refining made it worse (-logL %.3f -> %.3f, %s), '
                    'so the screening point was kept. It is not a certified '
                    'stationary point.'
                    % (j, r['obj_in'], r['obj_end'], r['status_name']))
        except Exception as ex:                        # noqa: BLE001
            res.warnings.append('stage-2 solve %d failed: %s' % (j, str(ex)[:100]))
            u2.append(np.full(len(u0), np.nan))
            f2.append(np.inf)
            meta2.append({})
        progress('stage2', j + 1, len(order),
                 float(np.nanmin(f2)) if f2 else None)
    res.stage2_us = np.array(u2) if u2 else np.zeros((0, len(u0)))
    res.stage2_fs = np.array(f2) if f2 else np.zeros(0)
    res.stage2_meta = meta2
    res.timings['stage2'] = time.time() - t

    # --------------------------------------------------- certify and package
    t = time.time()
    cls = []
    for idx_e, (u, f) in enumerate(zip(res.stage2_us, res.stage2_fs)):
        meta = res.stage2_meta[idx_e] if idx_e < len(res.stage2_meta) else {}
        ep = dict(u=u, obj=float(f), stage1_rank=int(idx_e),
                  start_index=int(order[idx_e]) if idx_e < len(order) else -1,
                  hess=None, gn=None, cls=M.UNCERTIFIED, eigmin=None, cond=None,
                  null=None, converged=bool(meta.get('converged', False)),
                  regressed=bool(meta.get('regressed', False)),
                  status_name=meta.get('status_name', ''),
                  cpu_limited=bool(meta.get('cpu_limited', False)))
        # A regressed endpoint is the screening point, not a stationary point, so
        # certifying it would attach a curvature claim to something that has no
        # right to one.
        if np.isfinite(f) and certify and not ep['regressed'] and not cancelled():
            try:
                H = np.asarray(xb.hess(u), dtype=float)
                Hf = -H[np.ix_(free, free)]      # curvature of -logL over free cols
                lab, emin, cond, nullv = M.classify(Hf, np.asarray(u)[list(free)])
                ep.update(hess=Hf, cls=lab, eigmin=emin, cond=cond, null=nullv)
            except Exception as ex:              # noqa: BLE001
                res.warnings.append('certification failed for endpoint %d: %s'
                                    % (idx_e, str(ex)[:100]))
            if gauss_newton:
                try:
                    J = np.asarray(xb.jac(u, active=list(free)), dtype=float)
                    W = 1.0 / np.asarray(err, dtype=float)
                    Jw = J * W[:, None]
                    ep['gn'] = Jw.T.dot(Jw)
                except Exception as ex:          # noqa: BLE001
                    res.warnings.append('Gauss-Newton failed for endpoint %d: %s'
                                        % (idx_e, str(ex)[:100]))
        cls.append(ep['cls'])
        res.endpoints.append(ep)
        progress('certify', idx_e + 1, len(res.stage2_us), None)
    res.stage2_cls = np.array(cls)
    res.timings['certify'] = time.time() - t
    res.timings['total'] = time.time() - t00
    res.n1_ok = n_ok

    if res.n_cpu_limited:
        res.warnings.append(
            '%d solve(s) stopped on the CPU-time limit. Such an endpoint sits at an '
            'arbitrary point on its descent path, so the RANKING is suspect -- raise '
            'the limit or reduce N1.' % res.n_cpu_limited)
    if len(res.stage2_fs) and not res.n_converged:
        res.warnings.append(
            'NO endpoint converged: all %d ran out of iterations or CPU time. The '
            'ranking may still be useful, but nothing here is a certified minimum '
            '-- raise the stage-2 iteration limit or the CPU limit, and treat the '
            'classifications as provisional.' % len(res.stage2_fs))
    return res


def to_modes(res, model, prov, err=None):
    """Turn a :class:`ScanResult` into deduplicated :class:`~fastscan.modes.Mode`
    records, ranked and weighted.
    """
    lo, hi, free, names = res.box
    mkey = model.__mkeys__[0]
    num_boxes = len(model.__mpar__[mkey]['d']) - 2
    out = []
    # De-normalise from the box carried in the result, NOT via xb.to_physical:
    # that reads process-global backend state, so it would give wrong answers in
    # the parent process when the scan itself ran in a spawn child.
    lo = np.asarray(lo, dtype=float)
    span = np.asarray(hi, dtype=float) - lo
    for ep in res.endpoints:
        if not np.isfinite(ep['obj']):
            continue
        p = lo + span * np.asarray(ep['u'], dtype=float)
        params = {names[c]: float(p[c]) for c in range(len(names)) if names[c]}
        railed = tuple(names[c] for c in free
                       if ep['u'][c] <= 1e-6 or ep['u'][c] >= 1 - 1e-6)
        m = M.Mode(u=np.asarray(ep['u'], dtype=float), neglogL=float(ep['obj']),
                   params=params,
                   theta=float(p[xb.IDX_THETA]), phi=float(p[xb.IDX_PHI]),
                   protins=float(p[xb.IDX_PROTINS]), cov=float(p[xb.IDX_COV]),
                   chi2=2.0 * float(ep['obj']), n_active=prov.n_active,
                   hess_class=ep['cls'],
                   hess_source=('exact' if ep['hess'] is not None else 'none'),
                   hessian=ep['hess'], gn=ep['gn'], eigmin=ep['eigmin'],
                   cond=ep['cond'], null_vector=ep['null'],
                   active_bounds=railed, stage1_rank=ep['stage1_rank'],
                   start_index=ep['start_index'],
                   template_label=prov.template_label)
        flags = []
        if m.hess_class == M.SADDLE:
            flags.append('saddle')
        if not ep.get('converged', False):
            flags.append('not_converged')
        if ep.get('regressed', False):
            flags.append('refine_regressed')
        if ep.get('cpu_limited', False):
            flags.append('cpu_limited')
        if railed:
            flags.append('railed:' + ','.join(railed))
        for i in range(num_boxes):
            rv = params.get(names[xb.IDX_BOXES + num_boxes + i])
            if rv is not None and not (prov.rho_top <= rv <= S.RHO_PHYS_MAX):
                flags.append('unphysical_rho%d' % (i + 1))
        if res.cancelled:
            flags.append('unconverged')
        m.flags = tuple(flags)
        out.append(m)
    kept = M.dedup(out)
    M.laplace_weights(kept)
    return kept
