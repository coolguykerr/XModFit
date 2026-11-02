"""``spawn``-child entry point for the fast global search.

Why a child process at all, rather than a QThread or the Qt thread:

* **The XPR backend is process-wide single-slot** (``_jl``, ``_loaded``, ``_alloc``
  module globals; ``XPR.GLOBAL_SD`` a single Julia ``Ref``). During a scan on the Qt
  thread, ``QApplication.processEvents()`` -- which this codebase calls in 20-odd
  places -- can dispatch any queued slot that calls ``func.y()``, and that calls
  ``xb.load`` **with no** ``data_R``. The objective would silently become
  ``0.5*sum((R/1)^2)`` for the remainder of the run, with no error, because the
  fingerprint check is on the load side and not the eval side. A child process
  cannot be raced by the GUI.
* A single stage-2 solve is tens of seconds inside one blocking Julia call, and the
  first exact Hessian in a process costs ~35-47 s of JIT. There is no Python frame
  to run a progress callback from, so on the Qt thread the window would simply stop
  repainting.
* ``terminate()`` gives real cancellation. Ipopt is called without an intermediate
  callback, so there is nothing to poll inside a solve.
* It keeps the Julia session's Hessian-time RSS (~1.2 GB) out of the GUI process,
  and it makes a later multi-template run a fan-out rather than a redesign.

Protocol. The parent sends ``(model, job)`` and drains dicts:

    {'t': 'progress', 'stage': str, 'i': int, 'n': int, 'best': float|None}
    {'t': 'log',      'msg': str}
    {'t': 'result',   'res': ScanResult, 'modes': [Mode, ...]}
    {'t': 'error',    'msg': str}          <- traceback, never a silent exit

``'result'`` or ``'error'`` is always the last message. The model instance is shipped
by pickle, which is the same thing ``pdb_seed_worker`` does and is why the model must
stay picklable; a ``Fit`` is a ``QObject`` and would not pickle, so it is never sent.

This module must not import Qt, ``xmodfit`` or ``Fit_Routines``.
"""
import os
import sys
import traceback


def _bootstrap(repo=None):
    """Make a bare spawn child able to import the repo, headlessly."""
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    try:
        os.chdir(repo)
    except OSError:
        pass
    for p in (repo, os.path.join(repo, 'Functions')):
        if p not in sys.path:
            sys.path.insert(0, p)
    # the bundled xraydb/ directory carries no source, and importing the model
    # package reaches it. A stub is enough: nothing on the scan path uses it.
    if 'xraydb' not in sys.modules:
        try:
            import xraydb  # noqa: F401
        except Exception:
            import types

            import numpy as _np
            m = types.ModuleType('xraydb')
            m.f0 = lambda *a, **k: _np.array([1.0])
            m.atomic_number = lambda s: 1
            m.mirror_reflectivity = lambda *a, **k: _np.array([1.0])
            m.xray_delta_beta = lambda *a, **k: (0.0, 0.0, 0.0)
            sys.modules['xraydb'] = m


def run(conn, model, job):
    """Run one scan and stream results back over ``conn``. Never raises."""
    try:
        _run(conn, model, job)
    except BaseException:                      # noqa: BLE001 -- must not exit mute
        try:
            conn.send({'t': 'error', 'msg': traceback.format_exc()[-4000:]})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run(conn, model, job):
    _bootstrap(job.get('repo'))

    import numpy as np

    from fastscan import core as C
    from fastscan import modes as M
    from fastscan import seeds as S

    q = np.asarray(job['q'], dtype=float)
    y = np.asarray(job['y'], dtype=float)
    err = np.asarray(job['err'], dtype=float)

    def progress(stage, i, n, best):
        conn.send({'t': 'progress', 'stage': stage, 'i': int(i), 'n': int(n),
                   'best': (None if best is None else float(best))})

    membrane = [np.asarray(s, dtype=float) for s in (job.get('membrane_seeds') or [])]

    res = C.scan(model, q, y, err,
                 n1=int(job.get('n1', 64)), k=int(job.get('k', 6)),
                 seed=int(job.get('seed', 2024)),
                 membrane_seeds=membrane or None,
                 iter1=int(job.get('iter1', 30)), tol1=float(job.get('tol1', 1e-3)),
                 max_cpu1=float(job.get('max_cpu1', 15.0)),
                 iter2=int(job.get('iter2', 300)), tol2=float(job.get('tol2', 1e-8)),
                 max_cpu2=float(job.get('max_cpu2', 60.0)),
                 band_stratified=bool(job.get('band_stratified', True)),
                 certify=bool(job.get('certify', True)),
                 gauss_newton=bool(job.get('gauss_newton', True)),
                 progress=progress)

    lo, hi, free, names = res.box
    prov = M.Provenance(
        algorithm=('twostage-bandstrat-v1' if job.get('band_stratified', True)
                   else 'twostage-v1'),
        n1=int(job.get('n1', 64)), n1_ok=int(res.n1_ok), k=int(job.get('k', 6)),
        sobol_seed=int(job.get('seed', 2024)),
        band_stratified=bool(job.get('band_stratified', True)),
        iter1=int(job.get('iter1', 30)), tol1=float(job.get('tol1', 1e-3)),
        iter2=int(job.get('iter2', 300)), tol2=float(job.get('tol2', 1e-8)),
        n_cpu_limited=int(res.n_cpu_limited),
        seed_source=job.get('seed_source', 'current-params'),
        n_seeds=len(membrane) or 1,
        seeds_dropped=int(job.get('seeds_dropped', 0)),
        template_path=str(getattr(model, 'fname', '')),
        template_label=job.get('template_label') or
        os.path.splitext(os.path.basename(str(getattr(model, 'fname', ''))))[0],
        num_boxes=len(model.__mpar__[model.__mkeys__[0]]['d']) - 2,
        spacez=float(model.spacez), box_ref_length=float(model.box_ref_length),
        vol_calc=str(getattr(model.vol_calc, '__name__', None) or model.vol_calc),
        rho_top=float(job.get('rho_top', 0.0)),
        rho_bottom=float(job.get('rho_bottom', 0.334)),
        electron_scale=float(job.get('electron_scale', 1.0)),
        objective='xpr_logL', fit_scale=job.get('fit_scale', 'Linear'),
        dataset=job.get('dataset', ''),
        x_window=tuple(job.get('x_window', (float('nan'), float('nan')))),
        n_active=len(q), data_sha256=M.data_fingerprint(q, y, err),
        varying_names=tuple(names[c] for c in free), free=tuple(free),
        param_lower=np.asarray(lo), param_upper=np.asarray(hi),
        hessian=('exact' if job.get('certify', True) else 'none'),
        timings=dict(res.timings),
        scipy_version=_scipy_version(),
    )
    mode_list = C.to_modes(res, model, prov, err=err)

    # the curve for each surviving mode, for the overlay and for the .xmodes cache.
    # Cheap (one sim each) and it makes a saved set plottable with no Julia at all.
    try:
        from Functions import xpr_backend as xb
        for m in mode_list:
            m.R = np.asarray(xb.sim(m.u)[1], dtype=float)
    except Exception as ex:                    # noqa: BLE001
        res.warnings.append('could not cache mode curves: %s' % str(ex)[:100])

    conn.send({'t': 'result', 'res': res, 'modes': mode_list, 'prov': prov})


def _scipy_version():
    try:
        import scipy
        return scipy.__version__
    except Exception:
        return ''


def build_job(fit, model, q, y, err, **kw):
    """Assemble the picklable job dict from live GUI state.

    Runs in the PARENT. Kept here, next to the consumer, so the two cannot drift.
    """
    job = dict(
        repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        q=list(map(float, q)), y=list(map(float, y)), err=list(map(float, err)),
        rho_top=float(model.__rho__[model.__mkeys__[0]][0]),
        rho_bottom=float(model.__rho__[model.__mkeys__[0]][-1]),
    )
    job.update(kw)
    return job
