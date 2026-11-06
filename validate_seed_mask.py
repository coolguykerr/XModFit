"""
validate_seed_mask.py -- A1.3: the (theta, phi) orientation seed scans for
XLayersPDB MCMC honour the frozen per-column fit mask.

The serial scan profiles each cell through Fit.residual / JointFit.residual, which
apply the frozen window mask. The parallel scan ships the model(s) to
pdb_seed_worker / pdb_joint_seed_worker, and those must apply the same mask in the
same order: evaluate on the true window x[imin:imax+1], then select fit[mask],
y[mask], e[mask] (fit-side skipped for a __masked_x__ model), before any
per-dataset max-normalisation or log.

Both scans are driven through the REAL Minimizer.emcee seed block (via
Fit.perform_fit / JointFit.perform_fit with fit_method='emcee'), restricted to a
coarse subset of grid cells by interception, and stopped before sampling:

    (a) single model, mask on:  parallel costs == serial costs   (rel 1e-9)
    (b) joint both-shared, 2 datasets, mask on: the same
    (c) mask on: parallel costs differ from a mask-off parallel run
    (d) mask=None: the costs the minimizer gets == in-process costs computed with
        the mask argument omitted (the parent commit's call)
    (e) JointFit._build_orientation_seed (seed_mode='individual') passes each
        dataset's frozen imask and its costs match an independent masked residual
    (f) a __masked_x__ model (yfit already active-only) is not masked twice, in
        both workers

    QT_QPA_PLATFORM=offscreen python validate_seed_mask.py
"""
import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scipy.optimize import least_squares

from tests.harness import check, finish

REL = 1e-9
SCALE = 'Linear'
XMIN, XMAX = 0.026, 0.550
THETAS = (30.0, 60.0, 90.0, 120.0)      # 4 x 4 = 16 cells of the standard grid


class _Stop(BaseException):
    """Raised by the interceptors once the scan is captured. A BaseException, so
    the minimizer's `except Exception` fallback cannot swallow it."""


# ------------------------------------------------------------------ setup --
def load_data():
    raw = np.loadtxt(os.path.join(ROOT, 'Data', 'i2_tim3_1000_test.txt'))
    return raw[:, 0], raw[:, 1], raw[:, 2]


def build_model(x, theta=40.88, phi=74.08):
    from importlib import import_module
    XLayersPDB = import_module("Functions.XRR.XLayersPDB").XLayersPDB
    mpar = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                      'd':   [0.0, 11.3182806, 8.75928561, 1.0],
                      'rho': [0.0, 0.23396671, 0.46419943, 0.334],
                      'mu':  [0.0, 0.0, 0.0, 0.0],
                      'sig': [3.4, 3.4, 3.4, 3.4]}}
    func = XLayersPDB(x=x, E=10.0, fname='./Data/1ubq.pdb',
                      mpar=mpar, dz=0.5, fslice=0.5, rrf=True, fix_sig=True,
                      qoff=9.830348e-05, yscale=1.0, bkg=0.0,
                      theta=theta, phi=phi, protins=6.094141, cov=0.1582754)
    func.x = x
    return func


def set_varying(params, names):
    for k in params:
        params[k].vary = k in names


def window(x):
    from Fit_Routines import _window_indices
    return _window_indices(x, XMIN, XMAX)


def block_mask(n, imin, start, width=10):
    """Full-length mask dropping a contiguous block of `width` points that starts
    `start` points into the window."""
    m = np.ones(n, bool)
    m[imin + start:imin + start + width] = False
    return m


def subset_cells():
    from lmfit.minimizer import build_orientation_grid
    _, _, _, cells = build_orientation_grid(thetares=10)
    out = []
    for th in THETAS:
        ring = [c for c in cells if c[2] == th]
        idx = np.linspace(0, len(ring) - 1, 4).round().astype(int)
        out.extend(ring[i] for i in idx)
    return out


SUB = None          # subset cells, filled in main()
SUBKEY = None       # {(ti, pi)}
SUBVAL = None       # {(round theta, round phi): (ti, pi)}


def _vkey(th, ph):
    return (round(float(th), 6), round(float(ph), 6))


# ----------------------------------------------------------- interceptors --
def capture_serial(run):
    """Run `run()` with lmfit.minimizer.least_squares intercepted: for the subset
    cells compute the real cost, skip the rest, stop at the first call after the
    grid (the walker loop)."""
    import lmfit.minimizer as MZ
    orig = MZ.least_squares
    got = {}
    calls = [0]
    from lmfit.minimizer import build_orientation_grid
    ncells = len(build_orientation_grid(thetares=10)[3])

    def ls(fun, x0, *a, **k):
        calls[0] += 1
        if calls[0] > ncells:
            raise _Stop()
        res = k.get('args', a[0] if a else None)[0]
        key = SUBVAL.get(_vkey(res.params['theta'].value, res.params['phi'].value))
        if key is None:
            return types.SimpleNamespace(cost=0.0)
        r = orig(fun, x0, *a, **k)
        got[key] = float(r.cost)
        return r

    MZ.least_squares = ls
    try:
        run()
    except _Stop:
        pass
    finally:
        MZ.least_squares = orig
    return got


def capture_parallel(module, run):
    """Run `run()` with `module.map_cell_costs` intercepted: evaluate the subset
    cells with the caller's real arguments, record them, then stop."""
    orig = module.map_cell_costs
    rec = {}

    def mcc(cells, *a, **k):
        sub = [c for c in cells if (c[0], c[1]) in SUBKEY]
        rec['args'] = a
        rec['kwargs'] = dict(k)
        rec['costs'] = {(ti, pi): float(c) for (ti, pi, c) in orig(sub, *a, **k)}
        rec['orig'] = orig
        raise _Stop()

    module.map_cell_costs = mcc
    try:
        run()
    except _Stop:
        pass
    finally:
        module.map_cell_costs = orig
    return rec


def max_rel(a, b):
    keys = sorted(set(a) | set(b))
    if set(a) != set(b) or not keys:
        return np.inf
    return max(abs(a[k] - b[k]) / max(1.0, abs(a[k])) for k in keys)


# --------------------------------------------------------------- builders --
def make_fit(x, y, e, mask):
    from Fit_Routines import Fit
    func = build_model(x)
    fit = Fit(lambda: func, x, y, e, mask=mask)
    set_varying(fit.fit_params, ('theta', 'phi', 'protins', 'cov'))
    fit.result = types.SimpleNamespace(params=fit.fit_params.copy())
    return fit


def run_fit_emcee(fit, cores):
    return lambda: fit.perform_fit(XMIN, XMAX, fit_scale=SCALE, fit_method='emcee',
                                   emcee_walkers=10, emcee_steps=1, emcee_cores=cores)


def make_joint(x, y, e, masks):
    from Joint_Fit_Routines import JointFit
    jf = JointFit()
    for i, (s, m) in enumerate(zip((1.0, 1.3), masks)):
        func = build_model(x)
        set_varying(func.params, ('theta', 'phi', 'protins', 'cov'))
        jf.add_dataset('D%d' % i, func=func, x=x, y=s * y, yerr=s * e, mask=m)
    groups = [{'members': [(0, 'theta'), (1, 'theta')], 'bare': True, 'name': None},
              {'members': [(0, 'phi'), (1, 'phi')], 'bare': True, 'name': None},
              {'members': [(0, 'protins'), (1, 'protins')], 'bare': True, 'name': None}]
    jf.build_master_params(groups=groups)
    jf.result = types.SimpleNamespace(params=jf.master_params.copy())
    return jf


def run_joint_emcee(jf, cores, mode='joint'):
    return lambda: jf.perform_fit(XMIN, XMAX, fit_scale=SCALE, fit_method='emcee',
                                  emcee_walkers=10, emcee_steps=1, emcee_cores=cores,
                                  seed_mode=mode)


# ------------------------------------------------------- (f) synthetic --------
class _Pointwise:
    """Pointwise-in-x stand-in for an XLayersPDB: y = s*(1 + a cos(theta) x + b
    sin(phi) x^2). `masked_x` twin asserts it was handed only the active points."""
    __masked_x__ = False
    expect_len = None

    def __init__(self):
        from lmfit import Parameters
        self.params = Parameters()
        self.params.add('theta', value=40.0)
        self.params.add('phi', value=70.0)
        self.params.add('s', value=1.0)
        self.x = None
        self.output_params = {}

    def y(self):
        if self.expect_len is not None:
            assert len(self.x) == self.expect_len, (len(self.x), self.expect_len)
        p = self.params
        th, ph = np.radians(p['theta'].value), np.radians(p['phi'].value)
        return p['s'].value * (1 + 0.7 * np.cos(th) * self.x + 0.4 * np.sin(ph) * self.x ** 2)


class _PointwiseMasked(_Pointwise):
    __masked_x__ = True


def check_masked_x():
    import pdb_seed_worker as PSW
    import pdb_joint_seed_worker as PJS
    from lmfit import Parameters
    x = np.linspace(0.1, 2.0, 60)
    rng = np.random.RandomState(1)
    y = 1.2 * (1 + 0.7 * np.cos(0.9) * x + 0.4 * np.sin(1.4) * x ** 2) + 0.01 * rng.randn(60)
    y[25] = 50.0                                   # hot point, masked out below
    e = 0.01 * np.ones_like(x)
    m = np.ones(60, bool); m[20:30] = False
    cells = [(0, 0, 20.0, 40.0), (0, 1, 60.0, 200.0), (1, 0, 110.0, 100.0)]  # model > 0 for Log

    plain, mx = _Pointwise(), _PointwiseMasked()
    mx.expect_len = int(m.sum())
    c_plain = PSW.map_cell_costs(cells, plain, x, y, e, 'Log', ['s'], [1.0], 1,
                                 mask=m, yfit_masked=False)
    c_mx = PSW.map_cell_costs(cells, mx, x, y, e, 'Log', ['s'], [1.0], 1,
                              mask=m, yfit_masked=True)
    d = max(abs(a[2] - b[2]) / max(1, abs(a[2])) for a, b in zip(c_plain, c_mx))
    check("(f1) single worker: __masked_x__ model costs == full-window model costs",
          d < REL and all(np.isfinite(c[2]) for c in c_mx), 'max rel %.2e' % d)

    # joint worker: two pointwise datasets sharing theta/phi, local s
    mp = Parameters()
    mp.add('theta', value=40.0); mp.add('phi', value=70.0)
    mp.add('d0_s', value=1.0); mp.add('d1_s', value=1.0)
    pmap = {'theta': [(0, 'theta'), (1, 'theta')], 'phi': [(0, 'phi'), (1, 'phi')],
            'd0_s': [(0, 's')], 'd1_s': [(1, 's')]}
    vn = ['theta', 'phi', 'd0_s', 'd1_s']

    def joint(models, ym):
        datas = [(x, y, e, m, ym), (x, 1.5 * y, 1.5 * e, m, ym)]
        scan = PJS.JointSeedScan(vn, pmap, 'Log', mp.copy(), {})
        return PJS.map_cell_costs(cells, scan, models, datas, 1)

    a = joint([_Pointwise(), _Pointwise()], False)
    m0, m1 = _PointwiseMasked(), _PointwiseMasked()
    m0.expect_len = m1.expect_len = int(m.sum())
    b = joint([m0, m1], True)
    d = max(abs(p[2] - q[2]) / max(1, abs(p[2])) for p, q in zip(a, b))
    check("(f2) joint worker: __masked_x__ models not masked twice",
          d < REL and all(np.isfinite(c[2]) for c in b), 'max rel %.2e' % d)


# ------------------------------------------------------------------ main --
def main():
    global SUB, SUBKEY, SUBVAL
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)    # noqa: F841
    import pdb_seed_worker as PSW
    import pdb_joint_seed_worker as PJS
    t00 = time.time()

    SUB = subset_cells()
    SUBKEY = {(c[0], c[1]) for c in SUB}
    SUBVAL = {_vkey(c[2], c[3]): (c[0], c[1]) for c in SUB}
    x, y, e = load_data()
    imin, imax = window(x)
    n = len(x)
    print('data %d pts, window [%d, %d] (%d pts), %d subset cells'
          % (n, imin, imax, imax - imin + 1, len(SUB)))

    # ---------------------------------------------------- (a) single model --
    mask = block_mask(n, imin, start=30)
    fit = make_fit(x, y, e, mask)
    t0 = time.time()
    ser = capture_serial(run_fit_emcee(fit, 1))
    t_ser = time.time() - t0
    t0 = time.time()
    par = capture_parallel(PSW, run_fit_emcee(fit, 2))
    print('single: serial %.1f s, parallel %.1f s' % (t_ser, time.time() - t0))
    check("(a) single model, mask on: frozen mask was the window mask",
          fit._imask_frozen is not None and np.array_equal(fit._imask_frozen, mask[imin:imax + 1]),
          'n_active %d of %d' % (fit.n_active, imax - imin + 1))
    d = max_rel(par.get('costs', {}), ser)
    check("(a) single model, mask on: parallel costs == serial costs (rel 1e-9)",
          d < REL, 'max rel %.3e over %d cells' % (d, len(ser)))

    # ---------------------------------------- (c) mask on != mask off (single) --
    fit_off = make_fit(x, y, e, None)
    par_off = capture_parallel(PSW, run_fit_emcee(fit_off, 2))
    d_off = max_rel(par.get('costs', {}), par_off.get('costs', {}))
    check("(c) single model: mask-on parallel costs differ from mask-off",
          np.isfinite(d_off) and d_off > 1e-6, 'max rel diff %.3e' % d_off)

    # ------------------------------------------ (d) mask=None == parent call --
    a = par_off['args']
    ref = {(ti, pi): float(c) for (ti, pi, c) in
           par_off['orig']([c for c in SUB], *a[:7], 1)}      # no mask argument
    km = par_off['kwargs'].get('mask', None)
    check("(d) single model, mask=None: minimizer passes no mask",
          km is None, 'mask kwarg %r' % (type(km).__name__,))
    d = max_rel(par_off['costs'], ref)
    check("(d) single model, mask=None: costs == in-process call without the mask "
          "argument", d < 1e-12, 'max rel %.3e' % d)

    # -------------------------------------------- (b) joint, both-shared --
    # dataset 1's block covers the window maximum of y, so the mask has to be
    # applied BEFORE the per-dataset max-normalisation to match the serial scan
    jmax = int(np.argmax(y[imin:imax + 1]))
    s1 = max(0, min(jmax - 4, imax - imin - 10))
    masks = [block_mask(n, imin, start=30), block_mask(n, imin, start=s1)]
    jf = make_joint(x, y, e, masks)
    t0 = time.time()
    jser = capture_serial(run_joint_emcee(jf, 1))
    t_ser = time.time() - t0
    t0 = time.time()
    jpar = capture_parallel(PJS, run_joint_emcee(jf, 2))
    print('joint: serial %.1f s, parallel %.1f s' % (t_ser, time.time() - t0))
    check("(b) joint: each objective's imask is its frozen window mask",
          all(o['imask'] is not None and np.array_equal(o['imask'], mk[o['imin']:o['imax'] + 1])
              for o, mk in zip(jf.objectives, masks)), 'block starts %d, %d' % (30, s1))
    d = max_rel(jpar.get('costs', {}), jser)
    check("(b) joint both-shared, mask on: parallel costs == serial costs (rel 1e-9)",
          d < REL, 'max rel %.3e over %d cells' % (d, len(jser)))

    # ------------------------------------------------- (c)/(d) joint --
    jf_off = make_joint(x, y, e, [None, None])
    jpar_off = capture_parallel(PJS, run_joint_emcee(jf_off, 2))
    d_off = max_rel(jpar.get('costs', {}), jpar_off.get('costs', {}))
    check("(c) joint: mask-on parallel costs differ from mask-off",
          np.isfinite(d_off) and d_off > 1e-6, 'max rel diff %.3e' % d_off)
    scan, models, datas = jpar_off['args'][:3]
    check("(d) joint, mask=None: every datas entry carries mask None",
          all(len(dd) == 3 or dd[3] is None for dd in datas), '%d datasets' % len(datas))
    datas3 = [tuple(dd[:3]) for dd in datas]
    ref = {(ti, pi): float(c) for (ti, pi, c) in
           jpar_off['orig']([c for c in SUB], scan, models, datas3, 1)}
    d = max_rel(jpar_off['costs'], ref)
    check("(d) joint, mask=None: costs == in-process call with 3-tuple datas",
          d < 1e-12, 'max rel %.3e' % d)

    # ------------------------------- (e) _build_orientation_seed (individual) --
    jf_i = make_joint(x, y, e, masks)
    rec = {'calls': []}
    orig = PSW.map_cell_costs

    def mcc(cells, func, xs, ys, es, fs, nuis, start, workers, **k):
        sub = [c for c in cells if (c[0], c[1]) in SUBKEY]
        rec['calls'].append(dict(func=func, xs=xs, ys=ys, es=es, nuis=list(nuis),
                                 start=list(start), kw=dict(k),
                                 costs={(ti, pi): float(c) for (ti, pi, c) in
                                        orig(sub, func, xs, ys, es, fs, nuis, start, 1, **k)}))
        if len(rec['calls']) == len(jf_i.objectives):
            raise _Stop()
        return [(c[0], c[1], 1.0) for c in cells]

    PSW.map_cell_costs = mcc
    try:
        run_joint_emcee(jf_i, 1, mode='individual')()
    except _Stop:
        pass
    finally:
        PSW.map_cell_costs = orig
    ok_mask, worst = len(rec['calls']) == 2, 0.0
    for i, cl in enumerate(rec['calls']):
        o = jf_i.objectives[i]
        im = cl['kw'].get('mask')
        ok_mask &= im is not None and np.array_equal(im, o['imask'])
        ok_mask &= cl['kw'].get('yfit_masked') is False
        # independent reference: evaluate on the window, then mask, (y-f)/e
        func = cl['func']
        func.x = np.asarray(cl['xs'])
        yy, ee = np.asarray(cl['ys'])[o['imask']], np.asarray(cl['es'])[o['imask']]
        for (ti, pi), cost in cl['costs'].items():
            th, ph = [c for c in SUB if (c[0], c[1]) == (ti, pi)][0][2:]

            def resid(v):
                for nm, val in zip(cl['nuis'], v):
                    setattr(func, nm, val); func.params[nm].value = val
                func.theta = th; func.params['theta'].value = th
                func.phi = ph; func.params['phi'].value = ph
                func.params.update_constraints()
                return (yy - np.asarray(func.y()).ravel()[o['imask']]) / ee
            rc = float(least_squares(resid, cl['start']).cost)
            worst = max(worst, abs(rc - cost) / max(1.0, abs(rc)))
    check("(e) _build_orientation_seed passes each dataset's frozen imask",
          ok_mask, '%d per-dataset scans' % len(rec['calls']))
    check("(e) _build_orientation_seed costs == independent masked residual (rel 1e-9)",
          worst < REL, 'max rel %.3e' % worst)

    # ------------------------------------------------------ (f) __masked_x__ --
    try:
        check_masked_x()
    except Exception as exc:
        check("(f) __masked_x__ checks ran", False, '%s: %s' % (type(exc).__name__, exc))
    print('total %.1f s' % (time.time() - t00))


if __name__ == '__main__':
    main()
    sys.exit(finish())
