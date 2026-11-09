"""
validate_joint_ws1.py -- WS1 (joint-path correctness) regression checks.

One function per task section; later tasks append T2..T5.

    T1  one PDB model-name list (A1.1): Fit_Routines and xmodfit hold exactly
        lmfit.minimizer.PDB_FUNCNAMES
    T2  .npz sidecars carry the shared-beam footprint term (B2a)
    T3  share-group mismatches (01 6.8): differing start values warn once per
        group; differing min/max/vary raise ShareMismatchError unless
        on_mismatch='first'; JointFit.warnings does not grow across rebuilds
        or re-fits and is printed in the report

    T4  warn on per-max weighting (01 6.7) and dropped sub-curves (6.14)
    T5  joint-vs-individual chi-square on the joint fit's own active points (A1.4)
    N2  the report's derived-quantity keymap follows share-groups (audit N2)

    QT_QPA_PLATFORM=offscreen python validate_joint_ws1.py
"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Functions'))

# the bundled xraydb/ directory carries no source; stub it before importing xmodfit
try:
    from xraydb import XrayDB as _R  # noqa
except Exception:
    _xr = types.ModuleType("xraydb")
    class _XrayDB:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB; _xr.f0 = lambda *a, **k: 0; _xr.__path__ = []
    sys.modules["xraydb"] = _xr

from PyQt5.QtWidgets import QApplication

from tests.harness import check, finish


def test_t1():
    import lmfit.minimizer as mn
    import Fit_Routines
    import xmodfit
    ref = getattr(mn, 'PDB_FUNCNAMES', None)
    check("T1 lmfit.minimizer.PDB_FUNCNAMES exists", ref is not None)
    check("T1 Fit_Routines.PDB_FUNCNAMES == minimizer set",
          ref is not None and getattr(Fit_Routines, 'PDB_FUNCNAMES', None) == ref)
    check("T1 xmodfit _PDB_MCMC_FUNCS == minimizer set",
          ref is not None and xmodfit.XModFit._PDB_MCMC_FUNCS == ref)
    check("T1 XLayersPDB_XPR in xmodfit _PDB_MCMC_FUNCS",
          'XLayersPDB_XPR' in xmodfit.XModFit._PDB_MCMC_FUNCS)
    check("T1 XLayersPDB_box in Fit_Routines set",
          'XLayersPDB_box' in getattr(Fit_Routines, 'PDB_FUNCNAMES', ()))
    check("T1 legacy alias _PDB_FUNCNAMES is the same object",
          getattr(mn, '_PDB_FUNCNAMES', None) is ref)


class _Line:
    def __init__(self, x=0, a=1.0, b=0.0, mpar={}):
        from lmfit import Parameters
        self.x = x
        self.a, self.b = a, b
        self.__mpar__ = mpar
        self.choices = {}
        self.filepaths = {}
        self.__fit__ = False
        self.output_params = {'scaler_parameters': {}}
        self.params = Parameters()
        self.params.add('a', value=a, vary=0, min=-1e9, max=1e9)
        self.params.add('b', value=b, vary=0, min=-1e9, max=1e9)

    def y(self):
        import numpy as np
        return self.a * np.asarray(self.x, float) + self.b


def test_t2():
    import numpy as np
    import joint_covariance as jcov
    from Joint_Fit_Routines import JointFit
    n = 40
    x = np.linspace(0.1, 2.0, n)
    y = 3.0 * x + 0.5
    e = 0.05 * np.ones(n)
    base = dict(x=x, y_corr=y, sigma_meas=e,
                k_index=(np.arange(n) >= n // 2).astype(int),
                SF_LIST=np.array([1.0, 10.0]),
                Sigma_logc=np.array([[0.01, 0.002], [0.002, 0.02]]))
    v = 0.3 * np.exp(-x / 0.5)
    d = os.path.join(ROOT, 'tests', '_logs')
    os.makedirs(d, exist_ok=True)
    pa, pb, pc = (os.path.join(d, 'ws1_t2_%s.npz' % s) for s in 'abc')
    np.savez(pa, **base)
    np.savez(pb, v=v, var_fwhm=4e-4, **base)
    np.savez(pc, v=v, **base)

    def build(path):
        cov = jcov.load_dataset_cov(path, x=x)
        jf = JointFit()
        jf.add_dataset('A', funcClass=_Line, x=x, y=y, yerr=e, cov=dict(cov))
        jf.add_dataset('B', funcClass=_Line, x=x, y=y, yerr=e, cov=dict(cov))
        jf.build_master_params()
        jf._set_ranges(x[0], x[-1])
        jf.residual(jf.master_params, 'Linear')
        return cov, jf

    cov, jf = build(pa)
    check("T2a no footprint keys -> no v / var_fwhm in cov",
          'v' not in cov and 'var_fwhm' not in cov)
    check("T2a _use_footprint is False", jf._use_footprint is False)

    cov, jf = build(pb)
    check("T2b v and var_fwhm passed through",
          'v' in cov and len(cov['v']) == n and cov['var_fwhm'] == 4e-4)
    check("T2b _use_footprint is True", jf._use_footprint is True)
    cat = lambda a: np.concatenate([a, a])
    C = jcov.build_full_covariance(cat(y), cat(e), cat(base['k_index']),
                                   base['Sigma_logc'], v=cat(v), var_fwhm=4e-4)
    LL = jf._L @ jf._L.T
    rel = np.max(np.abs(LL - C) / np.maximum(np.abs(C), 1e-300))
    check("T2b L L^T == build_full_covariance(v, var_fwhm) to 1e-12",
          LL.shape == C.shape and rel < 1e-12, rel)

    try:
        jcov.load_dataset_cov(pc, x=x)
        check("T2c only v -> ValueError", False, 'no exception')
    except ValueError as ex:
        check("T2c only v -> ValueError naming the file",
              pc in str(ex) and "'v' and 'var_fwhm'" in str(ex), str(ex))


def test_t3():
    import numpy as np
    from validate_joint_gls import Line, X, make_data, E
    from Joint_Fit_Routines import JointFit, ShareMismatchError
    ys = make_data(3)
    err = E * np.ones_like(X)
    names = ('DS_one', 'DS_two', 'DS_three')

    def build(starts=(3.0, 3.0, 3.0), max2=np.inf, vary2=True, **kw):
        jf = JointFit()
        for i, (nm, y) in enumerate(zip(names, ys)):
            jf.add_dataset(nm, funcClass=Line, x=X, y=y, yerr=err)
            pa = jf.objectives[i]['func'].params['a']
            pa.set(value=starts[i], vary=(vary2 if i == 1 else True),
                   max=(max2 if i == 1 else np.inf))
            jf.objectives[i]['func'].params['b'].set(vary=True)
        grp = jf.share_within('a', list(names))
        jf.build_master_params(groups=[grp], **kw)
        return jf, grp

    # (a) same bounds, different values -> builds, exactly one warning
    jf, grp = build(starts=(2.5, 3.0, 3.5))
    w = list(jf.warnings)
    check("T3a value mismatch builds with exactly 1 warning", len(w) == 1, w)
    check("T3a warning names all three datasets",
          len(w) == 1 and all(nm in w[0] for nm in names), w)
    check("T3a warning text follows the brief",
          len(w) == 1 and w[0].startswith('shared ')
          and 'endpoints start at different values' in w[0]
          and w[0].endswith('using DS_one=2.5'), w)
    mk = list(jf.param_map)[0]
    check("T3a master takes the first endpoint's value",
          jf.master_params[mk].value == 2.5, jf.master_params[mk].value)

    # (e) a second build does not grow the list
    jf.build_master_params(groups=[grp])
    check("T3e rebuilding does not grow warnings", len(jf.warnings) == 1,
          jf.warnings)

    # (f) after perform_fit the report carries the warnings block; re-fit doesn't grow
    rep, msg = jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T3f report contains 'Joint-fit warnings:'",
          'Joint-fit warnings:' in rep and names[2] in rep, rep[-400:])
    rep2, _ = jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T3f re-fit keeps the build warning and does not grow",
          len(jf.warnings) == 1 and rep2.count('Joint-fit warnings:') == 1,
          jf.warnings)
    # run-time entries (perform_fit's own, via _add_run_warning) are replaced
    jf._add_run_warning('stale run-time note')
    jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T3f perform_fit drops its previous run-time warnings only",
          not any('stale' in w for w in jf.warnings)
          and len(jf.warnings) == 1, jf.warnings)
    jf.clear()
    check("T3 clear() resets warnings", jf.warnings == [], jf.warnings)

    # (b) dataset 2 has max=10, the others inf -> ShareMismatchError
    try:
        build(max2=10.0)
        check("T3b bound mismatch raises ShareMismatchError", False, 'no exception')
    except ShareMismatchError as ex:
        s = str(ex)
        check("T3b ShareMismatchError names 'a' and the dataset",
              "'a'" in s and 'DS_two' in s and isinstance(ex, ValueError), s)

    # a refused rebuild leaves the object as it was (check before mutate)
    jf, grp = build(starts=(2.5, 3.0, 3.5))
    mp0, pm0, w0 = jf.master_params, jf.param_map, list(jf.warnings)
    jf.objectives[1]['func'].params['a'].set(max=10.0)
    try:
        jf.build_master_params(groups=[grp])
        check("T3b refused rebuild raises", False, 'no exception')
    except ShareMismatchError:
        check("T3b refused rebuild leaves master_params/param_map/warnings intact",
              jf.master_params is mp0 and jf.param_map is pm0 and jf.warnings == w0,
              jf.warnings)

    # (c) on_mismatch='first' -> builds, master max == inf, one warning
    try:
        jf, _ = build(max2=10.0, on_mismatch='first')
        mk = list(jf.param_map)[0]
        check("T3c on_mismatch='first' builds with master max == inf",
              jf.master_params[mk].max == np.inf, jf.master_params[mk].max)
        check("T3c on_mismatch='first' leaves exactly 1 warning",
              len(jf.warnings) == 1, jf.warnings)
    except Exception as ex:
        check("T3c on_mismatch='first' builds", False, repr(ex))

    # (c2) final review I1: under 'first' a narrower later endpoint must not clip.
    # master 50, DS_two bounded at max=30 -> every local value must equal 50.
    try:
        jf, _ = build(max2=30.0, on_mismatch='first')
        mk = list(jf.param_map)[0]
        p = jf.master_params.copy()
        p[mk].value = 50.0
        jf.push_params(p)
        loc = [jf.objectives[i]['func'].params['a'].value for i in range(3)]
        check("T3c2 'first': every local value == master 50 after push_params",
              p[mk].value == 50.0 and all(v == 50.0 for v in loc), loc)
        check("T3c2 'first': local bounds/vary unified to the first endpoint's",
              all(jf.objectives[i]['func'].params['a'].max == np.inf
                  and jf.objectives[i]['func'].params['a'].vary for i in range(3)),
              [(jf.objectives[i]['func'].params['a'].max,
                jf.objectives[i]['func'].params['a'].vary) for i in range(3)])
        check("T3c2 'first' warning says the bounds were unified",
              len(jf.warnings) == 1 and 'bounds unified' in jf.warnings[0],
              jf.warnings)
    except Exception as ex:
        check("T3c2 'first' unifies bounds", False, repr(ex))

    # (d) vary differs -> ShareMismatchError
    try:
        build(vary2=False)
        check("T3d vary mismatch raises ShareMismatchError", False, 'no exception')
    except ShareMismatchError as ex:
        check("T3d vary mismatch raises ShareMismatchError", True, str(ex))

    # the legacy shared= path obeys the same rule
    try:
        jf = JointFit()
        for i, (nm, y) in enumerate(zip(names, ys)):
            jf.add_dataset(nm, funcClass=Line, x=X, y=y, yerr=err)
        jf.objectives[1]['func'].params['a'].set(min=0.0)
        jf.build_master_params(shared=['a'])
        check("T3 shared= path raises on a bound mismatch", False, 'no exception')
    except ShareMismatchError as ex:
        check("T3 shared= path raises on a bound mismatch", True, str(ex))

    # JointFitSpec.compile passes on_mismatch through (default 'raise')
    from joint_fit_spec import JointFitSpec
    data = {nm: {'c': {'x': X, 'y': y, 'yerr': err}} for nm, y in zip(names, ys)}

    def spec():
        sp = JointFitSpec(default_model_class=Line)
        for i, nm in enumerate(names):
            m = Line(x=X)
            m.params['a'].set(max=(10.0 if i == 1 else np.inf))
            sp.add_dataset(nm, model=m)
        sp.set_shared('a')
        return sp
    try:
        spec().compile(data)
        check("T3 compile() raises ShareMismatchError by default", False,
              'no exception')
    except ShareMismatchError as ex:
        check("T3 compile() raises ShareMismatchError by default", True, str(ex))
    try:
        jf = spec().compile(data, on_mismatch='first')
        check("T3 compile(on_mismatch='first') builds with 1 warning",
              len(jf.warnings) == 1, jf.warnings)
    except Exception as ex:
        check("T3 compile(on_mismatch='first') builds", False, repr(ex))

def test_t4():
    """01 6.7 / 6.14: warn on per-max weighting and on dropped sub-curves."""
    import numpy as np
    from validate_joint_gls import Line, X, make_data, E
    from Joint_Fit_Routines import JointFit
    ys = make_data(2)
    err = E * np.ones_like(X)
    WARN = ("fit scale 'Linear w/o error' divides each dataset by its own maximum, "
            "so datasets are weighted by 1/max(y); use 'Linear' (error-weighted) "
            "for a comparable joint objective")

    def build(n=2):
        jf = JointFit()
        for i in range(n):
            jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=ys[i], yerr=err)
            jf.objectives[i]['func'].params['b'].set(vary=True)
        jf.build_master_params(groups=[jf.share_within('a', ['D%d' % i for i in range(n)])])
        return jf

    jf = build()
    rep, _ = jf.perform_fit(X[0], X[-1], fit_scale='Linear w/o error',
                            fit_method='leastsq')
    check("T4a 'Linear w/o error' 2-dataset no-cov -> warning present",
          WARN in list(jf.warnings), list(jf.warnings))
    check("T4a warning appears in the report", WARN in rep)
    jf.perform_fit(X[0], X[-1], fit_scale='Linear w/o error', fit_method='leastsq')
    check("T4a re-fit does not duplicate the warning",
          list(jf.warnings).count(WARN) == 1, list(jf.warnings))
    jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T4b 'Linear' on the same object drops the warning",
          WARN not in list(jf.warnings), list(jf.warnings))
    jf2 = build()
    jf2.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    check("T4b 'Linear' fresh fit -> no such warning",
          WARN not in list(jf2.warnings), list(jf2.warnings))
    jf3 = build()
    jf3.perform_fit(X[0], X[-1], fit_scale='Log w/o error', fit_method='leastsq')
    check("T4b 'Log w/o error' -> no such warning",
          WARN not in list(jf3.warnings), list(jf3.warnings))

    # (c) compile(): two curve keys -> warning, first curve only
    from joint_fit_spec import JointFitSpec
    names = ('M_one', 'M_two')
    other = {'x': X[:5], 'y': np.zeros(5), 'yerr': np.ones(5)}
    data = {nm: {'curveA': {'x': X, 'y': y, 'yerr': err}, 'curveB': dict(other)}
            for nm, y in zip(names, ys)}
    sp = JointFitSpec(default_model_class=Line)
    for nm in names:
        sp.add_dataset(nm, model=Line(x=X))
    sp.set_shared('a')
    jf = sp.compile(data)
    ws = list(jf.warnings)
    check("T4c compile warns for each multi-curve dataset",
          all(any(w.startswith('dataset %s has 2 curves; only the first (curveA) '
                               'enters the joint fit' % nm) for w in ws)
              for nm in names), ws)
    check("T4c build-time warnings are plain str (not run-time)",
          all(type(w) is str for w in ws), [type(w) for w in ws])
    check("T4c only the first curve's points are fitted",
          all(len(o['x']) == len(X) for o in jf.objectives),
          [len(o['x']) for o in jf.objectives])
    sp1 = JointFitSpec(default_model_class=Line)
    one = {nm: {'curveA': {'x': X, 'y': y, 'yerr': err}} for nm, y in zip(names, ys)}
    for nm in names:
        sp1.add_dataset(nm, model=Line(x=X))
    sp1.set_shared('a')
    check("T4c single-curve datasets -> no multi-curve warning",
          not any('curves' in w for w in sp1.compile(one).warnings))

    # (c') audit gap: the script orchestrator (joint_fit_orchestration) dropped the
    # later sub-curves silently. build_jointfit_from_selection now records the
    # same message in JointFit.warnings; a bare _primary_curve uses warnings.warn.
    import warnings as _w
    import joint_fit_orchestration as JO
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter('always')
        jo = JO.build_jointfit_from_selection(data, list(names), Line, shared=['a'])
    wo = list(jo.warnings)
    check("T4c' build_jointfit_from_selection warns for each multi-curve dataset",
          all(any(w.startswith('dataset %s has 2 curves; only the first (curveA) '
                               'enters the joint fit' % nm) for w in wo)
              for nm in names), wo)
    check("T4c' ... in JointFit.warnings only (no duplicate warnings.warn)",
          not any('curves' in str(r.message) for r in rec), [str(r.message) for r in rec])
    check("T4c' ... and fits the first curve only",
          all(len(o['x']) == len(X) for o in jo.objectives))
    jo1 = JO.build_jointfit_from_selection(one, list(names), Line, shared=['a'])
    check("T4c' single-curve selection -> no multi-curve warning",
          not any('curves' in w for w in jo1.warnings), list(jo1.warnings))
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter('always')
        JO._primary_curve(data['M_one'], name='M_one')
        JO._primary_curve(one['M_one'], name='M_one')
    msgs = [str(r.message) for r in rec]
    check("T4c' bare _primary_curve: warnings.warn once, for the multi-curve dataset",
          len(msgs) == 1 and msgs[0].startswith('dataset M_one has 2 curves'), msgs)

def test_t5():
    """A1.4 (01 6.6): joint-vs-individual chi-square on the joint fit's own points."""
    import numpy as np
    from validate_joint_gls import Line, X, make_data, E
    import Joint_Fit_Routines as J
    from Joint_Fit_Routines import JointFit
    ys = make_data(2)
    err = E * np.ones_like(X)
    mask = np.ones(len(X), bool)
    mask[[12, 13, 20]] = False

    def build(mask1=mask):
        jf = JointFit()
        for i in range(2):
            jf.add_dataset('D%d' % i, funcClass=Line, x=X, y=ys[i], yerr=err,
                           mask=(mask1 if i == 1 else None))
            jf.objectives[i]['func'].params['b'].set(vary=True)
        jf.build_master_params(groups=[jf.share_within('a', ['D0', 'D1'])])
        return jf

    jf = build()
    ranges = {'D0': (X[5], X[30]), 'D1': (X[8], X[35])}
    jf._set_ranges(X[0], X[-1], ranges)
    o = jf.objectives[1]
    want = np.arange(o['imin'], o['imax'] + 1)
    want = want[mask[o['imin']:o['imax'] + 1]]
    got = np.asarray(jf.active_index(1))
    check("T5a active_index(1) == arange(i0,i1+1)[mask[i0:i1+1]]",
          np.array_equal(got, want) and len(want) < o['imax'] - o['imin'] + 1,
          (got, want))
    check("T5a masked-out point 12 absent, 13 absent",
          12 not in got and 13 not in got and 20 not in got)

    # (b) no-cov Linear fit: summed chisqr_on == result.chisqr
    jf = build()
    jf.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq',
                   ranges=ranges)
    tot = nd = tot0 = 0
    for i, ob in enumerate(jf.objectives):
        # M1: evaluate on the objective's own [imin, imax] window
        c, n = J.chisqr_on(ob['func'], ob['x'], ob['y'], ob['yerr'],
                           jf.active_index(i), masked_x=ob['yfit_masked'],
                           imin=ob['imin'], imax=ob['imax'])
        tot += c; nd += n
        c0, _ = J.chisqr_on(ob['func'], ob['x'], ob['y'], ob['yerr'],
                            jf.active_index(i), masked_x=ob['yfit_masked'])
        tot0 += c0
    rc = jf.result.chisqr
    check("T5b sum chisqr_on == result.chisqr to 1e-10",
          abs(tot - rc) <= 1e-10 * abs(rc), (tot, rc))
    check("T5b chisqr_on without imin/imax (idx-span fallback) agrees for a "
          "pointwise model", abs(tot0 - rc) <= 1e-10 * abs(rc), (tot0, rc))
    check("T5b ndata == result.ndata", nd == jf.result.ndata, (nd, jf.result.ndata))

    # (c) count follows the active points, not the scalar window
    n_scalar = sum(1 for o in jf.objectives
                   for _ in range(o['imin'], o['imax'] + 1))
    check("T5c ndata == sum len(active_index) and < scalar-window count",
          nd == sum(len(jf.active_index(i)) for i in range(2)) and nd < n_scalar,
          (nd, n_scalar))
    full = jf.active_index(0)
    check("T5c narrower range on D0 shortens it", len(full) == 26, len(full))

    # (e) final review I2: a compiled, never-fitted JointFit (the .jpar / .par
    # load paths) gets its fit ranges and column mask frozen by the same entry
    # point those load paths call, XModFit._apply_joint_ranges.
    import types as _types
    import xmodfit
    from joint_fit_spec import JointFitSpec
    sp = JointFitSpec(default_model_class=Line)
    for i in range(2):
        m = Line(x=X)
        m.params['b'].set(vary=True)
        sp.add_dataset('D%d' % i, model=m, fit_xmin=ranges['D%d' % i][0],
                       fit_xmax=ranges['D%d' % i][1])
    sp.set_shared('a')
    data = {'D%d' % i: {'c': {'x': X, 'y': ys[i], 'yerr': err}} for i in range(2)}
    jl = sp.compile(data, masks={'D0': None, 'D1': mask})
    check("T5e before: a compiled, unfitted JointFit scores every point",
          len(jl.active_index(1)) == len(X), len(jl.active_index(1)))
    fake = _types.SimpleNamespace(_joint_range=lambda: (X[0], X[-1]))
    ret = xmodfit.XModFit._apply_joint_ranges(fake, jl, sp)
    check("T5e _apply_joint_ranges returns the JointFit", ret is jl)
    check("T5e after: active_index honours the fit ranges and mask (no perform_fit)",
          all(np.array_equal(jl.active_index(i), jf.active_index(i)) for i in range(2))
          and np.array_equal(jl.active_index(1), want),
          [len(jl.active_index(i)) for i in range(2)])

    # __masked_x__ model: fed active points only, not masked twice
    class MX(Line):
        __masked_x__ = True
    jm = JointFit()
    for i in range(2):
        jm.add_dataset('D%d' % i, funcClass=MX, x=X, y=ys[i], yerr=err, mask=mask)
        jm.objectives[i]['func'].params['b'].set(vary=True)
    jm.build_master_params(groups=[jm.share_within('a', ['D0', 'D1'])])
    jm.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq')
    t = 0
    for i, ob in enumerate(jm.objectives):
        c, n = J.chisqr_on(ob['func'], ob['x'], ob['y'], ob['yerr'],
                           jm.active_index(i), masked_x=ob['yfit_masked'])
        t += c
    check("T5d __masked_x__ model: chisqr_on == result.chisqr",
          ob['yfit_masked'] and abs(t - jm.result.chisqr) <= 1e-10 * jm.result.chisqr,
          (t, jm.result.chisqr))

    # (f) audit gap: drive the REAL XModFit.computeModelComparison (and _eval_chisqr
    # through it) on a joint fit with per-dataset fit ranges and a column mask. The
    # helpers above are not enough: reverting the call site to the scalar window
    # with no mask leaves T5a-T5e green. Host is a SimpleNamespace, as in T5e.
    from PyQt5.QtWidgets import QTextEdit
    jc = sp.compile(data, masks={'D0': None, 'D1': mask})
    jc.perform_fit(X[0], X[-1], fit_scale='Linear', fit_method='leastsq', ranges=ranges)
    jc._is_fitted = True
    ind = {}
    for i in range(2):      # "individual fits": fixed lines, deliberately off the data
        ind['D%d' % i] = _types.SimpleNamespace(
            func=Line(x=X, a=3.0 + 0.01 * i, b=0.5), _is_fitted=True, result=None)
    te = QTextEdit()
    host = _types.SimpleNamespace(jointFit=jc, data=data, dataset_fits=ind,
                                  compareTextEdit=te, _short_name=lambda n: n,
                                  _joint_range=lambda: (X[0], X[-1]))
    host._eval_chisqr = _types.MethodType(xmodfit.XModFit._eval_chisqr, host)
    xmodfit.XModFit.computeModelComparison(host)
    rows = {}
    for ln in te.toPlainText().splitlines():
        for key in ('data points N', 'chi-square'):
            if ln.startswith(key + ' '):
                rows[key] = ln.split()[-2:]
    n_act = sum(len(jc.active_index(i)) for i in range(2))
    ok = len(rows) == 2
    nj = ni = cj = ci = None
    if ok:
        nj, ni = int(rows['data points N'][0]), int(rows['data points N'][1])
        cj, ci = float(rows['chi-square'][0]), float(rows['chi-square'][1])
    # independent expectation for the individuals: plain sum over the active points
    ci_want = 0.0
    for i in range(2):
        f = Line(x=X, a=3.0 + 0.01 * i, b=0.5)   # fresh: chisqr_on re-windows func.x
        r = (ys[i] - f.y()) / err
        ci_want += float(np.sum(r[jc.active_index(i)] ** 2))
    check("T5f computeModelComparison: N (joint and individuals) == active points",
          ok and nj == n_act and ni == n_act and n_act < 2 * len(X),
          (nj, ni, n_act, 2 * len(X)))
    check("T5f computeModelComparison: joint chi-square == result.chisqr",
          ok and abs(cj - jc.result.chisqr) <= 1e-5 * jc.result.chisqr,
          (cj, jc.result.chisqr))
    check("T5f computeModelComparison: individual chi-square on the same points",
          ok and abs(ci - ci_want) <= 1e-5 * ci_want, (ci, ci_want))


def test_n2():
    """Audit N2: JointFit._report builds the derived-quantity keymap from
    param_map, so a pose shared through JointFitSpec groups (bare, or a subset
    group) still reaches XLayersPDB_XPR.derived_report and its protins_top block
    appears. The model is synthetic; derived_params/derived_report are the real
    XLayersPDB_XPR methods (pure-Python geometry on Data/1ubq.pdb, no Julia)."""
    import numpy as np
    from importlib import import_module
    from lmfit import Parameters
    from joint_fit_spec import JointFitSpec
    XPR = import_module('Functions.XRR.XLayersPDB_XPR').XLayersPDB_XPR
    PI = np.pi
    xg = np.linspace(0.02, 0.5, 40)

    def shape(t, p, ins, x):
        return (1 + 0.3 * np.cos(t * PI / 180) * np.cos(2 * PI * x)
                + 0.3 * np.sin(t * PI / 180) * np.cos(p * PI / 180) * np.sin(2 * PI * x)
                + 0.02 * ins * x)

    class SynPose:
        derived_params = XPR.derived_params
        derived_report = XPR.derived_report

        def __init__(s, x=xg):
            s.x = x
            s.use_protein = True
            s.fname = os.path.join(ROOT, 'Data', '1ubq.pdb')
            s.params = Parameters()
            s.params.add('theta', value=50., vary=True)
            s.params.add('phi', value=130., vary=True)
            s.params.add('protins', value=5., vary=True)
            s.params.add('scale', value=1., vary=True)
            s.output_params = {}

        def init_params(s):
            pass

        def y(s):
            P = s.params
            return P['scale'].value * shape(P['theta'].value, P['phi'].value,
                                            P['protins'].value, s.x)

    rng = np.random.RandomState(1)
    err = 0.01 * np.ones_like(xg)
    scales = (1.0, 1.3, 0.8)

    def build(n, share):
        names = ['P%d' % i for i in range(n)]
        data = {nm: {'c': {'x': xg, 'y': scales[i] * shape(50, 130, 5, xg)
                           + 0.01 * rng.randn(len(xg)), 'yerr': err}}
                for i, nm in enumerate(names)}
        sp = JointFitSpec(default_model_class=SynPose)
        for nm in names:
            sp.add_dataset(nm, model=SynPose())
        for k in ('protins', 'theta', 'phi'):
            share(sp, k, names)
        jf = sp.compile(data)
        jf.perform_fit(xg[0], xg[-1], fit_scale='Linear', fit_method='leastsq')
        return jf

    # (a) pose shared by every dataset (bare master keys protins/theta/phi)
    jf = build(2, lambda sp, k, names: sp.set_shared(k))
    km = jf._derived_keymaps()
    check("N2a spec-shared pose: no legacy shared list (the case _master_key missed)",
          list(jf.shared) == [], jf.shared)
    check("N2a keymap points at master keys that exist",
          all(v in jf.master_params for m in km for v in m.values()), km)
    check("N2a both datasets map to the same (shared) keys", km[0] == km[1], km)
    rep = jf._report()
    check("N2a protins_top block present in the joint report",
          rep.count('[[Derived]]') == 1 and 'protins_top:' in rep,
          rep.count('[[Derived]]'))

    # (b) pose shared by a SUBSET (P0, P1) of three datasets: g-prefixed master keys
    jf = build(3, lambda sp, k, names: sp.set_shared(k, dataset_names=names[:2]))
    km = jf._derived_keymaps()
    check("N2b subset group: keymap points at master keys that exist",
          all(v in jf.master_params for m in km for v in m.values()), km)
    check("N2b P0/P1 share keys, P2 has its own",
          km[0] == km[1] and km[2] != km[0]
          and not set(km[2].values()) & set(km[0].values()), km)
    rep = jf._report()
    check("N2b two derived blocks (P0/P1 collapsed, P2 separate)",
          rep.count('[[Derived]]') == 2 and '(P2)' in rep, rep.count('[[Derived]]'))

    # (c) the legacy scripted shared= path is unchanged
    from Joint_Fit_Routines import JointFit
    jl = JointFit()
    for i in range(2):
        jl.add_dataset('L%d' % i, func=SynPose(), x=xg,
                       y=scales[i] * shape(50, 130, 5, xg), yerr=err)
    jl.build_master_params(shared=['protins', 'theta', 'phi'])
    jl.perform_fit(xg[0], xg[-1], fit_scale='Linear', fit_method='leastsq')
    check("N2c legacy shared=: one derived block",
          jl._report().count('[[Derived]]') == 1)


def test_t3_gui():
    """The GUI guards spec.compile (final review I3/I4) and freezes the fit
    ranges of a loaded joint fit (I2).

    A ShareMismatchError asks Yes/No (No: nothing changes; Yes: recompile with
    on_mismatch='first'); any other ValueError is shown in a warning box; neither
    reaches the Qt slot, and no success box follows a refusal."""
    import numpy as np
    import xmodfit
    from PyQt5.QtCore import Qt
    import joint_fit_io
    from Fit_Routines import _window_indices
    calls = {'warning': [], 'information': [], 'question': []}

    class _MB:
        Ok, Yes, No = 0x400, 0x4000, 0x10000
        answer = [0x10000]
        warning = staticmethod(lambda *a, **k: calls['warning'].append(a) or 0)
        information = staticmethod(lambda *a, **k: calls['information'].append(a) or 0)
        question = staticmethod(
            lambda *a, **k: calls['question'].append(a) or _MB.answer[0])

    def reset():
        for v in calls.values():
            del v[:]

    def run(fn, *a, **k):
        try:
            fn(*a, **k)
            return None
        except Exception as ex:
            return ex
    saved = (xmodfit.QMessageBox, joint_fit_io.parse_par, joint_fit_io.apply_block)
    xmodfit.QMessageBox = _MB
    try:
        w = xmodfit.XModFit()
        check("T3gui XModFit has _compile_joint_guarded / _apply_joint_ranges",
              callable(getattr(w, '_compile_joint_guarded', None))
              and callable(getattr(w, '_apply_joint_ranges', None)))
        d = os.path.join(ROOT, 'tests', '_logs')
        os.makedirs(d, exist_ok=True)
        q = np.linspace(0.02, 0.30, 40)
        paths = [os.path.join(d, 'ws1_t3_ds%d.txt' % i) for i in (1, 2)]
        for p, amp in zip(paths, (1.0, 2.0)):
            np.savetxt(p, np.c_[q, amp * np.exp(-((q - 0.1) ** 2) / 0.002) + 0.01,
                                np.full_like(q, 0.02)], header="x y yerr")
        w.addData(fnames=paths)
        w.categoryListWidget.setCurrentItem(
            w.categoryListWidget.findItems("XRR", Qt.MatchExactly)[0])
        w.funcListWidget.setCurrentItem(
            w.funcListWidget.findItems("XLayers", Qt.MatchExactly)[0])
        w.dataListWidget.clearSelection()
        for i in range(w.dataListWidget.count()):
            w.dataListWidget.item(i).setSelected(True)
        w.jointFitPanel.add_selected_datasets()
        spec = w.jointFitPanel.spec
        check("T3gui two datasets in the joint spec", len(spec.datasets) == 2)
        for i, e in enumerate(spec.datasets):
            e.model.params['qoff'].set(vary=1, max=(0.5 if i == 1 else np.inf))
        spec.set_shared('qoff')
        w.fitIterationLineEdit.setText("5")
        w.jointFitEnabled = True
        fit0, jf0 = w.fit, getattr(w, 'jointFit', None)

        # (I4, No) the mismatch is asked about; declining changes nothing
        _MB.answer[0] = _MB.No
        raised = run(w.run_joint_fit)
        check("T3gui run_joint_fit returns without raising", raised is None, repr(raised))
        check("T3gui the mismatch is shown in a Yes/No question naming qoff",
              len(calls['question']) == 1 and 'qoff' in str(calls['question'][0][2])
              and "Apply the first dataset's bounds/vary to all members of each "
                  "share group and continue?" in str(calls['question'][0][2])
              and not calls['warning'], calls)
        check("T3gui No: self.fit / self.jointFit unchanged",
              w.fit is fit0 and getattr(w, 'jointFit', None) is jf0)

        # .par-into-joint load path: question shown, No -> no success box
        par = os.path.join(d, 'ws1_t3_dummy.par')
        open(par, 'w').write('')
        joint_fit_io.parse_par = lambda txt: {}
        joint_fit_io.apply_block = lambda *a, **k: None
        reset()
        raised = run(w.loadParIntoJoint, fname=par)
        check("T3gui loadParIntoJoint asks, No -> no success box",
              raised is None and len(calls['question']) == 1
              and not calls['information'], (repr(raised), calls))

        # (I3) any other ValueError from compile: warning box, no crash,
        # no success box, nothing replaced
        def _boom(*a, **k):
            raise ValueError("attached covariances use different var_fwhm")
        spec.compile = _boom
        try:
            reset()
            raised = run(w.run_joint_fit)
            check("T3gui ValueError: run_joint_fit shows a warning, does not raise",
                  raised is None and len(calls['warning']) == 1
                  and 'var_fwhm' in str(calls['warning'][0][2])
                  and w.fit is fit0, (repr(raised), calls))
            reset()
            raised = run(w.loadParIntoJoint, fname=par)
            check("T3gui ValueError: loadParIntoJoint warns, no success box",
                  raised is None and len(calls['warning']) == 1
                  and not calls['information'], (repr(raised), calls))
        finally:
            del spec.compile

        # (I4, Yes) recompile with on_mismatch='first' and fit
        _MB.answer[0] = _MB.Yes
        reset()
        raised = run(w.run_joint_fit)
        jf = getattr(w, 'jointFit', None)
        check("T3gui Yes: run_joint_fit fits with the first dataset's bounds",
              raised is None and jf is not None and jf is not jf0
              and getattr(jf, 'result', None) is not None
              and any('bounds unified' in str(x) for x in jf.warnings),
              (repr(raised), getattr(jf, 'warnings', None)))
        check("T3gui Yes: both members' local qoff bounds now match",
              jf is not None and all(o['func'].params['qoff'].max == np.inf
                                     for o in jf.objectives),
              None if jf is None else [o['func'].params['qoff'].max
                                       for o in jf.objectives])

        # (I2) .jpar round trip with a narrowed #FitRange: the loaded, never
        # re-fitted joint fit's active_index honours it
        joint_fit_io.parse_par, joint_fit_io.apply_block = saved[1], saved[2]
        spec.datasets[1].fit_xmin, spec.datasets[1].fit_xmax = 0.05, 0.2
        jpar = os.path.join(d, 'ws1_i2_roundtrip.jpar')
        reset()
        w.saveJointFit(fname=jpar)
        reset()
        raised = run(w.loadJointFit, fname=jpar)
        jl = getattr(w, 'jointFit', None)
        ok = raised is None and jl is not None and jl is not jf
        if ok:
            i0, i1 = _window_indices(q, 0.05, 0.2)
            a0, a1 = _window_indices(q, *w._joint_range())
            got1, got0 = np.asarray(jl.active_index(1)), np.asarray(jl.active_index(0))
        check("T3gui I2: loaded .jpar's active_index honours #FitRange=",
              ok and np.array_equal(got1, np.arange(i0, i1 + 1)) and len(got1) < len(q),
              (repr(raised), None if not ok else (len(got1), i0, i1)))
        check("T3gui I2: the other dataset uses the scalar Xrange window",
              ok and np.array_equal(got0, np.arange(a0, a1 + 1)),
              None if not ok else (len(got0), a0, a1))
        check("T3gui I2: load succeeded (success box, no question)",
              ok and len(calls['information']) == 1 and not calls['question'], calls)
    finally:
        xmodfit.QMessageBox, joint_fit_io.parse_par, joint_fit_io.apply_block = saved


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    for t in (test_t1, test_t2, test_t3, test_t4, test_t5, test_n2,
              test_t3_gui):
        try:
            t()
        except Exception as ex:
            import traceback; traceback.print_exc()
            check("%s runs" % t.__name__, False, repr(ex))
    print()
    return finish()


if __name__ == '__main__':
    sys.exit(main())
