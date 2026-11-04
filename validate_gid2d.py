#!/usr/bin/env python3
"""
validate_gid2d.py -- regression tests for Functions/gid2d_core.py and
Functions/GISAXS/GID2D.py.

    python validate_gid2d.py

Every target below is either an algebraic identity or a published value, so the
tolerances are tight on purpose. A drifting residual means the model changed,
not that the arithmetic got unlucky.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, 'Functions'), os.path.join(HERE, 'Functions', 'GISAXS')):
    if p not in sys.path:
        sys.path.insert(0, p)

import gid2d_core as GC                                          # noqa: E402
from GID2D import GID2D                                          # noqa: E402

RAW = r"T:\SURFACE\NX_school\2026sep\extracted_data\dppc_q2d_gid.txt"
PACKED = os.path.join(HERE, 'Data', 'dppc_q2d_gid_indexed.txt')

NN = dict(a0=4.8743, t=18.85, psi=0.0, eps=-0.0321, L_dom=200.0, L_ch=19.0)

from tests.harness import check, finish                          # noqa: E402


def mk(**kw):
    # Background pinned to ZERO unless a test asks otherwise: these are Bragg
    # sum-rule / lineshape-width checks, and GID2D's shipped bkg0 default is the
    # DPPC map's mean background (~40), which would bias every one of them. The
    # (11):(02) ratio came out 1.632 instead of 1.998 when this was left implicit.
    d = dict(NN)
    d.setdefault('bkg0', 0.0)
    d.update(kw)
    f = GID2D(x=np.array([0.0]), **d)
    f.__fit__ = True                    # suppress the reporting path in tests
    return f


# ---------------------------------------------------------------- 1. core --
def test_core():
    print('\n[1] gid2d_core selfcheck')
    n = GC.selfcheck(verbose=False)
    check('core selfcheck (%d assertions)' % n, n > 50)


# ------------------------------------------------- 2. the 2:1 sum rule -----
def test_sum_rule():
    """Integrated (11)+(1-1) : (02) = 2:1 over the accessible half space, with
    NO abs() anywhere. The mechanism is that the (02) rod sits at q_z = 0 and is
    HALVED by the horizon, while the (11)/(1-1) rods sit at |q_z| = 0.43 so
    their Friedel mates at +0.43 are nearly whole. Getting 3.8 instead means
    somebody folded q_z."""
    print('\n[2] integrated (11)+(1-1):(02) sum rule')
    qxy = np.linspace(1.30, 1.65, 1401)
    tot = {}
    for tag, keep in (('11', {(1, 1)}), ('02', {(0, 2)})):
        f = mk(mode='inplane', powder=False, vineyard=False, bkg_trans=False,
               alpha_i=0.0, qzmin=0.0, qzmax=1.4, nq=2001)
        f.__hk_filter__ = keep
        f.update_parameters()
        tot[tag] = float(np.trapz(f._reduce_inplane(qxy), qxy))
    r = tot['11'] / tot['02']
    check('ratio = 1.998 +/- 0.005 (published 2:1)', abs(r - 1.998) < 5e-3,
          'got %.5f' % r)

    # The failure mode this guards against: taking abs() of the ROD CENTRE q_z0,
    # so all four (11)/(1-1) rods pile up at +0.43 instead of two of them sitting
    # below the horizon. Note it is abs(q_z0) that does the damage, not abs() of
    # the evaluation point -- folding the evaluation axis is symmetric and leaves
    # the ratio alone, which is exactly why this bug survives casual checking.
    z = np.linspace(0.0, 1.4, 4001)
    tot_f = {}
    for tag, keep in (('11', {(1, 1)}), ('02', {(0, 2)})):
        lat = GC.Lattice2D(NN['a0'], NN['t'], NN['psi'], NN['eps'])
        acc = np.zeros((len(qxy), len(z)))
        for R in lat.reflections(six=True):
            if (abs(R['hk'][0]), abs(R['hk'][1])) not in keep:
                continue
            acc += (GC.radial_profile(qxy[:, None], R['qxy'], NN['L_dom'],
                                      powder=False)
                    * GC.rod_profile(z[None, :], abs(R['qz']),      # the bug
                                     np.radians(NN['t']), NN['L_ch']))
        tot_f[tag] = float(np.trapz(np.trapz(acc, z, axis=1), qxy))
    rf = tot_f['11'] / tot_f['02']
    check('abs(q_z0) inflates the ratio to ~4 (the guarded bug)', rf > 3.0,
          'folding the rod centre gives %.3f instead of 2.0' % rf)


# ----------------------------------------- 3. reduction self-consistency ---
def test_reduction_consistency():
    """Reducing the 2D mesh must equal evaluating the 1D integral directly, or
    the 'one model, three views' claim is empty."""
    print('\n[3] map reduction vs direct 1D evaluation')
    f = mk(mode='inplane', qzmin=0.0, qzmax=1.4, nq=801)
    f.update_parameters()
    qxy = np.linspace(1.30, 1.65, 121)
    z = f._qz_grid(0.0, 1.4)
    red = np.trapz(f._model(qxy[:, None], z[None, :]), z, axis=1)
    direct = np.array([np.trapz(f._model(np.full_like(z, q), z), z) for q in qxy])
    e = float(np.max(np.abs(red - direct)) / np.max(direct))
    check('in-plane: max rel diff < 1e-12', e < 1e-12, '%.2e' % e)

    qz = np.linspace(0.0, 1.0, 121)
    xg = np.linspace(1.44, 1.52, 401)
    red = np.trapz(f._model(xg[:, None], qz[None, :]), xg, axis=0)
    direct = np.array([np.trapz(f._model(xg, np.full_like(xg, q)), xg) for q in qz])
    e = float(np.max(np.abs(red - direct)) / np.max(direct))
    check('rod: max rel diff < 1e-12', e < 1e-12, '%.2e' % e)


# --------------------------------------- 4. the section-9 quadrature fix ---
def test_quadrature():
    """The graded q_z grid must converge MONOTONICALLY and beat the uniform grid
    it replaces by orders of magnitude. Uniform is left in as the control: it is
    the thing that produced a ~1% grid-dependent error looking like physics."""
    print('\n[4] q_z quadrature convergence (the graded-grid fix)')
    f = mk(mode='inplane', L_ch=35.0, qzmin=0.0, qzmax=1.4)
    f.update_parameters()
    qxy = np.linspace(1.30, 1.65, 141)

    def red(z):
        return np.trapz(f._model(qxy[:, None], z[None, :]), z, axis=1)

    NS = (51, 101, 201, 401, 801)
    ref = red(f._qz_grid(0.0, 1.4, n=40001))
    graded, uniform = [], []
    for n in NS:
        zg = f._qz_grid(0.0, 1.4, n=n)
        graded.append(float(np.max(np.abs(red(zg) - ref)) / np.max(ref)))
        zu = np.linspace(0.0, 1.4, len(zg))
        uniform.append(float(np.max(np.abs(red(zu) - ref)) / np.max(ref)))
    print('       n:        ' + ''.join('%10d' % n for n in NS))
    print('       graded:   ' + ''.join('%10.2e' % v for v in graded))
    print('       uniform:  ' + ''.join('%10.2e' % v for v in uniform))
    i4 = NS.index(401)
    check('graded converges monotonically',
          all(graded[i + 1] < graded[i] for i in range(len(graded) - 1)),
          str(['%.1e' % v for v in graded]))
    check('graded reaches < 1e-4 by n=401', graded[i4] < 1e-4, '%.2e' % graded[i4])
    check('graded beats uniform by >20x at n=401',
          uniform[i4] / graded[i4] > 20.0, '%.0fx' % (uniform[i4] / graded[i4]))
    # The original symptom: the uniform grid's error is large enough at practical
    # node counts to masquerade as physics (order 1%), which is the whole reason
    # the graded grid exists. Non-monotonicity is the tell but which n shows it
    # depends on where the nodes land relative to the cusp, so assert the
    # magnitude -- that is the part that actually corrupts a fit.
    check('uniform carries a >1e-3 error at n<=401 (masquerades as physics)',
          max(uniform[:i4 + 1]) > 1e-3,
          'worst %.2e over n<=401' % max(uniform[:i4 + 1]))
    check('uniform is non-monotone somewhere in the sweep',
          not all(uniform[i + 1] < uniform[i] for i in range(len(uniform) - 1)),
          str(['%.1e' % v for v in uniform]))


# ------------------------------------ 5. Rod_Lipid convention agreement ----
def test_rod_lipid_agreement():
    """GID2D and Rod_Lipid must agree on |T_f|^2 for the same data, or the two
    functions silently disagree about q_c."""
    print('\n[5] Vineyard convention matches Rod_Lipid')
    sys.path.insert(0, os.path.join(HERE, 'Functions', 'GISAXS'))
    from Rod_Lipid import Rod_Lipid
    qz = np.linspace(0.0, 0.5, 501)
    E, ai, qc = 15.126, 0.10, 0.0217
    rl = Rod_Lipid(x=qz, E=E, alpha=ai, qc=qc)
    k0 = 2 * np.pi * E / 12.3984                      # Rod_Lipid's own HC
    mine = GC.transmission(qz, E, ai, qc)
    theirs = rl.trans(qz - k0 * np.sin(np.radians(ai)), qc / 2)
    e = float(np.max(np.abs(mine - theirs)))
    check('|T_f|^2 identical to Rod_Lipid.trans(qbeta, qc/2)', e < 2e-3,
          'max abs diff %.2e (HC differs in the 5th digit: 12.39842 vs 12.3984)' % e)


# --------------------------------------------------- 6. pack / load / fit --
def test_pack_and_map():
    print('\n[6] pack_map / load_map / map mode')
    if not os.path.exists(RAW):
        check('raw map present', False, 'SKIPPED, not found: %s' % RAW)
        return
    if not os.path.isdir(os.path.dirname(PACKED)):
        os.makedirs(os.path.dirname(PACKED))
    info = GC.pack_map(RAW, PACKED, drop_masked=True, qz_min=0.0)
    check('packed %d rows (dropped %d)' % (info['kept'], info['dropped']),
          info['kept'] > 30000, '')
    print('        %s' % info['sigma_note'])
    mp = GC.load_map(PACKED)
    check('no non-positive errors survive packing', bool(np.all(mp['Ierr'] > 0)))
    check('no non-positive intensities survive packing', bool(np.all(mp['I'] > 0)))
    # the sigma floor must remove the mask residue that would otherwise carry
    # essentially all of chi^2
    rel = mp['Ierr'] / np.median(mp['Ierr'])
    check('no near-zero-sigma pixels survive (they would dominate chi^2)',
          float(rel.min()) > 1e-5, 'min sigma/median = %.3e' % rel.min())
    check('index column strictly increasing',
          bool(np.all(np.diff(mp['idx']) == 1)))
    check('load_map is cached on the file stamp',
          GC.load_map(PACKED) is mp)

    # float row index must work: XModFit's x line-edit accepts np.linspace(...)
    xf = np.linspace(0, mp['n'] - 1, 257)
    r = GC.row_lookup(xf, mp)
    check('float row index -> in-range integer rows',
          r.min() >= 0 and r.max() < mp['n'] and r.dtype.kind == 'i')

    # full-map evaluation, and the windowing contract
    x = np.arange(mp['n'], dtype=float)
    f = GID2D(x=x, mapfile=PACKED, mode='map2d', E=15.126, alpha_i=0.10,
              qc=0.0217, yscale=1.0, bkg0=1.0, fisher=False, **NN)
    f.__fit__ = True
    y = f.y()
    check('map mode returns one value per row', len(y) == mp['n'],
          '%d vs %d' % (len(y), mp['n']))
    check('map mode is finite everywhere', bool(np.all(np.isfinite(y))))

    # a q_xy/q_z window must zero the residual OUTSIDE it, exactly
    f2 = GID2D(x=x, mapfile=PACKED, mode='map2d', qxymin=1.42, qxymax=1.52,
               qzmin=0.05, qzmax=0.60, yscale=1.0, bkg0=1.0, fisher=False, **NN)
    f2.__fit__ = True
    y2 = f2.y()
    out = ~((mp['qxy'] >= 1.42) & (mp['qxy'] <= 1.52) &
            (mp['qz'] >= 0.05) & (mp['qz'] <= 0.60))
    check('windowed-out rows have exactly zero residual',
          bool(np.all(y2[out] == mp['I'][out])),
          '%d of %d rows neutralised' % (int(out.sum()), mp['n']))
    check('windowed-in rows are NOT just the data',
          not np.allclose(y2[~out], mp['I'][~out]),
          '%d active rows' % int((~out).sum()))


# ------------------------------------------- 7. outputs and scalers --------
def test_outputs():
    print('\n[7] output_params reductions and scalers')
    if not os.path.exists(PACKED):
        check('packed map present', False, 'SKIPPED')
        return
    mp = GC.load_map(PACKED)
    x = np.arange(mp['n'], dtype=float)
    f = GID2D(x=x, mapfile=PACKED, mode='map2d', qxymin=1.40, qxymax=1.56,
              qzmin=0.02, qzmax=0.90, yscale=100.0, bkg0=40.0, **NN)
    f.__fit__ = False
    f.y()
    op = f.output_params
    s = op['scaler_parameters']
    for k in ('inplane_data', 'inplane_fit', 'residual_inplane',
              'inplane_model_fine'):
        check('output_params has %s' % k, k in op)
    rods = sorted(k for k in op if k.startswith('rod_') and k.endswith('_data'))
    check('one rod output per family', len(rods) == 3, str(rods))
    check('band narrower than half the peak separation',
          s['band_over_half_separation'] < 1.0,
          '%.3f (half-width %.5f, separation %.5f)'
          % (s['band_over_half_separation'], s['rod_band_halfwidth'],
             s['peak_separation']))
    check('A/molecule round-trips through the peak inversion',
          abs(s['A_per_molecule'] - s['A_per_molecule_from_peaks']) < 1e-6,
          '%.6f vs %.6f' % (s['A_per_molecule'], s['A_per_molecule_from_peaks']))
    check('both chi2/N are reported (labelled LINEAR)',
          'chi2_per_N_reported_LINEAR' in s and 'chi2_per_N_active_only_LINEAR' in s,
          'raw %.4g  active-only %.4g'
          % (s['chi2_per_N_reported_LINEAR'], s['chi2_per_N_active_only_LINEAR']))
    # the weighting alarm: this data has multiplicative errors, so Linear scale
    # concentrates essentially all the weight on a few near-zero-intensity pixels
    check('weight concentration reported for both scales',
          'weight_frac_top_10_px_LINEAR' in s and 'weight_frac_top_10_px_LOG' in s,
          'Linear top-10 %.1f%%  Log top-10 %.2f%%  (median sigma/I %.3f)'
          % (100 * s['weight_frac_top_10_px_LINEAR'],
             100 * s['weight_frac_top_10_px_LOG'], s['median_sigma_over_I']))
    check('Log scale is recommended on this data',
          'RECOMMENDED_fit_scale' in s,
          s.get('RECOMMENDED_fit_scale', s.get('WARNING_weighting', 'neither'))[:96])
    check('identifiability report present',
          any(k.startswith('varinflation_') for k in s) or
          'identifiability' in s or 'fisher_condition_number' in s,
          str(s.get('identifiability', 'fisher cond = %.3g'
                    % s.get('fisher_condition_number', float('nan')))))
    check('self-arming flag set after y()', f.__fit__ is True)

    # psi must be reported as unidentifiable at t = 0, not silently fitted
    f0 = mk(mode='inplane', t=0.0, qzmin=0.0, qzmax=1.4)
    f0.__fit__ = False
    f0.y()
    check("psi reported unidentifiable at t=0",
          'no' in f0.output_params['scaler_parameters']['psi_identifiable'],
          f0.output_params['scaler_parameters']['psi_identifiable'])


# ---------------------------------------------- 8. tilt-spread behaviour ---
def test_tilt_spread():
    """sig_t must broaden the off-plane rods while leaving q_z(02) = 0 sharp --
    that asymmetry is the signature that separates a tilt distribution from two
    coexisting populations, and it is why sig_t has to be in the model."""
    print('\n[8] sig_t broadens off-plane rods, not the in-plane one')
    qz = np.linspace(0.0, 1.0, 401)

    def rod_at(q0, sig_t, n_tilt):
        f = mk(mode='rod', sig_t=sig_t, n_tilt=n_tilt, vineyard=False,
               bkg_trans=False, qxymin=q0 - 0.012, qxymax=q0 + 0.012, nq=201)
        f.update_parameters()
        return f._reduce_rod(qz, q0 - 0.012, q0 + 0.012)

    def width(y):
        c = float(np.trapz(y * qz, qz) / np.trapz(y, qz))
        return float(np.sqrt(np.trapz(y * (qz - c) ** 2, qz) / np.trapz(y, qz)))

    w0_off, w0_in = width(rod_at(1.4637, 0.0, 1)), width(rod_at(1.4885, 0.0, 1))
    w5_off, w5_in = width(rod_at(1.4637, 5.0, 9)), width(rod_at(1.4885, 5.0, 9))
    check('sig_t=5 broadens the off-plane (11) rod', w5_off > 1.05 * w0_off,
          'sigma_qz %.4f -> %.4f' % (w0_off, w5_off))
    check('sig_t=5 leaves the in-plane (02) rod much less affected',
          (w5_in / w0_in - 1) < 0.5 * (w5_off / w0_off - 1),
          '(02) %+.1f%% vs (11) %+.1f%%'
          % (100 * (w5_in / w0_in - 1), 100 * (w5_off / w0_off - 1)))

    # n_tilt=1 must collapse to the single-tilt model EXACTLY
    a = rod_at(1.4637, 0.0, 7)
    b = rod_at(1.4637, 0.0, 1)
    check('sig_t=0 collapses regardless of n_tilt',
          float(np.max(np.abs(a - b))) == 0.0)


if __name__ == '__main__':
    test_core()
    test_sum_rule()
    test_reduction_consistency()
    test_quadrature()
    test_rod_lipid_agreement()
    test_pack_and_map()
    test_outputs()
    test_tilt_spread()
    sys.exit(finish())
