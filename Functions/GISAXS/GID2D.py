####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
####Please do not remove lines above####

####Import your modules below if needed####
import gid2d_core as GC
from gid2d_core import (Lattice2D, transmission, radial_profile, rod_profile,
                        qz_nodes, tilt_quadrature, load_map, row_lookup,
                        sigma_size, cell_from_peaks_oblique, tilt_from_peaks_nn,
                        trapz, TAU)


class GID2D:  # Please put the class name same as the function name
    # The engine may hand y() only the ACTIVE rows. Safe here: map mode resolves x
    # through row_lookup (nearest-neighbour + clip), and the 1-D reductions
    # integrate over grids they build themselves -- self.x is never used as a
    # quadrature axis. See Fit.evaluate for the general safety taxonomy.
    __masked_x__ = True

    def __init__(self, x=0, mapfile='', mode='map2d',
                 a0=4.95, t=30.5, psi=0.0, eps=0.041,
                 L_dom=400.0, L_ch=35.0, sig_t=0.0, sig_z=0.0, eta=0.5,
                 E=15.126, alpha_i=0.10, qc=0.0217,
                 qxyoff=0.0, qzoff=0.0,
                 yscale=100.0, bkg0=40.0, bkg1=0.0, bkg2=0.0, bkg3=0.0,
                 arc_amp=0.0, arc_Q0=1.50, arc_sig=0.10,
                 qxymin=0.0, qxymax=100.0, qzmin=0.0, qzmax=100.0,
                 lineshape='Gaussian', powder=True, vineyard=True,
                 bkg_trans=True, arc=False, dwba='kinematic', fisher=True,
                 window='model', n_tilt=1, nq=401, mpar={}):
        """
        Fits a full 2D (q_xy, q_z) grazing-incidence-diffraction map from a
        monolayer/thin-film chain lattice with ONE physical model, and emits the
        1D in-plane Bragg peak and the Bragg rods as derived outputs rather than
        as independent fits.

        x         : row index into mapfile (mode='map2d'), q_xy (mode='inplane')
                    or q_z (mode='rod'). See 'mode'.
        mapfile   : indexed 2D map, columns 'idx I Ierr qxy qz'. Produce it with
                    gid2d_core.pack_map(). Required for mode='map2d'.
        mode      : 'map2d'   -- x is a float row index; the residual IS the 2D map
                    'inplane' -- x is q_xy; model = integral over [qzmin,qzmax]
                    'rod'     -- x is q_z; model = integral over [qxymin,qxymax]
        a0        : nearest-neighbour spacing in the plane PERPENDICULAR to the
                    chains, Angstrom
        t         : chain tilt from the surface normal, degrees
        psi       : tilt azimuth from the a axis, degrees. EXACTLY a no-op at
                    t = 0 and degenerate at small t -- see scaler_parameters,
                    which reports 'psi_identifiable'.
        eps       : uniaxial strain along a, standing in for backbone-plane
                    ordering. With eps = 0 the perpendicular net is exactly
                    hexagonal for any t, so eps is cleanly separated from tilt.
        L_dom     : in-plane coherence length, Angstrom (sets the radial width)
        L_ch      : coherently scattering chain length, Angstrom (sets rod width)
        sig_t     : r.m.s. spread in chain tilt, degrees. Raise n_tilt with it.
        sig_z     : vertical r.m.s. displacement, Angstrom (Debye-Waller in q_z)
        eta       : Lorentzian fraction, used only when lineshape='PseudoVoigt'
        E         : photon energy, keV
        alpha_i   : incident angle, degrees. Imposes a rigid q_z offset k sin(ai)
        qc        : critical TOTAL momentum transfer (single-side value is qc/2)
        qxyoff    : instrument q_xy zero offset, Angstrom^-1
        qzoff     : instrument q_z zero offset, Angstrom^-1
        yscale    : overall scale. Defaults to a value of the same ORDER as the
                    DPPC example map rather than 1.0 -- with yscale=1 and bkg0=0
                    the model is ~1e-9 against data of ~50, so the model panel of
                    the 2D view opens completely black and the first fit starts
                    far outside the basin. The rest of the defaults are likewise
                    this dataset's starting values.
        bkg0..3   : 2D background, bkg0 + bkg1*q_xy + bkg2*q_z + bkg3*q_z^2.
                    bkg0 defaults to the DPPC map's approximate mean background.
        arc_amp   : amplitude of a broad amorphous arc at constant |Q|
        arc_Q0    : |Q| centre of the arc, Angstrom^-1
        arc_sig   : |Q| width of the arc, Angstrom^-1
        qxymin/max: q_xy bounds, Angstrom^-1 (CONFIGURATION, keep vary=0). TWO
                    jobs: the map-mode fit window when window='model', AND the
                    q_xy integration band for mode='rod'. The second job is not
                    optional, so these parameters stay even when the windowing
                    moves into the engine.
        qzmin/max : q_z bounds, Angstrom^-1 (CONFIGURATION, keep vary=0). Likewise
                    two jobs: the map-mode fit window when window='model', AND the
                    q_z quadrature limits for mode='inplane' (see _qz_grid).
        lineshape : 'Gaussian' (finite domain), 'Lorentzian' (finite positional
                    correlation length) or 'PseudoVoigt'
        powder    : True applies the analytic 2D-powder ring-average Lorentz factor
        vineyard  : True applies the exit-side transmission factor |T_f|^2
        bkg_trans : True multiplies the polynomial background by |T_f|^2 too
        arc       : True enables the amorphous arc term
        dwba      : 'kinematic' (|T_f|^2 |V|^2). 'dwba' is an unimplemented stub.
        window    : how the map-mode fit window is enforced.
                    'model'  -- inside y(): rows outside the q window return the
                       DATA value so their residual is zero. Self-contained, but
                       the residual vector is padded, so the reduced chi^2 lmfit
                       reports is optimistic (both numbers are in
                       scaler_parameters). This is the default: it is what earlier
                       .par files assume.
                    'engine' -- y() returns the raw model everywhere and the fit
                       engine drops the excluded rows, giving an HONEST dof. This
                       needs the window set as engine column limits (the Limits
                       button, on the qxy/qz columns) -- with 'engine' selected and
                       no column limits set, EVERY row is fitted, including the
                       sub-critical ones.
        fisher    : True adds a cheap identifiability report to scaler_parameters
        n_tilt    : Gauss-Hermite nodes for the sig_t average. 1 = single tilt.
        nq        : quadrature nodes for the 1D reductions

        Conventions, all enforced in gid2d_core (run its selfcheck after edits):
        q_z is SIGNED and summed over the SIX first-order reflections, evaluated
        only on q_z >= 0 -- never folded. Areas use |b1 x b2|. The Vineyard
        factor matches Rod_Lipid.trans(qbeta, qc/2) so the two agree on the same
        data. The rod carries the cos(t) factor that Rod_Lipid omits, because
        here the same t also sets the peak positions.
        """
        if type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        # An empty default would leave the Fixed-Parameters file button with no
        # path, and mode='map2d' would then raise on the very first update_plot().
        # Fall back to the packed example map if it is present, the same way
        # XLayersPDB ships with a default PDB.
        if not mapfile:
            _d = os.path.join(os.path.abspath(os.curdir), 'Data',
                              'dppc_q2d_gid_indexed.txt')
            mapfile = _d if os.path.exists(_d) else ''
        self.mapfile = mapfile
        self.mode = mode
        self.a0, self.t, self.psi, self.eps = a0, t, psi, eps
        self.L_dom, self.L_ch, self.sig_t, self.sig_z = L_dom, L_ch, sig_t, sig_z
        self.eta = eta
        self.E, self.alpha_i, self.qc = E, alpha_i, qc
        self.qxyoff, self.qzoff = qxyoff, qzoff
        self.yscale = yscale
        self.bkg0, self.bkg1, self.bkg2, self.bkg3 = bkg0, bkg1, bkg2, bkg3
        self.arc_amp, self.arc_Q0, self.arc_sig = arc_amp, arc_Q0, arc_sig
        self.qxymin, self.qxymax = qxymin, qxymax
        self.qzmin, self.qzmax = qzmin, qzmax
        self.lineshape, self.powder, self.vineyard = lineshape, powder, vineyard
        self.bkg_trans, self.arc, self.dwba, self.fisher = bkg_trans, arc, dwba, fisher
        self.window = window
        self.n_tilt, self.nq = n_tilt, nq
        self.__mpar__ = mpar
        self.choices = {'mode': ['map2d', 'inplane', 'rod'],
                        'lineshape': ['Gaussian', 'Lorentzian', 'PseudoVoigt'],
                        'powder': [True, False],
                        'vineyard': [True, False],
                        'bkg_trans': [True, False],
                        'arc': [True, False],
                        'dwba': ['kinematic', 'dwba'],
                        'fisher': [True, False],
                        'window': ['model', 'engine']}
        self.filepaths = {'mapfile': self.mapfile}
        # Diagnostic hook: restrict _peak to a set of |(h,k)| families, e.g.
        # {(1,1)} or {(0,2)}. Used by validate_gid2d.py to check the integrated
        # (11)+(1-1):(02) = 2:1 sum rule family by family. None = all six.
        self.__hk_filter__ = None
        self.__fit__ = False
        self.output_params = {'scaler_parameters': {}}
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
        self.params = Parameters()
        p = self.params.add
        # -- lattice: these are the quantities the GID explorer takes as input
        p('a0', value=self.a0, vary=0, min=1.0, max=20.0, expr=None, brute_step=0.01)
        p('t', value=self.t, vary=0, min=0.0, max=60.0, expr=None, brute_step=0.1)
        p('psi', value=self.psi, vary=0, min=-90.0, max=90.0, expr=None, brute_step=0.5)
        p('eps', value=self.eps, vary=0, min=-0.5, max=0.5, expr=None, brute_step=0.001)
        # -- lineshape widths
        p('L_dom', value=self.L_dom, vary=0, min=5.0, max=1e5, expr=None, brute_step=1.0)
        p('L_ch', value=self.L_ch, vary=0, min=1.0, max=1e4, expr=None, brute_step=0.5)
        p('sig_t', value=self.sig_t, vary=0, min=0.0, max=30.0, expr=None, brute_step=0.1)
        p('sig_z', value=self.sig_z, vary=0, min=0.0, max=20.0, expr=None, brute_step=0.05)
        p('eta', value=self.eta, vary=0, min=0.0, max=1.0, expr=None, brute_step=0.02)
        # -- optics. alpha_i and E are normally fixed from the beamline log.
        p('E', value=self.E, vary=0, min=1.0, max=100.0, expr=None, brute_step=0.01)
        p('alpha_i', value=self.alpha_i, vary=0, min=0.0, max=5.0, expr=None, brute_step=0.005)
        p('qc', value=self.qc, vary=0, min=0.0, max=0.2, expr=None, brute_step=0.0005)
        # -- instrument zero offsets
        p('qxyoff', value=self.qxyoff, vary=0, min=-0.1, max=0.1, expr=None, brute_step=0.0005)
        p('qzoff', value=self.qzoff, vary=0, min=-0.1, max=0.1, expr=None, brute_step=0.0005)
        # -- scale and background
        p('yscale', value=self.yscale, vary=0, min=0.0, max=np.inf, expr=None, brute_step=0.1)
        p('bkg0', value=self.bkg0, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        p('bkg1', value=self.bkg1, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        p('bkg2', value=self.bkg2, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        p('bkg3', value=self.bkg3, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        p('arc_amp', value=self.arc_amp, vary=0, min=0.0, max=np.inf, expr=None, brute_step=0.1)
        p('arc_Q0', value=self.arc_Q0, vary=0, min=0.1, max=10.0, expr=None, brute_step=0.005)
        p('arc_sig', value=self.arc_sig, vary=0, min=1e-3, max=2.0, expr=None, brute_step=0.005)
        # -- fit window. CONFIGURATION, not physics: a window edge enters the
        #    objective discontinuously, so its derivative is a delta function and
        #    TRF/LM cannot move it; a floating edge also trades directly against
        #    yscale and background. The PEAK POSITIONS are the parameters, and
        #    they move continuously. Keep these vary=0.
        p('qxymin', value=self.qxymin, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.01)
        p('qxymax', value=self.qxymax, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.01)
        p('qzmin', value=self.qzmin, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.01)
        p('qzmax', value=self.qzmax, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.01)

    def update_parameters(self):
        for key in self.params.keys():
            setattr(self, key, self.params[key].value)

    # ------------------------------------------------------------ geometry --
    def _lattices(self):
        """(weight, tilt_deg, Lattice2D) per tilt-quadrature node.

        q_z0 is RECOMPUTED per node, which is the whole point of sig_t: a spread
        in t smears off-plane reflections along the envelope q_z = q_xy tan t
        while leaving q_z = 0 reflections sharp, because q_z(02) = 0
        independently of t under NN tilt. That is the signature that
        distinguishes a tilt distribution from two coexisting populations.
        """
        n = 1 if abs(self.sig_t) < 1e-9 else max(1, int(self.n_tilt))
        xs, ws = tilt_quadrature(n)
        out = []
        for xj, wj in zip(xs, ws):
            tj = float(self.t + xj * self.sig_t)
            out.append((float(wj), tj, Lattice2D(self.a0, tj, self.psi, self.eps)))
        return out

    def _rod_lobe(self):
        """sinc^2 lobe width in q_z, 2 pi / (L_ch cos t). Sets dqz_max so the
        graded q_z grid never gets coarser than the rod it has to resolve."""
        return TAU / max(self.L_ch * np.cos(np.radians(self.t)), 1e-9)

    # ----------------------------------------------------------- intensity --
    def _peak(self, qxy, qz):
        """Bragg term only: signed-q_z sum over the SIX first-order points,
        tilt-averaged. No background, no scale, no q_z >= 0 gate."""
        out = np.zeros(np.broadcast(qxy, qz).shape, float)
        keep = self.__hk_filter__
        for wj, tj, lat in self._lattices():
            trad = np.radians(tj)
            for R in lat.reflections(six=True):
                if keep is not None and (abs(R['hk'][0]), abs(R['hk'][1])) not in keep:
                    continue
                out = out + wj * (
                    radial_profile(qxy, R['qxy'], self.L_dom,
                                   lineshape=self.lineshape, eta=self.eta,
                                   powder=self.powder)
                    * rod_profile(qz, R['qz'], trad, self.L_ch))
        return out

    def _model(self, qxy, qz):
        """Full intensity on (q_xy, q_z), evaluated only on q_z >= 0.

        I = yscale |T_f|^2 SUM_six W_hk(q_xy) F_rod(q_z - q_z0_hk) exp(-q_z^2 sig_z^2)
            + background
        """
        if self.dwba != 'kinematic':
            raise NotImplementedError(
                "dwba='%s' is a stub. Only the kinematic model |T_f|^2 |V|^2 is "
                "implemented; the full 4-channel DWBA buys accuracy only near "
                "alpha_f ~ alpha_c." % (self.dwba,))
        qxy = np.asarray(qxy, float)
        qz = np.asarray(qz, float)
        tr = transmission(qz, self.E, self.alpha_i, self.qc) if self.vineyard \
            else np.ones(np.broadcast(qxy, qz).shape, float)

        res = self._peak(qxy, qz) * np.exp(-(qz * self.sig_z) ** 2) * tr
        res = self.yscale * res

        bkg = self.bkg0 + self.bkg1 * qxy + self.bkg2 * qz + self.bkg3 * qz ** 2
        res = res + (bkg * tr if self.bkg_trans else bkg)
        if self.arc:
            Q = np.sqrt(qxy ** 2 + qz ** 2)
            res = res + self.arc_amp * np.exp(
                -(Q - self.arc_Q0) ** 2 / (2.0 * max(self.arc_sig, 1e-6) ** 2))
        # never emit intensity below the horizon; |T_f|^2 already does this when
        # vineyard=True, this makes it unconditional
        return np.where(np.broadcast_to(qz, res.shape) >= 0.0, res, 0.0)

    # ---------------------------------------------------------- reductions --
    def _qz_grid(self, lo, hi, n=None):
        return qz_nodes(lo, hi, self.E, self.alpha_i, self.qc,
                        n=n or self.nq, dqz_max=self._rod_lobe() / 8.0)

    def _reduce_inplane(self, qxy_vals, z=None):
        """Integral over the q_z window -> in-plane profile vs q_xy."""
        if z is None:
            z = self._qz_grid(self.qzmin, self.qzmax)
        if len(z) < 2:
            return np.zeros(len(qxy_vals))
        qxy_vals = np.asarray(qxy_vals, float)
        return trapz(self._model(qxy_vals[:, None], z[None, :]), z, axis=1)

    def _reduce_rod(self, qz_vals, lo, hi, n=None):
        """Integral over a q_xy band -> rod profile vs q_z."""
        xg = np.linspace(lo, hi, n or self.nq)
        if len(xg) < 2 or hi <= lo:
            return np.zeros(len(qz_vals))
        qz_vals = np.asarray(qz_vals, float)
        return trapz(self._model(xg[:, None], qz_vals[None, :]), xg, axis=0)

    def _rod_bands(self):
        """Non-overlapping q_xy bands, one per first-order family.

        Band-width rule, enforced, from references/pitfalls.md: bin in |q_xy| and
        keep the band NARROWER THAN HALF the peak separation. The recorded
        failures were a 2-theta window wider than the separation (ratio 1.00), a
        band wider than half the separation (0.94) and a fixed-column
        integration of a curving feature (0.65). Only a |q_xy| band narrower
        than half the separation returned 1.98.
        """
        lat = Lattice2D(self.a0, self.t, self.psi, self.eps)
        cen = {}
        for R in lat.reflections(six=False):
            cen[R['hk']] = R['qxy']
        qs = sorted(set(round(v, 10) for v in cen.values()))
        sep = min(np.diff(qs)) if len(qs) > 1 else np.inf
        # width from the radial lineshape, then capped well inside half-separation
        half = min(3.0 * sigma_size(self.L_dom), 0.4 * sep) if np.isfinite(sep) \
            else 3.0 * sigma_size(self.L_dom)
        assert not np.isfinite(sep) or half < 0.5 * sep, (
            "rod band half-width %.5f is not narrower than half the peak "
            "separation %.5f -- see the band-width rule" % (half, 0.5 * sep))
        return {hk: (q - half, q + half) for hk, q in cen.items()}, half, sep

    # ------------------------------------------------------- data reduction --
    @staticmethod
    def _group_integrate(key, val, wgt):
        """Integrate val over wgt within each distinct key. Returns (keys, ints).
        Used for the DATA reductions, so they use exactly the pixels the fit
        used -- reductions of the model are taken on the same nodes, so a
        reduction can never disagree with the fit."""
        ks = np.unique(key)
        out = np.zeros(len(ks))
        for i, kk in enumerate(ks):
            m = key == kk
            if m.sum() < 2:
                out[i] = np.nan
                continue
            o = np.argsort(wgt[m])
            out[i] = trapz(val[m][o], wgt[m][o])
        return ks, out

    # ---------------------------------------------------------------- main --
    def y(self):
        """
        Define the function in terms of x to return some value
        """
        self.update_parameters()
        # Self-arming: xmodfit sets func.__fit__ = False on every explicit
        # Calculate/plot path and never sets it True, while the fit loop does
        # neither. Setting it True at the end of y() therefore computes the
        # expensive extras once per Calculate and skips them for every residual
        # evaluation inside a fit.
        #
        # ...but that alone is NOT enough, and getting it wrong produced an empty
        # 2D map. update_plot calls evaluate() BEFORE set_x() (xmodfit.py:6207),
        # so the first evaluation runs on the x LINE-EDIT expression, and only the
        # second one -- via residual() after set_x -- sees the real dataset. Other
        # functions never self-arm, so they simply report twice and the second
        # (correct) result wins. Here the flag suppressed the second one, leaving
        # outputs built from a q-shaped default x that, read as row indices,
        # collapses to about two pixels of the map.
        #
        # So also report whenever x has CHANGED since the last report: that catches
        # update_plot's evaluate -> set_x -> residual sequence exactly, while a fit
        # loop (constant x) still reports only once.
        _x = np.atleast_1d(np.asarray(self.x, float))
        _xsig = (len(_x), float(_x[0]), float(_x[-1])) if len(_x) else (0, 0.0, 0.0)
        report = (not self.__fit__) or (_xsig != getattr(self, '_last_xsig', None))
        self._last_xsig = _xsig

        # xmodfit coerces every Fixed-Parameters cell with eval() before falling
        # back to str() (update_plot / fixedParamChanged), so a choice string that
        # happens to be a Python name comes back as that object -- 'map' arrived
        # as the builtin <class 'map'>. The mode values are chosen to avoid that
        # ('map2d', not 'map'), and this normalises whatever does arrive.
        mode = getattr(self.mode, '__name__', None) or str(self.mode)
        for _k in ('lineshape', 'dwba'):
            _v = getattr(self, _k)
            if not isinstance(_v, str):
                setattr(self, _k, getattr(_v, '__name__', None) or str(_v))

        if mode == 'inplane':
            qxy = np.asarray(self.x, float) + self.qxyoff
            res = self._reduce_inplane(qxy)
            if report:
                self._drop_map_images()
                self._scalers(n_active=len(qxy))
            self.__fit__ = True
            return res

        if mode == 'rod':
            qz = np.asarray(self.x, float) + self.qzoff
            bands, half, sep = self._rod_bands()
            lo = self.qxymin if np.isfinite(self.qxymin) else min(b[0] for b in bands.values())
            hi = self.qxymax if np.isfinite(self.qxymax) else max(b[1] for b in bands.values())
            res = self._reduce_rod(qz, lo, hi)
            if report:
                self._drop_map_images()
                self._scalers(n_active=len(qz), extra={'rod_band_qxy': '%.4f:%.4f' % (lo, hi)})
            self.__fit__ = True
            return res

        # ---------------------------------------------------------- map mode
        if not self.mapfile or not os.path.exists(str(self.mapfile)):
            msg = ("mode='map2d' needs a readable 'mapfile': click the mapfile "
                   "button in Fixed Parameters, or build one with "
                   "gid2d_core.pack_map(raw, indexed). Got %r" % (self.mapfile,))
            if not report:
                # Inside a fit, fail loudly rather than fit against zeros.
                raise IOError(msg)
            # On Calculate/plot, return zeros and say so in the parameter panel,
            # so selecting the function never throws a dialog at load time.
            self._drop_map_images()
            self._scalers(n_active=0, extra={'ERROR_mapfile': msg})
            self.__fit__ = True
            return np.zeros(np.shape(np.asarray(self.x, float)))
        mp = load_map(self.mapfile)
        rows = row_lookup(self.x, mp)
        qxy = mp['qxy'][rows] + self.qxyoff
        qz = mp['qz'][rows] + self.qzoff
        dat = mp['I'][rows]
        err = mp['Ierr'][rows]

        # Keep the RAW model: the line below overwrites it with the data outside the
        # window, so a model image built from `res` would show data in the wings and
        # a residual image would show an identically-zero (fake-perfect) region.
        # Seeing what the model does OUTSIDE the fit window is the main reason to
        # look at a 2D map, and capturing it here is free.
        mdl = self._model(qxy, qz)

        # Window as CONFIGURATION. XModFit cannot mask: Fit_Routines.residual
        # computes (data-fit)/err outside the model, so the only way to neutralise
        # a row from inside y() is to return the data value there, making its
        # residual identically zero. sigma <= 0 rows have no in-model escape at
        # all and are dropped by pack_map instead.
        active = ((qxy >= self.qxymin) & (qxy <= self.qxymax) &
                  (qz >= self.qzmin) & (qz <= self.qzmax) & (err > 0))
        if str(self.window) == 'engine':
            # the engine drops the excluded rows, so return the raw model and let
            # the reported dof be honest
            res = mdl
            active = err > 0
        else:
            res = np.where(active, mdl, dat)

        if report:
            self._map_outputs(mp, rows, qxy, qz, dat, err, res, active, mdl=mdl)
        self.__fit__ = True
        return res

    # -------------------------------------------------------- 2D map images --
    MAP_IMAGE_KEYS = ('map2d_data', 'map2d_model', 'map2d_residual')

    def _drop_map_images(self):
        """Remove stale 2D images when this evaluation cannot produce them.

        output_params is built once in __init__ and never reset, and Fit.params IS
        func.__dict__ (Fit_Routines.py:16), so switching mode away from 'map2d'
        would otherwise leave the previous map on screen indefinitely.
        """
        for k in self.MAP_IMAGE_KEYS:
            self.output_params.pop(k, None)

    def _coverage_message(self, mp, rows, cov):
        """Diagnose an under-filled map, naming the actual cause.

        The most likely cause is not a too-coarse x expression but the WRONG FILE
        loaded as the dataset. In map2d mode x must be a ROW INDEX into mapfile, so
        the dataset has to be the indexed file that pack_map writes. Loading the raw
        (q_xy, q_z, I, Ierr) file instead makes Data_Dialog pick q_xy as x, and
        row_lookup then rounds every one of those q values into rows 0-2: 41797
        points collapse onto 2 pixels of a 39739-cell grid, so the panel looks
        blank. Worse, the residual is then comparing the model against the
        dataset's SECOND column (q_z), so the fit is meaningless too -- which is
        why this reads as an error and not a cosmetic note.
        """
        if cov >= 0.5:
            return ''
        xa = np.atleast_1d(np.asarray(self.x, float))
        n_hit = len(np.unique(np.asarray(rows)))
        frac_distinct = n_hit / float(max(1, len(xa)))
        n = int(mp['n'])
        if len(xa) > 100 and frac_distinct < 0.5:
            return ('x IS NOT A ROW INDEX into mapfile: %d x-points land on only '
                    '%d distinct rows (x runs %.4g..%.4g, but row indices run '
                    '0..%d). You have almost certainly loaded the RAW '
                    '(q_xy, q_z, I, Ierr) file as the dataset, so its q_xy column '
                    'became x -- the fit is comparing the model against q_z and is '
                    'meaningless. Load the INDEXED file instead; build one with:  '
                    'python -c "import sys; sys.path.insert(0,\'Functions\'); '
                    'import gid2d_core as G; '
                    'G.pack_map(r\'<raw file>\', r\'Data/<name>_indexed.txt\')"'
                    % (len(xa), n_hit, xa.min(), xa.max(), n - 1))
        return ('x covers only %.1f%% of the grid; set x to np.arange(0.,%d.) '
                '(or load the indexed file as the dataset) to see the whole map'
                % (100 * cov, n))

    def _map_images(self, mp, rows, dat, err, res, active, mdl):
        """Scatter the per-row vectors onto the (n_qxy, n_qz) MEASUREMENT grid.

        A bare reshape is invalid: pack_map DROPS masked rows (38992 of 41797 in the
        DPPC file), so mp['rectangular'] is False and idx != i_qz*n_qxy + i_qxy --
        exactly the arithmetic load_map's docstring already warns against. Index
        lookup on uqxy/uqz is exact instead, because those come from np.unique of
        these very floats.

        NaN marks "no measured pixel" and renders fully transparent in the viewer,
        so holes read as background rather than as the bottom of the colour scale.
        """
        ux, uz = mp['uqxy'], mp['uqz']
        # index with the UNOFFSET q, since ux/uz are the file's own values
        ix = np.clip(np.searchsorted(ux, mp['qxy'][rows]), 0, len(ux) - 1)
        iz = np.clip(np.searchsorted(uz, mp['qz'][rows]), 0, len(uz) - 1)
        shape = (len(ux), len(uz))

        def grid(v, where=None):
            Z = np.full(shape, np.nan)
            v = np.asarray(v, float)
            if where is None:
                Z[ix, iz] = v
            else:
                Z[ix[where], iz[where]] = v[where]
            return Z

        with np.errstate(divide='ignore', invalid='ignore'):
            r = (dat - res) / np.where(err > 0, err, np.nan)

        ax = dict(x=ux + self.qxyoff, y=uz + self.qzoff,
                  plotType='image', panel='map2d')
        op = self.output_params
        op['map2d_data'] = dict(ax, z=grid(dat), role='data', cmap='CET-L3',
                               zscale='log',
                               names=['q_xy (1/Angs)', 'q_z (1/Angs)', 'I data'])
        op['map2d_model'] = dict(ax, z=grid(mdl), role='model', cmap='CET-L3',
                                 zscale='log',
                                 names=['q_xy (1/Angs)', 'q_z (1/Angs)', 'I model'])
        # Residual masked to `active`: windowed-out pixels satisfy res == dat by
        # construction, so plotting them would paint an identically-zero
        # "perfect fit" region. NaN there instead makes the fit window VISIBLE.
        op['map2d_residual'] = dict(ax, z=grid(r, where=active), role='residual',
                                    cmap='CET-D1A', zscale='signed',
                                    names=['q_xy (1/Angs)', 'q_z (1/Angs)',
                                           '(data-model)/sigma'])
        return float(np.isfinite(op['map2d_data']['z']).sum()) / float(shape[0] * shape[1])

    # ------------------------------------------------------------- outputs --
    def _map_outputs(self, mp, rows, qxy, qz, dat, err, res, active, mdl=None):
        """1D reductions of the SAME model and SAME data over identical windows,
        plus the 2D images."""
        op = self.output_params
        m = active
        cov = self._map_images(mp, rows, dat, err, res, active,
                              mdl if mdl is not None else res)
        if m.sum() < 4:
            self._scalers(n_active=int(m.sum()),
                          extra={'WARNING': 'fewer than 4 active pixels',
                                 'map_coverage_frac': cov})
            return

        qxy_a, qz_a, dat_a, res_a = qxy[m], qz[m], dat[m], res[m]

        # -- in-plane peak: integrate over the q_z window at each q_xy column
        kq, ip_dat = self._group_integrate(qxy_a, dat_a, qz_a)
        _, ip_fit = self._group_integrate(qxy_a, res_a, qz_a)
        good = np.isfinite(ip_dat) & np.isfinite(ip_fit)
        op['inplane_data'] = {'x': kq[good], 'y': ip_dat[good],
                              'names': ['qxy (1/Angs)', 'I integrated over qz']}
        op['inplane_fit'] = {'x': kq[good], 'y': ip_fit[good],
                             'names': ['qxy (1/Angs)', 'I integrated over qz']}
        with np.errstate(divide='ignore', invalid='ignore'):
            op['residual_inplane'] = {
                'x': kq[good],
                'y': (ip_dat[good] - ip_fit[good]) / np.where(ip_dat[good] != 0,
                                                              np.abs(ip_dat[good]), np.nan),
                'names': ['qxy (1/Angs)', '(data-fit)/|data|']}

        # -- Bragg rods: integrate over a q_xy band around each |G_hk|
        bands, half, sep = self._rod_bands()
        for hk, (lo, hi) in sorted(bands.items()):
            b = (qxy_a >= lo) & (qxy_a <= hi)
            if b.sum() < 4:
                continue
            kz, rd = self._group_integrate(qz_a[b], dat_a[b], qxy_a[b])
            _, rf = self._group_integrate(qz_a[b], res_a[b], qxy_a[b])
            g = np.isfinite(rd) & np.isfinite(rf)
            tag = '%d%d' % (hk[0], hk[1]) if hk[1] >= 0 else '%d-%d' % (hk[0], -hk[1])
            op['rod_%s_data' % tag] = {'x': kz[g], 'y': rd[g],
                                       'names': ['qz (1/Angs)', 'I integrated over qxy band']}
            op['rod_%s_fit' % tag] = {'x': kz[g], 'y': rf[g],
                                      'names': ['qz (1/Angs)', 'I integrated over qxy band']}

        # -- pure-model reduction on the graded grid, for the quadrature check
        z = self._qz_grid(self.qzmin, self.qzmax)
        if len(z) > 1:
            op['inplane_model_fine'] = {
                'x': kq[good], 'y': self._reduce_inplane(kq[good], z=z),
                'names': ['qxy (1/Angs)', 'model integrated over qz (graded grid)']}

        # -- chi^2. Report BOTH: lmfit sees the padded count because neutralised
        #    rows are still in the residual vector, so raw chi2/N is optimistic.
        with np.errstate(divide='ignore', invalid='ignore'):
            r_all = (dat - res) / np.where(err > 0, err, np.nan)
        n_all = np.count_nonzero(np.isfinite(r_all))
        chi2_raw = float(np.nansum(r_all ** 2) / max(n_all, 1))
        r_act = r_all[m]
        n_act = np.count_nonzero(np.isfinite(r_act))
        chi2_eff = float(np.nansum(r_act ** 2) / max(n_act, 1))

        r2 = np.where(np.isfinite(r_act), r_act, 0.0) ** 2
        tot = float(r2.sum())
        k = max(1, n_act // 1000)
        top = np.sort(r2)[::-1]
        self._scalers(n_active=int(m.sum()),
                      extra={'n_rows_passed': int(len(rows)),
                             'n_rows_in_file': int(mp['n']),
                             'map_coverage_frac': cov,
                             # x must span the whole file for the 2D panel to be
                             # complete. The default xLineEdit is a 300-point
                             # linspace, which fills 300 of ~39739 cells and makes
                             # the map look broken for a reason nothing on screen
                             # explains.
                             'WARNING_map_coverage': self._coverage_message(mp, rows, cov),
                             'chi2_per_N_reported_LINEAR': chi2_raw,
                             'chi2_per_N_active_only_LINEAR': chi2_eff,
                             'chi2_frac_top_0.1pct_px_LINEAR':
                                 float(top[:k].sum() / tot) if tot > 0 else 0.0,
                             'chi2_frac_top_10_px_LINEAR':
                                 float(top[:10].sum() / tot) if tot > 0 else 0.0,
                             'rod_band_halfwidth': float(half),
                             'peak_separation': float(sep),
                             'band_over_half_separation': float(half / (0.5 * sep))
                             if np.isfinite(sep) else float('nan')},
                      weights=(dat[m], err[m]),
                      jac_data=(qxy, qz, dat, err, m) if self.fisher else None)

    @staticmethod
    def _weight_report(dat, err):
        """How concentrated the fit's WEIGHT is, per fit_scale. Data-only, so it
        does not depend on the model or on which scale was actually used -- which
        is what makes it a usable alarm.

        Measured on the DPPC ROI (q_xy 1.40-1.54, q_z 0.02-0.60, 9875 pixels):
        sigma is MULTIPLICATIVE here, sigma/I median 0.41 and capped at 1.0, so
        Linear weight 1/sigma^2 diverges wherever I -> 0 below the critical
        angle. ONE pixel at (1.4018, 0.0841) held 72.2% of the total weight and
        the top ten held 99.1% -- and that is real data, not mask residue that
        pack_map's sigma floor could remove.

        XModFit's 'Log' residual is (log10 d - log10 f) * d/err, so its weight
        is (I/sigma)^2, which is FLAT for a multiplicative error model. Measured:
        top pixel 0.08%, top ten 0.66%. It also changes the answer -- L_dom
        144 -> 313 A, L_ch 9.4 -> 15.3 A, bkg0 -4.3 -> +53 -- so this is a
        correctness issue, not a cosmetic one. Fit this data in Log scale.
        """
        out = {}
        ok = err > 0
        if ok.sum() < 10:
            return out
        d, e = np.asarray(dat)[ok], np.asarray(err)[ok]
        rel = e / np.where(d != 0, np.abs(d), np.nan)
        out['median_sigma_over_I'] = float(np.nanmedian(rel))
        for tag, w in (('LINEAR', 1.0 / e ** 2), ('LOG', (d / e) ** 2)):
            w = w[np.isfinite(w)]
            if not len(w) or w.sum() <= 0:
                continue
            o = np.sort(w)[::-1]
            out['weight_frac_top_px_%s' % tag] = float(o[0] / w.sum())
            out['weight_frac_top_10_px_%s' % tag] = float(o[:10].sum() / w.sum())
        lin = out.get('weight_frac_top_10_px_LINEAR', 0.0)
        log = out.get('weight_frac_top_10_px_LOG', 0.0)
        if lin > 0.5 and log < 0.5 * lin:
            out['RECOMMENDED_fit_scale'] = (
                "Log -- errors are multiplicative (median sigma/I = %.2f), so in "
                "Linear scale the top 10 pixels carry %.0f%% of the weight "
                "(vs %.1f%% in Log) and the fit follows a handful of low-q_z "
                "pixels, not the map" % (out['median_sigma_over_I'],
                                         100 * lin, 100 * log))
        elif lin > 0.5:
            out['WARNING_weighting'] = (
                "top 10 pixels carry %.0f%% of the Linear weight; inspect the "
                "error model before trusting any parameter" % (100 * lin))
        return out

    def _scalers(self, n_active=0, extra=None, weights=None, jac_data=None):
        lat = Lattice2D(self.a0, self.t, self.psi, self.eps)
        R = {r['hk']: r for r in lat.reflections(six=True)}
        a_vec, b_vec = lat.A1, lat.A2
        gamma = np.degrees(np.arccos(
            float(a_vec @ b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec))))
        s = {'A_per_molecule': lat.area_per_molecule(),
             'a': float(np.linalg.norm(a_vec)),
             'b': float(np.linalg.norm(b_vec)),
             'gamma_deg': float(gamma),
             'b_over_a': float(np.linalg.norm(b_vec) / np.linalg.norm(a_vec)),
             'sqrt3_cos_t': float(np.sqrt(3.0) * np.cos(np.radians(self.t))),
             'n_active_pixels': int(n_active),
             'n_tilt_nodes': 1 if abs(self.sig_t) < 1e-9 else max(1, int(self.n_tilt))}
        for hk in lat.FAMILIES:
            tag = '%d%d' % (hk[0], hk[1]) if hk[1] >= 0 else '%d-%d' % (hk[0], -hk[1])
            sh = lat.sheet(*hk)
            s['qxy_%s' % tag] = R[hk]['qxy']
            s['qz_%s' % tag] = R[hk]['qz']          # SIGNED
            s['d_sheet_%s' % tag] = sh['d_sheet']
            s['chi_%s_deg' % tag] = float(np.degrees(sh['chi']))
        # inversion of the model's own peaks, as a round-trip consistency check
        try:
            inv = cell_from_peaks_oblique(R[(1, 1)]['qxy'], R[(1, -1)]['qxy'],
                                          R[(0, 2)]['qxy'])
            s['A_per_molecule_from_peaks'] = 2.0 * inv['area_per_chain']
            s['tilt_nn_from_peaks_deg'] = float(np.degrees(tilt_from_peaks_nn(
                R[(1, 1)]['qz'], R[(1, 1)]['qxy'], R[(0, 2)]['qxy'])))
        except Exception:
            pass
        # psi is EXACTLY a no-op at t = 0 (1/cos 0 - 1 = 0), so report it rather
        # than silently fitting a parameter that does nothing
        s['psi_identifiable'] = ('no (t~0: psi is exactly a no-op)'
                                 if abs(self.t) < 1e-6 else
                                 'weak (small t)' if abs(self.t) < 3.0 else 'yes')
        if abs(self.sig_t) > 1e-9 and int(self.n_tilt) < 3:
            s['WARNING_n_tilt'] = ('sig_t = %.3g with n_tilt = %d: the tilt '
                                   'average is unconverged, raise n_tilt to 5-7'
                                   % (self.sig_t, int(self.n_tilt)))
        if extra:
            s.update(extra)
        if weights is not None:
            s.update(self._weight_report(*weights))
        if jac_data is not None:
            s.update(self._identifiability(*jac_data))
        self.output_params['scaler_parameters'] = s

    def _identifiability(self, qxy, qz, dat, err, m):
        """Cheap Fisher report: condition number and per-parameter variance
        inflation from correlation. More useful on this problem than a nominal
        error bar -- L_dom and sig_t are expected to correlate strongly, and t
        and eps are identifiable only if the broad feature is accepted as the
        (11)/(1-1) family, which is a model choice rather than a measurement.
        """
        names = [k for k in self.params if self.params[k].vary]
        if not names or m.sum() < len(names) + 2:
            return {'identifiability': 'not computed (no varying parameters)'}
        keep = {k: getattr(self, k) for k in names}
        w = 1.0 / np.where(err[m] > 0, err[m], np.inf)
        try:
            J = np.zeros((int(m.sum()), len(names)))
            for j, k in enumerate(names):
                v0 = float(keep[k])
                h = max(abs(v0) * 1e-4, 1e-7)
                setattr(self, k, v0 + h)
                fp = self._model(qxy[m], qz[m])
                setattr(self, k, v0 - h)
                fm = self._model(qxy[m], qz[m])
                setattr(self, k, v0)
                J[:, j] = (fp - fm) / (2.0 * h) * w
            F = J.T @ J
            d = np.sqrt(np.clip(np.diag(F), 1e-300, None))
            cond = float(np.linalg.cond(F))
            out = {'fisher_condition_number': cond,
                   'fisher_note': ('>1e8 means at least one direction is '
                                   'unconstrained by this data')}
            try:
                Ci = np.linalg.inv(F)
                for j, k in enumerate(names):
                    # sigma_j / sigma_j^(no correlation)
                    out['varinflation_%s' % k] = float(
                        np.sqrt(max(Ci[j, j], 0.0)) * d[j])
            except np.linalg.LinAlgError:
                out['identifiability'] = 'Fisher matrix singular'
            return out
        except Exception as e:
            return {'identifiability': 'not computed (%s)' % e}
        finally:
            for k, v in keep.items():
                setattr(self, k, v)


if __name__ == '__main__':
    # single-tilt DPPC NN preset, in-plane mode: no map file needed
    x = np.linspace(1.30, 1.65, 141)
    fun = GID2D(x=x, mode='inplane', a0=4.8743, t=18.85, psi=0.0, eps=-0.0321,
                L_dom=200.0, L_ch=19.0, qzmin=0.0, qzmax=1.4)
    fun.__fit__ = False
    y = fun.y()
    print('in-plane reduction: %d points, max %.4g at qxy=%.4f'
          % (len(y), y.max(), x[int(np.argmax(y))]))
    for k in sorted(fun.output_params['scaler_parameters']):
        print('   %-28s %s' % (k, fun.output_params['scaler_parameters'][k]))
