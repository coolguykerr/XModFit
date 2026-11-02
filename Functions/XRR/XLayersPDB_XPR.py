####Please do not remove lines below####
import copy
import os
import sys

import numpy as np
from lmfit import Parameters

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines/'))

####Please do not remove lines above####

####Import your modules below if needed####
try:
    from Functions import xpr_backend as xb
    from Functions import pdb_geometry as pg
except Exception:
    import xpr_backend as xb
    import pdb_geometry as pg


class XLayersPDB_XPR:  # Please put the class name same as the function name
    """Smoothed, autodifferentiable XLayersPDB backed by the XPR Julia model.

    Same physics as :mod:`XLayersPDB` -- specular reflectivity of an n-box film with
    an optional rigid protein whose electron density comes from a PDB structure --
    but every step of the forward model is continuously differentiable, so the Julia
    backend can hand back exact gradients and Hessians of the log-likelihood via
    ForwardDiff.  Stage 1 uses only the forward evaluation, through the ordinary
    lmfit/emcee machinery; :mod:`Functions.xpr_backend` already exposes ``grad`` and
    ``hess`` for the gradient-based seeding and HMC work that follows.

    Differences from :mod:`XLayersPDB` that matter when comparing fits:

    * **One global roughness.**  XPR smears every interface with a single ``sig``,
      so roughness is a scalar parameter here rather than a per-layer ``mpar``
      column.
    * **No absorption, so E does nothing.**  Note this is NOT because the critical
      edge is energy independent in angle -- it is that the edge is energy
      independent in *q*.  Parratt is written directly in q here (as it is in
      XModFit's own ``XLayers``), with ``Qc = 16*pi*r_e*drho``, so
      ``q_c = 4*sqrt(pi*r_e*rho)`` carries no wavelength at all; E moves the
      critical ANGLE, ``theta_c = q_c*lambda/4pi``, not the critical q.  The only
      place lambda enters the recursion is ``-32*pi^2/lambda^2`` multiplying beta,
      and XPR allocates its absorption array without ever writing to it, so
      beta == 0 and there is no ``mu`` column.  ``E`` is therefore inert; it is kept
      because every other XModFit XRR function has it, and the Julia side is already
      wired to use it the day absorption (or anomalous f') is added.
    * **Layer table.**  Row 0 is the top medium and the last row is the bulk; only
      their ``rho`` is used (``d`` is ignored, as in XLayersPDB).  The rows between
      them are the boxes, ordered top to bottom.
    * **Euler convention.**  ``theta``/``phi`` follow XPR's hand-written rotation
      matrix, which IS the ``scipy`` ``'zx'`` convention XLayersPDB uses --
      ``pdb_geometry.xpr_rotation_matrix`` equals
      ``from_euler('zx', [phi, theta], degrees=True)`` to 4e-16, and both sides
      take degrees.  Angles therefore transfer between the two models, and so does
      ``cov`` after rescaling by ``frac*cov_scale``; only the absolute ``logL`` does
      not.  See ``handoff/05-xpr-julia.md`` 4.8 and its claims 10/11 -- an earlier
      version of this bullet asserted the opposite.
    * **``protins`` does NOT transfer; ``protins_top`` does.**  Upstream XPR
      redefined the insertion parameter on 2026-09-08: it now places the protein's
      MASS CENTROID above the bottom of the box stack, where it used to place the
      TOP of the protein (and where every numba ``XLayersPDB*`` still does,
      ``ztop_grid = -sum(d[:-1]) + protins``).  The reason is smoothness: the top of
      the protein is ``max_i (r3(theta,phi).c_i + rad_i)``, a maximum over atoms,
      so anchoring on it put a kink into the forward model at every (theta, phi)
      where the topmost atom changes -- which defeats the whole point of a
      differentiable backend.  Anchoring on the centroid is exactly linear in
      ``protins`` and exactly smooth in the angles.

      The old quantity is still what one reads off a figure, so it is reported
      rather than dropped: ``protins_top = protins + maxh(theta, phi)``, published
      as a scaler output by ``y()`` and, with its propagated error and its
      covariance against every other varying parameter, by
      :meth:`derived_params` after a fit.  Converting an existing setup, in either
      direction, is :meth:`protins_top` / :meth:`protins_from_top`.
    * **The bounding media are load-time constants.**  XPR bakes ``rho`` of row 0
      and of the last row into its preallocated ``SimData``, so varying either one
      forces a full Julia reallocation on every iteration.  Keep them fixed unless
      you are prepared for a very slow fit.
    * **Why 'exact' costs ~300x.**  ``vol_calc='approx'`` sums each atom's
      spherical-segment volume analytically -- one arithmetic pass over
      (slabs x atoms), no geometry at all.  ``vol_calc='exact'`` instead computes the
      true area of the UNION of circles in each z-plane by Green's theorem and
      integrates it over z.  Because that area kinks whenever a sphere enters or
      leaves the plane, the quadrature is split at every such event (two per atom)
      and a 7-point Gauss-Legendre rule applied to each smooth piece -- so a model
      evaluation costs ``7 * (slabs + 2*atoms)`` union-of-circles solves where
      'approx' costs none.  Measured on a 1697-atom protein over 259 slabs: 25557
      solves, 3.5 ms -> 1160 ms per ``y()`` and 12 ms -> 1634 ms per ForwardDiff
      gradient.
    * **``cov`` is XPR's raw coverage; the physical fraction is an OUTPUT.**  The
      fitted parameter reaches Julia untouched, so the whole dependence of the model
      on the parameters stays inside ForwardDiff's tape and the gradient and Hessian
      are exact.  Raw ``cov`` is ``n_s * A``, surface number density times whatever
      area XPR normalises by -- ``sum(pi r^2)`` for 'approx', the true footprint for
      'exact' -- so it is not a covered fraction and is not even on the same scale in
      the two ``vol_calc`` modes (~70x apart for a 1700-atom protein).  The fraction
      is derived downstream and published as the ``coverage`` scaler output alongside
      the ``cov_scale`` that produced it; :meth:`coverage_fraction` converts a whole
      MCMC chain after the fact.
    * **``cov_norm`` fixes a dimensional gap, with a constant.**  XPR's fast path
      builds ``alpha[j] = sum_i f_ij / (A*dz)`` from dimensionless segment
      *fractions*, so it carries units of 1/volume where the slow path's
      ``volume/(A*dz)`` is properly dimensionless.  The model's electron term and its
      excluded-volume term therefore imply surface densities differing by
      ``V_union/N`` -- a factor of 6.1 here -- which is precisely the mismatch
      XLayersPDB removes with its ``protvol`` normalisation.  ``cov_norm='protvol'``
      (default) closes it by scaling the loaded electron array by ``N/V_union``: a
      per-structure CONSTANT, rotation invariant, folded into the data at load time,
      hence invisible to the parameters and to AD.  ``cov_norm='xpr'`` reproduces
      upstream XPR exactly, dimensional gap included.
    * **Finite parameter bounds are now REQUIRED, not just faster.**  Until
      2026-09-09 this was a performance argument: evaluation cost tracked the z-span
      of the profile on a fixed 0.5 A lattice, so an unbounded ``d`` let a walker sit
      at ``d = [210, 320]`` and cost 15-20 ms instead of 1.8 ms.  That is obsolete --
      the slab count is a load-time constant now and cost is flat across parameter
      space (measured 6.6-9.1 ms everywhere).  The bounds matter for a different and
      stricter reason: ``_param_box`` derives XPR's grid from them.
      ``max(d)`` sets ``cells_per_box``, hence the in-box mesh, hence THE ANSWER;
      and the ``protins`` bounds become ``d_protein_bounds``, which Julia enforces
      with a hard ``error("protein out of bounds")``.  An infinite bound would give
      an absurd mesh, so ``_param_box`` substitutes the class constants ``D_MAX``,
      ``SIG_MAX`` and ``PROTINS_MAX`` for any non-finite one -- which also rescues a
      .par saved before those defaults existed.  Widen a bound and you change the
      mesh: expect logL to move a little (see ``spacez`` below), so do it between
      fits rather than during one.
    * **``spacez`` is the dominant discretization error, and it is yours to set.**
      The mesh is ``spacez`` everywhere outside the boxes and ``d_b/ceil(max(d)/
      spacez)`` inside box ``b`` -- so it is always at least as fine as ``spacez``
      in the boxes, and exactly ``spacez`` in the two regions where most of an
      inserted protein actually sits.  Measured on the 1697-atom test protein at its
      published fit: refining ``spacez`` 0.5 -> 0.25 -> 0.125 moves logL by -0.76
      then -0.25, extrapolating to a continuum limit ~1.1 below the default.  The
      error is a smooth q^2.23 distortion of R -- 3.9e-5 at q < 0.1 rising to
      1.5e-3 at q > 0.5 -- so it is concentrated where the protein signal is.  It
      biases fitted parameters by <= 0.002 A, i.e. negligibly, but it does NOT
      cancel from absolute chi2.  **Never compare chi2 across different ``spacez``
      or different box bounds.**  Halve ``spacez`` once at the end to confirm a
      result is mesh-converged.
    """

    def __init__(self, x=0.1, E=10.0, fname='./Data/1ubq.pdb',
                 mpar={'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                                 'd': [0.0, 13.4, 10.0, 1.0],
                                 'rho': [0.0, 0.243, 0.453, 0.334]}},
                 rrf=True, use_protein=True, vol_calc='approx', cov_norm='protvol',
                 fslice=0.5, spacez=0.5, box_ref_length=30.0,
                 qoff=0.0, yscale=1.0, bkg=0.0, sig=3.4,
                 theta=0.0, phi=0.0, protins=0.0, cov=0.1):
        """
        x         : array of wave-vector transfer along z-direction
        E         : Energy of x-rays in keV
        fname     : PDB structure of the protein
        rrf       : True for Fresnel normalized reflectivity, False for reflectivity
        use_protein: False evaluates the bare box model (theta/phi/protins/cov idle)
        vol_calc  : 'approx' (spherical segments, fast) or 'exact' (Gauss-Legendre
                    union of spheres, ~300x slower)
        cov_norm  : 'protvol' scales the loaded electron array by N/V_union, a
                    per-structure constant that repairs XPR's dimensional gap the way
                    XLayersPDB does; 'xpr' reproduces upstream exactly. Neither
                    touches the fitted cov, which always reaches Julia raw.
        fslice    : grid step (Angs) for the union volume and projected area. Only
                    the constant N/V_union enters the model; the projected area is
                    used purely to report the covered fraction.
        spacez    : z mesh (Angs) for XPR's profile. Sets the spacing everywhere
                    OUTSIDE the boxes (above the top interface, below the stack, and
                    in the Gaussian tails) and, through
                    cells_per_box = ceil(max(d)/spacez), the number of cells each box
                    is divided into. THE DOMINANT DISCRETIZATION ERROR -- see the
                    class docstring. 0.5 is upstream's default; halve it to check a
                    result is mesh-converged, at ~2x the cost.
        box_ref_length: box length (Angs) at which the in-box mesh equals spacez.
                    Each box is cut into ceil(box_ref_length/spacez) cells whatever
                    its length, so the in-box spacing is d_b*spacez/box_ref_length --
                    finer than spacez for a thinner box, coarser for a thicker one.
                    DELIBERATELY not derived from the d bounds: it sets the answer,
                    so it must not change when someone widens a bound. 30.0 is
                    upstream's default and reproduces its published numbers.
        qoff      : q-offset to correct the zero q of the instrument
        yscale    : a scale factor for R or R/Rf
        bkg       : Incoherent background
        sig       : single interfacial roughness applied to the whole profile, Angs
        theta/phi : Euler angles of the protein in degrees (XPR convention)
        protins   : insertion of the protein into the box stack, Angs, measured to
                    the protein's CENTRE OF MASS: protins puts the centroid that far
                    above the bottom of the stack, so it is routinely NEGATIVE. The
                    old top-of-protein insertion is reported as 'protins_top'; see
                    :meth:`protins_top`
        cov       : surface coverage of the protein
        mpar      : Dictionary of Phases where,
                    Layers: Layer description,
                    d: thickness of each layer in Angs (row 0 and the last row are
                       the semi-infinite media and their d is ignored),
                    rho: Electron density of each layer in el/Angs^3
        """
        if type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        self.fname = fname if (fname and os.path.exists(fname)) else None
        self.E = E
        # deep-copied because `mpar` is a mutable default argument: XModFit builds
        # every instance as ``funcClass()``, so without this two datasets in a joint
        # fit would share -- and overwrite -- one layer table.
        self.__mpar__ = copy.deepcopy(mpar)
        self.rrf = rrf
        self.use_protein = use_protein
        self.vol_calc = vol_calc
        self.cov_norm = cov_norm
        self.fslice = fslice
        self.spacez = spacez
        self.box_ref_length = box_ref_length
        self.qoff = qoff
        self.yscale = yscale
        self.bkg = bkg
        self.sig = sig
        self.theta = theta
        self.phi = phi
        self.protins = protins
        self.cov = cov
        # NB 'approx'/'exact' are not Python names, so they survive XModFit's eval()
        # of Fixed-Parameter cells as strings; True/False are meant to become bools.
        # spacez is a free float, not a choice -- it appears as a Fixed Parameter
        # cell because __init__ takes it, and XModFit eval()s that cell.
        self.choices = {'rrf': [True, False], 'use_protein': [True, False],
                        'vol_calc': ['approx', 'exact'],
                        'cov_norm': ['protvol', 'xpr']}
        self.filepaths = {'fname': self.fname}

        self.__d__ = {}
        self.__rho__ = {}
        self.__fit__ = False
        self.__mkeys__ = list(self.__mpar__.keys())
        self.init_params()
        self.output_params = {'scaler_parameters': {}}

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
        self.params = Parameters()
        self.params.add('qoff', self.qoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('yscale', self.yscale, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('bkg', self.bkg, vary=0, min=0, max=1, expr=None, brute_step=0.1)
        # DEFAULT BOUNDS ARE FINITE ON PURPOSE -- see the class docstring. The cost of
        # one evaluation scales with the z-span of the profile, so an unbounded d,
        # protins or sig lets an MCMC walker park somewhere physically meaningless and
        # 10x slower. Widen them in the GUI if your system really needs it.
        self.params.add('sig', self.sig, vary=0, min=0, max=self.SIG_MAX, expr=None, brute_step=0.1)
        # theta/phi are UNBOUNDED, exactly as in XLayersPDB: y() folds every proposal
        # into [0,180]x[0,360] before it reaches the model, and CustomSampler folds
        # the stored emcee coords the same way. Finite bounds here would instead make
        # lmfit's prior reject a proposal at, say, theta=185 that is really theta=175
        # -- a reflecting wall at the poles that would bias the chain.
        self.params.add('theta', self.theta, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=1)
        self.params.add('phi', self.phi, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=1)
        # protins places the MASS CENTROID above the bottom of the stack, so unlike
        # the old top-of-protein convention it is negative for a protein sitting on
        # or in the film -- the floor cannot be 0. Either sign moves the protein off
        # the stack and grows the z-span of the profile, so the bound is symmetric.
        self.params.add('protins', self.protins, vary=1, min=-self.PROTINS_MAX,
                        max=self.PROTINS_MAX, expr=None, brute_step=0.1)
        # cov is XPR's RAW coverage, whose scale depends on vol_calc and on the
        # protein's size (see the class docstring), so there is no useful universal
        # upper bound to impose -- 100% coverage sits at whatever the 'cov_scale'
        # scaler output reports. Bound it yourself once you have read that number.
        self.params.add('cov', self.cov, vary=1, min=0, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key != 'Layers':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i),
                                        value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0,
                                        max=self.D_MAX if key == 'd' else self.RHO_MAX,
                                        expr=None, brute_step=0.05)

    def update_parameters(self):
        for mkey in self.__mpar__.keys():
            Nlayers = len(self.__mpar__[mkey]['d'])
            self.__d__[mkey] = np.array([self.params['__%s_%s_%03d' % (mkey, 'd', i)].value
                                         for i in range(Nlayers)])
            self.__rho__[mkey] = np.array([self.params['__%s_%s_%03d' % (mkey, 'rho', i)].value
                                           for i in range(Nlayers)])

    def _fold_angles(self):
        """Fold (theta, phi) into XPR's [0,180] x [0,360] fundamental domain.

        Identical to the fold in XLayersPDB.y() and to the one CustomSampler applies
        to emcee walkers, so a chain and a least-squares fit agree on which of the
        equivalent (theta, phi) pairs they report.
        """
        phitemp = self.params['phi'].value % 360
        thetatemp = self.params['theta'].value % 360
        self._theta_reflected = thetatemp > 180
        if thetatemp > 180:
            self.params['theta'].value = 360 - thetatemp
            self.params['phi'].value = (phitemp + 180) % 360
        else:
            self.params['theta'].value = thetatemp
            self.params['phi'].value = phitemp
        self.theta = self.params['theta'].value
        self.phi = self.params['phi'].value

    def canonical_params(self, params=None):
        """Fold ``theta``/``phi`` in ``params`` into [0,180] x [0,360], in place.

        theta and phi are deliberately UNBOUNDED (see :meth:`init_params`), and the
        fold that ``y()`` applies is local to the evaluation: lmfit sets the
        parameters from the optimiser's own variable vector at the start of every
        residual call, and writes that vector back into ``result.params`` when the
        fit ends. So whatever ``y()`` folded is discarded, and the REPORTED angles
        are the raw optimiser coordinates -- which, since the objective is exactly
        periodic in phi and mirror-symmetric in theta, can drift arbitrarily far.
        Observed in practice: a fit reporting theta=379.3, phi=-8208.2 while the
        model was evaluating theta=19.3, phi=71.8.

        The fit is correct either way, but the number written to a .par file, shown
        in the table, or read by the structure viewer should be the conventional
        one. Fit.perform_fit calls this once the fit is over.

        The uncertainties are unaffected (|d(folded)/d(raw)| == 1), but a reflected
        theta flips the SIGN of its correlations with the other parameters; the
        magnitudes stand.
        """
        params = self.params if params is None else params
        if 'theta' not in params or 'phi' not in params:
            return None
        th = params['theta'].value % 360
        ph = params['phi'].value % 360
        if th > 180:
            th, ph = 360 - th, (ph + 180) % 360
        params['theta'].value = th
        params['phi'].value = ph
        self.theta, self.phi = th, ph
        return th, ph

    def _norm_mode(self):
        return getattr(self.cov_norm, '__name__', None) or str(self.cov_norm)

    def electron_scale(self, vol_calc=None):
        """Constant multiplying the loaded electron array; 1.0 unless normalising.

        ``N/V_union`` -- atom count over the true union volume, i.e. the reciprocal
        of the mean effective atomic volume once overlaps are removed. This is the
        volume that XPR's fast path is missing from ``alpha`` (see the class
        docstring), supplied on the other side of the ratio so a single number fixes
        both terms of ``rho = C*rho_prot + (1 - C*alpha)*rho_box``.

        Rotation invariant and computed once per structure, so it never varies during
        a fit and never reaches ForwardDiff as anything but a constant. 'exact'
        already has consistent units and is left alone.
        """
        vol_calc = vol_calc if vol_calc is not None else (
            getattr(self.vol_calc, '__name__', None) or str(self.vol_calc))
        if vol_calc == 'exact' or self._norm_mode() != 'protvol':
            return 1.0
        if not (self.fname and os.path.isfile(self.fname)):
            return 1.0
        n_atoms, _a_xpr, _v = pg.sphere_sums(self.fname)
        v_union = pg.union_volume(self.fname, self.fslice)
        return (n_atoms / v_union) if (v_union > 0 and n_atoms) else 1.0

    def cov_scale(self, vol_calc=None, theta=None, phi=None):
        """Factor converting XPR's raw ``cov`` to the fraction of surface covered.

        ``coverage = cov / cov_scale``.  Purely diagnostic -- nothing in ``y()`` uses
        it, so the model handed to Julia depends on the fitted parameters and nothing
        else, and the ForwardDiff gradient/Hessian are exact.

        XPR builds the mixed profile as ``rho = C*rho_prot + (1 - C*alpha)*rho_box``,
        where with ``vol_calc='approx'``

            rho_prot[j] = sum_i Z_i f_ij / (A_xpr*dz)      f_ij = V_seg,ij / V_sph,i
            alpha[j]    = sum_i      f_ij / (A_xpr*dz)     A_xpr = sum_i pi r_i^2

        Two INDEPENDENT things separate that ``C`` from a covered fraction:

        1. NUMERATOR. ``alpha``'s numerator is a sum of dimensionless *fractions*,
           not volumes. Every atom's fractions sum to 1 over the slabs, so ``alpha``
           integrates to exactly ``N/A_xpr`` where the truth integrates to
           ``V_union/A_proj``. The ratio is XLayersPDB's ``protvol`` correction:
           one constant, ``V_union/N``, rotation invariant (measured spread over
           orientations 0.13%, i.e. grid noise).

        2. DENOMINATOR, and not a volume effect at all. Writing the physical film as
           ``rho = n_s*E_j/dz + (1 - n_s*V_j/dz)*rho_box`` for a surface number
           density ``n_s`` and matching term by term gives

               C = n_s * A

           so A cancels out of the *physics* but not out of the *parameter*: every
           choice of A gives identical reflectivity and merely relabels C. 'exact'
           and XLayersPDB use the real footprint ``A_proj(theta, phi)``, making C the
           covered fraction; 'approx' uses ``A_xpr``, the sum of atomic disc areas
           with no overlap removed, which for a compact protein is ``N*a^2/R^2``
           times bigger -- 11.7x for the 1697-atom test protein, growing like R with
           protein size, and swinging 8.8-13.6 across orientation.

        Hence, with ``cov_norm='protvol'`` and 'approx',

            cov_scale = (V_union/N) * (A_xpr / A_proj(theta, phi))

        and 1.0 for 'exact', which already normalises by the true footprint. Setting
        ``cov = fraction * cov_scale`` and running 'approx' reproduces 'exact' to
        0.03-0.25% rms in R (2.8% if the raw cov is used directly); the residual is
        term 1's replacement of each atom's volume by the mean.

        Under ``cov_norm='xpr'`` the two terms of the model imply different surface
        densities, so no single number converts cov to a fraction and this returns
        NaN rather than a misleading one.

        Pass explicit ``theta``/``phi`` to convert a whole MCMC chain after the fact.
        """
        vol_calc = vol_calc if vol_calc is not None else (
            getattr(self.vol_calc, '__name__', None) or str(self.vol_calc))
        if vol_calc == 'exact':
            return 1.0
        if not (self.fname and os.path.isfile(self.fname)):
            return 1.0
        if self._norm_mode() != 'protvol':
            return float('nan')
        n_atoms, a_xpr, _v_spheres = pg.sphere_sums(self.fname)
        v_union = pg.union_volume(self.fname, self.fslice)
        a_proj = pg.projected_area(self.fname,
                                   self.theta if theta is None else theta,
                                   self.phi if phi is None else phi, self.fslice)
        if v_union <= 0 or a_proj <= 0 or n_atoms == 0:
            return 1.0
        return (v_union / n_atoms) * (a_xpr / a_proj)

    def coverage_fraction(self, cov=None, theta=None, phi=None, vol_calc=None):
        """The fraction of the interface covered by protein, from a raw ``cov``.

        Accepts arrays, so an emcee chain converts in one call::

            frac = model.coverage_fraction(chain['cov'].values,
                                           chain['theta'].values,
                                           chain['phi'].values)
        """
        cov = self.cov if cov is None else cov
        th = self.theta if theta is None else theta
        ph = self.phi if phi is None else phi
        cov_a, th_a, ph_a = np.broadcast_arrays(np.asarray(cov, dtype=float),
                                                np.asarray(th, dtype=float),
                                                np.asarray(ph, dtype=float))
        out = np.empty(cov_a.shape, dtype=float)
        flat = out.reshape(-1)
        for k, (c, t, p) in enumerate(zip(cov_a.reshape(-1), th_a.reshape(-1),
                                          ph_a.reshape(-1))):
            flat[k] = c / self.cov_scale(vol_calc=vol_calc, theta=float(t),
                                         phi=float(p))
        return float(out) if out.ndim == 0 else out

    # ---- the two insertion conventions ------------------------------------- #

    def maxh(self, theta=None, phi=None):
        """``maxh(theta, phi)``: mass centroid to the top of the protein, Angs.

        The offset between the fitted ``protins`` (centroid-referenced, what XPR
        uses) and ``protins_top`` (top-referenced, the old convention and the one
        every numba ``XLayersPDB*`` uses). Computed from the structure alone -- no
        model evaluation, no Julia -- so a whole MCMC chain costs one pass.
        """
        if not (self.fname and os.path.isfile(self.fname)) or not self.use_protein:
            return 0.0
        return pg.com_extent(self.fname,
                             self.theta if theta is None else float(theta),
                             self.phi if phi is None else float(phi))[0]

    def protins_top(self, protins=None, theta=None, phi=None):
        """The insertion measured to the TOP of the protein: ``protins + maxh``.

        Broadcasts, so an emcee chain converts in one call::

            top = model.protins_top(chain['protins'].values,
                                    chain['theta'].values, chain['phi'].values)

        which is the right way to get a credible interval on it -- the propagated
        error from :meth:`derived_params` linearises a ``max`` over atoms, this does
        not.
        """
        return self._map_insertion(protins, theta, phi, +1)

    def protins_from_top(self, protins_top, theta=None, phi=None):
        """Inverse of :meth:`protins_top`: a top-referenced insertion -> ``protins``.

        Use this to carry a starting value, or a whole saved setup, over from
        ``XLayersPDB`` or from a pre-2026-09-08 XPR fit.
        """
        return self._map_insertion(protins_top, theta, phi, -1)

    def _map_insertion(self, value, theta, phi, sign):
        value = self.protins if value is None else value
        th = self.theta if theta is None else theta
        ph = self.phi if phi is None else phi
        v, t, p = np.broadcast_arrays(np.asarray(value, dtype=float),
                                      np.asarray(th, dtype=float),
                                      np.asarray(ph, dtype=float))
        out = np.empty(v.shape, dtype=float)
        flat = out.reshape(-1)
        for k, (vi, ti, pi) in enumerate(zip(v.reshape(-1), t.reshape(-1),
                                             p.reshape(-1))):
            flat[k] = vi + sign * self.maxh(float(ti), float(pi))
        return float(out) if out.ndim == 0 else out

    def derived_params(self, result=None, keymap=None):
        """``protins_top`` with its error and covariances, from a finished fit.

        ``protins_top = protins + maxh(theta, phi)`` is a function of three fitted
        parameters, so its variance follows from the fit covariance by the usual
        linearisation, with

            d/d(protins) = 1,   d/d(theta), d/d(phi) = d(maxh)/d(theta, phi)

        taken analytically from the atom that currently attains the maximum
        (``pdb_geometry.com_extent_grad``). Also returns the covariance of
        ``protins_top`` with every other varying parameter,
        ``cov(top, p_k) = sum_j J_j Sigma_jk``, which is what a downstream
        propagation (a shared insertion in a joint fit, say) needs.

        MUST be called on the RAW optimiser coordinates, i.e. BEFORE
        :meth:`canonical_params` folds the angles: ``result.covar`` is in those
        coordinates, and although ``maxh`` is exactly invariant under the fold
        (``r3(360-theta, phi+180) == r3(theta, phi)``) so its VALUE does not care,
        the sign of ``d(maxh)/d(theta)`` does. ``Fit._canonicalize_result`` calls it
        in that order.

        Caveat, and the reason ``nties`` is reported: ``maxh`` is a maximum over
        atoms and is only piecewise smooth, so this error bar is a linearisation
        around one active atom. Where the fit sits on a kink (two atoms tied for
        topmost) there is no two-sided derivative and the number is indicative
        only; :meth:`protins_top` over an MCMC chain is the honest interval.
        Returns None when there is no protein or no usable covariance.

        ``keymap`` renames the three parameters this reads, for a result whose
        parameters are not called ``protins``/``theta``/``phi`` -- a joint fit
        namespaces the un-shared ones as ``d{i}_protins`` and so on. Pass
        ``{'protins': 'd0_protins', ...}``; Joint_Fit_Routines builds it from
        ``_master_key``.
        """
        if not (self.use_protein and self.fname and os.path.isfile(self.fname)):
            return None
        if result is None or getattr(result, 'params', None) is None:
            return None
        km = {'protins': 'protins', 'theta': 'theta', 'phi': 'phi'}
        km.update(keymap or {})
        params = result.params
        if not all(km[k] in params for k in ('protins', 'theta', 'phi')):
            return None
        ins = float(params[km['protins']].value)
        th, ph = float(params[km['theta']].value), float(params[km['phi']].value)
        h, dh_dth, dh_dph, iatom, nties = pg.com_extent_grad(self.fname, th, ph)
        out = {'protins_top': ins + h, 'maxh': h, 'maxh_atom': iatom + 1,
               'maxh_ties': nties, 'protins_top_err': None, 'cov': {}}

        covar = getattr(result, 'covar', None)
        names = list(getattr(result, 'var_names', []) or [])
        if covar is None or not names or np.shape(covar) != (len(names), len(names)):
            return out
        J = np.zeros(len(names))
        for k, nm in enumerate(names):
            if nm == km['protins']:
                J[k] = 1.0
            elif nm == km['theta']:
                J[k] = dh_dth
            elif nm == km['phi']:
                J[k] = dh_dph
        cov = np.asarray(covar, dtype=float)
        var = float(J @ cov @ J)
        out['protins_top_err'] = float(np.sqrt(var)) if var >= 0 else None
        row = cov @ J
        out['cov'] = {nm: float(row[k]) for k, nm in enumerate(names)}
        return out

    def derived_report(self, result=None, keymap=None, label=''):
        """:meth:`derived_params` as a block of text to append to a fit report."""
        info = self.derived_params(result, keymap=keymap)
        if info is None:
            return ''
        L = ['[[Derived]]' + (' %s' % label if label else '')]
        err = info['protins_top_err']
        L.append('    protins_top:  %-12.6g %s (= protins + maxh, insertion to the '
                 'TOP of the protein)'
                 % (info['protins_top'],
                    '+/- %-10.4g' % err if err is not None else '(no covariance)'))
        L.append('    maxh:         %-12.6g       centroid -> top at theta/phi, '
                 'atom %d%s' % (info['maxh'], info['maxh_atom'],
                                '' if info['maxh_ties'] <= 1
                                else ', %d ATOMS TIED -- kink, error indicative only'
                                     % info['maxh_ties']))
        if info['cov']:
            pairs = ['%s=%.4g' % (k, v) for k, v in info['cov'].items()]
            L.append('    cov(protins_top, .):  ' + ', '.join(pairs))
        L.append('    NB the fitted protins is measured to the MASS CENTROID; '
                 'protins_top is the pre-2026-09-08 / XLayersPDB convention.')
        return '\n'.join(L)

    def _full_params(self, d, rho, num_boxes):
        """Assemble XPR's 8 + 2n parameter vector.

        Every entry is a fitted parameter passed straight through -- no derived or
        orientation-dependent rescaling -- so Julia's ForwardDiff sees the whole
        dependence of the model on the parameters.
        """
        p = np.zeros(8 + 2 * num_boxes)
        p[xb.IDX_QOFF] = self.qoff
        p[xb.IDX_THETA] = self.theta
        p[xb.IDX_PHI] = self.phi
        p[xb.IDX_PROTINS] = self.protins
        p[xb.IDX_COV] = self.cov
        p[xb.IDX_SIG] = self.sig
        p[xb.IDX_YSCALE] = self.yscale
        p[xb.IDX_BKG] = self.bkg
        p[xb.IDX_BOXES:xb.IDX_BOXES + num_boxes] = d[1:-1]
        p[xb.IDX_BOXES + num_boxes:] = rho[1:-1]
        return p

    # Default upper bounds. Generous but FINITE: evaluation cost is proportional to
    # the z-span of the profile, which these three drive.
    D_MAX = 100.0        # Angstroms per box
    RHO_MAX = 5.0        # el/Angs^3 (water 0.334, protein ~0.44, gold 4.66)
    SIG_MAX = 10.0       # Angstroms of interfacial roughness
    PROTINS_MAX = 100.0  # Angstroms of insertion

    # lmfit scalar parameter -> column of XPR's 8+2n vector
    _SCALAR_COLUMNS = {'qoff': xb.IDX_QOFF, 'theta': xb.IDX_THETA,
                       'phi': xb.IDX_PHI, 'protins': xb.IDX_PROTINS,
                       'cov': xb.IDX_COV, 'sig': xb.IDX_SIG,
                       'yscale': xb.IDX_YSCALE, 'bkg': xb.IDX_BKG}

    def parameter_columns(self, names):
        """Map lmfit parameter names to (column, sign) in XPR's parameter vector.

        Returns None if ANY name cannot be differentiated, so the caller falls back
        to finite differences for the whole Jacobian rather than mixing exact and
        missing columns. A column of ``-1`` means the parameter provably has no
        effect (the bounding media's ``d``), so its derivative is exactly zero.

        The unsupported case is ``rho`` of the first or last layer: XPR bakes those
        into ``SimData`` as plain Float64 at load time, so ForwardDiff cannot see
        them.
        """
        mkey = self.__mkeys__[0]
        nlayer = len(self.__mpar__[mkey]['d'])
        nbox = nlayer - 2
        cols, signs = [], []
        # theta's raw value is folded into [0,180] before it reaches Julia; a
        # reflected fold means d(folded)/d(raw) = -1.
        tsign = -1.0 if getattr(self, '_theta_reflected', False) else 1.0
        for nm in names:
            if nm in self._SCALAR_COLUMNS:
                cols.append(self._SCALAR_COLUMNS[nm])
                signs.append(tsign if nm == 'theta' else 1.0)
                continue
            hit = None
            for key, base in (('d', xb.IDX_BOXES), ('rho', xb.IDX_BOXES + nbox)):
                pre = '__%s_%s_' % (mkey, key)
                if nm.startswith(pre):
                    try:
                        i = int(nm[len(pre):])
                    except ValueError:
                        return None
                    if 1 <= i <= nbox:
                        hit = base + i - 1
                    elif key == 'd':
                        hit = -1          # top/bulk thickness is ignored by XPR
                    else:
                        return None       # rho_top / rho_bottom: load-time constant
                    break
            if hit is None:
                return None
            cols.append(hit)
            signs.append(1.0)
        return np.asarray(cols, dtype=int), np.asarray(signs, dtype=float)

    def _prepare(self):
        """Everything y() does before calling Julia. Returns the parameter vector."""
        x = np.atleast_1d(np.asarray(self.x, dtype=float))
        self._fold_angles()
        self.update_parameters()
        mkey = self.__mkeys__[0]
        d, rho = self.__d__[mkey], self.__rho__[mkey]
        num_boxes = len(d) - 2
        if num_boxes < 1:
            raise ValueError('XLayersPDB_XPR needs at least 3 layers (top medium, '
                             'one box, bulk); the layer table has %d.' % len(d))

        include_protein = bool(self.use_protein) and bool(self.fname) \
            and os.path.isfile(self.fname)
        # XModFit eval()s Fixed-Parameter cells, so a choices value that happens to
        # be a Python name arrives as that object rather than the string.
        vol_calc = getattr(self.vol_calc, '__name__', None) or str(self.vol_calc)

        # Sizing arguments for XPR's grid, derived ONCE from the lmfit parameter
        # BOUNDS. This is not an optimisation any more, it is required: since the
        # 2026-09-09 revision `cells_per_box = ceil(max_box_length/spacez)` sets the
        # in-box mesh, so max_box_length and spacez CHANGE THE ANSWER. Sizing them
        # from live values (as this used to, quantized to ratchet upwards) would step
        # the objective in the middle of a descent every time the ratchet clicked.
        # d_protein_bounds must additionally bracket the fitted protins, because
        # Julia enforces it with a hard error(); see xb.check_headroom.
        sizing = xb.alloc_bounds_for(*self._param_box(num_boxes),
                                     num_boxes=num_boxes, spacez=float(self.spacez),
                                     box_ref_length=float(self.box_ref_length))

        xb.load(x, num_boxes, pdbfile=self.fname if include_protein else None,
                vol_calc=vol_calc, rho_top=float(rho[0]),
                rho_bottom=float(rho[-1]), E=float(self.E), rrf=bool(self.rrf),
                electron_scale=self.electron_scale(vol_calc) if include_protein else 1.0,
                **sizing)

        p = self._full_params(d, rho, num_boxes)
        xb.check_headroom(p, num_boxes, include_protein)
        return p, num_boxes, include_protein, vol_calc, mkey

    def _param_box(self, num_boxes):
        """(lower, upper) in XPR's 8+2n layout, from the lmfit parameter bounds.

        Infinite or absent bounds fall back to the class ``*_MAX`` constants, which
        are the same finite defaults ``init_params`` installs -- so a .par saved
        before those defaults existed still yields a usable, finite grid rather than
        an infinite one. A bound TIGHTER than the current value is widened to
        include it, so a hand-edited value outside its own bounds cannot make the
        declared window exclude the point we are about to evaluate.
        """
        lo = np.zeros(8 + 2 * num_boxes)
        hi = np.zeros(8 + 2 * num_boxes)

        def box(name, col, floor, ceil_):
            v = float(self.params[name].value) if name in self.params else 0.0
            a, b = (float(self.params[name].min), float(self.params[name].max)) \
                if name in self.params else (floor, ceil_)
            if not np.isfinite(a):
                a = min(floor, v)
            if not np.isfinite(b):
                b = max(ceil_, v)
            lo[col], hi[col] = min(a, v), max(b, v)

        box('protins', xb.IDX_PROTINS, -self.PROTINS_MAX, self.PROTINS_MAX)
        box('sig', xb.IDX_SIG, 0.0, self.SIG_MAX)
        mkey = self.__mkeys__[0]
        for i in range(num_boxes):
            box('__%s_d_%03d' % (mkey, i + 1), xb.IDX_BOXES + i, 0.0, self.D_MAX)
        return lo, hi

    def dydp(self, names):
        """dR/dp for the named lmfit parameters: (len(x), len(names)), or None.

        Exact, by forward-mode AD in Julia, and seeded only on the parameters asked
        for. Returns None when :meth:`parameter_columns` cannot map them all, so the
        caller can fall back to finite differences instead of getting a wrong
        column. Evaluated at the CURRENT parameter values, so call it right after
        (or instead of) ``y()``.
        """
        mapped = self.parameter_columns(names)
        if mapped is None:
            return None
        cols, signs = mapped
        p, num_boxes, include_protein, vol_calc, mkey = self._prepare()
        live = cols >= 0
        out = np.zeros((len(np.atleast_1d(self.x)), len(cols)))
        if np.any(live):
            out[:, live] = xb.jac(p, cols[live]) * signs[live]
        return out

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        p, num_boxes, include_protein, vol_calc, mkey = self._prepare()

        if self.__fit__:
            logL, R = xb.sim(p)
        else:
            # One Julia call for both. Asking for the reflectivity and the profile
            # separately evaluates the model twice, and XModFit never actually sets
            # __fit__ on the model, so that doubling would apply to every single
            # likelihood evaluation in a fit or an MCMC chain.
            logL, R, prot_top, z, edp = xb.sim_edp(p)
            self.output_params['%s_EDP' % mkey] = {
                'x': z, 'y': edp, 'names': ['z (Angs)', 'rho (el/Angs^3)'],
                'plotType': 'line'}
            self.output_params['%s_dEDP' % mkey] = {
                'x': z[:-1], 'y': np.diff(edp) / np.diff(z),
                'names': ['z (Angs)', 'drho/dz (el/Angs^4)'], 'plotType': 'line'}
            self.output_params['scaler_parameters']['Nboxes'] = num_boxes
            # Both are load-time constants; eval_smeared! is O(B^2), so this is what
            # the protins/sig BOUNDS are costing. Tighten them if slabs is large.
            _M, _B = xb.slab_count()
            if _B is not None:
                self.output_params['scaler_parameters']['slabs'] = _B
                self.output_params['scaler_parameters']['mesh_box_Angs'] = float(
                    np.max(self.__d__[mkey][1:-1]) * self.spacez
                    / max(self.box_ref_length, 1e-9))
            self.output_params['scaler_parameters']['protein'] = int(include_protein)
            if include_protein:
                # Downstream diagnostics only -- computed here, never fed back into
                # the model, so they cannot perturb the fit or the derivatives.
                scale = self.cov_scale(vol_calc)
                self.output_params['scaler_parameters']['cov_scale'] = scale
                self.output_params['scaler_parameters']['coverage'] = self.cov / scale
                self.output_params['scaler_parameters']['footprint_Angs2'] = (
                    pg.projected_area(self.fname, self.theta, self.phi, self.fslice))
                self.output_params['scaler_parameters']['prot_volume_Angs3'] = (
                    pg.union_volume(self.fname, self.fslice))
                # The insertion in the OLD (top-of-protein) convention. Julia hands
                # it back from the rotation it just did, so it is free and cannot
                # drift from what the model actually evaluated; the Python
                # com_extent path reproduces it and is what carries the error bar
                # and converts a chain. Cross-checked in validate_xpr.py.
                self.output_params['scaler_parameters']['protins_top'] = prot_top
                self.output_params['scaler_parameters']['maxh'] = (
                    prot_top - self.protins)
        return R


if __name__ == '__main__':
    x = np.linspace(0.02, 0.7, 200)
    fun = XLayersPDB_XPR(x=x, fname=None, use_protein=False)
    print(fun.y())
