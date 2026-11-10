import numpy as np
from lmfit import fit_report, Minimizer, conf_interval, printfuncs
from PyQt5.QtCore import pyqtSignal, QObject
from lmfit.minimizer import PDB_FUNCNAMES
from PyQt5.QtWidgets import QMessageBox
import traceback
import os
import covariance_core as cc


class _CovDowngrade(str):
    """A ``Fit.warnings`` entry recording a covariance downgrade (spec 6.5): the
    fit continued in a weaker mode. Any of these puts ``DEGRADED FIT:`` first."""

def _window_indices(x, lo, hi):
    """(imin, imax) inclusive for lo <= x <= hi, clamped to the full extent.

    Replaces three inconsistent copies of this logic. The dict branch of
    perform_fit and JointFit._set_ranges used ``0.99*lo`` while the scalar branch
    used ``lo`` exactly; for a NEGATIVE lo the 0.99 factor moves the bound the
    wrong way (0.99 * -1 = -0.99, i.e. tighter, where for positive lo it is
    looser), so the two branches disagreed in opposite directions.

    The clamping is inherited from JointFit._set_ranges, which already guarded a
    degenerate window this way. It is more forgiving than the individual path used
    to be -- that raised IndexError when the window missed the data entirely -- so
    a typo'd window would now silently fit EVERYTHING. It says so instead.
    """
    x = np.asarray(x)
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    a = np.where(x >= lo)[0]
    b = np.where(x <= hi)[0]
    clamped = (len(a) == 0) or (len(b) == 0)
    imin = int(a[0]) if len(a) else 0
    imax = int(b[-1]) if len(b) else len(x) - 1
    if imax < imin:                     # degenerate window: use everything
        imin, imax = 0, len(x) - 1
        clamped = True
    if clamped and len(x):
        print('XModFit: fit window %r:%r does not overlap the data '
              '(x spans %.6g..%.6g) -- falling back to points %d:%d, i.e. the '
              'whole range. Check Xmin:Xmax.'
              % (lo, hi, float(np.min(x)), float(np.max(x)), imin, imax))
    return imin, imax


class Fit(QObject):
    functionCalled=pyqtSignal(dict,int,np.ndarray,str)
    covariance_mode='model'     # 'model' | 'data' (spec 4.4); perform_fit may override
    cov_max_rounds=5            # model-based outer-iteration round limit (spec 4.4)
    def __init__(self,funClass,x,y=None,yerr=None,mask=None,parent=None):
        QObject.__init__(self,parent)

        self.set_x(x,y=y,yerr=yerr,mask=mask)
        self.func=funClass()
        self.func.x=self.x
        self.params=self.func.__dict__
        self.func.init_params()
        self.fit_params=self.func.params
        self.function_calls=0
        # Opt out with fit.use_jacobian = False to force finite differences.
        self.use_jacobian=True
        self.fitter=None
        self.result=None
        self.Niter=0


    def sync_param(self):
        for key in self.fit_params.keys():
            self.fit_params[key].value=self.params[key]

    def set_x(self,x,y=None,yerr=None,mask=None,cov=None):
        """Install data. ``mask`` is a full-length boolean array (or dict of them,
        keyed like x) marking points ELIGIBLE for the fit.

        mask=None CLEARS any previously installed mask: set_x means "new data",
        and keeping a stale mask sized for the old x is a silent misalignment.

        cov=None likewise CLEARS any attached covariance (spec 6.2): the GUI moves
        one Fit between datasets, and a covariance belongs to the dataset. A
        non-None cov goes through attach_covariance (CovarianceError on mismatch).
        """
        self._imask_frozen=None
        self._yfit_masked=False
        if type(x)==dict:
            self.x=x
            self.yerr={}
            self.y={}
            self.imin={}
            self.imax={}
            for key in self.x:
                if yerr is None:
                    self.yerr[key]=np.ones_like(self.x[key])
                else:
                    self.yerr[key]=yerr[key]
                if y is None:
                    self.y[key]=np.zeros_like(self.x[key])
                else:
                    self.y[key]=y[key]
                self.imin[key] = 0
                self.imax[key] = len(self.x[key])
        else:
            self.x=x
            if yerr is None:
                self.yerr=np.ones_like(x)
            else:
                self.yerr=yerr
            if y is None:
                self.y=np.zeros_like(x)
            else:
                self.y=y
            self.imin=0
            self.imax=len(self.x)
        self.set_mask(mask)
        self._cov_reset(cov)

    # ------------------------------------------------------------- masking ---
    # A per-point eligibility mask that COMPOSES with imin/imax rather than
    # replacing it. imin/imax keep their meaning (the contiguous x window), so the
    # many GUI sites that slice with them stay correct; the mask is an additional
    # refinement for limits that are NOT intervals in index space -- e.g. a
    # q_xy x q_z rectangle on a 2D map whose x is a row index.
    #
    # The mask is FROZEN once per perform_fit and never recomputed inside
    # residual(): lmfit's leastsq requires a constant residual length across
    # iterations, so a mask that could change per-iteration is a latent crash.
    def set_mask(self,mask):
        """Install/replace the mask without touching x/y/yerr. Validates length."""
        if mask is None:
            self.mask=None
            self._imask_frozen=None
            return
        if type(self.x)==dict:
            m={}
            for key in self.x:
                mk=mask.get(key) if hasattr(mask,'get') else None
                if mk is None:
                    m[key]=None
                    continue
                mk=np.asarray(mk,bool)
                if len(mk)!=len(self.x[key]):
                    raise ValueError('mask for %r has %d points but x has %d'
                                     %(key,len(mk),len(self.x[key])))
                m[key]=mk
            self.mask=m if any(v is not None for v in m.values()) else None
        else:
            mask=np.asarray(mask,bool)
            if len(mask)!=len(self.x):
                raise ValueError('mask has %d points but x has %d'
                                 %(len(mask),len(self.x)))
            self.mask=mask
        self._imask_frozen=None

    def imask_for(self,key=None):
        """The mask restricted to the current [imin, imax+1] window, or None.

        Returns the value frozen by perform_fit when there is one, else derives it
        on the fly. That is what lets both call paths work: runFit passes FULL
        arrays and lets perform_fit compute imin/imax and freeze, while update_plot
        pre-slices the arrays and calls residual() directly with imin=0.
        """
        if self._imask_frozen is not None:
            return self._imask_frozen[key] if key is not None else self._imask_frozen
        if self.mask is None:
            return None
        if key is not None:
            mk=self.mask.get(key) if hasattr(self.mask,'get') else None
            if mk is None:
                return None
            return mk[self.imin[key]:self.imax[key]+1]
        return self.mask[self.imin:self.imax+1]

    def _freeze_imask(self):
        if self.mask is None:
            self._imask_frozen=None
            return
        if type(self.x)==dict:
            self._imask_frozen={}
            for key in self.x:
                mk=self.mask.get(key) if hasattr(self.mask,'get') else None
                self._imask_frozen[key]=(None if mk is None else
                                         mk[self.imin[key]:self.imax[key]+1])
        else:
            self._imask_frozen=self.mask[self.imin:self.imax+1]

    def _count_active(self):
        """Number of points that will enter the residual vector."""
        if type(self.x)==dict:
            n=0
            for key in self.x:
                w=self.imax[key]-self.imin[key]+1
                im=self.imask_for(key)
                n+=w if im is None else int(np.count_nonzero(im[:w]))
            return n
        w=self.imax-self.imin+1
        im=self.imask_for()
        return w if im is None else int(np.count_nonzero(im[:w]))

    def active_index(self,key=None):
        """Integer indices of the points actually fitted: arange(imin, imax+1)
        filtered by the window mask. Single source of truth for "what is in the
        fit" -- use it for chi2/dof reporting and for the MCMC data fingerprint,
        which would otherwise not notice a change of mask."""
        if type(self.x)==dict:
            if key is None:
                raise ValueError('active_index needs a key for dict-x data')
            idx=np.arange(self.imin[key],self.imax[key]+1)
        else:
            idx=np.arange(self.imin,self.imax+1)
        im=self.imask_for(key)
        return idx if im is None else idx[im[:len(idx)]]

    def evaluate(self):
        # self.func.__fit__=False
        # A model may declare __masked_x__ = True to receive ONLY the points that
        # will enter the residual, which saves evaluating the excluded ones (for a
        # 2D map windowed from 39k rows to 5k that is ~8x). Opt-in, because a gappy
        # x is only safe for models that are pointwise in x or that look x up as an
        # index (e.g. GID2D's row_lookup). It is NOT safe for anything integrating
        # or differencing over self.x (np.trapz, np.diff, FFT), nor for anything
        # using self.x[0]/[-1] as a normalization anchor.
        self._install_func_x()
        self.yfit = self.func.y()
        return self.yfit

    def _install_func_x(self):
        """Point func.x at the windowed (and possibly masked) abscissa.

        Split out of evaluate() so residual_jac can reproduce exactly the same x
        without also evaluating the model.
        """
        masked_x = bool(getattr(self.func, '__masked_x__', False)) and self.mask is not None
        self._yfit_masked = False
        if type(self.x) == dict:
            self.func.x = {}
            for key in self.x.keys():
                seg = self.x[key][self.imin[key]:self.imax[key] + 1]
                im = self.imask_for(key) if masked_x else None
                self.func.x[key] = seg if im is None else seg[im[:len(seg)]]
            self._yfit_masked = masked_x
        else:
            seg = self.x[self.imin:self.imax + 1]
            im = self.imask_for() if masked_x else None
            if im is not None:
                seg = seg[im[:len(seg)]]
                self._yfit_masked = True
            self.func.x = seg
        
    # def evaluate4fit(self):
    #     self.func.__fit__=True
    #     if type(self.x)==dict:
    #         self.func.x={}
    #         for key in self.x.keys():
    #             self.func.x[key]=self.x[key][self.imin[key]:self.imax[key]+1]
    #     else:
    #         self.func.x = self.x[self.imin:self.imax + 1]
    #     self.yfit=self.func.y()
    #     return self.yfit
    
    def residual(self, fit_params, fit_scale):
        for key in self.fit_params.keys():
            self.params[key]=fit_params[key].value
            self.fit_params[key].value=fit_params[key].value
        self.__fit__=True
        yfit=self.evaluate()
        if type(yfit)==dict:
            y=[]
            for key in yfit.keys():
                data=self.y[key][self.imin[key]:self.imax[key]+ 1]
                err=self.yerr[key][self.imin[key]:self.imax[key]+1]
                fit=yfit[key]
                # Mask BEFORE maxd and before any log. Two independent reasons:
                #  * maxd is a data-derived normalization, so an excluded hot pixel
                #    would otherwise still rescale the whole residual vector --
                #    exactly the failure the window exists to prevent;
                #  * the Log branch takes log10(data), and excluded rows with
                #    data <= 0 give -inf/NaN, which nan_policy='raise' turns into
                #    an aborted fit. Masking first makes them unreachable.
                im=self.imask_for(key)
                if im is not None:
                    im=im[:len(data)]
                    data,err=data[im],err[im]
                    if not getattr(self,'_yfit_masked',False):
                        fit=fit[im]      # else evaluate() already masked it
                maxd=np.max(data)
                data=data/maxd
                err=err/maxd
                fit=fit/maxd
                dnorm = data[0]
                fnorm = fit[0]
                if fit_scale=='Log':
                    y.append((np.log(data)-np.log(fit))*data/err)
                elif fit_scale=='Log w/o error':
                    y.append(np.log10(data) - np.log10(fit))
                elif fit_scale=='Linear w/o error':
                    y.append(data - fit)
                else:
                    y.append((data-fit)/err)
            y=np.concatenate(y)
        else:
            if self._cov_active(fit_scale):  # whitened path (spec 6.2); see _cov_residual
                return self._cov_residual(fit_params, yfit, fit_scale)
            data=self.y[self.imin:self.imax+1]
            err=self.yerr[self.imin:self.imax+1]
            fit = yfit
            im=self.imask_for()          # see the dict branch for why this is here
            if im is not None:
                im=im[:len(data)]
                data,err=data[im],err[im]
                if not getattr(self,'_yfit_masked',False):
                    fit=np.asarray(fit)[im]
            if fit_scale=='Log':
                # Natural log, so the residual is normalised: sigma_ln(y) = err/y.
                # log10 here understated every residual by ln10 (chi2 by ~5.3x).
                # pdb_logprob / pdb_seed_worker / pdb_joint_logprob and
                # JointFit.residual mirror this and must stay identical.
                y= (np.log(data)-np.log(fit))*data/err
            elif fit_scale=='Log w/o error':
                y = (np.log10(data)-np.log10(fit))
            elif fit_scale=='Linear w/o error':
                y = (data-fit)
            else:
                y = (data-fit)/err
        self.Niter+=1
        self.functionCalled.emit(fit_params, self.Niter, y, fit_scale)
        return y

    # ------------------------------------------------------- analytic Jacobian ---
    # A model may expose dydp(names) -> d(model)/d(param), shape (npts, len(names)),
    # and parameter_columns(names) -> None when it cannot differentiate them all.
    # XLayersPDB_XPR does, via forward-mode AD in Julia. Supplying it to leastsq
    # replaces the (nvarys+1) finite-difference evaluations per iteration with one
    # evaluation plus one exact Jacobian.
    def can_jac(self, fit_params, fit_scale):
        """Whether an analytic Jacobian is available for THIS fit."""
        if not getattr(self, 'use_jacobian', True):
            return False
        if type(self.x) == dict:
            return False                      # joint fits: not wired up
        if not hasattr(self.func, 'dydp'):
            return False
        if any(p.expr is not None for p in fit_params.values()):
            return False                      # constraints need a chain rule we skip
        names = [n for n, p in fit_params.items() if p.vary]
        # Below ~4 varying parameters one AD pass costs about what the (nvarys+1)
        # finite-difference evaluations do, and the model pays a one-off Julia
        # specialisation on top. Measured on XLayersPDB_XPR: a clear win from 4 up,
        # a loss at 2.
        if len(names) < 4:
            return False
        cols = getattr(self.func, 'parameter_columns', None)
        try:
            return cols is not None and cols(names) is not None
        except Exception:
            return False

    def residual_jac(self, fit_params, fit_scale):
        """d(residual)/d(varying parameter), shaped (nresid, nvarys).

        Mirrors residual()'s scaling term by term, so the two stay consistent:

            Linear            r = (data-fit)/err       dr/dp = -(1/err) dfit/dp
            Log               r = (ln(data)-ln(fit))*data/err
                                                       dr/dp = -(data/(err*fit)) dfit/dp
            Log w/o error     r = lg(data)-lg(fit)     dr/dp = -(1/(fit*ln10)) dfit/dp
            Linear w/o error  r = data-fit             dr/dp = -dfit/dp
        """
        for key in self.fit_params.keys():
            self.params[key] = fit_params[key].value
            self.fit_params[key].value = fit_params[key].value
        names = [n for n, p in fit_params.items() if p.vary]
        self._install_func_x()
        jac = self.func.dydp(names)
        if self._cov_active(fit_scale):      # whitened, same rows as _cov_residual
            return self._cov_residual_jac(jac, fit_scale)
        data = self.y[self.imin:self.imax + 1]
        err = self.yerr[self.imin:self.imax + 1]
        im = self.imask_for()
        if im is not None:
            im = im[:len(data)]
            data, err = data[im], err[im]
            if not getattr(self, '_yfit_masked', False):
                jac = jac[im]
        if fit_scale == 'Linear':
            return -jac / err[:, None]
        if fit_scale == 'Linear w/o error':
            return -jac
        fit = np.asarray(self.func.y())
        if im is not None and not getattr(self, '_yfit_masked', False):
            fit = fit[im]
        ln10 = np.log(10.0)
        if fit_scale == 'Log':
            return -(data / (err * fit))[:, None] * jac
        return -(1.0 / (fit * ln10))[:, None] * jac

    def log_likelihood(self, fit_params, fit_scale):
        residuals = self.residual(fit_params, fit_scale)
        return -0.5 * np.sum(residuals**2)  # + np.log(sigma2))

    def log_prior(self, paramkeys):
        window = True
        for key in paramkeys:
            window = window and (self.func.params[key].min < self.func.params[key].value < self.func.params[key].max)
        if window:
            return 0.0
        return -np.inf

    def log_probability(self, fit_params, fit_keys, fit_scale):
        """
        Calculates logarithm of the Chisqr probablity distribution
        :param fit_params: list of values for all the fitting parameters
        :param fit_keys:  list of names/keys of fitting parameters
        :param fit_scale:  scale used for fitting between ['Linear', 'Log', 'Log w/o error', 'Linear w/o error']
        :return:
        """
        for i, key in enumerate(fit_keys):
            self.func.params[key].value = fit_params[i]
            self.fit_params[key].value=fit_params[i]
        lp = self.log_prior(fit_keys)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(self.fit_params, fit_scale)
        
    def callback(self,params,iterations,residual,fit_scale):
        """
        """
        #self.Niter += 1
        #print(self.Niter, ': I m here')
        #self.functionCalled.emit(params,iterations,residual,fit_scale)
        if self.fit_abort:
            return True
        else:
            return None
    
    
    def perform_fit(self,xmin,xmax,fit_scale='Linear',fit_method='leastsq',maxiter=1,emcee_walkers=100, emcee_steps=100,
                    emcee_burn=0, emcee_cores=1, reuse_sampler=False, emcee_thin=1,backend=None,seed_temperature=1.0,
                    mask=None, covariance_mode=None):
        self.Niter=0
        #self.sync_param()
        self.fit_abort=False
        if mask is not None:
            self.set_mask(mask)
        if type(self.x)==dict:
            self.imin = {}
            self.imax = {}
            for key in self.x.keys():
                self.imin[key],self.imax[key]=_window_indices(self.x[key],xmin,xmax)
        else:
            self.imin, self.imax = _window_indices(self.x, xmin, xmax)
        # freeze the window mask so len(residual) is constant across iterations
        self._freeze_imask()
        self.n_active=self._count_active()
        if self.n_active==0:
            raise ValueError('the fit window and column limits leave 0 active '
                             'points (x window %r:%r) -- widen them before fitting'
                             %(xmin,xmax))
        if self._cov_begin(fit_scale, covariance_mode):   # covariance in use: see _cov_perform_fit
            return self._cov_perform_fit(fit_scale, fit_method, maxiter, emcee_walkers, emcee_steps,
                                         emcee_burn, emcee_cores, reuse_sampler, emcee_thin, backend,
                                         seed_temperature)
        if fit_method=='leastsq':
            # Only leastsq: lmfit wraps Dfun in Minimizer.__jacobian, which applies
            # the bounded->internal gradient rescaling. The least_squares path takes
            # its own `jac` with a different signature and is left on finite
            # differences.
            jkws = {'Dfun': self.residual_jac} if self.can_jac(self.fit_params, fit_scale) else {}
            self.fitter=Minimizer(self.residual,self.fit_params,fcn_args=(fit_scale,),iter_cb=self.callback,nan_policy='raise',max_nfev=maxiter,**jkws)
        elif fit_method=='differential_evolution':
            self.fitter=Minimizer(self.residual,self.fit_params,fcn_args=(fit_scale,),iter_cb=self.callback,
                                  nan_policy='raise', calc_covar=True, max_nfev=maxiter, popsize=300, updating='immediate')
        elif fit_method=='brute':
            self.fitter=Minimizer(self.residual,self.fit_params,fcn_args=(fit_scale,),iter_cb=self.callback,nan_policy='raise')
        elif fit_method =='emcee':
            self.fitter=self.set_emcee_minimizer(fit_scale=fit_scale,emcee_walkers=emcee_walkers,emcee_steps=emcee_steps,
                                     emcee_cores=emcee_cores,emcee_burn=emcee_burn,reuse_sampler=reuse_sampler,
                                     emcee_thin=emcee_thin,backend=backend,seed_temperature=seed_temperature)
        else:
            self.fitter = Minimizer(self.residual, self.fit_params, fcn_args=(fit_scale,), iter_cb=self.callback,
                                    nan_policy='raise',max_nfev=maxiter)
        if fit_method!='emcee':
            self.result = self.fitter.minimize(method=fit_method, params=self.fit_params)
            extra=self._canonicalize_result()
            return self._report(extra),self.result.message
        else:
            self.result = self.fitter.minimize(method=fit_method, params=self.emcee_params, burn=0, steps=emcee_steps, thin=emcee_thin, is_weighted=True,
                                        nwalkers=emcee_walkers, workers=emcee_cores, reuse_sampler=reuse_sampler)
            extra=self._canonicalize_result()
            return self._report(extra), 'None'

    def _report(self, extra=''):
        """lmfit's fit_report, plus whatever the model derived from the result."""
        rep = fit_report(self.result)
        if getattr(self, '_cov_reported', False):    # a covariance was attached for this fit
            return self._cov_report(rep + '\n\n' + extra if extra else rep)
        return rep + '\n\n' + extra if extra else rep

    def _canonicalize_result(self):
        """Let the model put periodic/degenerate parameters back in their window.

        A model may define canonical_params(params) to fold parameters whose value
        is only defined modulo something -- XLayersPDB_XPR's Euler angles, which are
        unbounded so the optimiser and the emcee moves can cross the poles freely.
        Called before fit_report so the report, the parameter table and any saved
        .par all show the conventional value rather than the optimiser's raw
        coordinate. It re-labels an identical model, so chi2 and the residual are
        untouched.

        Returns the model's derived_report(result) text, if it has one, for
        appending to the fit report -- computed here because it has to run BEFORE
        the fold (see below) and this is the only place that owns that ordering.
        """
        # Derived quantities FIRST: a model may report a function of the fitted
        # parameters with an error propagated through result.covar, and covar is in
        # the optimiser's raw coordinates -- the ones canon is about to relabel. See
        # XLayersPDB_XPR.derived_params, which needs the un-folded angles to get the
        # SIGN of d(maxh)/d(theta) right.
        extra = ''
        if self.result is not None:
            derive = getattr(self.func, 'derived_report', None)
            if derive is not None:
                try:
                    extra = derive(self.result) or ''
                except Exception:
                    extra = ''
        canon = getattr(self.func, 'canonical_params', None)
        if canon is None or self.result is None:
            return extra
        try:
            canon(self.result.params)
            canon(self.fit_params)
        except Exception:
            pass
        return extra


    def set_emcee_minimizer(self,fit_scale='Linear',emcee_walkers=100, emcee_steps=100, emcee_cores=1,
                            emcee_burn=0, reuse_sampler=False, emcee_thin=1,backend=None,seed_temperature=1.0):

        funcname = type(self.func).__name__
        if funcname in PDB_FUNCNAMES:
            if any([self.result.params[key].stderr is None for key in self.result.params.keys() if
                    self.result.params[key].vary == 1]):
                fiterr = np.zeros(len([self.result.params[key].value for key in self.result.params.keys() if
                                       self.result.params[key].vary == 1]))
            else:
                fiterr = np.array([self.result.params[key].stderr / self.result.params[key].value
                                   for key in self.result.params.keys() if self.result.params[key].vary == 1])
        else:
            fiterr = 0
        self.emcee_params = self.result.params.copy()
        if 'w/o' in fit_scale:
            self.emcee_params.add('__lnsigma', value=-1.0, vary=True, min=np.log(0.001), max=np.log(2.0))
            fitter = Minimizer(self.residual, self.emcee_params, fcn_args=(fit_scale,),
                           iter_cb=self.callback,
                           nan_policy='propagate', burn=emcee_burn, steps=emcee_steps, thin=emcee_thin,
                           is_weighted=False, backend=backend, fiterr=fiterr, funcname=funcname,
                           nwalkers=emcee_walkers, workers=emcee_cores, reuse_sampler=reuse_sampler,
                           run_mcmc_kwargs={'store':True}, seed_temperature=seed_temperature)
        else:
            fitter = Minimizer(self.residual, self.emcee_params, fcn_args=(fit_scale,), iter_cb=self.callback,
                           nan_policy='propagate', burn=emcee_burn, steps=emcee_steps, thin=emcee_thin,
                           is_weighted=True, backend=backend, fiterr=fiterr, funcname=funcname,
                           nwalkers=emcee_walkers, workers=emcee_cores, reuse_sampler=reuse_sampler,
                           run_mcmc_kwargs={'store':True}, seed_temperature=seed_temperature)
        return fitter

    # ---------------------------------------------------------- covariance ---
    # General correlated-error fitting (handoff/17-spec-general-covariance.md 6.2).
    # Everything below is reached only through one guarded line in set_x,
    # residual, residual_jac, perform_fit and _report, so with no covariance
    # attached the upstream code path is untouched. A one-dataset JointFit with
    # the same covariance is the reference (validate_covariance.py S1).
    #
    # Rows of the whitened residual: slice [imin, imax] -> imask -> finite mask
    # (finite y, sigma > 0, finite intensity; Log also y > 0, intensity > 0).
    # The finite mask is built together with L and both are frozen for a whole
    # perform_fit (CLAUDE.md rule 4). Called outside perform_fit (update_plot's
    # pre-sliced arrays), L is built lazily on the first residual call -- in
    # model mode from the model at those parameters -- and cached on
    # (fit_scale, n_active, mode); set_x, attach_covariance and every
    # outer-iteration round invalidate it.
    def _cov_reset(self, cov=None):
        """Clear any covariance and its cached factor, then attach ``cov``."""
        self.cov = None
        self._cov_cache = None
        self._cov_dec = None
        self._cov_fit_mode = None
        # new data -> no report state from the previous dataset survives (the GUI
        # moves one Fit between datasets; _report/chi2_stat_for_ci read these)
        for k, v in (('warnings', []), ('cov_info', {}), ('_cov_last_active', False),
                     ('_cov_reported', False), ('_cov_warnings', []), ('_cov_notes', []),
                     ('_cov_factor_warnings', []), ('_cov_jitter', 0.0),
                     ('pending_warnings', [])):
            setattr(self, k, v)
        if cov is not None:
            self.attach_covariance(cov)

    def attach_covariance(self, cov):
        """Attach a covariance_core.DatasetCovariance (or a legacy cov dict) to the
        current data, or detach with None. For array-x data it must belong to the
        data: CovarianceError if its length differs or match_problems(x, y) is not
        empty. For dict-x data it is stored unchecked and the fit ignores it with
        a warning (spec 6.5)."""
        self._cov_cache = None
        self._cov_dec = None
        if cov is None:
            self.cov = None
            return
        if type(self.x) == dict:
            self.cov = cov
            return
        if not isinstance(cov, cc.DatasetCovariance):
            cov = cc.from_legacy_meta(cov, self.x, source='<dict>', loader='legacy-dict')
        probs = cov.match_problems(np.asarray(self.x, float), self.y)
        if cov.n != len(self.x) and not probs:
            probs = ['covariance has %d points but x has %d' % (cov.n, len(self.x))]
        if probs:
            self.cov = None
            raise cc.CovarianceError('covariance %s does not match the data: %s'
                                     % (cov.source or '<in memory>', '; '.join(probs)))
        self.cov = cov

    def chi2_stat_for_ci(self):
        """('chisqr', chi2) when a covariance was active in the last fit (absolute
        errors: the CI threshold is chi2_min + delta-chi2), else ('redchi', redchi)."""
        if getattr(self, '_cov_last_active', False):
            return 'chisqr', self.result.chisqr
        return 'redchi', self.result.redchi

    def _cov_name(self):
        return getattr(self, 'name', '') or 'dataset'

    def _cov_decide(self, fit_scale):
        """(covariance to use or None, [downgrade reasons]) for the current state."""
        cov = getattr(self, 'cov', None)
        if cov is None:
            return None, []
        src = getattr(cov, 'source', None) or '<in memory>'   # a legacy dict has none
        if type(self.x) == dict:
            return None, ['covariance %s ignored: %s is a multi-curve (dict-x) dataset, '
                          'which the covariance path does not support; fitted without it'
                          % (src, self._cov_name())]
        if fit_scale not in ('Linear', 'Log'):
            return None, ["covariance ignored: fit scale %r is unweighted, so the fit runs "
                          "exactly as without a covariance (use 'Linear' or 'Log')"
                          % (fit_scale,)]
        probs = cov.match_problems(np.asarray(self.x, float), self.y)
        if cov.n != len(self.x) and not probs:
            probs = ['covariance has %d points but x has %d' % (cov.n, len(self.x))]
        if probs:
            if cov.replaces_y is not None and cov.n == len(self.x):
                return cov.without_terms(), [
                    'covariance %s does not match the data (%s); fitted without its '
                    'correlated terms (corrected y with the measurement errors only)'
                    % (src, '; '.join(probs))]
            return None, ['covariance %s does not match the data (%s); fitted without it'
                          % (src, '; '.join(probs))]
        return cov, []

    def _cov_active(self, fit_scale):
        """True when the whitened residual applies to this call."""
        if getattr(self, 'cov', None) is None or type(self.x) == dict \
                or fit_scale not in ('Linear', 'Log'):
            return False
        if self._cov_dec is None or self._cov_dec[0] != fit_scale:
            self._cov_dec = (fit_scale,) + self._cov_decide(fit_scale)
        return self._cov_dec[1] is not None

    def _cov_index(self):
        """Indices (into the full arrays) of the rows residual() fits: the
        [imin, imax] slice, then the window mask -- exactly the stock slicing."""
        idx = np.arange(len(self.x))[self.imin:self.imax + 1]
        im = self.imask_for()
        return idx if im is None else idx[im[:len(idx)]]

    def _cov_masked(self, arr):
        """A model-side array (yfit or dydp) restricted to the imask rows, unless
        evaluate() already did that (__masked_x__)."""
        arr = np.asarray(arr, dtype=float)
        im = self.imask_for()
        if im is None or getattr(self, '_yfit_masked', False):
            return arr
        n = len(self.y[self.imin:self.imax + 1])
        return arr[im[:n]]

    def _cov_rows(self):
        use = self._cov_dec[1]
        idx = self._cov_index()
        d = np.asarray(use.y_for_fit(self.y), dtype=float)[idx]
        sig = np.asarray(use.sigma_meas, dtype=float)[idx]
        return idx, d, sig

    def _cov_mode(self):
        return self._cov_fit_mode or self.covariance_mode

    def _cov_needs(self, fit_scale):
        return cc.needs_intensity([cc.Block(self._cov_dec[1], np.zeros(0), np.zeros(0),
                                            np.zeros(0))], fit_scale)

    def _cov_build(self, fit_scale, idx, d, sig, inten, rows=None):
        """Assemble C on the active rows, factorise, cache (L, rows). ``rows`` is a
        frozen finite mask (model rebuilds within one perform_fit); a model
        intensity unusable on a frozen row falls back to the data there."""
        scale = fit_scale
        inten = np.asarray(inten, dtype=float)
        if rows is not None:
            m = rows
            with np.errstate(invalid='ignore'):
                bad = m & ~np.isfinite(inten)
                if scale == 'Log':
                    bad |= m & ~(inten > 0)
            if bad.any():
                w = ('model intensity not usable at %d active points (non-finite%s); the '
                     'data are used there' % (int(bad.sum()), ' or <= 0' if scale == 'Log' else ''))
                if w not in self._cov_warnings:
                    self._cov_warnings.append(w)
                inten = np.where(bad, d, inten)
        else:
            with np.errstate(invalid='ignore'):
                m = np.isfinite(d) & np.isfinite(sig) & (sig > 0) & np.isfinite(inten)
                if scale == 'Log':
                    m &= (d > 0) & (inten > 0)
            nd = int(len(m) - m.sum())
            if nd:
                w = ('covariance fit (%s): points excluded (non-finite, sigma <= 0%s): %s: %d'
                     % (scale, ' or y <= 0' if scale == 'Log' else '', self._cov_name(), nd))
                if w not in self._cov_notes:
                    self._cov_notes.append(w)
        keep = np.flatnonzero(m)
        cs = self._cov_dec[1].subset(idx[keep])
        C, awarn = cc.assemble([cc.Block(cs, d[keep], sig[keep], inten[keep],
                                         name=self._cov_name())], scale)
        L, fwarn = cc.factorize(C)
        self._cov_jitter = 0.0
        if any('jitter' in w for w in fwarn):
            from joint_covariance import cholesky_with_jitter
            self._cov_jitter = float(cholesky_with_jitter(C)[1])
        for w in awarn + fwarn:
            if w not in self._cov_warnings:
                self._cov_warnings.append(w)
        self._cov_factor_warnings = list(fwarn)
        self._cov_cache = {'key': (fit_scale, idx.tobytes(), self._cov_mode()), 'L': L, 'rows': m}

    def _cov_state(self, fit_scale, idx, d, sig, f):
        """The cached (L, rows), built now if missing or keyed differently. A lazy
        build in model mode uses the model ``f`` when a term depends on it."""
        st = self._cov_cache
        if st is None or st['key'] != (fit_scale, idx.tobytes(), self._cov_mode()):
            model = self._cov_mode() == 'model' and self._cov_needs(fit_scale)
            self._cov_build(fit_scale, idx, d, sig, f if model else d)
            st = self._cov_cache
        if len(st['rows']) != len(d):
            raise RuntimeError('covariance rows %d != residual rows %d: the window mask '
                               'changed without a rebuild' % (len(st['rows']), len(d)))
        return st

    def _cov_residual(self, fit_params, yfit, fit_scale):
        """Whitened residual L^-1 (d - f) (Linear) or L^-1 (ln d - ln f) (Log), in
        absolute units; d is cov.y_for_fit(y) (legacy: y_corr). The masked rows are
        dropped before any log."""
        idx, d, sig = self._cov_rows()
        f = self._cov_masked(yfit)
        st = self._cov_state(fit_scale, idx, d, sig, f)
        m = st['rows']
        d, f = d[m], f[m]
        r = d - f if fit_scale == 'Linear' else np.log(d) - np.log(f)
        y = cc.whiten(st['L'], r)
        self.Niter += 1
        self.functionCalled.emit(fit_params, self.Niter, y, fit_scale)
        return y

    def _cov_residual_jac(self, jac, fit_scale):
        """L^-1 times -df/dp (Linear) or -(1/f) df/dp (Log), on _cov_residual's rows."""
        idx, d, sig = self._cov_rows()
        jac = self._cov_masked(jac)
        f = self._cov_masked(self.func.y())
        st = self._cov_state(fit_scale, idx, d, sig, f)
        m = st['rows']
        J = -jac[m] if fit_scale == 'Linear' else -(1.0 / f[m])[:, None] * jac[m]
        return cc.whiten_jac(st['L'], J)

    def _cov_push(self, params):
        for key in self.fit_params.keys():
            self.params[key] = params[key].value
            self.fit_params[key].value = params[key].value

    def _cov_rebuild_model(self, fit_scale, params):
        """Rebuild C from the model at ``params`` on the frozen rows."""
        rows = self._cov_cache['rows']
        self._cov_push(params)
        f = self._cov_masked(self.evaluate())
        idx, d, sig = self._cov_rows()
        self._cov_build(fit_scale, idx, d, sig, f, rows=rows)

    def _cov_begin(self, fit_scale, covariance_mode):
        """Start-of-fit bookkeeping (every perform_fit). Decides once whether the
        covariance applies and records each downgrade; True -> covariance path."""
        if covariance_mode is not None and covariance_mode not in ('model', 'data'):
            raise ValueError("covariance_mode must be 'model' or 'data', got %r"
                             % (covariance_mode,))
        self.warnings = list(getattr(self, 'pending_warnings', None) or [])   # e.g. a reload note
        self.pending_warnings = []
        self.cov_info = {}
        self._cov_warnings = []
        self._cov_notes = []
        self._cov_factor_warnings = []
        self._cov_jitter = 0.0
        self._cov_last_active = False
        self._cov_cache = None
        self._cov_dec = None
        self._cov_fit_mode = None
        self._cov_reported = getattr(self, 'cov', None) is not None
        if not self._cov_reported:
            return False
        use, reasons = self._cov_decide(fit_scale)
        for w in reasons:
            self.warnings.append(_CovDowngrade(w))
        if use is None:
            return False
        self._cov_dec = (fit_scale, use, reasons)
        self._cov_fit_mode = self.covariance_mode if covariance_mode is None else covariance_mode
        return True

    def _cov_finish(self, fit_scale, rounds, converged, mcmc_frozen=None):
        self.cov_info = {'mode': self._cov_fit_mode, 'scale': fit_scale, 'rounds': int(rounds),
                         'converged': bool(converged),
                         'factor_warnings': list(self._cov_factor_warnings),
                         'jitter': float(self._cov_jitter)}
        if mcmc_frozen:
            self.cov_info['mcmc_frozen'] = mcmc_frozen
        for w in self._cov_warnings:
            self.warnings.append(_CovDowngrade(w))
        for w in self._cov_notes:
            self.warnings.append(w)
        self._cov_last_active = True
        self.result._cov_stamp = self._cov_data_stamp()   # model-mode MCMC freezes C only on this

    def _cov_data_stamp(self):
        """Identity of the data on the active rows (x and y), for the MCMC freeze."""
        idx = self._cov_index()
        return hash((np.asarray(self.x, float)[idx].tobytes(),
                     np.asarray(self.y, float)[idx].tobytes()))

    def cov_residual_index(self, fit_scale):
        """Indices into x of the rows of the whitened residual the last residual()
        call returned, or None when no covariance applied (the GUI's residual plot)."""
        if not self._cov_active(fit_scale) or self._cov_cache is None:
            return None
        return self._cov_index()[self._cov_cache['rows']]

    def _cov_perform_fit(self, fit_scale, fit_method, maxiter, emcee_walkers, emcee_steps,
                         emcee_burn, emcee_cores, reuse_sampler, emcee_thin, backend,
                         seed_temperature):
        """perform_fit with a covariance in use. Mirrors the stock branches, plus:
        scale_covar=False on every Minimizer (absolute errors, spec 6.2); model mode
        with an intensity-dependent term -> covariance_core.outer_iterate; emcee
        serial (sampling and seed scan) with C frozen before sampling (spec 6.4)."""
        idx, d, sig = self._cov_rows()
        self._cov_build(fit_scale, idx, d, sig, d)       # round 0: data-based; freezes the rows
        needs = self._cov_fit_mode == 'model' and self._cov_needs(fit_scale)
        if fit_method == 'emcee':
            frozen = 'data'
            if needs:
                if self.result is None:
                    self.warnings.append(_CovDowngrade(
                        'model-based covariance needs a prior fit to freeze C at; none '
                        'exists, so MCMC uses the data-based covariance'))
                elif getattr(self.result, '_cov_stamp', None) != self._cov_data_stamp():
                    self.warnings.append(_CovDowngrade(
                        'model-based covariance needs a prior covariance fit of these data '
                        'to freeze C at; the last result belongs to other data (or was '
                        'fitted without the covariance), so MCMC uses the data-based '
                        'covariance'))
                else:
                    self._cov_rebuild_model(fit_scale, self.result.params)
                    frozen = 'model'
            if isinstance(emcee_cores, int) and emcee_cores > 1:
                self.warnings.append(_CovDowngrade(
                    'parallel MCMC requested with a covariance: sampling (and any '
                    'orientation seed scan) runs serial'))
            self.fitter = self.set_emcee_minimizer(fit_scale=fit_scale, emcee_walkers=emcee_walkers,
                                                   emcee_steps=emcee_steps, emcee_cores=1,
                                                   emcee_burn=emcee_burn, reuse_sampler=reuse_sampler,
                                                   emcee_thin=emcee_thin, backend=backend,
                                                   seed_temperature=seed_temperature)
            self.fitter.scale_covar = False
            self.result = self.fitter.minimize(method=fit_method, params=self.emcee_params, burn=0,
                                               steps=emcee_steps, thin=emcee_thin, is_weighted=True,
                                               nwalkers=emcee_walkers, workers=1,
                                               reuse_sampler=reuse_sampler)
            self._cov_finish(fit_scale, 0, True, mcmc_frozen=frozen)
            extra = self._canonicalize_result()
            return self._report(extra), 'None'
        common = dict(fcn_args=(fit_scale,), iter_cb=self.callback, nan_policy='raise',
                      scale_covar=False)
        if fit_method == 'leastsq':
            jkws = {'Dfun': self.residual_jac} if self.can_jac(self.fit_params, fit_scale) else {}
            self.fitter = Minimizer(self.residual, self.fit_params, max_nfev=maxiter, **common, **jkws)
        elif fit_method == 'differential_evolution':
            self.fitter = Minimizer(self.residual, self.fit_params, calc_covar=True, max_nfev=maxiter,
                                    popsize=300, updating='immediate', **common)
        elif fit_method == 'brute':
            self.fitter = Minimizer(self.residual, self.fit_params, **common)
        else:
            self.fitter = Minimizer(self.residual, self.fit_params, max_nfev=maxiter, **common)
        if needs:
            state = {'params': self.fit_params, 'last': None}

            def fit_once():
                if self.fit_abort and state['last'] is not None:
                    return state['last']             # aborted: stop iterating
                res = self.fitter.minimize(method=fit_method, params=state['params'])
                state['params'], state['last'] = res.params, res
                return res

            def rebuild(res):
                self._cov_rebuild_model(fit_scale, res.params)

            self.result, rounds, converged = cc.outer_iterate(fit_once, rebuild,
                                                              max_rounds=self.cov_max_rounds)
            if not converged:
                self.warnings.append(_CovDowngrade(
                    'model-based covariance did not converge in %d rounds (some varying '
                    'parameter still moved >= 0.1 stderr); the last result is kept' % rounds))
        else:
            self.result = self.fitter.minimize(method=fit_method, params=self.fit_params)
            rounds, converged = 0, True
        self._cov_finish(fit_scale, rounds, converged)
        extra = self._canonicalize_result()
        return self._report(extra), self.result.message

    def _cov_report(self, rep):
        """The stock report plus, with a covariance attached for this fit, the
        Covariance block (when it was used), a warnings block, and DEGRADED FIT
        on the first line if anything was downgraded (spec 6.5, 6.6)."""
        downs = [w for w in self.warnings if isinstance(w, _CovDowngrade)]
        if self.cov_info:
            rep = rep + '\n\n' + cc.report_block([(self._cov_name(), self._cov_dec[1])],
                                                 self.cov_info, downs, self.result)
        if self.warnings:
            rep = rep + '\n\nFit warnings:\n' + ''.join('  - %s\n' % w for w in self.warnings)
        if downs:
            rep = 'DEGRADED FIT: %s\n\n' % '; '.join(downs) + rep
        return rep

    def perform_pymc(self):
        pass



    def perform_emcee(self):
        pass



        
        