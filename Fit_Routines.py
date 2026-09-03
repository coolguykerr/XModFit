import numpy as np
from lmfit import fit_report, Minimizer, conf_interval, printfuncs
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox
import traceback
import os

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

    def set_x(self,x,y=None,yerr=None,mask=None):
        """Install data. ``mask`` is a full-length boolean array (or dict of them,
        keyed like x) marking points ELIGIBLE for the fit.

        mask=None CLEARS any previously installed mask: set_x means "new data",
        and keeping a stale mask sized for the old x is a silent misalignment.
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
                    y.append((np.log10(data)-np.log10(fit))*data/err)
                elif fit_scale=='Log w/o error':
                    y.append(np.log10(data) - np.log10(fit))
                elif fit_scale=='Linear w/o error':
                    y.append(data - fit)
                else:
                    y.append((data-fit)/err)
            y=np.concatenate(y)
        else:
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
                y= (np.log10(data)-np.log10(fit))*data/err
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
            Log               r = (lg(data)-lg(fit))*data/err
                                                       dr/dp = -(data/(err*fit*ln10)) dfit/dp
            Log w/o error     r = lg(data)-lg(fit)     dr/dp = -(1/(fit*ln10)) dfit/dp
            Linear w/o error  r = data-fit             dr/dp = -dfit/dp
        """
        for key in self.fit_params.keys():
            self.params[key] = fit_params[key].value
            self.fit_params[key].value = fit_params[key].value
        names = [n for n, p in fit_params.items() if p.vary]
        self._install_func_x()
        jac = self.func.dydp(names)
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
            return -(data / (err * fit * ln10))[:, None] * jac
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
                    mask=None):
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
            self._canonicalize_result()
            return fit_report(self.result),self.result.message
        else:
            self.result = self.fitter.minimize(method=fit_method, params=self.emcee_params, burn=0, steps=emcee_steps, thin=emcee_thin, is_weighted=True,
                                        nwalkers=emcee_walkers, workers=emcee_cores, reuse_sampler=reuse_sampler)
            self._canonicalize_result()
            return fit_report(self.result), 'None'

    def _canonicalize_result(self):
        """Let the model put periodic/degenerate parameters back in their window.

        A model may define canonical_params(params) to fold parameters whose value
        is only defined modulo something -- XLayersPDB_XPR's Euler angles, which are
        unbounded so the optimiser and the emcee moves can cross the poles freely.
        Called before fit_report so the report, the parameter table and any saved
        .par all show the conventional value rather than the optimiser's raw
        coordinate. It re-labels an identical model, so chi2 and the residual are
        untouched.
        """
        canon = getattr(self.func, 'canonical_params', None)
        if canon is None or self.result is None:
            return
        try:
            canon(self.result.params)
            canon(self.fit_params)
        except Exception:
            pass


    def set_emcee_minimizer(self,fit_scale='Linear',emcee_walkers=100, emcee_steps=100, emcee_cores=1,
                            emcee_burn=0, reuse_sampler=False, emcee_thin=1,backend=None,seed_temperature=1.0):

        funcname = type(self.func).__name__
        if funcname in ('XLayersPDB', 'XLayersPDB_dev_smooth', 'XLayersPDB_XPR'):
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

    def perform_pymc(self):
        pass



    def perform_emcee(self):
        pass



        
        