from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrices
import scipy.stats
from sklearn import linear_model
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

try:
    import pymc3 as pm
    backend = 'pymc3'
    print('Backend pymc3')
except:
    print('pymc3 not available, attempting to import pymc')
    try:
        import pymc as pm
        backend = 'pymc'
        print('Backend pymc')
    except:
        backend = None
        print('pymc3 and pymc not available; unable to use fully Bayesian methods')
        print('Backend None')


__all__ = ["MediationModel"]

#TODO: ADD SAVING MODEL OPTION FOR PYMC3 BACKEND

class MediationModel(object):
    """ Estimates an unconditional indirect effect based on a simple mediation model:

                                                  M
                                             (a) / \ (b)
                                                /   \   
                                               X ---- Y
                                                 (c)
                Models
                ------
                1. Mediator:   M ~ 1 + a*X
                2. Endogenous: Y ~ 1 + c*X + b*M
                Indirect effect: a*b
    
    Parameters
    ----------
    method : string
        Method for calculating confidence interval for unconditional indirect effect.
        Valid methods include: 'delta' : multivariate delta method
                               'boot' : nonparametric bootstrap
                               'bayesboot' : Bayesian bootstrap
                               'bayes-norm' : fully Bayesian model with normal priors
                               'bayes-robust' fully Bayesian model with robust Cauchy priors

    interval : string
        Method for interval estimation. Valid arguments depend on the method.
            - method = delta : 'first' or 'second' for order of Taylor series approximation
            - method = boot : 'perc' (percentile) or 'bc' (bias-corrected)
            - method = bayesboot : 'cred' (credible) or 'hpd' (highest posterior density)
            - method = bayes-norm or bayes-robust : 'cred' (credible) or 'hpd' (highest posterior density)

    mediator_type : string
        Variable indicating whether mediator variable is continuous or categorical

    endogenous_type : string
        Variable indicating whether endogenous variable is continuous or categorical

    alpha : float, default .05
        Type I error rate - corresponds to generating (1-alpha)*100 intervals

    plot : boolean, default False
        Whether to plot distribution (empirical sampling or posterior) of indirect effect. Need to specify
        bootstrap or Bayesian estimation.

    parameters : dict
        Dictionary of parameters for different estimation methods

        Expected keys for method = 'boot'
            - boot_samples : int
                Number of bootstrap samples
            - estimator : str
                Estimator for indirect effect. Currently supports 'sample', 'mean', and 'median'

        Expected keys for method = 'bayesboot'
            - boot_samples : int
                Number of bootstrap samples
            - resample_size : int
                Size of Bayesian bootstrap samples
            - estimator : str
                Estimator for indirect effect. Currently supports 'sample', 'mean', and 'median'

        Expected keys for method = 'bayes-norm' or method = 'bayes-robust'
            - iter : int
                Number of simulations for MCMC sampler
            - burn : int
                Number of burn-in samples
            - thin : int
                Factor by which to thin chain
            - estimator : str
                Estimator for indirect effect. Currently supports 'mean' and 'median'
            - n_chains : int
                Number of chains to run
            - standardize : boolean (optional)
                Whether to standardize variables for robust (Cauchy) priors
            - save_models : boolean (optional)
                Whether to save model for further analysis (e.g., convergence testing)

    Returns
    -------
    self : object
        Instance of MediationModel class
    """
    def __init__(self, method = None, interval = None, mediator_type = None, endogenous_type = None, 
                 alpha = .05, plot = False, parameters = None):

        _valid_bool = [True, False]
        _valid_var = ['continuous', 'categorical']
        _valid_methods = ['delta', 'boot', 'bayesboot', 'bayes-norm', 'bayes-robust']
        _valid_interval = ['first', 'second', 'perc', 'bc', 'cred', 'hpd']

        # Define global variables
        if method in _valid_methods:
            self.method = method
        else:
            raise ValueError('%s not a valid method; valid methods are %s' % (method, _valid_methods))

        if interval in _valid_interval:
            self.interval = interval
        else:
            raise ValueError('%s not a valid interval; valid intervals are %s' % (interval, _valid_interval))

        if mediator_type in _valid_var:
            self.mediator_type = mediator_type
        else:
            raise ValueError('%s not a valid mediator type; valid types are %s' % (mediator_type, _valid_var))

        if endogenous_type in _valid_var:
            self.endogenous_type = endogenous_type
        else:
            raise ValueError('%s not a valid endogenous type; valid types are %s' % (endogenous_type, _valid_var))

        if alpha <= 0 or alpha >= 1:
            raise ValueError('%.3f is not a valid value for alpha; should be in interval (0, 1)' % alpha)
        else:
            self.alpha = alpha

        if plot in _valid_bool:
			self.plot = plot
        else:
            raise ValueError('%s not a valid value for plot; should be a boolean argument' % plot)

        if parameters is not None:
            assert(isinstance(parameters, dict) == True), 'parameters argument should be a dictionary'
        else:
            parameters = {}

        self.parameters = self._check_parameters(parameters)

        self.fit_ran = False


    # ..helper functions (all start with underscore _)
    def _check_parameters(self, parameters):
        """Check keys in parameters and set to default values if none provided

        Parameters
        ----------
        parameters : dict
        	Dictionary of parameters for different estimation methods

        Returns
        -------
        parameters : dict
            Dictionary of parameters for different estimation methods with default values if none provided
        """
        # Bootstrap methods
        if self.method in ['boot', 'bayesboot']:
            if 'boot_samples' not in parameters:
                parameters['boot_samples'] = 2000
            if 'estimator' not in parameters:
                parameters['estimator'] = 'sample'
        	if self.method == 'bayesboot':
    			if 'resample_size' not in parameters:
    				parameters['resample_size'] = parameters.get('boot_samples')

        # Fully Bayesian methods
        elif self.method in ['bayes-norm', 'bayes-robust']:
        	if 'iter' not in parameters:
        		parameters['iter'] = 10000
        	if 'burn' not in parameters:
        		parameters['burn'] = int(parameters.get('iter')/2)
        	if 'thin' not in parameters:
        		parameters['thin'] = 1
        	if 'n_chains' not in parameters:
        		parameters['n_chains'] = 1
        	if 'estimator' not in parameters:
        		parameters['estimator'] = 'mean'

        else: # Ignore parameters for delta methods
        	pass

        return parameters


    @staticmethod
    def _bayes_probs(n = None):
        """Draw Bayesian bootstrap probabilities

        Parameters
        ----------
        n : int
            Number of probabilities to draw

        Returns
        -------
        probs : numpy array with dimension = [n]
            Array of Bayesian bootstrap probabilities to use for resampling data
        """
        # Random uniform draws
        u = np.random.uniform(low = 0, high = 1, size = n - 1)

        # Pad beginning and end of array
        u = np.insert(u, 0, 0)
        u = np.append(u, 1)

        # Sort and calculate first-order differences
        u.sort()
        return np.diff(u)


    @staticmethod
    def _invlogit(var = None):
        """Inverse logit transform

        Parameters
        ----------
        var : 1d array-like
            Array of inputs to be transformed

        Returns
        -------
        probs : 1d array-like
            Array with inverse logit applied to each element
        """
        return np.exp(var) / (1 + np.exp(var))


    def _standardize(self, exog = None, med = None, endog = None):
        """Standardizes continuous variables to have mean 0, sd = 0.5 and binary input variables to be mean
        centered (assumes a 0/1 coding scheme). This scaling is the recommended standardization for
        Bayesian analysis with robust (Cauchy) priors. 

        NOTE: Do not standardize binary mediator or endogenous variables because values cannot be used in 
              logistic regression

        Parameters
        ----------
        exog : 1d array-like
            Exogenous variable

        med :1d array-like
            Mediator variable

        endog : 1d array-like
            Endogenous variable

        Returns
        -------
        exog_std : 1d array-like
            Standardized exogenous variable

        med_std : 1d array-like
            Standardized mediator variable

        endog_std : 1d array-like
            Standardized endogenous variable
        """
        assert(self.method == 'bayes-robust'), "Method should be bayes-robust for this scaling"

        # Standardize exogenous variable
        exog_vals = np.unique(exog)
        if len(exog_vals) == 2 and 0 in exog_values and 1 in exog_values:
            exog_std = exog - np.mean(exog)
        else:
            exog_std = (exog - np.mean(exog))/(2*np.std(exog))
        
        # Standardize mediator variable
        if self.mediator_type == 'continuous':
            med_std = (med - np.mean(med))/(2*np.std(med))
        else:
            med_std = med

        # Standardize endogenous variable
        if self.endogenous_type == 'continuous':
            endog_std = (endog - np.mean(endog))/(2*np.std(endog)) 
        else:
            endog_std = endog

        return exog_std, med_std, endog_std      


    def _delta_method(self):
        """Estimate indirect effect with confidence interval using multivariate delta method

        Parameters
        ----------
        None

        Returns
        -------
        indirect : dictionary
            Dictionary containing: (1) point estimate, (2) confidence intervals
        """
        indirect = {}

        # Mediator variable model
        if self.mediator_type == 'continuous':
            clf_mediator = sm.GLM(self.m, self.design_m, family = sm.families.Gaussian())
        else:
            clf_mediator = sm.GLM(self.m, self.design_m, family = sm.families.Binomial())

        # Estimate model and get coefficients
        clf_mediator_results = clf_mediator.fit()
        beta_m = clf_mediator_results.params.reshape(2,)
        vcov_m = -np.linalg.inv(clf_mediator.information(beta_m)) # Get variance/covariance matrix
            
        # Endogenous variable model
        if self.endogenous_type == 'continuous':
            clf_endogenous = sm.GLM(self.y, self.design_y, family = sm.families.Gaussian())
        else:
            clf_endogenous = sm.GLM(self.y, self.design_y, family = sm.families.Binomial())

        # Estimate model and get coefficients
        clf_endogenous_results = clf_endogenous.fit()
        beta_y = clf_endogenous_results.params.reshape(3,)
        vcov_y = -np.linalg.inv(clf_endogenous.information(beta_y)) # Get variance/covariance matrix

        # Save estimates for calculations
        a = beta_m[1]
        b = beta_y[2]

        # Calculate conditional indirect effect
        ab = a*b
     
        # Variance estimate for mediator variable model
        var_a = vcov_m[1, 1]
         
        # Variance estimate for endogenous variable model
        var_b = vcov_y[2, 2]

        # First-order approximation
        if self.interval == 'first':
            MM_var = b**2*var_a + a**2*var_b

        # Second-order approximation
        else:
            MM_var = b**2*var_a + a**2*var_b + var_a*var_b

        # Compute 100(1 - alpha)% CI
        z_score = scipy.stats.norm.ppf(1 - self.alpha/2)
        ll, ul = ab - z_score * np.sqrt(MM_var), ab + z_score * np.sqrt(MM_var)
        indirect['point'] = ab; indirect['ci'] = np.array([ll, ul])
        return indirect  


    def _point_estimate(self, m = None, design_m = None, y = None, design_y = None):
        """Point estimate

        Parameters
        ----------
        m : 1d array-like
            Dependent variable for mediator model

        design_m : 2d array-like
            Design matrix for mediator model

        y : 1d array-like
            Dependent variable for endogenous model

        design_y : 2d array-like
            Design matrix for endogenous model

        Returns
        -------
        point : float
            Point estimate of indirect effect based on bootstrap sample or full sample
        """
        # Mediator variable model
        if self.mediator_type == 'continuous':
            clf_mediator = linear_model.LinearRegression(fit_intercept = False)
        else:
            clf_mediator = linear_model.LogisticRegression(fit_intercept = False)

        # Estimate model and get coefficients
        try:
            clf_mediator.fit(design_m, m.ravel())
            beta_m = clf_mediator.coef_.reshape(2,)
        except:
            beta_m = np.array([np.nan, np.nan])
            
        # Endogenous variable model
        if self.endogenous_type == 'continuous':
            clf_endogenous = linear_model.LinearRegression(fit_intercept = False)
        else:
            clf_endogenous = linear_model.LogisticRegression(fit_intercept = False)

        # Estimate model and get coefficients
        try:
            clf_endogenous.fit(design_y, y.ravel())
            beta_y = clf_endogenous.coef_.reshape(3,)
        except:
            beta_y = np.array([np.nan, np.nan, np.nan])

        # Save estimates for calculations
        a = beta_m[1]
        b = beta_y[2]

        # Point estimate
        return a*b


    def _pymc_bayes_method(self, exog = None, med = None, endog = None):
        """Estimate indirect effect and intervals with fully Bayesian model (pymc as backend).
           Default sampler is Slice sampling for all parameters
           Better to use pymc3 as backend if convergence problems appear since it uses NUTS sampler

        Parameters
        ----------
        exog : 1d array-like
            Exogenous variable

        med :1d array-like
            Mediator variable

        endog : 1d array-like
            Endogenous variable

        Returns
        -------
        indirect : dictionary
            Dictionary containing: (1) point estimate, (2) interval estimates
        """
        indirect = {}

        # Standardize data if specified
        if 'standardize' in self.parameters:    # Check if key exists rather than throw error if omitted
            if self.parameters['standardize']:
                exog, med, endog = self._standardize(exog = exog, med = med, endog = endog)

        # Priors
        if self.method == 'bayes-norm':
            # Mediator model: M ~ i_M + a*X
            i_M = pm.Normal('i_M', mu = 0, tau = 1e-10, value = 0)
            a = pm.Normal('a', mu = 0, tau = 1e-10, value = 0)

            # Endogenous model: Y ~ i_Y + c*X + b*M
            i_Y = pm.Normal('i_Y', mu = 0, tau = 1e-10, value = 0)
            c = pm.Normal('c', mu = 0, tau = 1e-10, value = 0)
            b = pm.Normal('b', mu = 0, tau = 1e-10, value = 0)

        else:
           # Mediator model: M ~ i_M + a*X
            i_M = pm.Cauchy('i_M', alpha = 0, beta = 10, value = 0)
            a = pm.Cauchy('a', alpha = 0, beta = 2.5, value = 0)    

            # Endogenous model: Y ~ i_Y + c*X + b*M
            i_Y = pm.Cauchy('i_Y', alpha = 0, beta = 10, value = 0)
            c = pm.Cauchy('c', alpha = 0, beta = 2.5, value = 0)
            b = pm.Cauchy('b', alpha = 0, beta = 2.5, value = 0) 

        # Expected values (linear combos)
        expected_med = i_M + a*exog
        expected_endog = i_Y + c*exog + b*med

        if self.mediator_type == 'continuous':
            tau_M = pm.Gamma('tau_M', alpha = .001, beta = .001, value = 1)
            response_M = pm.Normal('response_M', mu = expected_med, tau = tau_M, value = med, observed = True)
            med_model = [i_M, a, tau_M, response_M]
        else:
            p_M = pm.InvLogit('p_M', expected_med)
            response_M = pm.Bernoulli('response_M', value = med, p = p_M, observed = True) 
            med_model = [i_M, a, response_M]

        if self.endogenous_type == 'continuous':
            tau_Y = pm.Gamma('tau_Y', alpha = .001, beta = .001, value = 1)
            response_Y = pm.Normal('response_Y', mu = expected_endog, tau = tau_Y, value = endog, observed = True)
            endog_model = [i_Y, b, c, tau_Y, response_Y]
        else:
            p_Y = pm.InvLogit('p_Y', expected_endog)
            response_Y = pm.Bernoulli('response_Y', value = endog, p = p_Y, observed = True)   
            endog_model = [i_Y, b, c, response_Y]         

        # Build MCMC mediator model and estimate model
        med_model = pm.Model(med_model)
        med_mcmc = pm.MCMC(med_model)

        # Specify samplers for mediator model (slice sampling)
        med_mcmc.use_step_method(pm.Slicer, a, w = 10, m = 10000, doubling = True)
        med_mcmc.use_step_method(pm.Slicer, i_M, w = 10, m = 10000, doubling = True)
        if self.mediator_type == 'continuous':
            med_mcmc.use_step_method(pm.Slicer, tau_M, w = 10, m = 10000, doubling = True)

        # Run multiple chains if specified
        for i in xrange(self.parameters['n_chains']):
            med_mcmc.sample(iter = self.parameters['iter'], 
                            burn = self.parameters['burn'], 
                            thin = self.parameters['thin'], 
                            progress_bar = False)

        # Build MCMC endogenous model and estimate model
        endog_model = pm.Model(endog_model)
        endog_mcmc = pm.MCMC(endog_model)

        # Specify samplers for mediator model (slice sampling)
        endog_mcmc.use_step_method(pm.Slicer, b, w = 10, m = 10000, doubling = True)
        endog_mcmc.use_step_method(pm.Slicer, c, w = 10, m = 10000, doubling = True)
        endog_mcmc.use_step_method(pm.Slicer, i_Y, w = 10, m = 10000, doubling = True)
        if self.mediator_type == 'continuous':
            endog_mcmc.use_step_method(pm.Slicer, tau_Y, w = 10, m = 10000, doubling = True)

        # Run multiple chains if specified
        for i in xrange(self.parameters['n_chains']):
            endog_mcmc.sample(iter = self.parameters['iter'], 
                              burn = self.parameters['burn'], 
                              thin = self.parameters['thin'], 
                              progress_bar = False)

        # Get posterior distribution of a and b then create indirect effect
        a_path = a.trace(chain = None)
        b_path = b.trace(chain = None)
        ab_estimates = a_path*b_path

        # If plotting distribution of ab
        if self.plot:
            self.ab_estimates = a_path*b_path

        # Save mcmc models if specified
        if 'save_models' in self.parameters:    # Check if key exists rather than throw error if omitted
            if self.parameters['save_models']:
                self.med_mcmc = med_mcmc
                self.endog_mcmc = endog_mcmc

        # Point estimate
        if self.parameters['estimator'] == 'mean':
            indirect['point'] = np.mean(ab_estimates)
        else:
            indirect['point'] = np.median(ab_estimates)

        # Interval estimate
        if self.interval == 'cred':
            indirect['ci'] = self._boot_interval(ab_estimates = ab_estimates)
        else:
            indirect['ci'] = self._hpd_interval(ab_estimates = ab_estimates)
        return indirect


    def _pymc3_bayes_method(self, exog = None, med = None, endog = None):
        """Estimate indirect effect and intervals with fully Bayesian model (pymc3 as backend)

        Parameters
        ----------
        exog : 1d array-like
            Exogenous variable

        med :1d array-like
            Mediator variable

        endog : 1d array-like
            Endogenous variable

        Returns
        -------
        indirect : dictionary
            Dictionary containing: (1) point estimate, (2) interval estimates
        """
        indirect = {}

        # Standardize data if specified
        if 'standardize' in self.parameters:    # Check if key exists rather than throw error if omitted
            if self.parameters['standardize']:
                exog, med, endog = self._standardize(exog = exog, med = med, endog = endog)

        # Mediator model: M ~ i_M + a*X
        with pm.Model() as med_mcmc:

            # Define priors
            if self.method == 'bayes-norm':
                i_M = pm.Normal('i_M', mu = 0, tau = 1e-10)
                a = pm.Normal('a', mu = 0, tau = 1e-10)
            else:
                i_M = pm.Cauchy('i_M', alpha = 0, beta = 10)
                a = pm.Cauchy('a', alpha = 0, beta = 2.5)    

            # Expected values (linear combos)
            expected_med = i_M + a*exog

            # Define likelihood
            if self.mediator_type == 'continuous':
                tau_M = pm.Gamma('tau_M', alpha = .001, beta = .001)
                response_M = pm.Normal('response_M', mu = expected_med, tau = tau_M, observed = med)
            
            else:
                p_M = self._invlogit(expected_med)
                response_M = pm.Binomial('response_M', n = 1, p = p_M, observed = med)

            # Fit model
            start = pm.find_MAP()
            trace_M = pm.sample(draws = self.parameters['iter'], 
                                chain = self.parameters['n_chains'], 
                                step = pm.NUTS(scaling = start),
                                start = start,
                                progressbar = False)
            a_path = trace_M.get_values('a', burn = self.parameters['burn'], thin = self.parameters['thin'], combine = True)

        # Endogenous model: Y ~ i_Y + c*X + b*M
        with pm.Model() as endog_mcmc:
            # Define priors
            if self.method == 'bayes-norm':
                i_Y = pm.Normal('i_Y', mu = 0, tau = 1e-10)
                c = pm.Normal('c', mu = 0, tau = 1e-10)
                b = pm.Normal('b', mu = 0, tau = 1e-10)
            
            else:
                i_Y = pm.Cauchy('i_Y', alpha = 0, beta = 10)
                c = pm.Cauchy('c', alpha = 0, beta = 2.5)
                b = pm.Cauchy('b', alpha = 0, beta = 2.5) 

            # Expected values (linear combos)
            expected_endog = i_Y + c*exog + b*med

            # Define likelihood
            if self.endogenous_type == 'continuous':
                tau_Y = pm.Gamma('tau_Y', alpha = .001, beta = .001)
                response_Y = pm.Normal('response_Y', mu = expected_endog, tau = tau_Y, observed = endog)
            
            else:
                p_Y = self._invlogit(expected_endog)
                response_Y = pm.Binomial('response_Y', n = 1, p = p_Y, observed = endog)

            # Fit model
            start = pm.find_MAP()
            trace_Y = pm.sample(draws = self.parameters['iter'], 
                                chain = self.parameters['n_chains'], 
                                step = pm.NUTS(scaling = start),
                                start = start,
                                progressbar = False)
            b_path = trace_Y.get_values('b', burn = self.parameters['burn'], thin = self.parameters['thin'], combine = True)

        # Get posterior distribution of a and b then create indirect effect
        ab_estimates = a_path*b_path

        # If plotting distribution of ab
        if self.plot:
            self.ab_estimates = a_path*b_path

        # Point estimate
        if self.parameters['estimator'] == 'mean':
            indirect['point'] = np.mean(ab_estimates)
        else:
            indirect['point'] = np.median(ab_estimates)

        # Interval estimate
        if self.interval == 'cred':
            indirect['ci'] = self._boot_interval(ab_estimates = ab_estimates)
        else:
            indirect['ci'] = self._hpd_interval(ab_estimates = ab_estimates)
        return indirect


    def _boot_point(self, ab_estimates = None):
        """Get bootstrap point estimate

        Parameters
        ----------
        m : 1d array-like
            Dependent variable for mediator model

        design_m : 2d array-like
            Design matrix for mediator model

        y : 1d array-like
            Dependent variable for endogenous model

        design_y : 2d array-like
            Design matrix for endogenous model

        ab_estimates : 1d array-like
            Array with bootstrap estimates for each sample

        Returns
        -------
        point : float
            Bootstrap point estimate for indirect effect
        """

        # Get posterior point estimate based on estimator 
        if self.parameters['estimator'] == 'mean':
            return np.mean(ab_estimates)
        elif self.parameters['estimator'] == 'median':
            return np.median(ab_estimates)
        else: 
            return self._point_estimate(m = self.m, design_m = self.design_m, y = self.y, design_y = self.design_y)
       

    def _boot_interval(self, ab_estimates = None, sample_point = None):
        """Get (1-alpha)*100 interval estimates based on specified method

        Parameters
        ----------
        ab_estimates : 1d array-like
            Array with bootstrap estimates for each sample

        sample_point : float
            Indirect effect estimate based on full sample. Note, this is only used by the
            bias-corrected confidence interval

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit interval estimates
        """
        if self.interval in ['perc', 'cred']:
            return self._percentile_interval(ab_estimates)
        elif self.interval == 'bc':
            return self._bias_corrected_interval(ab_estimates, sample_point = sample_point)
        else: 
            return self._hpd_interval(ab_estimates)


    def _percentile_interval(self, ab_estimates = None):
        """Get (1-alpha)*100 percentile (nonparametric) or credible (Bayesian) interval estimate

        Parameters
        ----------
        ab_estimates : 1d array-like
            Array with bootstrap estimates for each sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit percentile interval estimates
        """
        ll = np.percentile(ab_estimates, q = (self.alpha/2)*100)
        ul = np.percentile(ab_estimates, q = (1 - self.alpha/2)*100)
        return np.array([ll, ul])


    def _bias_corrected_interval(self, ab_estimates = None, sample_point = None):
        """Get (1-alpha)*100 bias-corrected confidence interval estimate

        Parameters
        ----------
        ab_estimates : 1d array-like
            Array with bootstrap estimates for each sample

        sample_point : float
            Indirect effect point estimate based on full sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit bias-corrected confidence interval estimates
        """
        if self.parameters['estimator'] == 'sample':
            warnings.warn('The estimator is usually sample for bias-corrected intervals')
        
        # Bias of bootstrap estimates
        z0 = scipy.stats.norm.ppf(np.sum(ab_estimates < sample_point)/self.parameters['boot_samples'])

        # Adjusted intervals
        adjusted_ll = scipy.stats.norm.cdf(2*z0 + scipy.stats.norm.ppf(self.alpha/2))*100
        adjusted_ul = scipy.stats.norm.cdf(2*z0 + scipy.stats.norm.ppf(1 - self.alpha/2))*100
        ll, ul = np.percentile(ab_estimates, q = adjusted_ll), np.percentile(ab_estimates, q = adjusted_ul)
        return np.array([ll, ul])


    """
    Next two functions taken form the PyMC library https://github.com/pymc-devs/pymc -> utils.py
    """
    def _calc_min_interval(self, ab_estimates = None):
        """Determine the minimum interval of a given width

        Parameters
        ----------
        ab_estimates : SORTED 1d array-like
            Array with bootstrap estimates for each sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit for highest density interval estimates
        """
        B = len(ab_estimates)
        cred_mass = 1.0 - self.alpha

        interval_idx_inc = int(np.floor(cred_mass*B))
        n_intervals = B - interval_idx_inc
        interval_width = ab_estimates[interval_idx_inc:] - ab_estimates[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError('Too few elements for interval calculation')

        min_idx = np.argmin(interval_width)
        hdi_min, hdi_max = ab_estimates[min_idx], ab_estimates[min_idx+interval_idx_inc]
        return np.array([hdi_min, hdi_max])


    def _hpd_interval(self, ab_estimates = None):
        """Get (1-alpha)*100 highest posterior density estimates

        Parameters
        ----------
        ab_estimates : 1d array-like
            Array with bootstrap estimates for each sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit for highest density interval estimates
        """

        # Make a copy of trace
        ab_estimates = ab_estimates.copy()

        # For multivariate node
        if ab_estimates.ndim > 1:

            # Transpose first, then sort
            tx = np.transpose(ab_estimates, list(range(ab_estimates.ndim))[1:]+[0])
            dims = np.shape(tx)

            # Container list for intervals
            intervals = np.resize(0.0, dims[:-1]+(2,))

            for index in make_indices(dims[:-1]):
                try:
                    index = tuple(index)
                except TypeError:
                    pass

                # Sort trace
                sx = np.sort(tx[index])

                # Append to list
                intervals[index] = self._calc_min_interval(sx)

            # Transpose back before returning
            return np.array(intervals)

        else:
            # Sort univariate node
            sx = np.sort(ab_estimates)
            return np.array(self._calc_min_interval(sx))


    def _boot_method(self):
        """Estimate indirect effect with confidence interval using nonparametric or Bayesian bootstrap

        Parameters
        ----------
        None

        Returns
        -------
        indirect : dictionary
            Dictionary containing: (1) point estimate, (2) interval estimates
        """
        indirect = {}
        ab_estimates = np.zeros((self.parameters['boot_samples']))

        # Nonparametric bootstrap. Note, p = None implies uniform distribution over np.arange(n)
        if self.method == 'boot':
            for i in xrange(self.parameters['boot_samples']):
                idx = np.random.choice(np.arange(self.n), 
                                                 replace = True, 
                                                 p = None, 
                                                 size = self.n)
                ab_estimates[i] = self._point_estimate(m = self.m[idx], design_m = self.design_m[idx], 
                                                       y = self.y[idx], design_y = self.design_y[idx])
        else:
            # Bayesian bootstrapping
            for i in xrange(self.parameters['boot_samples']):
                probs = self._bayes_probs(self.n)
                idx = np.random.choice(np.arange(self.n), 
                                       replace = True, 
                                       p = probs, 
                                       size = self.parameters['resample_size'])
                ab_estimates[i] = self._point_estimate(m = self.m[idx], design_m = self.design_m[idx], 
                                                       y = self.y[idx], design_y = self.design_y[idx])

        # Check to make sure ab_estimates does not contain np.nan values (failed bootstrap samples)
        if np.any(np.isnan(ab_estimates)):
            self.boot_failed = np.sum(np.isnan(ab_estimates))
            ab_estimates = ab_estimates[~np.isnan(ab_estimates)]
        else:
            self.boot_failed = 0

        # Save estimates if plotting
        if self.plot:
            self.ab_estimates = ab_estimates

        # Bootstrap point estimate and confidence interval
        indirect['point'] = self._boot_point(ab_estimates = ab_estimates)
        indirect['ci'] = self._boot_interval(ab_estimates = ab_estimates, sample_point = indirect['point'])
        return indirect


    def _estimate_paths(self):
        """Estimate all coefficients from mediation model

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Creates a dictionary that contains the point estimates, standard errors, and confidence intervals
            for each structural path in the model
        """        
        # Estimate mediator model
        self.all_paths = {}
        if self.mediator_type == 'continuous':
            clf_mediator = sm.GLM(self.m, self.design_m, family = sm.families.Gaussian())
        else:
            clf_mediator = sm.GLM(self.m, self.design_m, family = sm.families.Binomial())
        results_mediator = clf_mediator.fit()

        # Get coefficients
        coefs_mediator = results_mediator.params
        self.all_paths['a0'] = coefs_mediator[0]
        self.all_paths['a'] = coefs_mediator[1]

        # Get standard errors
        self.all_paths['se_a0'] = np.sqrt(results_mediator.cov_params()[0, 0])
        self.all_paths['se_a'] = np.sqrt(results_mediator.cov_params()[1, 1])

        # Get confidence intervals
        self.all_paths['ci_a0'] = results_mediator.conf_int(alpha = self.alpha)[0, :]
        self.all_paths['ci_a'] = results_mediator.conf_int(alpha = self.alpha)[1, :]

        # Estimate endogenous model
        if self.endogenous_type == 'continuous':
            clf_endogenous = sm.GLM(self.y, self.design_y, family = sm.families.Gaussian())
        else:
            clf_endogenous = sm.GLM(self.y, self.design_y, family = sm.families.Binomial())
        results_endogenous = clf_endogenous.fit()

        # Get coefficients
        coefs_endogenous = results_endogenous.params
        self.all_paths['b0'] = coefs_endogenous[0]
        self.all_paths['c'] = coefs_endogenous[1]
        self.all_paths['b'] = coefs_endogenous[2]

        # Get standard errors
        self.all_paths['se_b0'] = np.sqrt(results_endogenous.cov_params()[0, 0])
        self.all_paths['se_c'] = np.sqrt(results_endogenous.cov_params()[1, 1])
        self.all_paths['se_b'] = np.sqrt(results_endogenous.cov_params()[2, 2])

        # Get confidence intervals
        self.all_paths['ci_b0'] = results_endogenous.conf_int(alpha = self.alpha)[0, :]
        self.all_paths['ci_c'] = results_endogenous.conf_int(alpha = self.alpha)[1, :]
        self.all_paths['ci_b'] = results_endogenous.conf_int(alpha = self.alpha)[2, :]


    # ..main functions that are callable
    def fit(self, exog = None, med = None, endog = None):
        """Fit model and estimate indirect effect

        Parameters
        ----------
        exog : 1d array-like
            Exogenous variable

        med :1d array-like
            Mediator variable

        endog : 1d array-like
            Endogenous variable

        Returns
        -------
        self : object
            A fitted object of class MediationModel
        """
        assert(exog.shape[0] == med.shape[0] == endog.shape[0]), "All variables should have same shape for first dimension"
        
        # Create pandas dataframe of data
        combined = np.hstack((exog.reshape(-1, 1), med.reshape(-1, 1), endog.reshape(-1, 1)))
        data = pd.DataFrame(combined, columns = ['x', 'm', 'y'])

        # Define variables
        self.n = med.shape[0]
        self.m, self.design_m = dmatrices('m ~ x', data = data)
        self.y, self.design_y = dmatrices('y ~ x + m', data = data)

        # Estimate indirect effect based on method
        if self.method == 'delta':
            self.indirect = self._delta_method()
        elif self.method in ['boot', 'bayesboot']:
            self.indirect = self._boot_method()
        else:
            if backend == 'pymc3':
                self.indirect = self._pymc3_bayes_method(exog = exog, med = med, endog = endog)
            elif backend == 'pymc':
                self.indirect = self._pymc_bayes_method(exog = exog, med = med, endog = endog)
            else:
                raise ValueError('fully Bayesian methods not available due to import errors with pymc3 and pymc')
        self.fit_ran = True


    def indirect_effect(self):
        """Get point estimate and confidence interval for indirect effect based on specified method

        Parameters
        ----------
        None

        Returns
        -------
        estimates : 1d array-like
            Array with point estimate, lower limit, and upper limit of interval estimate
        """
        assert(self.fit_ran == True), 'Need to run .fit() method before getting indirect effect'
        point = np.array([self.indirect['point']]).ravel()
        ci = self.indirect['ci'].ravel()
        return np.concatenate((point, ci)).reshape((3,))


    def summary(self, exog_name = None, med_name = None, endog_name = None):
        """Print summary of parameter estimates

        Parameters
        ----------
        exog_name : str
            Name of exogenous variable

        med_name : str
            Name of mediator variable

        endog_name : str
            Name of endogenous variable

        Returns
        -------
        None 
        """
        # Error checking
        assert(self.fit_ran == True), 'Need to run .fit() method before printing summary'

        # Estimate all paths
        self._estimate_paths()

        # Define method strings
        if self.method == 'delta':
            str_method = 'Multivariate Delta Method'
            if self.interval == 'first':
                str_interval = 'First-Order Approximation'
            else:
                str_interval = 'Second-Order Approximation'
        elif self.method == 'boot':
            str_method = 'Nonparametric Bootstrap'
            if self.interval == 'perc':
                str_interval = 'Percentile'
            else:
                str_interval = 'Bias-Corrected'
        elif self.method == 'bayesboot':
            str_method = 'Bayesian Bootstrap'
            if self.interval == 'cred':
                str_interval = 'Credible'
            else:
                str_interval = 'Highest Posterior Density'
        else:
            str_method = 'Fully Bayesian'
            if self.interval == 'cred':
                str_interval = 'Credible'
            else:
                str_interval = 'Highest Posterior Density'

        # Define models
        if self.mediator_type == 'continuous':
            med_model = 'Linear Regression'
        else:
            med_model = 'Logistic Regression'

        if self.endogenous_type == 'continuous':
            endog_model = 'Linear Regression'
        else:
            endog_model = 'Logistic Regression'

        # Overall summary
        print('{:-^71}'.format(''))
        print('{:^71}'.format('MEDIATION MODEL SUMMARY'))
        print('{:-^71}\n'.format(''))

        if exog_name is not None and med_name is not None and endog_name is not None:
            print('{0:<20}{1:<14}{2:<10}'.format('Exogenous:', exog_name, '-->  ' + exog_name[:3]))
            print('{0:<20}{1:<14}{2:<10}'.format('Mediator:', med_name, '-->  ' + med_name[:3]))
            print('{0:<20}{1:<14}{2:<10}'.format('Endogenous:', endog_name, '-->  ' + endog_name[:3]))

            # Truncate names
            exog_name = '{:.3}'.format(exog_name)
            med_name = '{:.3}'.format(med_name)
            endog_name = '{:.3}'.format(endog_name)

        else:
            exog_name = 'X'
            med_name = 'M'
            endog_name = 'Y'
            print('{0:<20}{1:<14}'.format('Exogenous:', exog_name))
            print('{0:<20}{1:<14}'.format('Mediator:', med_name))
            print('{0:<20}{1:<14}'.format('Endogenous:', endog_name))

        print('\n{0:<20}{1:<14}'.format('Mediator Model:', med_model))
        print('{0:<20}{1:<14}'.format('Endogenous Model:', endog_model))

        print('\n{0:<20}{1:<14}'.format('Sample Size:', self.n))
        print('{0:<20}{1:<14}'.format('Alpha:', self.alpha))

        print('\n{0:<20}{1:<14}'.format('Method:', str_method))
        print('{0:<20}{1:<14}'.format('Interval:', str_interval))

        if self.method in ['boot', 'bayesboot']:
            print('{0:<20}{1:<3}'.format('Boot Samples:', self.parameters['boot_samples']))
            print('{0:<20}{1:<3}'.format('Failed Samples:', self.boot_failed))
            if self.method == 'bayesboot':
                print('{0:<20}{1:<3}'.format('Resample Size:', self.parameters['resample_size']))
                print('{0:<20}{1:<10}'.format('Estimator:', self.parameters['estimator']))
        
        elif self.method in ['bayes-norm', 'bayes-robust']:
            if self.method == 'bayes-norm':
                prior_type = 'Normal'
            else:
                prior_type = 'Robust'

            print('{0:<20}{1:<3}'.format('Prior Type:', prior_type))
            print('{0:<20}{1:<3}'.format('Iterations:', self.parameters['iter']))
            print('{0:<20}{1:<3}'.format('Burn-in:', self.parameters['burn']))
            print('{0:<20}{1:<3}'.format('Thin:', self.parameters['thin']))
            print('{0:<20}{1:<3}'.format('Number of Chains:', self.parameters['n_chains']))

        # Parameter estimates summary
        print('\n{:-^71}'.format(''))
        print('{:^95}'.format(str(int((1-self.alpha)*100)) + '% Intervals'))
        print('{:^96}'.format('-----------------'))
        print('{path:^12}{coef:^12}{point:^12}{ll:^12}{ul:^12}{sig:^12}'.format(path = 'Path',
                                                                                coef = 'Coef',
                                                                                point = 'Point',
                                                                                ll = 'LL',
                                                                                ul = 'UL',
                                                                                sig = 'Sig'))
        print('{:-^71}'.format(''))

        # a effect
        if np.sign(self.all_paths['ci_a'][0]) == np.sign(self.all_paths['ci_a'][1]):
            sig = 'Yes'
        else:
            sig = 'No'
        print('{path:^12}{coef:^12}{point:^12.4f}{ll:^12.4f}{ul:^12.4f}{sig:^12}'.format(path = ' %s -> %s' % (exog_name, med_name),
                                                                                         coef = 'a',
                                                                                         point = self.all_paths['a'],
                                                                                         ll = self.all_paths['ci_a'][0],
                                                                                         ul = self.all_paths['ci_a'][1],
                                                                                         sig = sig))
        # b effect
        if np.sign(self.all_paths['ci_b'][0]) == np.sign(self.all_paths['ci_b'][1]):
            sig = 'Yes'
        else:
            sig = 'No'
        print('{path:^12}{coef:^12}{point:^12.4f}{ll:^12.4f}{ul:^12.4f}{sig:^12}'.format(path = ' %s -> %s' % (med_name, endog_name),
                                                                                         coef = 'b',
                                                                                         point = self.all_paths['b'],
                                                                                         ll = self.all_paths['ci_b'][0],
                                                                                         ul = self.all_paths['ci_b'][1],
                                                                                         sig = sig))
        # c effect
        if np.sign(self.all_paths['ci_c'][0]) == np.sign(self.all_paths['ci_c'][1]):
            sig = 'Yes'
        else:
            sig = 'No'
        print('{path:^12}{coef:^12}{point:^12.4f}{ll:^12.4f}{ul:^12.4f}{sig:^12}'.format(path = ' %s -> %s' % (exog_name, endog_name),
                                                                                         coef = 'c',
                                                                                         point = self.all_paths['c'],
                                                                                         ll = self.all_paths['ci_c'][0],
                                                                                         ul = self.all_paths['ci_c'][1],
                                                                                         sig = sig))
        # indirect effect
        if np.sign(self.indirect['ci'].ravel()[0]) == np.sign(self.indirect['ci'].ravel()[1]):
            sig = 'Yes'
        else:
            sig = 'No'
        print('\n{path:^12}{coef:^12}{point:^12.4f}{ll:^12.4f}{ul:^12.4f}{sig:^12}'.format(path = 'Indirect',
                                                                                           coef = 'a*b',
                                                                                           point = self.indirect['point'],
                                                                                           ll = self.indirect['ci'][0],
                                                                                           ul = self.indirect['ci'][1],
                                                                                           sig = sig))
        print('{:-^71}'.format(''))


    def plot_indirect(self):
        """Plot histogram of bootstrap or posterior distribution of indirect effect

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Error checking
        assert(self.fit_ran == True), 'Need to run .fit() method before generating histogram'
        assert(self.plot == True and hasattr(clf, 'ab_estimates')), 'Need to specify plot == True in constructor function to enable plotting'
        if self.method == 'delta':
            raise ValueError('Plotting not available for multivariate delta method')
        
        # Create figure
        plt.figure()
        plt.hist(self.ab_estimates, bins = 100, color = 'gray')
        plt.axvline(self.indirect['point'], color = 'blue', label = 'Point', linewidth = 3)
        plt.axvline(self.indirect['ci'][0], color = 'blue', label = 'Interval', linestyle = 'dashed', linewidth = 3)
        plt.axvline(self.indirect['ci'][1], color = 'blue', linestyle = 'dashed', linewidth = 3)
        
        # Check method for title of histogram
        if self.method == 'boot':
            str_method = 'Bootstrap Distribution'
            if self.interval == 'perc':
                str_interval = 'Percentile Intervals'
            else:
                str_interval = 'Bias-Corrected Intervals'
        elif self.method == 'bayesboot':
            str_method = 'BayesBoot Posterior Distribution'
            if self.interval == 'cred':
                str_interval = 'Credible Intervals'
            else:
                str_interval = 'HPD Intervals'
        else:
            str_method = 'Bayesian Posterior Distribution'
            if self.interval == 'cred':
                str_interval = 'Credible Intervals'
            else:
                str_interval = 'HPD Intervals'
        title_str = '{title:} with {alpha:}% {int_type:}\nPoint = {point:.3f}, Interval = [{ll:.3f}, {ul:.3f}]'.format(
                                                                                        title = str_method, 
                                                                                        alpha = int((1-self.alpha)*100), 
                                                                                        int_type = str_interval,
                                                                                        point = self.indirect['point'],
                                                                                        ll = self.indirect['ci'][0],
                                                                                        ul = self.indirect['ci'][1])
        plt.title(title_str)
        plt.legend()
        plt.show()