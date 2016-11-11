from __future__ import division, print_function

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrices
import pymc as pm
import scipy.stats
from sklearn import linear_model
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# TODO
#   - Add documentation Bayesian modeling
#   - Rewrite API for cleaner modeling
#   - Test Bayesian modeling

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
            - method = delta : 'first' or 'second'
            - method = boot : 'perc' or 'bc'
            - method = bayesboot : 'cred' or 'hpd'
            - method = bayes : 'cred' or 'hpd'

    mediator_type : string
                    Variable indicating whether mediator variable is continuous or categorical

    endogenous_type : string
                      Variable indicating whether endogenous variable is continuous or categorical

    alpha : float, default .05
        Type I error rate - corresponds to generating (1-alpha)*100 intervals

    fit_intercept : boolean, default True
        Whether to fit an intercept terms

    plot : boolean, default False
        Whether to plot distribution (empirical sampling or posterior) of indirect effect. Need to specify
        bootstrap or Bayesian estimation.

    parameters : dict
        Dictionary of parameters for different estimation methods.
        Expected keys for bootstrap:
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
        Expected keys for method = 'bayes'
            - iter : int
                Number of simulations for MCMC sampler
            - burn : int
                Number of burn-in samples
            - thin : int
                Factor to thin chain by
            - estimator : str
                Estimator for indirect effect. Currentl supports 'mean' and 'median'
            - n_chains : int
                Number of chains to run
            - check_convergence : boolean
                Run standard tests for convergence

    Returns
    -------
    self : object
        Instance of MediationModel class
    """
    def __init__(self, method = None, interval = None, mediator_type = None, endogenous_type = None, 
                 alpha = .05, fit_intercept = True, plot = False, parameters = None):

        _valid_bool = [True, False]
        _valid_var = ['continuous', 'categorical']
        _valid_methods = ['delta', 'boot', 'bayesboot', 'bayes-norm', 'bayes-robust']

        # Define global variables
        if method in _valid_methods:
            self.method = method
        else:
            raise ValueError('%s not a valid method; valid methods are %s' % (method, _valid_methods))

        self.interval = interval

        if mediator_type in _valid_var:
            self.mediator_type = mediator_type
        else:
            raise ValueError('%s not a valid mediator type' % mediator_type)

        if endogenous_type in _valid_var:
            self.endogenous_type = endogenous_type
        else:
            raise ValueError('%s not a valid endogenous type' % endogenous_type)

        if alpha <= 0 or alpha >= 1:
            raise ValueError('%.3f is not a valid value for alpha; should be in interval (0, 1)' % alpha)
        else:
            self.alpha = alpha

        if fit_intercept in _valid_bool:
            self.fit_intercept = fit_intercept
        else:
            raise ValueError('%s not a valid value for fit_intercept; should be a boolean argument' % fit_intercept)

        if plot in _valid_bool:
            self.plot = plot
        else:
            raise ValueError('%s not a valid value for plot; should be a boolean argument' % plot)

        self.parameters = parameters

        self.fit_ran = False


    # ..helper functions (all start with underscore _)
    def _bayes_probs(self, n = None):
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
        clf_mediator.fit(design_m, m.ravel())
        beta_m = clf_mediator.coef_.reshape(2,)
            
        # Endogenous variable model
        if self.endogenous_type == 'continuous':
            clf_endogenous = linear_model.LinearRegression(fit_intercept = False)
        else:
            clf_endogenous = linear_model.LogisticRegression(fit_intercept = False)

        # Estimate model and get coefficients
        clf_endogenous.fit(design_y, y.ravel())
        beta_y = clf_endogenous.coef_.reshape(3,)

        # Save estimates for calculations
        a = beta_m[1]
        b = beta_y[2]

        # Point estimate
        return a*b


    def _bayes_method(self, exog = None, med = None, endog = None):
        """Estimate indirect effect and intervals with fully Bayesian model

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

        if self.method == 'bayes-norm':
            # Mediator model: M ~ i_M + a*X
            i_M = pm.Normal('i_M', mu = 0, tau = 1e-10, value = 0)
            a = pm.Normal('a1', mu = 0, tau = 1e-10, value = 0)

            # Endogenous model: Y ~ i_Y + c*X + b*M
            i_Y = pm.Normal('i_Y', mu = 0, tau = 1e-10, value = 0)
            c = pm.Normal('c', mu = 0, tau = 1e-10, value = 0)
            b = pm.Normal('b', mu = 0, tau = 1e-10, value = 0)

        else:
           # Mediator model: M ~ i_M + a*X
            i_M = pm.Cauchy('i_M', alpha = 0, beta = 10, value = 0)
            a = pm.Cauchy('a1', alpha = 0, beta = 2.5, value = 0)    

            # Endogenous model: Y ~ i_Y + c*X + b*M
            i_Y = pm.Cauchy('i_Y', alpha = 0, beta = 10, value = 0)
            c = pm.Cauchy('c', alpha = 0, beta = 2.5, value = 0)
            b = pm.Cauchy('b', alpha = 0, beta = 2.5, value = 0) 

        # Expected values (linear combos)
        expected_med = i_M + a*exog
        expected_endog = i_Y + c*exog + b*med

        if self.mediator_type == 'continuous':
            tau_M = pm.Gamma('tau_M', alpha = .001, beta = .001)
            response = pm.Normal('response', mu = expected_med, tau = tau_M, value = med, observed = True)
            med_model = [i_M, a, tau_M, response]
        else:
            response = pm.Bernoulli('response', value = med, p = pm.invlogit(expected_med), observed = True) 
            med_model = [i_M, a, response]

        if self.endogenous_type == 'continuous':
            tau_Y = pm.Gamma('tau_Y', alpha = .001, beta = .001)
            response = pm.Normal('response', mu = expected_endog, tau = tau_Y, value = endog, observed = True)
            endog_model = [i_Y, b, c, tau_Y, response]
        else:
            response = pm.Bernoulli('response', value = endog, p = pm.invlogit(expected_endog), observed = True)   
            endog_model = [i_Y, b, c, response]         

        # Build MCMC model and estimate model
        bayes_model = pm.Model(med_model + endog_model)
        mcmc = pm.MCMC(bayes_model)

        # Run multiple chains if specified
        for i in xrange(self.parameters['n_chains']):
            mcmc.sample(iter = self.parameters['iter'], 
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


    def _boot_point(self, m = None, design_m = None, y = None, design_y = None, ab_estimates = None):
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
            return self._point_estimate(m = m, design_m = design_m, y = y, design_y = design_y)
       

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
        assert(self.parameters['estimator'] == 'sample'), 'The estimator must be sample for bias-corrected intervals'
        
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
        ab_estimates : SORTED numpy array with dimensions = [b1, 1]
            Array with ab estimates based on bootstrap or Bayesian method

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
        ab_estimates : numpy array with dimensions = [b1, 1]
            Array with ab estimates based on bootstrap or Bayesian method

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
            Dictionary containing: (1) point estimate, (2) confidence intervals
        """
        indirect = {}
        ab_estimates = np.zeros((self.b1))

        # Nonparametric bootstrap. Note, p = None implies uniform distribution over np.arange(n)
        if self.method == 'boot':
            for i in xrange(self.parameters['boot_samples']):
                idx = np.random.choice(np.arange(self.n), 
                                                 replace = True, 
                                                 p = None, 
                                                 size = self.n)
                ab_estimates[i] = self._point_estimate(m = m[idx], design_m = design_m[idx], 
                                                       y = y[idx], design_y = design_y[idx])
        else:
            # Bayesian bootstrapping
            for i in xrange(self.parameters['boot_samples']):
                probs = self._bayes_probs(self.n)
                idx = np.random.choice(np.arange(self.n), 
                                       replace = True, 
                                       p = probs, 
                                       size = self.parameters['resample_size'])
                ab_estimates[i] = self._point_estimate(m = m[idx], design_m = design_m[idx], 
                                                       y = y[idx], design_y = design_y[idx])

        if self.plot:
            self.ab_estimates = ab_estimates

        # Bootstrap point estimate and confidence interval
        indirect['point'] = self._boot_point(m = m, design_m = design_m, y = y, 
                                             design_y = design_y, ab_estimates = ab_estimates)
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
        self.n = m.shape[0]
        self.m, self.design_m = dmatrices('m ~ x', data = data)
        self.y, self.design_y = dmatrices('y ~ x + m', data = data)

        # If no intercept, then drop from both design matrices
        if self.fit_intercept == False:
            self.design_m = np.delete(self.design_m, [0], axis = 1)
            self.design_y = np.delete(self.design_y, [0], axis = 1)

        # Estimate indirect effect based on method
        if self.method == 'delta':
            self.indirect = self._delta_method()
        elif self.method == 'boot':
            self.indirect = self._boot_method()
        else:
            self.indirect = self._bayes_method(exog = exog, med = med, endog = endog)
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
            print('{0:<20}{1:<14}'.format('Mediator:', exog_name))
            print('{0:<20}{1:<14}'.format('Endogenous:', exog_name))

        print('\n{0:<20}{1:<14}'.format('Mediator Model:', med_model))
        print('{0:<20}{1:<14}'.format('Endogenous Model:', endog_model))

        print('\n{0:<20}{1:<14}'.format('Sample Size:', self.n))
        print('{0:<20}{1:<14}'.format('Alpha:', self.alpha))

        print('\n{0:<20}{1:<14}'.format('Method:', str_method))
        print('{0:<20}{1:<14}'.format('Interval:', str_interval))

        if self.method in ['boot', 'bayesboot']:
            print('{0:<20}{1:<3}'.format('Boot Samples:', self.parameters['boot_samples']))
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
        print('{path:<12}{coef:^12}{point:^12.4f}{ll:^12.4f}{ul:^12.4f}{sig:^12}'.format(path = ' %s -> %s' % (exog_name, med_name),
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
        print('{path:<12}{coef:^12}{point:^12.4f}{ll:^12.4f}{ul:^12.4f}{sig:^12}'.format(path = ' %s -> %s' % (med_name, endog_name),
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
        print('{path:<12}{coef:^12}{point:^12.4f}{ll:^12.4f}{ul:^12.4f}{sig:^12}'.format(path = ' %s -> %s' % (exog_name, endog_name),
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

if __name__ == "__main__":
    
    # Simulate data
    x = np.random.normal(0, 1, (100, 1))
    m = .4*x + np.random.normal(0, 1, (100, 1))
    y = .4*m + np.random.normal(0, 1, (100, 1))

    # Delta method (first-order)
    clf = MediationModel(method = 'delta', interval = 'first', mediator_type = 'continuous',
                         endogenous_type = 'continuous', plot = False)
    clf.fit(exog = x, med = m, endog = y)
    print(clf.indirect_effect())
    clf.summary(exog_name = 'depression', med_name = 'alcohol', endog_name = 'drugabuse')

    # Delta method (second-order)
    clf = MediationModel(method = 'delta', interval = 'second', mediator_type = 'continuous',
                         endogenous_type = 'continuous', plot = False)
    clf.fit(exog = x, med = m, endog = y)
    print(clf.indirect_effect())
    clf.summary(exog_name = 'depression', med_name = 'alcohol', endog_name = 'drugabuse')

    # Fully Bayesian method (HPD intervals)
    params = {'iter': 10000, 'burn': 500, 'thin': 1, 'n_chains': 2, 'estimator': 'mean'}
    clf = MediationModel(method = 'bayes-norm', interval = 'hpd', mediator_type = 'continuous', 
                         endogenous_type = 'continuous', plot = True, parameters = params)
    clf.fit(exog = x, med = m, endog = y)
    print(clf.indirect_effect())
    clf.summary(exog_name = 'depression', med_name = 'alcohol', endog_name = 'drugabuse')
    clf.plot_indirect()