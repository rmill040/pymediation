from __future__ import division, print_function

import numpy as np
from patsy import dmatrices
import pandas as pd
import statsmodels.api as sm


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
             Valid methods include: delta-1 (first-order delta method)
                                    delta-2 (second-order delta method)
                                    boot-perc (nonparametric bootstrap with percentile CIs)
                                    boot-bc (nonparametric bootstrap with bias-corrected CIs)
                                    bayes-credible (Bayesian bootstrap with credible intervals)
                                    bayes-hdi (Bayesian bootstrap with highest density intervals)

    mediator_type : string
                    Variable indicating whether mediator variable is continuous or categorical

    endogenous_type : string
                      Variable indicating whether endogenous variable is continuous or categorical

    b1 : int
        Stage one number of samples to draw - corresponds to the number of bootstrap or Bayesian bootstrap samples

    b2 : int (for bayesboot)
        Stage two number of samples to draw - corresponds to the size of Bayesian bootstrap samples to draw

    estimator : str (for bayesboot)
        Posterior estimator for point estimate. Currently supports sample, mean, median, and mode

    alpha : float, default .05
        Type I error rate - corresponds to generating (1-alpha)*100 intervals

    fit_intercept : boolean, default True
        Whether to fit an intercept terms

    save_boot_estimates : boolean, default False
        Whether to save the bootstrap estimates for plotting

    Returns
    -------
    self : object
        Instance of MediationModel class
    """
    def __init__(self, method = None, mediator_type = None, endogenous_type = None, b1 = None, 
                 b2 = None, estimator = None, alpha = .05, fit_intercept = True, save_boot_estimates = False,
                 estimate_all_paths = False):

        # Define global variables
        if method not in ['delta-1', 'delta-2', 'boot-perc', 'boot-bc', 'bayes-credible', 'bayes-hdi']:
            raise ValueError('%s not a valid method')
        else:
            self.method = method

        if mediator_type == 'continuous' or mediator_type == 'categorical':
            self.mediator_type = mediator_type
        else:
            raise ValueError('%s not a valid mediator type')

        if endogenous_type == 'continuous' or endogenous_type == 'categorical':
            self.endogenous_type = endogenous_type
        else:
            raise ValueError('%s not a valid endogenous type')

        if alpha <= 0 or alpha >= 1:
            raise ValueError('%.3f is not a valid value for alpha. Alpha should be in interval (0, 1)')
        else:
            self.alpha = alpha

        if fit_intercept == True or fit_intercept == False:
            self.fit_intercept = fit_intercept
        else:
            raise ValueError('fit_intercept should be a boolean argument')

        if estimate_all_paths == True or estimate_all_paths == False:
            self.estimate_all_paths = estimate_all_paths
        else:
            raise ValueError('estimate_all_paths should be a boolean argument')

        # Global variables to control bootstrap functionality
        if self.method in ['boot-perc', 'boot-bc', 'bayes-credible', 'bayes-hdi']:
            if estimator not in ['sample', 'mean', 'median', 'mode']:
                raise ValueError('%s is not a valid estimator')
            else:
                self.estimator = estimator
            if save_boot_estimates == True or save_boot_estimates == False:
                self.save_boot_estimates = save_boot_estimates
            else:
                raise ValueError('save_boot_estimates should be a boolean argument')
            self.b1 = int(b1)
            self.b2 = int(b2)
            self.fit_ran = False
            self.get_coefs_ran = False

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


    def _point_estimate(self, idx = None, exogenous = None, mediator = None, endogenous = None):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """

        # Get bootstrap or Bayesian bootstrap sample and check to make sure all variables are shaped the correct way
        n = len(idx)
        x = exogenous[idx].reshape((n, 1))
        m = mediator[idx].reshape((n, 1))
        y = endogenous[idx].reshape((n, 1))

        # Get design matrices
        design_m, design_y = self._create_design_matrix(exogenous = x, mediator = m)
    
    # Mediator variable model
        if self.mediator_type == 'continuous':
            clf_mediator = linear_model.LinearRegression(fit_intercept = False)
        else:
            clf_mediator = linear_model.LogisticRegression(fit_intercept = False)

        # Estimate model and get coefficients
        try:
            clf_mediator.fit(design_m, m.ravel())
            beta_m = clf_mediator.coef_.reshape(2,)
            clf_mediator_fail = 0
        except:
            clf_mediator_fail = 1
            
    # Endogenous variable model
        if self.endogenous_type == 'continuous':
            clf_endogenous = linear_model.LinearRegression(fit_intercept = False)
        else:
            clf_endogenous = linear_model.LogisticRegression(fit_intercept = False)

        # Estimate model and get coefficients
        try:
            clf_endogenous.fit(design_y, y.ravel())
            beta_y = clf_endogenous.coef_.reshape(3,)
            clf_endogenous_fail = 0
        except:
            clf_endogenous_fail = 1

        # Check for fitting errors
        if clf_mediator_fail or clf_endogenous_fail:
            return np.nan
        else:
            # Save estimates for calculations
            a = beta_m[1]
            b = beta_y[2]

            # Point estimate
            return a*b


    def _delta_method(self, m = None, design_m = None, y = None, design_y = None):
        """Estimate indirect effect with confidence interval using multivariate delta method

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
        estimates : dictionary
            Dictionary containing: (1) point estimate, (2) confidence intervals
        """

        # Empty dictionary
        estimates = {}

        # Mediator variable model
        if self.mediator_type == 'continuous':
            clf_mediator = sm.GLM(m, design_m, family = sm.families.Gaussian())
        else:
            clf_mediator = sm.GLM(m, design_m, family = sm.families.Binomial())

        # Estimate model and get coefficients
        try:
            clf_mediator_results = clf_mediator.fit()
            beta_m = clf_mediator_results.params.reshape(2,)
            vcov_m = -np.linalg.inv(clf_mediator.information(beta_m)) # Get variance/covariance matrix
            clf_mediator_fail = 0
        except:
            clf_mediator_fail = 1
            
        # Endogenous variable model
        if self.endogenous_type == 'continuous':
            clf_endogenous = sm.GLM(y, design_y, family = sm.families.Gaussian())
        else:
            clf_endogenous = sm.GLM(y, design_y, family = sm.families.Binomial())

        # Estimate model and get coefficients
        try:
            clf_endogenous_results = clf_endogenous.fit()
            beta_y = clf_endogenous_results.params.reshape(3,)
            vcov_y = -np.linalg.inv(clf_endogenous.information(beta_y)) # Get variance/covariance matrix
            clf_endogenous_fail = 0
        except:
            clf_endogenous_fail = 1

        # Check for fitting errors
        if clf_mediator_fail or clf_endogenous_fail:
            estimates['point'] = np.nan; estimates['CI'] = np.array([np.nan, np.nan]).reshape((1, 2))
            return estimates

        else:
            # Save estimates for calculations
            a = beta_m[1]
            b = beta_y[2]

            # Calculate conditional indirect effect
            point = a*b
         
            # Variance estimate for mediator variable model
            var_a = vcov_m[1, 1]
             
            # Variance estimate for endogenous variable model
            var_b = vcov_y[2, 2]

            # First-order approximation
            if self.method == 'delta-1':
                MM_var = b**2*var_a + a**2*var_b

            # Second-order approximation
            else:
                MM_var = b**2*var_a + a**2*var_b + var_a*var_b

            # Compute 100(1 - alpha)% CI
            z_score = scipy.stats.norm.ppf(1 - self.alpha/2)
            LL = point - z_score * np.sqrt(MM_var)
            UL = point + z_score * np.sqrt(MM_var)
            estimates['point'] = point; estimates['CI'] = np.array([LL, UL]).reshape((1, 2))
    

    def _boot_point_estimate(self, boot_estimates = None):
        """Get posterior point estimates

        Parameters
        ----------
        estimates : numpy array with dimensions = [b1, 1]
            Array with Bayesian bootstrap estimates for each sample

        Returns
        -------
        point : float
            Posterior point estimates 
        """

        # Get posterior point estimate based on estimator 
        if self.estimator == 'mean':
            return np.mean(estimates, axis = 0)
        elif self.estimator == 'median':
            return np.median(estimates, axis = 0)
        elif self.estimator == 'mode':
            return scipy.stats.mode(estimates, axis = 0)[0]
        else:
            raise ValueError("%s not a valid posterior estimator" % self.estimator)
       

    def _boot_interval(self, boot_estimates = None):
        """Get (1-alpha)*100 interval estimates

        Parameters
        ----------
        estimates : numpy array with dimensions = [b1, 1]
            Array with Bayesian bootstrap estimates for each sample

        Returns
        -------
        ll : float
            Lower limit interval estimate

        ul : float        
            Upper limit interval estimates
        """

        if self.method in ['boot-perc', 'bayes-credible']:
            ll = np.percentile(boot_estimates, q = (self.alpha/2)*100)
            ul = np.percentile(boot_estimates, q = (1 - self.alpha/2)*100)
        elif self.method == 'bc-boot':
            pass
        elif self.method == 'bayes-hdi':
            pass
        else:
            raise ValueError("ADD")
        return ll, ul
    

    def _bayes_hdi_interval(self, boot_estimates = None):
        """Get highest density intervals

        Parameters
        ----------
        estimates : numpy array with dimensions = [b1, 1]
            Array with Bayesian bootstrap estimates for each sample

        Returns
        -------
        ll : float
            Lower limit HDI interval estimate

        ul : float
            Upper limit HDI interval estimate
        """
        sorted_points = sorted(boot_estimates)
        ci_idx = np.ceil((1 - self.alpha) * len(sorted_points)).astype('int')
        n_cis = len(sorted_points) - ci_idx
        ci_width = [0]*n_cis
        for i in xrange(n_cis):
            ci_width[i] = sorted_points[i + ci_idx] - sorted_points[i]
            ll = sorted_points[ci_width.index(min(ci_width))]
            ul = sorted_points[ci_width.index(min(ci_width)) + ci_idx]
        return ll, ul

    def _estimate_paths(self):

        



    # ..main functions that are callable
    def fit(self, formula = None, data = None):
        """Fit model and estimate indirect effect

        Parameters
        ----------
        formula : str
            String that contains equations similar to R's lm() function or Python's statsmodel package.
            Requires two equations separated by semi-colon, where first equation is for mediator model, 
            second equation is for endogenous model

        Returns
        -------
        self : object
            A fitted object of class MediationModel
        """   
        # Parse equations and create predictors and outcomes
        formula = formula.split(';')

        # Error checking
        assert(len(formula == 2)), 'Need to provide two equations separated with semi-colon'
        if isinstance(data, pd.DataFrame) == False:
            raise ValueError('data needs to be a pandas dataframe, current data structure is %s' % type(data))

        # Mediator model
        self.m, self.design_m = dmatrices(formula[0])

        # Endogenous model
        self.y, self.design_y = dmatrices(formula[1])

        # If no intercept, then drop from both design matrices
        if self.fit_intercept == False:
            self.design_m = np.delete(design_m, [0], axis = 1)
            self.design_y = np.delete(design_y, [0], axis = 1)

        # Overall model estimates
        if self.estimate_all_paths:

            self.all_paths = {}

            # Estimate mediator model
            if self.mediator_type == 'continuous':
                clf_mediator = sm.GLM(self.m, self.design_m, family = sm.Gaussian())
            else:
                clf_mediator = sm.GLM(self.m, self.design_m, family = sm.Binomial())
            results_mediator = clf_mediator.fit()

            # Get coefficients
            coefs_mediator = results_mediator.params()
            self.all_paths['a0'] = coefs_mediator[0]
            self.all_paths['a'] = coefs_mediator[1]

            # Get standard errors
            self.all_paths['se_a0'] = np.sqrt(results_mediator().iloc[0, 0])
            self.all_paths['se_a'] = np.sqrt(results_mediator().iloc[1, 1])

            # Get confidence intervals
            self.all_paths['ci_a0'] = results_mediator.conf_int().values[0, :]
            self.all_paths['ci_a'] = results_mediator.conf_int().values[1, :]

            # Estimate endogenous model
            if self.endogenous_type == 'continuous':
                clf_endogenous = sm.GLM(y, design_y, family = sm.families.Gaussian())
            else:
                clf_endogenous = sm.GLM(y, design_y, family = sm.families.Binomial())
            results_endogenous = clf_endogenous.fit()

            # Get coefficients
            coefs_endogenous = results_endogenous.params()
            self.all_paths['b0'] = coefs_endogenous[0]
            self.all_paths['c'] = coefs_endogenous[1]
            self.all_paths['b'] = coefs_endogenous[2]

            # Get standard errors
            self.all_paths['se_b0'] = np.sqrt(results_endogenous().iloc[0, 0])
            self.all_paths['se_c'] = np.sqrt(results_endogenous().iloc[1, 1])
            self.all_paths['se_b'] = np.sqrt(results_endogenous().iloc[2, 2])

            # Get confidence intervals
            ci_b0 = results_endogenous.conf_int().values[0, :]
            self.all_paths['ci_c'] = results_endogenous.conf_int().values[1, :]
            self.all_paths['ci_b'] = results_endogenous.conf_int().values[2, :]


        # Estimate indirect effect 
        if self.method in ['delta-1', 'delta-2']:
            return self._delta_method(m = m, design_m = design_m, y = y, design_y = design_y)
        else:
            boot_estimates = np.zeros((self.b1, 1))


        # ADD OTHER FUNCTIONALITY HERE

        # Bayesian bootstrapping
        for i in xrange(self.b1):
            probs = self._bayes_boot_probs(n)
            idx = np.random.choice(np.arange(n), replace = True, p = probs, size = self.b2)
            self.boot_estimates[i, :] = self._get_indirect_boot_sample(idx = idx, 
                                                                       exogenous = exogenous,
                                                                       mediator = mediator,
                                                                       endogenous = endogenous)
        self.fit_ran = True

    def get_coefs(self):
        """Add

        Parameters
        ----------
        None

        Returns
        -------
        param_estimates : numpy array with dimensions = [n_covariates, 3]
            Array of lower limit interval, point, and upper limit interval estimates.
            Each row corresponds to a different covariate
        """
        assert(self.fit_ran == True), "Need to run .fit() method before getting coefficients"
        param_estimates = self._get_bayes_estimates(estimates = self.boot_estimates, exogenous = exogenous,)
        self.get_coefs_ran = True
        return param_estimates