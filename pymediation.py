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
                                    bayes-cred (Bayesian bootstrap with credible intervals)
                                    bayes-hdi (Bayesian bootstrap with highest density intervals)

    mediator_type : string
                    Variable indicating whether mediator variable is continuous or categorical

    endogenous_type : string
                      Variable indicating whether endogenous variable is continuous or categorical

    b1 : int
        Stage one number of samples to draw - corresponds to the number of bootstrap or Bayesian bootstrap samples

    b2 : int (for bayesboot)
        Stage two number of samples to draw - corresponds to the size of Bayesian bootstrap samples to draw

    estimator : str (for bootstrap), default sample
        Bootstrap point estimator. Currently supports sample, mean, and median

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
                 b2 = None, estimator = 'sample', alpha = .05, fit_intercept = True, save_boot_estimates = False,
                 estimate_all_paths = False):

        # Define global variables
        if method not in ['delta-1', 'delta-2', 'boot-perc', 'boot-bc', 'bayes-cred', 'bayes-hdi']:
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
        if self.method in ['boot-perc', 'boot-bc', 'bayes-cred', 'bayes-hdi']:
            if estimator not in ['sample', 'mean', 'median']:
                raise ValueError('%s is not a valid estimator' % estimator)
            else:
                self.estimator = estimator

            if save_boot_estimates == True or save_boot_estimates == False:
                self.save_boot_estimates = save_boot_estimates
            else:
                raise ValueError('save_boot_estimates should be a boolean argument')

            assert(isinstance(b1, int) == True), 'b1 should be an interger argument'
            self.b1 = b1

            if self.method in ['bayes-cred', 'bayes-hdi']:
                assert(isinstance(b2, int) == True), 'b2 should be an integer argument'
                self.b2 = b2
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
        indirect : dictionary
            Dictionary containing: (1) point estimate, (2) confidence intervals
        """
        indirect = {}

        # Mediator variable model
        if self.mediator_type == 'continuous':
            clf_mediator = sm.GLM(m, design_m, family = sm.families.Gaussian())
        else:
            clf_mediator = sm.GLM(m, design_m, family = sm.families.Binomial())

        # Estimate model and get coefficients
        clf_mediator_results = clf_mediator.fit()
        beta_m = clf_mediator_results.params.reshape(2,)
        vcov_m = -np.linalg.inv(clf_mediator.information(beta_m)) # Get variance/covariance matrix
            
        # Endogenous variable model
        if self.endogenous_type == 'continuous':
            clf_endogenous = sm.GLM(y, design_y, family = sm.families.Gaussian())
        else:
            clf_endogenous = sm.GLM(y, design_y, family = sm.families.Binomial())

        # Estimate model and get coefficients
        clf_endogenous_results = clf_endogenous.fit()
        beta_y = clf_endogenous_results.params.reshape(3,)
        vcov_y = -np.linalg.inv(clf_endogenous.information(beta_y)) # Get variance/covariance matrix

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
        ll, ul = point - z_score * np.sqrt(MM_var), point + z_score * np.sqrt(MM_var)
        indirect['point'] = point; indirect['ci'] = np.array([ll, ul]).reshape((1, 2))  
        return indirect  


    def _boot_point(self, m = None, design_m = None, y = None, design_y = None, boot_estimates = None):
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

        boot_estimates : 1d array-like
            Array with bootstrap estimates for each sample

        Returns
        -------
        point : float
            Bootstrap point estimate for indirect effect
        """

        # Get posterior point estimate based on estimator 
        if self.estimator == 'mean':
            return np.mean(boot_estimates)
        elif self.estimator == 'median':
            return np.median(boot_estimates)
        else: 
            return self._point_estimate(m = m, design_m = design_m, y = y, design_y = design_y)
       

    def _boot_interval(self, boot_estimates = None, sample_point = None):
        """Get (1-alpha)*100 interval estimates based on specified method

        Parameters
        ----------
        estimates : 1d array-like
            Array with bootstrap estimates for each sample

        sample_point : float
            Indirect effect estimate based on full sample. Note, this is only used by the
            bias-corrected confidence interval

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit interval estimates
        """
        if self.method in ['boot-perc', 'bayes-cred']:
            return self._percentile_interval(boot_estimates)
        elif self.method == 'boot-bc':
            return self._bias_corrected_interval(boot_estimates, sample_point = sample_point)
        else: 
            return self._hdi_interval(boot_estimates)


    def _percentile_interval(self, boot_estimates = None):
        """Get (1-alpha)*100 percentile (nonparametric) or credible (Bayesian) interval estimate

        Parameters
        ----------
        estimates : 1d array-like
            Array with bootstrap estimates for each sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit percentile interval estimates
        """
        ll = np.percentile(boot_estimates, q = (self.alpha/2)*100)
        ul = np.percentile(boot_estimates, q = (1 - self.alpha/2)*100)
        return np.array([ll, ul])


    def _bias_corrected_interval(self, boot_estimates = None, sample_point = None):
        """Get (1-alpha)*100 bias-corrected confidence interval estimate

        Parameters
        ----------
        estimates : 1d array-like
            Array with bootstrap estimates for each sample

        sample_point : float
            Indirect effect point estimate based on full sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit bias-corrected confidence interval estimates
        """
        assert(self.estimator == 'sample'), 'The estimator must be sample for bias-corrected intervals'
        z0 = scipy.stats.norm.ppf(np.sum(boot_estimates < sample_point)/self.b1)
        adjusted_ll = scipy.stats.norm.cdf(2*z0 + scipy.stats.norm.ppf(self.alpha/2))*100
        adjusted_ul = scipy.stats.norm.cdf(2*z0 + scipy.stats.norm.ppf(1 - self.alpha/2))*100
        ll = np.percentile(boot_estimates, q = adjusted_ll)
        ul = np.percentile(boot_estimates, q = adjusted_ul)
        return np.array([ll, ul])


    """
    Next two functions taken form the PyMC library https://github.com/pymc-devs/pymc -> utils.py
    """
    def _calc_min_interval(self, boot_estimates = None):
        """Determine the minimum interval of a given width

        Parameters
        ----------
        boot_estimates : SORTED numpy array with dimensions = [b1, 1]
            Array with Bayesian bootstrap estimates for each sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit for highest density interval estimates
        """
        n = len(boot_estimates)
        cred_mass = 1.0 - self.alpha

        interval_idx_inc = int(np.floor(cred_mass*n))
        n_intervals = n - interval_idx_inc
        interval_width = boot_estimates[interval_idx_inc:] - boot_estimates[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError('Too few elements for interval calculation')

        min_idx = np.argmin(interval_width)
        hdi_min = boot_estimates[min_idx]
        hdi_max = boot_estimates[min_idx+interval_idx_inc]
        return np.array([hdi_min, hdi_max])


    def _hdi_interval(self, boot_estimates = None):
        """Get (1-alpha)*100 highest posterior density estimates

        Parameters
        ----------
        boot_estimates : numpy array with dimensions = [b1, 1]
            Array with Bayesian bootstrap estimates for each sample

        Returns
        -------
        CI : 1d array-like
            Lower limit and upper limit for highest density interval estimates
        """

        # Make a copy of trace
        boot_estimates = boot_estimates.copy()

        # For multivariate node
        if boot_estimates.ndim > 1:

            # Transpose first, then sort
            tx = np.transpose(boot_estimates, list(range(boot_estimates.ndim))[1:]+[0])
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
            sx = np.sort(boot_estimates)
            return np.array(self._calc_min_interval(sx))


    def _boot_method(self, m = None, design_m = None, y = None, design_y = None):
        """Estimate indirect effect with confidence interval using nonparametric or Bayesian bootstrap

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
        indirect : dictionary
            Dictionary containing: (1) point estimate, (2) confidence intervals
        """
        indirect = {}
        boot_estimates = np.zeros((self.b1))
        n = m.shape[0]  # Assumes all arguments have same size

        # Nonparametric bootstrap. Note, p = None implies uniform distribution over np.arange(n)
        if self.method in ['boot-perc', 'boot-bc']:
            for i in xrange(self.b1):
                idx = np.random.choice(np.arange(n), replace = True, p = None, size = n)
                boot_estimates[i] = self._point_estimate(m = m[idx], design_m = design_m[idx], 
                                                         y = y[idx], design_y = design_y[idx])
        else:
            # Bayesian bootstrapping
            for i in xrange(self.b1):
                probs = self._bayes_probs(n)
                idx = np.random.choice(np.arange(n), replace = True, p = probs, size = self.b2)
                boot_estimates[i] = self._point_estimate(m = m[idx], design_m = design_m[idx], 
                                                            y = y[idx], design_y = design_y[idx])

        if self.save_boot_estimates:
            indirect['boot_estimates'] = boot_estimates

        # Bootstrap point estimate and confidence interval
        indirect['point'] = self._boot_point(m = m, design_m = design_m, y = y, 
                                             design_y = design_y, boot_estimates = boot_estimates)
        indirect['ci'] = self._boot_interval(boot_estimates = boot_estimates, sample_point = indirect['point'])
        return indirect


    def _estimate_paths(self, m = None, design_m = None, y = None, design_y = None):
        """Estimate all coefficients from mediation model

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
        self : object
            Creates a dictionary that contains the point estimates, standard errors, and confidence intervals
            for each structural path in the model
        """
        # Estimate mediator model
        self.all_paths = {}
        if self.mediator_type == 'continuous':
            clf_mediator = sm.GLM(m, design_m, family = sm.families.Gaussian())
        else:
            clf_mediator = sm.GLM(m, design_m, family = sm.families.Binomial())
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
            clf_endogenous = sm.GLM(y, design_y, family = sm.families.Gaussian())
        else:
            clf_endogenous = sm.GLM(y, design_y, family = sm.families.Binomial())
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
        m, design_m = dmatrices('m ~ x', data = data)
        y, design_y = dmatrices('y ~ x + m', data = data)

        # If no intercept, then drop from both design matrices
        if self.fit_intercept == False:
            design_m = np.delete(design_m, [0], axis = 1)
            design_y = np.delete(design_y, [0], axis = 1)

        # Estimates all paths if specified
        if self.estimate_all_paths:
            self.n = m.shape[0]
            self._estimate_paths(m = m, design_m = design_m, y = y, design_y = design_y)

        # Estimate indirect effect based on method
        if self.method in ['delta-1', 'delta-2']:
            self.indirect = self._delta_method(m = m, design_m = design_m, y = y, design_y = design_y)
        else:
            self.indirect = self._boot_method(m = m, design_m = design_m, y = y, design_y = design_y)
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
        assert(self.estimate_all_paths == True), 'Need to specify True for estimate_all_paths to get summary'

        # Define method strings
        if self.method == 'delta-1':
            str_method = 'Taylor Series Approximation'
            str_ci = 'First-Order Multivariate Delta'
        elif self.method == 'delta-2':
            str_method = 'Taylor Series Approximation'
            str_ci = 'Second-Order Multivariate Delta'
        elif self.method == 'boot-perc':
            str_method = 'Nonparametric Bootstrap'
            str_ci = 'Percentile'
        elif self.method == 'boot-bc':
            str_method = 'Nonparametric Bootstrap'
            str_ci = 'Bias-Corrected'
        elif self.method == 'bayes-cred':
            str_method = 'Bayesian Bootstrap'
            str_ci = 'Credible'
        else:
            str_method = 'Bayesian Bootstrap'
            str_ci = 'Highest Posterior Density'

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
        print('{0:<20}{1:<14}'.format('Interval:', str_ci))

        if self.method in ['boot-perc', 'boot-bc', 'bayes-cred', 'bayes-hdi']:
            print('{0:<20}{1:<3}'.format('Boot Samples:', self.b1))
            if self.method in ['bayes-cred', 'bayes-hdi']:
                print('{0:<20}{1:<3}'.format('Resample Size:', self.b2))
                print('{0:<20}{1:<10}'.format('Estimator:', self.estimator))

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
        """Plot histogram of bootstrap distribution of indirect effect

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Error checking
        assert(self.fit_ran == True), 'Need to run .plot() method before generating histogram'
        assert(self.save_boot_estimates == True), 'Need to specify True for save_boot_estimates to generate histogram'
        
        # Create figure
        plt.figure()
        plt.hist(self.indirect['boot_estimates'], bins = 100, color = 'gray')
        plt.axvline(self.indirect['point'], color = 'blue', label = 'Point', linewidth = 3)
        plt.axvline(self.indirect['ci'][0], color = 'blue', label = 'Interval', linestyle = 'dashed', linewidth = 3)
        plt.axvline(self.indirect['ci'][1], color = 'blue', linestyle = 'dashed', linewidth = 3)
        
        # Check method for title of histogram
        if self.method == 'boot-perc':
            str_method = 'Bootstrap Distribution'
            str_ci = 'Percentile CI'
        elif self.method == 'boot-bc':
            str_method = 'Bootstrap Distribution'
            str_ci = 'BC CI'
        elif self.method == 'bayes-cred':
            str_method = 'Bayesian Bootstrap Distribution'
            str_ci = 'Credible Interval'
        else:
            str_method = 'Bayesian Bootstrap Distribution'
            str_ci = 'HPD Interval'
        title_str = '{title:} with {alpha:}% {int_type:}\nPoint = {point:.3f}, Interval = [{ll:.3f}, {ul:.3f}]'.format(
                                                                                        title = str_method, 
                                                                                        alpha = int((1-self.alpha)*100), 
                                                                                        int_type = str_ci,
                                                                                        point = self.indirect['point'],
                                                                                        ll = self.indirect['ci'][0],
                                                                                        ul = self.indirect['ci'][1])
        
        plt.title(title_str)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    x = np.random.normal(0, 1, (100, 1))
    m = .4*x + np.random.normal(0, 1, (100, 1))
    y = .4*m + np.random.normal(0, 1, (100, 1))
    clf = MediationModel(method = 'bayes-hdi', b1 = 5000, b2 = 100, mediator_type = 'continuous', estimator = 'sample',
                         endogenous_type = 'continuous', estimate_all_paths = True, save_boot_estimates = True)

    clf.fit(exog = x, med = m, endog = y)
    print(clf.indirect_effect())
    clf.summary(exog_name = 'depression', med_name = 'alcohol', endog_name = 'drugabuse')
    clf.plot_indirect()