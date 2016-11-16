from __future__ import division, print_function

import numpy as np
import unittest

from ..core.mediation import MediationModel

__all__ = ["testMediationMethods"]


class testMediationMethods(unittest.TestCase):


    def _linear_to_binomial(self, linear_combo = None, n = None):

        # Convert to probabilities and sample random binomial variables
        tmp = 1 / (1 + np.exp(-linear_combo))
        return np.random.binomial(1, tmp, (n, 1))


    def _simulate_data(self, N = 5000, mediator_type = None, endogenous_type = None, effect = .59):
        
        # Exogenous variable
        self.X = np.random.normal(0, 1, (N, 1))  

        # Mediator variable
        if mediator_type == 'continuous':
            self.M = effect*self.X + np.random.normal(0, 1, (N, 1))
        elif mediator_type == 'categorical':
            self.M = self._linear_to_binomial(effect*self.X, N)
        else:
            raise ValueError('%s not a valid mediator_type' % mediator_type)

        # Endogenous variable
        if endogenous_type == 'continuous':
            self.Y = effect*self.X + effect*self.M + np.random.normal(0, 1, (N, 1))
        elif endogenous_type == 'categorical':
            self.Y = self._linear_to_binomial(effect*self.X + effect*self.M, N)
        else:
            raise ValueError('%s not a valid endogenous_type' % endogenous_type)

        # True indirect effect size
        self.TRUE_ab = effect*effect

        # Fudge factor for point estimates
        self.EPS = .10

    # Test Delta method
    def test_delta(self):

        # Define delta intervals
        intervals = ['first', 'second']
        med_types = ['continuous', 'categorical']
        end_types = ['continuous', 'categorical']

        for interval in intervals:
            for med_type in med_types:
                for end_type in end_types:
                    self._simulate_data(mediator_type = med_type, endogenous_type = end_type)
                    clf = MediationModel(method = 'delta', 
                                         interval = interval, 
                                         mediator_type = med_type,
                                         endogenous_type = end_type)

                    clf.fit(exog = self.X, med = self.M, endog = self.Y)
                    estimates = clf.indirect_effect()
                    self.assertTrue(estimates[0] > self.TRUE_ab - self.EPS and estimates[0] < self.TRUE_ab + self.EPS)

    def test_boot(self):

        # Define boot intervals
        intervals = ['perc', 'bc']
        estimator = ['sample', 'mean', 'median']
        med_types = ['continuous', 'categorical']
        end_types = ['continuous', 'categorical']

        for est in estimator:
            for interval in intervals:
                for med_type in med_types:
                    for end_type in end_types:
                        self._simulate_data(mediator_type = med_type, endogenous_type = end_type)
                        params = {'boot_samples': 1000, 'estimator': est}
                        clf = MediationModel(method = 'boot', 
                                             interval = interval, 
                                             mediator_type = med_type,
                                             endogenous_type = end_type,
                                             parameters = params)

                        clf.fit(exog = self.X, med = self.M, endog = self.Y)
                        estimates = clf.indirect_effect()
                        self.assertTrue(estimates[0] > self.TRUE_ab - self.EPS and estimates[0] < self.TRUE_ab + self.EPS)

    def test_bayesboot(self):
        pass

    def test_bayes(self):
        pass

"""
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

    # Fully Bayesian method with normal priors (HPD intervals)
    params = {'iter': 10000, 'burn': 500, 'thin': 1, 'n_chains': 2, 'estimator': 'mean'}
    clf = MediationModel(method = 'bayes-norm', interval = 'hpd', mediator_type = 'continuous', 
                         endogenous_type = 'continuous', plot = True, parameters = params)
    clf.fit(exog = x, med = m, endog = y)
    print(clf.indirect_effect())
    clf.summary(exog_name = 'depression', med_name = 'alcohol', endog_name = 'drugabuse')
    clf.plot_indirect()

    # Fully Bayesian method with robust priors (HPD intervals)
    params = {'iter': 10000, 'burn': 500, 'thin': 1, 'n_chains': 2, 'estimator': 'mean', 'standardize': True}
    clf = MediationModel(method = 'bayes-robust', interval = 'cred', mediator_type = 'continuous', 
                         endogenous_type = 'continuous', plot = True, parameters = params)
    clf.fit(exog = x, med = m, endog = y)
    print(clf.indirect_effect())
    clf.summary(exog_name = 'depression', med_name = 'alcohol', endog_name = 'drugabuse')
    clf.plot_indirect()
"""
if __name__ == "__main__":
    unittest.main()