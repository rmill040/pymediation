from __future__ import division, print_function

import numpy as np
import unittest

from ..core.mediation import MediationModel

__all__ = ["testMediationMethods"]


class testMediationMethods(unittest.TestCase):

    # Convert to probabilities and sample random binomial variables
    def _linear_to_binomial(self, linear_combo = None, n = None):
        tmp = 1 / (1 + np.exp(-linear_combo))
        return np.random.binomial(1, tmp, (n, 1))

    # Simulate data for testing
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
        self.EPS = .50

    # Test multivariate delta method
    def test_delta(self):
        print('\nTesting delta method...\n')

        # Define lists of interest
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

    # Test bootstrap method
    def test_boot(self):
        print('\nTesting bootstrap method...\n')

        # Define lists of interest
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


    # Test Bayesian bootstrap method
    def test_bayesboot(self):
        print('\nTesting Bayesian bootstrap method...\n')

        # Define lists of interest
        intervals = ['cred', 'hpd']
        estimator = ['sample', 'mean', 'median']
        med_types = ['continuous', 'categorical']
        end_types = ['continuous', 'categorical']

        for est in estimator:
            for interval in intervals:
                for med_type in med_types:
                    for end_type in end_types:
                        self._simulate_data(mediator_type = med_type, endogenous_type = end_type)
                        params = {'boot_samples': 1000, 'resample_size': 1000, 'estimator': est}
                        clf = MediationModel(method = 'bayesboot', 
                                             interval = interval, 
                                             mediator_type = med_type,
                                             endogenous_type = end_type,
                                             parameters = params)

                        clf.fit(exog = self.X, med = self.M, endog = self.Y)
                        estimates = clf.indirect_effect()
                        self.assertTrue(estimates[0] > self.TRUE_ab - self.EPS and estimates[0] < self.TRUE_ab + self.EPS)


    # Test fully Bayesian method
    def test_bayes(self):
        print('\nTesting fully Bayesian method...\n')

        # Define lists of interest
        methods = ['bayes-norm', 'bayes-robust']
        intervals = ['cred', 'hpd']
        estimator = ['mean', 'median']
        med_types = ['continuous', 'categorical']
        end_types = ['continuous', 'categorical']

        for method in methods:
            for est in estimator:
                for interval in intervals:
                    for med_type in med_types:
                        for end_type in end_types:
                            self._simulate_data(mediator_type = med_type, endogenous_type = end_type)
                            params = {'iter': 10000, 
                                      'burn': 5000, 
                                      'thin': 1,
                                      'estimator': est, 
                                      'n_chains': 1, 
                                      'standardize': False}
                            clf = MediationModel(method = method, 
                                                 interval = interval, 
                                                 mediator_type = med_type,
                                                 endogenous_type = end_type,
                                                 parameters = params)

                            clf.fit(exog = self.X, med = self.M, endog = self.Y)
                            estimates = clf.indirect_effect()
                            self.assertTrue(estimates[0] > self.TRUE_ab - self.EPS and estimates[0] < self.TRUE_ab + self.EPS)


if __name__ == "__main__":
    unittest.main()