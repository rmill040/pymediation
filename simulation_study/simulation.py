#!/usr/bin/env python
from __future__ import division, print_function

from mpi4py import MPI
import numpy as np
import os
import sys

from mediation import MediationModel


# Convert linear combination to binomial random variable
def linear_to_binomial(x = None, n = None):
	tmp = 1 / (1 + np.exp(-x))
	return np.random.binomial(1, tmp, (n, 1))


# Main simulation function
def simulation(iterations = None, rank = None):

	# Sample sizes
	sample_sizes = [50, 100, 200, 500, 1000]

	# Effect sizes
	a_list = [0, .14, .39, .59]
	b_list = [0, .14, .39, .59]
	c = 0

	# Variable types
	variable_type = ['continuous', 'categorical']

	# Different interval methods
	delta_interval = ['first', 'second']
	boot_interval = ['perc', 'bc']
	bayes_interval = ['cred', 'hpd']
	
	# Fully Bayesian methods
	bayes_methods = ['bayes-norm', 'bayes-robust']
	its = 20000
	burn = its/2

	# Different estimators
	bayesboot_estimators = ['sample', 'mean', 'median']
	bayes_estimators = ['mean', 'median']

	# Bootstrap parameters
	boot_samples = 5000
	boot_params = {'boot_samples': boot_samples, 'estimator': 'sample'}

	# Loop through parameters
	results = []
	for N in sample_sizes:
		for a in a_list:
			for b in b_list:
				for variable in variable_type:
					for i in xrange(iterations):

						print('Sample {0} | a {1} | b {2} | type {3} | iteration {4}\n'.format(N, a, b, variable, i))

						# Exogenous variable
						X = np.random.normal(0, 1, (N, 1))	

						# Mediator variable
						if variable == 'continuous':
							M = a*X + np.random.normal(0, 1, (N, 1))
						elif variable == 'categorical':
							M = linear_to_binomial(a*X, N)
						else:
							raise ValueError('%s not a valid variable_type' % variable)

						# Endogenous variable
						if variable == 'continuous':
							Y = c*X + b*M + np.random.normal(0, 1, (N, 1))
						elif variable == 'categorical':
							Y = linear_to_binomial(c*X + b*M, N)
						else:
							raise ValueError('%s not a valid variable_type' % variable)

						# Delta methods
						for interval in delta_interval:
							clf = MediationModel(method = 'delta',
												 interval = interval, 
												 mediator_type = variable,
											     endogenous_type = variable)
							clf.fit(exog = X, med = M, endog = Y)
							estimates = clf.indirect_effect()
							results.append([i, 				# Iteration number
											N, 				# Sample size
											variable, 		# Variable type
											a, 				# a effect size
											b, 				# b effect size
											'delta', 		# Method
											interval, 		# Interval type
											None, 			# Estimator
											None, 			# BayesBoot resample size
											estimates[0], 	# Point estimate
											estimates[1], 	# LL estimate
											estimates[2]])	# UL estimate

						# Bootstrap methods
						for interval in boot_interval:
							clf = MediationModel(method = 'boot',
												 interval = interval, 
												 mediator_type = variable,
											     endogenous_type = variable,
											     parameters = boot_params)
							clf.fit(exog = X, med = M, endog = Y)
							estimates = clf.indirect_effect()
							results.append([i, 				# Iteration number
											N, 				# Sample size
											variable, 		# Variable type
											a, 				# a effect size
											b, 				# b effect size
											'boot', 		# Method
											interval, 		# Interval type
											'sample', 		# Estimator
											None, 			# BayesBoot resample size
											estimates[0], 	# Point estimate
											estimates[1], 	# LL estimate
											estimates[2]])	# UL estimate


						# Bayesian bootstrap methods
						for interval in bayes_interval:
							b2_list = [N, 10*N, boot_samples]
							for b2 in b2_list:
								for estimator in bayesboot_estimators:
									bayesboot_params = {'boot_samples': boot_samples, 'resample_size': b2, 'estimator': estimator}
									clf = MediationModel(method = 'bayesboot',
														 interval = interval, 
														 mediator_type = variable,
													     endogenous_type = variable,
													     parameters = bayesboot_params)
									clf.fit(exog = X, med = M, endog = Y)
									estimates = clf.indirect_effect()
									results.append([i, 				# Iteration number
													N, 				# Sample size
													variable, 		# Variable type
													a, 				# a effect size
													b, 				# b effect size
													'bayesboot', 	# Method
													interval, 		# Interval type
													estimator, 		# Estimator
													b2, 			# BayesBoot resample size
													estimates[0], 	# Point estimate
													estimates[1], 	# LL estimate
													estimates[2]])	# UL estimate
						
						# Fully Bayesian methods
						for method in bayes_methods:
							for interval in bayes_interval:
								for estimator in bayes_estimators:
									bayes_params = {'iter': its, 'burn': burn, 'thin': 1, 'estimator': estimator, 'n_chains': 1}
									clf = MediationModel(method = method,
														 interval = interval, 
														 mediator_type = variable,
													     endogenous_type = variable,
													     parameters = bayes_params)
									clf.fit(exog = X, med = M, endog = Y)
									estimates = clf.indirect_effect()
									results.append([i, 				# Iteration number
													N, 				# Sample size
													variable, 		# Variable type
													a, 				# a effect size
													b, 				# b effect size
													method, 		# Method
													interval, 		# Interval type
													estimator, 		# Estimator
													None, 			# BayesBoot resample size
													estimates[0], 	# Point estimate
													estimates[1], 	# LL estimate
													estimates[2]])	# UL estimate
	# Concatenate results together
	results = np.vstack((results))
	file = 'worker_' + str(rank) + '.txt'
	np.savetxt(file, results, delimiter = ',', fmt = "%s")

def main():

	# Define MPI parameters
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	# Set seed for each cpu
	np.random.seed(rank*10 + 1)
	
	# Run function
	simulation(iterations = int(sys.argv[1]), rank = rank)

if __name__ == "__main__":
	main()
