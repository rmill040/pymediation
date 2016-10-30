#!/usr/bin/env python

from __future__ import division, print_function

from mpi4py import MPI
import numpy as np
import sys
import timeit
import os

from pymediation import MediationModel

# Convert linear combination to binomial random variable
def linear_to_binomial(x = None, n = None):
	tmp = 1 / (1 + np.exp(-x))
	return np.random.binomial(1, tmp, (n, 1))

def write_estimates(i = None, N = None, mediator_type = None, endogenous_type = None, 
					a = None, b = None, c = None, method = None, estimator = None, 
					b1 = None, b2 = None, point = None, ll = None, ul = None):
	tmp = [i, N, mediator_type, endogenous_type, a, b, c, method, estimator, b1, b2, point, ll, ul]
	print(','.join(map(str, tmp)))

# Main simulation function
def simulation(iterations = None):

	# Start timer and simulation counter
	local_start = timeit.default_timer()

	# Sample sizes
	sample_sizes = [50, 100, 200, 500, 1000]

	# Effect sizes
	a_list = [0, .14, .39, .59]
	b_list = [0, .14, .39, .59]
	c_list = [0, .14, .39, .59]

	# Variable types
	mediator_list = ['continuous', 'categorical']
	endogenous_list = ['continuous', 'categorical']

	# Preallocate lists for simulation
	delta_methods = ['delta-1', 'delta-2']
	boot_methods = ['boot-perc', 'boot-bc']
	bayes_methods = ['bayes-cred', 'bayes-hdi']
	b1_list = [1000, 2000, 5000]
	estimator_list = ['sample', 'mean', 'median']

	# Loop through parameters
	for N in sample_sizes:
		for a in a_list:
			for b in b_list:
				for c in c_list:
					for mediator_type in mediator_list:
						for endogenous_type in endogenous_list:
							for i in xrange(iterations):

								# Exogenous variable
								X = np.random.normal(0, 1, (N, 1))	

								# Mediator variable
								if mediator_type == 'continuous':
									M = a*X + np.random.normal(0, 1, (N, 1))
								elif mediator_type == 'categorical':
									M = linear_to_binomial(a*X, N)
								else:
									raise ValueError('%s not a valid mediator_type' % mediator_type)

								# Endogenous variable
								if endogenous_type == 'continuous':
									Y = c*X + b*M + np.random.normal(0, 1, (N, 1))
								elif endogenous_type == 'categorical':
									Y = linear_to_binomial(c*X + b*M, N)
								else:
									raise ValueError('%s not a valid endogenous_type' % endogenous_type)

								# Delta methods
								for method in delta_methods:
									clf = MediationModel(method = method, 
														 mediator_type = mediator_type,
													     endogenous_type = endogenous_type)
									clf.fit(exog = X, med = M, endog = Y)
									estimates = clf.indirect_effect()
									write_estimates(i = i,
													N = N,
													mediator_type = mediator_type,
													endogenous_type = endogenous_type,
													a = a,
													b = b,
													c = c,
													method = method,
													estimator = None,
													b1 = None,
													b2 = None,
													point = estimates[0],
													ll = estimates[1],
													ul = estimates[2])

								# Bootstrap methods
								for method in boot_methods:
									clf = MediationModel(method = method, 
														 mediator_type = mediator_type,
													     endogenous_type = endogenous_type,
													     b1 = 5000,
													     estimator = 'sample')
									clf.fit(exog = X, med = M, endog = Y)
									estimates = clf.indirect_effect()
									write_estimates(i = i,
													N = N,
													mediator_type = mediator_type,
													endogenous_type = endogenous_type,
													a = a,
													b = b,
													c = c,
													method = method,
													estimator = 'sample',
													b1 = 5000,
													b2 = None,
													point = estimates[0],
													ll = estimates[1],
													ul = estimates[2])

								# Bayesian bootstrap methods
								for method in bayes_methods:
									for b1 in b1_list:
										b2_list = [N, 10*N, b1]
										for b2 in b2_list:
											for estimator in estimator_list:
												clf = MediationModel(method = method, 
																	 mediator_type = mediator_type,
																     endogenous_type = endogenous_type,
																     b1 = b1,
																     b2 = b2,
																     estimator = estimator)
												clf.fit(exog = X, med = M, endog = Y)
												estimates = clf.indirect_effect()
												write_estimates(i = i,
																N = N,
																mediator_type = mediator_type,
																endogenous_type = endogenous_type,
																a = a,
																b = b,
																c = c,
																method = method,
																estimator = estimator,
																b1 = b1,
																b2 = b2,
																point = estimates[0],
																ll = estimates[1],
																ul = estimates[2])

def main():

	# Define MPI parameters
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	# Set seed for each cpu
	np.random.seed(rank*10 + 1)
	
	# Run function
	simulation(iterations = int(sys.argv[1]))

if __name__ == "__main__":
	time_start = timeit.default_timer()
	main()
	print('Simulation finished in {0} hours'.format((timeit.default_timer() - time_start))/3600)
