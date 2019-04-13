import numpy as N
import os

def edog_abc_ext2(
	p,
	channel_dim,
	patch_dim,
	x):
	xc = x % channel_dim
	px = xc % patch_dim
	py = xc / patch_dim	
	'''params: 0: cmu_x  1: cmu_y  
			   2: csigma_x  3: csigma_y  4: ctheta
			   5: ccdir_a 	6: ccdir_b   7: ccdir_c
			   8: smu_x  9: smu_y
			  10: ssigma_x  11: ssigma_y 12: stheta
			  13: scdir_a   14: scdir_b  15: scdir_c
			  16: k_s (k_c is implicitly fixed as 1)
			   '''
	def gauss2d(mux, muy, sigx, sigy, theta, scale=1):
		sigma_x = sigx**2*scale
		sigma_y = sigy**2*scale
		a =  N.cos(theta)**2/2/sigma_x + N.sin(theta)**2/2/sigma_y
		b = -N.sin(2*theta)/4/sigma_x + N.sin(2*theta)/4/sigma_y
		c =  N.sin(theta)**2/2/sigma_x + N.cos(theta)**2/2/sigma_y
		return N.exp( - ( a *(px-mux)**2 + 2*b*(px-mux)*(py-muy) + c*(py-muy)**2 ) )

	dog_c = 		gauss2d(p[0], p[1], p[2], p[3], p[4])
	dog_s = p[16] * gauss2d(p[8], p[9], p[10], p[11], p[12])	
	ret_A = p[5] * dog_c[0:channel_dim] 		      - p[13] * dog_s[0:channel_dim]
	ret_B = p[6] * dog_c[channel_dim:2*channel_dim]   - p[14] * dog_s[channel_dim:2*channel_dim]
	ret_C = p[7] * dog_c[2*channel_dim:3*channel_dim] - p[15] * dog_s[2*channel_dim:3*channel_dim]
 	ret = N.concatenate([ ret_A, ret_B, ret_C ])
	return ret


def reconstruct_ext2(
	p,
	channel_dim, 
	patch_dim, 
	shape):
	return edog_abc_ext2(
		p, 
		channel_dim, 
		patch_dim, 
		*N.indices(shape))


def permutate_mu_ext2( 
	p,
	patch_dim,
	channel_dim,
	rf
	):
	from . import arg_absmax_rfvector
	center_point = N.mean(
		arg_absmax_rfvector(rf, patch_dim, channel_dim), 
		axis=0)
	'''RF center point noise'''
	noise_c = (N.random.random_sample(2)-.5)*2 * patch_dim/8
	point_c = center_point + noise_c
	noise_s = (N.random.random_sample(2)-.5)*2 * patch_dim/8
	point_s = center_point + noise_s
	p[0] = point_c[0]
	p[1] = point_c[1]
	p[8] = point_s[0]
	p[9] = point_s[1]
	return p


def bestfit_ext2(
	rf, 
	channel_dim, 
	patch_dim, 
	num_attempts=10,
	maxiter=10000,
	num_pool_threads=4,
	min_error_accept=.5
	):
	'''params: 0: cmu_x  1: cmu_y  
			   2: csigma_x  3: csigma_y  4: ctheta
			   5: ccdir_a 	6: ccdir_b   7: ccdir_c
			   8: smu_x  9: smu_y
			  10: ssigma_x  11: ssigma_y 12: stheta
			  13: scdir_a   14: scdir_b  15: scdir_c
			  16: k_s (k_c is implicitly fixed as 1)
			   '''
	constraints = (
		{'type': 'ineq', 'fun': lambda x:  x[0]},				# cx > 0
		{'type': 'ineq', 'fun': lambda x:  patch_dim-1 - x[0]},	# cx < patch_dim-1
		{'type': 'ineq', 'fun': lambda x:  x[1]},				# cy > 0
		{'type': 'ineq', 'fun': lambda x:  patch_dim-1 - x[1]},	# cy < patch_dim-1

		{'type': 'ineq', 'fun': lambda x:  x[8]},				# cx > 0
		{'type': 'ineq', 'fun': lambda x:  patch_dim-1 - x[8]},	# cx < patch_dim-1
		{'type': 'ineq', 'fun': lambda x:  x[9]},				# cy > 0
		{'type': 'ineq', 'fun': lambda x:  patch_dim-1 - x[9]},	# cy < patch_dim-1


		{'type': 'ineq', 'fun': lambda x:  x[2]/x[3] + .6},
		{'type': 'ineq', 'fun': lambda x:  1.4 - x[2]/x[3]},
		{'type': 'ineq', 'fun': lambda x:  x[3]/x[2] + .6},
		{'type': 'ineq', 'fun': lambda x:  1.4 - x[3]/x[2]},
		{'type': 'ineq', 'fun': lambda x:  x[10]/x[11] + .6},
		{'type': 'ineq', 'fun': lambda x:  1.4 - x[10]/x[11]},
		{'type': 'ineq', 'fun': lambda x:  x[11]/x[10] + .6},
		{'type': 'ineq', 'fun': lambda x:  1.4 - x[11]/x[10]},


		{'type': 'ineq', 'fun': lambda x: .25 - abs(x[0] - x[8])},
		{'type': 'ineq', 'fun': lambda x: .25 - abs(x[1] - x[9])},

		{'type': 'ineq', 'fun': lambda x: x[2]-.17},
		{'type': 'ineq', 'fun': lambda x: x[3]-.17},		

		{'type': 'ineq', 'fun': lambda x: 1.5-abs(x[2]-x[10])},
		{'type': 'ineq', 'fun': lambda x: 1.5-abs(x[3]-x[11])},

		{'type': 'ineq', 'fun': lambda x: x[10]-x[2]-.03},
		{'type': 'ineq', 'fun': lambda x: x[11]-x[3]-.03},

	
		)
	bounds_p = []
	k_s = -.5 if N.abs(N.min(rf)) > N.abs(N.max(rf)) else .5
	init_p = [patch_dim/2, patch_dim/2]
	init_p += [patch_dim/2, patch_dim/2, N.pi/4.] + [0,0,0]
	init_p += [patch_dim/2, patch_dim/2]
	init_p += [patch_dim/2, patch_dim/2, N.pi/4.] + [0,0,0]
	init_p += [k_s]

	def __single_run(init_p):
		min_y, min_p = fit_slsqp(
			edog_abc_ext2, 
			rf, 
			init_p, 
			bounds_p, 
			channel_dim, 
			patch_dim, 
			maxiter=maxiter, 
			constraints=constraints)
		return (min_y, min_p)
	'''create list of inital model param'''
	runs_init_p = [permutate_mu_ext2(
		init_p, patch_dim, channel_dim, rf) for x in range(num_attempts)]
	'''run the list in parallel'''
	from multiprocessing import Pool
	from multiprocessing.dummy import Pool as ThreadPool
	pool = ThreadPool(num_pool_threads)
	runs = pool.map(__single_run, runs_init_p)
	pool.close()
	pool.join()
	'''choose best (min error) run'''
	import sys
	min_run = (sys.float_info.max,[])
	for run in runs:
		'''	- has to be smallest error yet
			- error must be below 'min_error_accept'!
			- mu xy must be inside the patch!'''
		if run[0] < min_run[0] and\
		 run[0] < min_error_accept and\
		 run[1][0] >= 0 and run[1][0] < patch_dim and\
		 run[1][1] >= 0 and run[1][1] < patch_dim:
			min_run = run
	return min_run


def print_params_ext2(
	ext2_params
	):
	p_names = ['cmu_x','cmu_y',
		'csigma_x','csigma_y','ctheta',
		'ccdir_a','ccdir_b','ccdir_c',
		'smu_x','smu_y',
		'ssigma_x','ssigma_y','stheta',
		'scdir_a','scdir_b','scdir_c',
		'k_s']
	print 'ext2_params:'
	for idx, p in enumerate(ext2_params):
		print p_names[idx], '    \t', N.round(p,2)
	print 


def best_fit_wrapper(
	rf,
	channel_dim, 
	patch_dim, 
	num_fit_attempts=10,
	maxiter=1000,
	num_pool_threads=3,
	debug_idx=None,
	min_error_accept=.5
	):
	from . import wrapper_value_of_rfvector_at
	from . import transpose_color_zero_to_one
	from scae import normalize_color

	print '__fit #rf', debug_idx

	result = []
	num_attempts_succ_run = 6
	while num_attempts_succ_run > 0:
		result = bestfit_ext2(
			rf, 
			channel_dim, 
			patch_dim, 
			num_attempts=num_fit_attempts,
			maxiter=maxiter,
			num_pool_threads=num_pool_threads,
			min_error_accept=min_error_accept
		)
		'''if a successful fit has been made...'''
		if result[1] != []:
			break
		else:
			print 'starting over', num_attempts_succ_run, 'tries left.'
			min_error_accept += .02
			print 'raising min error to', min_error_accept
		num_attempts_succ_run -= 1

	'''check if after n attempts resulted in successful fit'''
	if result[1] != []:
		print '__done #rf', debug_idx, 'e', N.round(result[0],2)
		print_params_ext2(result[1])
		'''retrieve center color'''
		rf_c_val = wrapper_value_of_rfvector_at(
			rf, 
			result[1][0], result[1][1],
			patch_dim, 
			channel_dim)
		return result[1], rf_c_val
	else:
		print '__NOT done #rf'
		return [], []





def f_to_min(
	p, 
	f,
	rf,
	channel_dim, 
	patch_dim
	):
	sqerr = (rf - f(p, channel_dim, patch_dim, *N.indices(rf.shape)))**2
	return N.sum(sqerr)

from scipy import optimize
''' SLSQP  Sequential Least Squares Programming
	http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html'''
def fit_slsqp(
	f,	
	rf, 
	init_p, 
	bounds_p, 
	channel_dim, 
	patch_dim,
	maxiter=10000,
	constraints=()
	):
	res = optimize.minimize(f_to_min, init_p, 
		args=(f, rf, channel_dim, patch_dim), 
		method='SLSQP', options={'maxiter':maxiter}, 
		bounds=bounds_p, constraints=constraints)
	return res.fun, res.x




