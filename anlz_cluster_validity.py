
import numpy as N
import argparse, os

parser = argparse.ArgumentParser(prog='anlz.cluster.val')
parser.add_argument('pkldfits_file')
parser.add_argument('-ntest', type=int, default=10, 
	help='number of test iterations.')
parser.add_argument('-mincl', type=int, default=2, 
	help='range min of clusters to test')
parser.add_argument('-maxcl', type=int, default=16, 
	help='range max of clusters to test')
parser.add_argument('-norm', '--normalize', type=float, default=1.1)

args = parser.parse_args()

'''check if ckpt file exists'''
if not os.path.exists(args.pkldfits_file):
	print 'file does not exist ', args.pkldfits_file
	exit()
abs_pkldfits_file = os.path.abspath(args.pkldfits_file)
working_dir = str(os.path.split(abs_pkldfits_file)[:-1][0])

'''load fits from file'''
from anlz import deserialize_results
loaded_result_dic = deserialize_results(
	filepath=os.path.join(working_dir, abs_pkldfits_file))
loaded_result = loaded_result_dic['fits']
channel_dim = int(loaded_result_dic['shape'][1]/3)
patch_dim = int(channel_dim**.5)
tfmax = float(loaded_result_dic['tfmax'])

from scae import normalize_color, transpose_color_zero_to_one, normalize_color2
'''generate cluster data'''
cluster_obs = []
for key in loaded_result.keys():
	if loaded_result[key]['c_color'] != []:	
		tmp = N.array([N.array([tfmax,tfmax,tfmax], dtype=N.float32)*args.normalize, 
			transpose_color_zero_to_one(loaded_result[key]['c_color']) ])
		cluster_obs.append( normalize_color(tmp)[1] )

		# cluster_obs.append(loaded_result[key]['c_color'])

'''prepare result data struc'''
import sys
results_num = len(xrange(args.mincl, args.maxcl))
results = [None]*results_num
results_davis = [None]*results_num
for i, c in enumerate(xrange(0,results_num)):
	results[i] = sys.float_info.max
	results_davis[i] = sys.float_info.max


def __apply_kmean(obs, prior_cl_num):
	from scipy.cluster.vq import whiten
	obs = whiten(obs)
	from scipy.cluster.vq import kmeans, vq
	centroids,_ = kmeans(obs, prior_cl_num, iter=1250)
	idx, dist = vq(obs,centroids)
	return obs, centroids, idx, dist

def __compact_data(obs, prior_cl_num, centroids, idx, dist):
	'''put result into explicit data structure'''
	compact = [None]*prior_cl_num
	for i, c in enumerate(centroids):
		compact[i] = []	
	for i, observation in enumerate(obs):
		compact[idx[i]].append([observation, dist[i]])
	return compact



def __davies_bouldin(prior_cl_num, centroids, dist_by_cluster):
	'''scatter inside a cluster c_i'''
	scatter_arr = [None]*prior_cl_num
	for i, c in enumerate(centroids):
		cluster_len = len(dist_by_cluster[i])
		scatter_inside_cl = 0.0
		for cluster in dist_by_cluster[i]:
			scatter_inside_cl += cluster[-1]
		scatter_arr[i] = scatter_inside_cl/cluster_len		

	'''separation between clusters i and j'''
	separation_max_arr = [None]*prior_cl_num
	for i, c in enumerate(centroids):
		separation_of_cl = []
		for j, c_other in enumerate(centroids):
			if i != j:
				separation_of_cl.append( (scatter_arr[i]+scatter_arr[j])/euclidean(c, c_other) )
		separation_max_arr[i] = N.max(separation_of_cl)

	dbidx = 0
	for i, c in enumerate(centroids):
		dbidx += separation_max_arr[i]
	return dbidx/prior_cl_num


from scipy.spatial.distance import euclidean
def __ray_turi(obs, centroids, dist_by_cluster):
	'''intra measure / compactness of clusters -> want MIN'''
	intra = 0
	for i, z in enumerate(centroids):
		for x in dist_by_cluster[i]:
			intra += euclidean(x[0], z)
	intra = intra/len(obs)		

	'''inter cluster measure; dist between cluster centres -> want MAX'''
	inter = sys.float_info.max
	for i, z in enumerate(centroids):
		for j, z_other in enumerate(centroids):
			if i != j and j >= i+1:
				eucl = euclidean(z, z_other)
				if inter > eucl:
					inter = eucl

	validity = intra / inter
	return validity


'''iterate through tests'''
for x in xrange(1,args.ntest):
	sys.stdout.write(str(x))
	sys.stdout.flush()

	'''for each num in range apply clsuter algo and test results'''
	for i, clnum in enumerate(xrange(args.mincl, args.maxcl)):

		'''cluster obs'''
		obs, centroids, idx, dist = __apply_kmean(cluster_obs, clnum)
		distances_by_cluster = __compact_data(obs, clnum, centroids, idx, dist)

		sys.stdout.write('.')
		sys.stdout.flush()

		rayturi = __ray_turi(obs, centroids, distances_by_cluster)
		davis = __davies_bouldin(clnum, centroids, distances_by_cluster)

		if results[i] > rayturi:
			results[i] = rayturi
		if results_davis[i] > davis:
			results_davis[i] = davis
		# results[i] += rayturi
		# results_davis[i] += davis


print 
print 'map', args.pkldfits_file.split('/')[-1], ',', 
# print 'map', args_file.split('/')[2], ',', 
print args.ntest, '''iterations'''
print '''clusters  \tray turi \tdavies bouldin'''
for i, c in enumerate(xrange(args.mincl, args.maxcl)):
	print c, '\t\t', N.round(results[i],4), '\t\t', N.round(results_davis[i],4)

