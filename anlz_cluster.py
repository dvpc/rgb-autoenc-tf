
import numpy as N
import argparse, os

parser = argparse.ArgumentParser(prog='anlz.cluster')
parser.add_argument('pkldfits_file')
parser.add_argument('-dbg', action='store_true')
parser.add_argument('-n', '--prior_cluster_num', default=None)
parser.add_argument('-alg', choices=['kmean','spec'], default='kmean')
parser.add_argument('-fold', default=None, 
	help='format e.g.: 0,5__1,4__2,6__3,7')
parser.add_argument('-a', '--alpha', type=float, default=1)
parser.add_argument('-norm', '--normalize', type=float, default=1.1)

args = parser.parse_args()

'''check if ckpt file exists'''
if not os.path.exists(args.pkldfits_file):
	print 'file does not exist', args.pkldfits_file
	exit()
abs_pkldfits_file = os.path.abspath(args.pkldfits_file)
working_dir = str(os.path.split(abs_pkldfits_file)[:-1][0])

if args.prior_cluster_num is None:
	print 'no number of clusters given.'
	exit()
else:
	prior_cluster_num = int(args.prior_cluster_num)

'''load fits from file'''
from anlz import deserialize_results
loaded_result_dic = deserialize_results(
	filepath=os.path.join(working_dir, abs_pkldfits_file))
loaded_result = loaded_result_dic['fits']
channel_dim = int(loaded_result_dic['shape'][1]/3)
patch_dim = int(channel_dim**.5)
tfmax = float(loaded_result_dic['tfmax'])
try:
	tfW = loaded_result_dic['W']
except Exception, e:
	print e
	tfW = None


import colorsys
from scae import transpose_color_zero_to_one, normalize_color
from anlz import normalize_color_weighted

'''generate cluster data'''
fit_keys = {}
cluster_obs = []
for key in loaded_result.keys():
	if loaded_result[key]['c_color'] != []:	
		fit_keys[key] = loaded_result[key]

		# tmp = N.array([N.array([tfmax,tfmax,tfmax], dtype=N.float32)*args.normalize, 
		# 	transpose_color_zero_to_one(fit_keys[key]['c_color']) ])
		# cluster_obs.append( normalize_color(tmp)[1] )
		
		cluster_obs.append(
			normalize_color_weighted(
				tfmax, 
				args.normalize, 
				fit_keys[key]['c_color']) )

		# cluster_obs.append(loaded_result[key]['c_color'])


# import pprint as pp
# print pp.pprint(cluster_obs)
# import pdb; pdb.set_trace()

'''apply cluster obs'''
if args.alg == 'kmean':
	from scipy.cluster.vq import whiten
	cluster_obs = whiten(cluster_obs)
	from scipy.cluster.vq import kmeans, vq
	centroids,_ = kmeans(
		cluster_obs, 
		prior_cluster_num,
		iter=1250)
	idx, dist = vq(cluster_obs,centroids)

elif args.alg == 'spec':
	from sklearn.preprocessing import StandardScaler
	cluster_obs = StandardScaler().fit_transform(cluster_obs)
	from sklearn import cluster
	spectral = cluster.SpectralClustering(
		n_clusters=prior_cluster_num, 
		eigen_solver='arpack')
	spectral.fit(cluster_obs)
	idx = spectral.labels_.astype(N.int)
else:
	raise("Not implemented")


'''move keys in new data struc,
calc mean color for each cluster and
calc mean spread.. '''
clustered_keys = [[] for cl in [None]*prior_cluster_num]
proto_colors_arr = [[] for cl in [None]*prior_cluster_num]
proto_spread_arr = [[] for cl in [None]*prior_cluster_num]
for c, key in enumerate(fit_keys.keys()):
	try:
		tmp = idx[c]
	except Exception:
		continue
	clustered_keys[idx[c]].append(key)
	res = fit_keys[key]
	if res['p'] != []:
		proto_colors_arr[idx[c]].append(transpose_color_zero_to_one(res['c_color']))
		proto_spread_arr[idx[c]].append(res['p'][10]*res['p'][11])	
	else:
		proto_colors_arr[idx[c]].append([-1,-1,-1])
		proto_spread_arr[idx[c]].append(-1)	

prototype_color = [N.mean(prot, axis=0) for prot in proto_colors_arr]
prototype_sprd = [N.mean(sprd, axis=0) for sprd in proto_spread_arr]

'''sort the prototype colors'''
import colorsys, copy
s_prototype_color = copy.copy(prototype_color)
s_prototype_color.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
'''move the clusters in the data struc accordingly''' 
s_clustered_keys = [[] for cl in [None]*prior_cluster_num]
s_prototype_sprd = [[] for cl in [None]*prior_cluster_num]
for i, pr_c in enumerate(prototype_color):
	for j, spr_c in enumerate(s_prototype_color):
		if pr_c[0] == spr_c[0] and \
		   pr_c[1] == spr_c[1] and \
		   pr_c[2] == spr_c[2]:
			s_clustered_keys[j] = clustered_keys[i]
			s_prototype_sprd[j] = prototype_sprd[i]
			break
'''fold / combine the clusters'''
if not args.fold is None:
	fold_arg = str(args.fold).split('__')
	fold_n = len(fold_arg)
	assert fold_n > 2, 'arg format does not fit.'
	fold_clustered_keys = [[] for cl in [None]*fold_n]
	for ip, token in enumerate(fold_arg):
		tk = token.split(',')
		fold_keys = []
		try:
			for t in tk:
				fold_keys += s_clustered_keys[int(t)]
		except Exception, e:
			print e
			# import pdb; pdb.set_trace()
		fold_clustered_keys[ip] = fold_keys
	s_clustered_keys = fold_clustered_keys
	prior_cluster_num = fold_n



'''plot debug RF map'''
from anlz.fit import reconstruct_ext2
from anlz import prepare_rf_tile

frame = 1
frame_v = 3
num_tiles = sum([len(a) for a in s_clustered_keys])
num_tiles_extent = int(num_tiles**.5) + int(args.prior_cluster_num)/2+1

all_tile_extent = frame + patch_dim * (num_tiles_extent+frame)
all_tile_extent_v = frame_v + patch_dim * (num_tiles_extent+frame_v)
all_tiles = N.zeros((all_tile_extent_v,all_tile_extent,3)) + tfmax

curr_rf = 0
curr_cluster = 0

for row in xrange(0,num_tiles_extent+2):
	# try:
	# 	print 'row', row, '___', len(s_clustered_keys[curr_cluster])
	# except Exception:
	# 	print 'row', row, '___'

	for col in xrange(0,num_tiles_extent):

		curr_tile = row*num_tiles_extent + col		
		try:
			cl_keys = s_clustered_keys[curr_cluster]
		except Exception:
			curr_cluster += 1
			continue
		try:
			key = cl_keys[curr_rf]
		except Exception:
			curr_rf = 0
			curr_cluster += 1
			break

		# print curr_rf, curr_cluster, 	
		# print '__key', key

		res = fit_keys[key]
		if res['p'] != []:
			if tfW is None:
				vec_rf_rec = reconstruct_ext2(res['p'], 
					channel_dim, patch_dim, N.zeros((patch_dim**2*3)).shape)
			else:
				vec_rf_rec = tfW[int(key)]
			rf_rec = prepare_rf_tile(vec_rf_rec, patch_dim)
			st = frame_v+row*(patch_dim+frame_v), frame+col*(patch_dim+frame)			
			try:
				all_tiles[st[0]:st[0]+patch_dim, st[1]:st[1]+patch_dim] = \
					rf_rec
				all_tiles[st[0]-2:st[0], st[1]:st[1]+patch_dim] = \
					transpose_color_zero_to_one(res['c_color'])
			except Exception, e:
				print e
				# import pdb; pdb.set_trace()

		curr_rf += 1

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_size_inches(2, 2)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(
	normalize_color(all_tiles),
	interpolation='nearest')
ax.set_axis_bgcolor = 'black'
plt.savefig(os.path.join(working_dir,
	'tmp_clustered_map.png'), dpi=2048, facecolor='black')
plt.close(fig)





'''plot cluster coverage'''
sq = int(prior_cluster_num**.5)+1
rows, cols = sq, sq
zeros = N.ones(patch_dim**2).reshape(patch_dim, patch_dim)*.5

import matplotlib.pyplot as plt
fig, subplots = plt.subplots(nrows=rows, ncols=cols, sharex=False, sharey=False, squeeze=False, 
	subplot_kw={'xticks':N.arange(0,patch_dim),'yticks':N.arange(0,patch_dim),'aspect':1})
for rs in subplots:
	ax.axis('off')
	for ax in rs:
		ax.set_title('')
		for m in [ax.title] + ax.get_xticklabels() + ax.get_yticklabels():
			m.set_fontsize(4)
		im = plt.imshow(zeros, interpolation='nearest', cmap='Greys')
		ax.images.append(im)

from anlz import add_ellipsoid
cscale = 1.7
sscale = 2.7

for cidx, c in enumerate(s_clustered_keys):

	ax = subplots[cidx/cols, cidx%rows]
	ax.axis('off')

	for key in c:
		if fit_keys[key]['p'] == []:
			continue

		ellip_par = fit_keys[key]['p']
		cmux = ellip_par[0]
		cmuy = ellip_par[1]
		rwc = ellip_par[2]*cscale
		rhc = ellip_par[3]*cscale
		ctheta = ellip_par[4]
		smux = ellip_par[8]
		smuy = ellip_par[9]
		rws = ellip_par[10]*sscale
		rhs = ellip_par[11]*sscale
		stheta = ellip_par[12]
		ellip_c = [cmuy, cmux, rwc, rhc, 180/N.pi*ctheta, .01]
		ellip_s = [smuy, smux, rws, rhs, 180/N.pi*stheta, .01]
		add_ellipsoid(ax, ellip_s, 
			edge_color='k', 
			face_color='None', 
			linewidth=.5, plain=True, alpha=args.alpha)
		add_ellipsoid(ax, ellip_c, 
			edge_color='k', 
			face_color=normalize_color_weighted(
				tfmax, args.normalize, fit_keys[key]['c_color']),
			linewidth=1., plain=True, alpha=args.alpha)

filename=os.path.join(working_dir, 'tmp_clustered_cov.png')
plt.tight_layout(pad=0, w_pad=1.5, h_pad=1.5)
plt.savefig(filename, dpi=768, bbox_inches='tight')
plt.close(fig)





'''print some stats'''
alive = len(fit_keys)
dead = loaded_result_dic['shape'][0] - alive
alive_perc = float(alive) / loaded_result_dic['shape'][0]

print args.pkldfits_file
print 'visible units:', loaded_result_dic['shape'][1]
print 'hidden units:', loaded_result_dic['shape'][0]
print 'dead hidden units:', dead
print 'alive hidden units:', alive
print 'alive %', alive_perc

print 'prior cluster num', prior_cluster_num
print '# \tnum \tspread \tprototype color:'
for n in xrange(prior_cluster_num):
	print n,\
		'\t', len(s_clustered_keys[n]),\
		'\t', N.round(s_prototype_sprd[n]*100),\
		'\t', N.round(s_prototype_color[n],1)


