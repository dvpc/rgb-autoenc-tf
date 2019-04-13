import numpy as N
import argparse, os

parser = argparse.ArgumentParser(prog='z.make_RF_crop')
parser.add_argument('pkldfits_file')
parser.add_argument('-rfids', default=None, help='format e.g. 1,112,13,29,50')
parser.add_argument('-norm', '--normalize', type=float, default=1.1)

args = parser.parse_args()

'''check if ckpt file exists'''
if not os.path.exists(args.pkldfits_file):
	print 'file does not exist', args.pkldfits_file
	exit()
abs_pkldfits_file = os.path.abspath(args.pkldfits_file)
working_dir = str(os.path.split(abs_pkldfits_file)[:-1][0])


'''build list of RF to process'''
rfs_to_crop = []
if args.rfids is None:
	print 'no list of cluster ids given.'
	exit()
else:
	tmp_rf_ids = args.rfids.split(',')
	'''check for valid input'''
	for strid in tmp_rf_ids:
		try:
			rfs_to_crop.append(int(strid))
		except Exception, e:
			print 'Value error in list of RF ids:', 
			print e; exit()


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
	tfmean = N.mean(tfW)
except Exception, e:
	print 'tfVariable', e, 'is missing, using reconstruction instead.'
	tfW = None
	tfmean = .5



'''plot RFs ...'''

'''shift rf valaues towards (global) tfmean'''
def shift_mean_towards_value(a, value):
	dmean = value - N.mean(a)
	return a + dmean

from scae import transpose_color_zero_to_one, normalize_color
from anlz import prepare_rf_tile, normalize_color_weighted
from anlz.fit import reconstruct_ext2
from anlz import add_ellipsoid
from anlz import wrapper_value_of_rfvector_at

frame = 1
frame_v = 5
num_tiles = len(rfs_to_crop)
def __map_extent(num_t):
	if num_t == 1:
		return (1,1)
	elif num_t == 2:
		return (1,2)
	elif num_t == 3:
		return (1,3)
	elif num_t == 4:
		return (1,4)
	elif num_t >= 5 and num_t <= 8:
		return (2,4)
	elif num_t >= 9 and num_t <= 12:
		return (3,4)
	else:
		ext = int(num_tiles**.5)+1
		return (ext,ext)
num_tiles_extent = __map_extent(num_tiles)

tile_extent = int(patch_dim) * 2
all_tile_extent = frame + tile_extent * (num_tiles_extent[1]+frame)
all_tile_extent_v = frame_v + tile_extent * (num_tiles_extent[0]+frame)
all_tiles = N.ones((all_tile_extent_v,all_tile_extent,3))

curr_index = 0
'''tmp storing fit params and tile offset
for later plotting when building the matplotlib fig''' 
tmp_fit_data = []

for row in xrange(0,num_tiles_extent[0]):
	for col in xrange(0,num_tiles_extent[1]):
		try:
			key = rfs_to_crop[curr_index]
		except Exception:
			continue
		res = loaded_result[key]
		'''if rf has fit data'''
		if res['p'] != []:
			rf_has_fit_data = True
			vec_rec_rf = reconstruct_ext2(res['p'], 
				channel_dim, patch_dim, N.zeros((patch_dim**2*3)).shape)
		else:
			rf_has_fit_data = False
			vec_rec_rf = None
		'''if weight maxreix is present'''
		if tfW is None:
			vec_rf = None if vec_rec_rf is None else vec_rec_rf
		else:
			vec_rf = tfW[int(key)]

		'''calc start positions for each sub-tile'''
		st = frame_v+row*(tile_extent+frame_v), frame+col*(tile_extent+frame)
		st2 = st[0], 			st[1]+patch_dim
		st3 = st[0]+patch_dim, 	st[1]
		st4 = st[0]+patch_dim, 	st[1]+patch_dim

		'''prepare rfs tiles'''
		rf = prepare_rf_tile(vec_rf, patch_dim)
		rf_norm = normalize_color_weighted(tfmax, args.normalize, rf)
		rf_norm = shift_mean_towards_value(rf_norm, tfmean)

		if rf_has_fit_data:
			rf_rec = prepare_rf_tile(vec_rec_rf, patch_dim)
			_, mu = wrapper_value_of_rfvector_at(
						vec_rf, res['p'][0], res['p'][1], 
						patch_dim, channel_dim, return_coord=True)
			tmp_fit_data.append( (key, st4, res['p'], st3, rf_norm, mu) )
		else:
			rf_rec = N.zeros((patch_dim,patch_dim,3)) + tfmax
			tmp_fit_data.append( (key, st4, None, st3, rf_norm) )
	
		rf_rec_norm = normalize_color_weighted(tfmax, args.normalize, rf_rec)
		rf_rec_norm = shift_mean_towards_value(rf_rec_norm, tfmean)
		rf_info = N.ones((patch_dim,patch_dim,3)) - .05

		all_tiles[st[0]:st[0]+patch_dim, st[1]:st[1]+patch_dim] = rf_norm
		all_tiles[st2[0]:st2[0]+patch_dim, st2[1]:st2[1]+patch_dim] = rf_rec_norm
		all_tiles[st3[0]:st3[0]+patch_dim, st3[1]:st3[1]+patch_dim] = rf_info
		all_tiles[st4[0]:st4[0]+patch_dim, st4[1]:st4[1]+patch_dim] = rf_info
		try:
			c_color = transpose_color_zero_to_one(loaded_result[key]['c_color'])
			all_tiles[st2[0]-2:st2[0], st2[1]:st2[1]+patch_dim] = c_color
		except Exception, e:
			print e

		curr_index += 1







import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
maxwhite = transpose_color_zero_to_one(N.array([tfmax,tfmax,tfmax], dtype=N.float32))
fig = plt.figure()
fig.set_size_inches(2, 2)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(
	normalize_color(all_tiles),
	# normalize_color_weighted(tfmax, args.normalize, all_tiles),
	interpolation='nearest')
ax.set_axis_bgcolor = 'black'
'''plot extra data onto all_tiles'''
cscale = 3.
sscale = 4.

def rot(angle, xy, p_rot_over):
	s = N.sin(angle)
	c = N.cos(angle)
	xy = (xy[0] - p_rot_over[1], xy[1] - p_rot_over[0])
	rx = xy[0]*c - xy[1]*s
	ry = xy[0]*s + xy[1]*c	
	return (rx+p_rot_over[1], ry+p_rot_over[0])

def plot_curve_of_rf_cut(xy, mrf, mu, theta, scale=-4.5, vertical=False):
	def rot_values(channel, rot_pairs):
		return N.array([mrf.T[channel][c[0],c[1]] for c in rot_pairs])
	from scipy.interpolate import interp1d
	def interp(values):
		return interp1d(X, values, kind='cubic')


	X = N.arange(0, patch_dim, 1)
	X_smooth = N.linspace(0, patch_dim-1, 140)		

	rng = xrange(0, patch_dim)
	if vertical:
		coords = [(mu[1], i) for i in rng]
	else:
		coords = [(i, mu[0]) for i in rng]
	r_coords = [N.round(rot(theta, (c[0],c[1]), mu)) for c in coords]

	def rot_values2(channel, rot_pairs):
		def ck_bnds(val, maxval):
			if val < 0 or val >= maxval:	return False
			else:							return True
		mean = N.mean(mrf)
		r_values = []
		for c in rot_pairs:
			if ck_bnds(c[0], patch_dim) and ck_bnds(c[1], patch_dim):
				r_values.append( mrf.T[channel][c[0],c[1]] ) 
			else:
				r_values.append( mean )
		return N.array(r_values)#[::-1]

	if len( rot_values2(0, r_coords) ) == 0:
		return

	'''shift curves towards center'''
	from scipy.ndimage.interpolation import shift
	b = rot_values2(2, r_coords)
	g = rot_values2(1, r_coords)
	r = rot_values2(0, r_coords)
	vsh = N.max( [N.argmax(N.abs(r)), N.argmax(N.abs(g)), N.argmax(N.abs(b))] )
	dbvsh = patch_dim/2 - vsh

	b = shift(b, dbvsh, cval=N.mean(mrf))
	g = shift(g, dbvsh, cval=N.mean(mrf))
	r = shift(r, dbvsh, cval=N.mean(mrf))

	ax.plot(xy[1]+X_smooth, xy[0]+interp(b)(X_smooth)*scale, 
		color=(.3,.3,.7), linewidth=linewidth*1.2, alpha=1.)
	ax.plot(xy[1]+X_smooth, xy[0]+interp(g)(X_smooth)*scale, 
		color=(.3,.7,.3), linewidth=linewidth*1.2, alpha=1.)
	ax.plot(xy[1]+X_smooth, xy[0]+interp(r)(X_smooth)*scale, 
		color=(.7,.3,.3), linewidth=linewidth*1.2, alpha=1.)


for fdata in tmp_fit_data:

	st3x = float(fdata[1][1])
	st3y = float(fdata[1][0])

	st4x = float(fdata[3][1])
	st4y = float(fdata[3][0]+patch_dim/2)

	if not fdata[2] is None:
		ellip_par = fdata[2]
		cmux = ellip_par[0]+st3y
		cmuy = ellip_par[1]+st3x
		rwc = ellip_par[2]*cscale
		rhc = ellip_par[3]*cscale
		ctheta = 180/N.pi*ellip_par[4]
		smux = ellip_par[8]+st3y
		smuy = ellip_par[9]+st3x
		rws = ellip_par[10]*sscale
		rhs = ellip_par[11]*sscale
		stheta = 180/N.pi*ellip_par[12]
		ellip_c = [cmuy, cmux, rwc, rhc, ctheta, .01]
		ellip_s = [smuy, smux, rws, rhs, stheta, .01]
		linewidth = .1
		add_ellipsoid(ax, ellip_s, 
			edge_color='k', face_color='None', linewidth=linewidth)
		add_ellipsoid(ax, ellip_c, 
			edge_color='k', face_color='None', linewidth=linewidth)

		'''plot smooth curve'''
		rfm = fdata[4]
		rf_mu = fdata[5]

		t_angle = ellip_par[4]#12

		plot_curve_of_rf_cut(
			(st4y-patch_dim*.225, st4x), rfm, 
			(N.round(rf_mu[0]), N.round(rf_mu[1])), 
			t_angle, vertical=False)

		plot_curve_of_rf_cut(
			(st4y+patch_dim*.2, st4x), rfm, 
			(N.round(rf_mu[0]), N.round(rf_mu[1])), 
			t_angle, vertical=True)

		'''plot line for indicating theta'''
		def make_line(angle, xmod=0, ymod=0):
			pc = (.64, .55)
			ps = (pc[0], pc[1]-patch_dim/16)
			pe = (pc[0], pc[1]+patch_dim/16)
			rps = rot(angle, ps, pc)
			rpe = rot(angle, pe, pc)
			ax.add_line(Line2D((xmod+rps[0], xmod+rpe[0]),
				(ymod+rps[1], ymod+rpe[1]), linewidth=.2, c='k')
			)

		make_line(t_angle-N.pi/2, 
			xmod=st3x-patch_dim*.25, 
			ymod=st4y-patch_dim*.45)
		make_line(t_angle, 
			xmod=st3x-patch_dim*.25, 
			ymod=st4y-patch_dim*.05)


	ax.annotate(xy=(st4x, st4y-patch_dim*1.56), s=str(fdata[0]), 
		fontsize=2.3, color='k')





plt.savefig(os.path.join(working_dir,
	'tmp_crop_map.png'), dpi=2048, facecolor='gray')
plt.close(fig)

