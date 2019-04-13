
import numpy as N
import os

from scae import normalize_color, normalize_color2
from scae import transpose_color_zero_to_one
from scae import transpose_color





def normalize_color_weighted(
	tfmax,
	normalize_factor,
	color_value,
	):
	tfmaxa = N.array([tfmax,tfmax,tfmax], dtype=N.float32)*normalize_factor
	cv_shape = color_value.shape	
	if len(cv_shape) == 1:
		tfmax_with_cvalue = N.array([
			tfmaxa, transpose_color_zero_to_one(color_value) ])
		return normalize_color(tfmax_with_cvalue)[1]
	else:
		cval_ext = N.insert(
			transpose_color_zero_to_one(color_value), 
			0, tfmaxa, axis=0)
		return normalize_color(cval_ext)[1:]







import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def write_rf_fit_map(
	list_rf_idx, 
	fit_results,
	patch_dim,
	channel_dim,
	tfvar,
	filename
	):
	from fit import reconstruct_ext2
	'''tmp storing fit params and tile offset
	for later plotting when building the matplotlib fig''' 
	tmp_fit_data = []

	frame = 1
	num_tiles = len(list_rf_idx)
	num_tiles_extent = int(num_tiles**.5) + 1

	tile_extent = int(patch_dim) * 2
	all_tile_extent = frame + tile_extent * (num_tiles_extent+frame)
	all_tiles = N.zeros((all_tile_extent,all_tile_extent,3)) + N.max(tfvar)

	for row in xrange(0,num_tiles_extent):
		for col in xrange(0,num_tiles_extent):
			
			curr_tile = row*num_tiles_extent + col
			if curr_tile >= len(fit_results):
				continue

			curr_result = fit_results[curr_tile]
			rf_idx = curr_result[0]
			vec_rf = tfvar[rf_idx]

			rf = prepare_rf_tile(vec_rf, patch_dim)

			if curr_result[1] == []:
				rf_rec = N.zeros((patch_dim,patch_dim,3)) + N.max(tfvar)
				rf_err = N.zeros((patch_dim,patch_dim,3)) + N.max(tfvar)
				rf_info = N.zeros((patch_dim,patch_dim,3)) + N.max(tfvar)
				rf_has_fit_data = False
			else:
				rec_p = curr_result[1]
				vec_rf_rec = reconstruct_ext2(rec_p, channel_dim, patch_dim, vec_rf.shape)				
				rf_rec = prepare_rf_tile(vec_rf_rec, patch_dim)
				rf_err = prepare_rf_tile(vec_rf_rec-vec_rf, patch_dim)
				rf_info = prepare_rf_tile(N.ones(vec_rf.shape)*-.1, 
					patch_dim, DEBUG_CORNER_COLOR=transpose_color_zero_to_one(curr_result[2]) )
				rf_has_fit_data = True

			'''starting points for the 4 sub-tiles'''
			st = frame+row*(tile_extent+frame), frame+col*(tile_extent+frame)
			st2 = st[0], 			st[1]+patch_dim
			st3 = st[0]+patch_dim, 	st[1]
			st4 = st[0]+patch_dim, 	st[1]+patch_dim

			all_tiles[st[0]:st[0]+patch_dim, st[1]:st[1]+patch_dim] = rf
			all_tiles[st2[0]:st2[0]+patch_dim, st2[1]:st2[1]+patch_dim] = rf_rec
			all_tiles[st3[0]:st3[0]+patch_dim, st3[1]:st3[1]+patch_dim] = rf_err
			all_tiles[st4[0]:st4[0]+patch_dim, st4[1]:st4[1]+patch_dim] = rf_info

			if rf_has_fit_data:
				tmp_fit_data.append( (rf_idx, st3, rec_p) )
	
	fig = plt.figure()
	fig.set_size_inches(2, 2)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(
		normalize_color(all_tiles),
		interpolation='nearest')
	ax.set_axis_bgcolor = 'black'
	'''plot fit data to figure'''
	cscale = 3.#3.0
	sscale = 4.#4.75
	for fdata in tmp_fit_data:
		ellip_par = fdata[2]
		cmux = ellip_par[0]+fdata[1][0]
		cmuy = ellip_par[1]+fdata[1][1]
		rwc = ellip_par[2]*cscale
		rhc = ellip_par[3]*cscale
		ctheta = ellip_par[4]
		smux = ellip_par[8]+fdata[1][0]
		smuy = ellip_par[9]+fdata[1][1]
		rws = ellip_par[10]*sscale
		rhs = ellip_par[11]*sscale
		stheta = ellip_par[12]
		ellip_c = [cmuy, cmux, rwc, rhc, 180/N.pi*ctheta, .01]
		ellip_s = [smuy, smux, rws, rhs, 180/N.pi*stheta, .01]
		linewidth = .1 if num_tiles < 100 else .03
		add_ellipsoid(ax, ellip_s, 
			edge_color='w', face_color='None', linewidth=linewidth)
		add_ellipsoid(ax, ellip_c, 
			edge_color='w', face_color='None', linewidth=linewidth)

		tx = float(fdata[1][0]+tile_extent/4) / all_tile_extent
		ty = float(fdata[1][1]+tile_extent/1.7) / all_tile_extent
		ax.text(ty, 1-tx, str(fdata[0]), transform=ax.transAxes, 
			fontsize=1.2, color=(.9,.9,.9))

	plt.savefig(filename, dpi=2048, facecolor='black')
	plt.close(fig)


def prepare_rf_tile(
	a,
	patch_dim,
	DEBUG_CORNER_COLOR=None
	):
	a_mrgb = convert_rfvector_to_rgbmatrix(a, patch_dim)
	a_mrgb = transpose_color_zero_to_one(a_mrgb)
	if DEBUG_CORNER_COLOR is None:
		pass
	else:
		a_mrgb[-2:][0:2] = DEBUG_CORNER_COLOR
	return a_mrgb






# debug
def write_rf(
	a,
	patch_dim,
	filename=None,
	ellip_par=None,
	DEBUG_CORNER_COLOR=None,
	):
	a_mrgb = convert_rfvector_to_rgbmatrix(a, patch_dim)
	a_mrgb = transpose_color_zero_to_one(a_mrgb)

	if DEBUG_CORNER_COLOR is None:
		pass
	else:
		a_mrgb[0:1][0:1] = DEBUG_CORNER_COLOR

	fig = plt.figure()
	fig.set_size_inches(2, 2)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(a_mrgb, interpolation='nearest')
	ax.set_axis_bgcolor = 'black'
	
	if not ellip_par is None:
		
		cscale = 3.0
		sscale = 4.75

		cmux = ellip_par[0]; 
		cmuy = ellip_par[1];
		rwc = ellip_par[2]*cscale; 	
		rhc = ellip_par[3]*cscale; 
		ctheta = ellip_par[4]
		smux = ellip_par[8]; 
		smuy = ellip_par[9]; 
		rws = ellip_par[10]*sscale; 	
		rhs = ellip_par[11]*sscale; 
		stheta = ellip_par[12]
		ellip_c = [cmuy, cmux, rwc, rhc, 180/N.pi*ctheta, .01]
		ellip_s = [smuy, smux, rws, rhs, 180/N.pi*stheta, .01]

		rf_c_val_t = transpose_color(N.array([ellip_par[5],ellip_par[6],ellip_par[7]]))
		rf_c_val_norm = normalize_color2(rf_c_val_t)
		rf_s_val_t = transpose_color(N.array([ellip_par[13],ellip_par[14],ellip_par[15]])*-1)
		rf_s_val_norm = normalize_color2(rf_s_val_t)

		add_ellipsoid(ax, ellip_s, edge_color=rf_s_val_norm, face_color='None')
		add_ellipsoid(ax, ellip_c, edge_color=rf_c_val_norm, face_color='None')

	plt.savefig(filename, dpi=1024, facecolor='black')
	plt.close(fig)


from matplotlib.patches import Ellipse
def add_ellipsoid(
	ax, 
	par, 
	alpha=1.0, 
	edge_color=[1.0,.0,1.0], 
	face_color='none', 
	linewidth=.1,
	plain=False):
	e = Ellipse(xy=(par[0], par[1]), 
		width=par[3], 
		height=par[2], 
		angle=par[4], 
		linewidth=linewidth)
	ax.add_artist(e)
	if plain:	e.set_clip_on(False)
	else: 		e.set_clip_box(ax.bbox)
	e.set_edgecolor(edge_color)
	e.set_facecolor(face_color)
	e.set_alpha(alpha)




###### RF vector helper f

def convert_rfvector_to_rgbmatrix(
	a,
	patch_dim,
	swap=True,
	flip=False,
	):
	r, g, b = decompose_vector(a)
	org = compose_vector(r, g, b)
	org = org.reshape(patch_dim,patch_dim,3)
	if swap: org = N.swapaxes(org, 0, 1)
	if flip: org = N.flipud(org)
	return org

def decompose_vector(
	a,
	):
	channel_magn = a.shape[0]/3
	r = a[0:channel_magn:1]
	g = a[channel_magn:2*channel_magn:1]
	b = a[2*channel_magn::1]
	return r, g, b

def compose_vector(
	r,
	g,
	b=None,
	):
	channel_magn = r.shape[0]
	if b.all() != None:
		rf = N.zeros((channel_magn, 3))
		for c in xrange(0, channel_magn):
			rf[c] = (r[c], g[c], b[c])		
	return N.copy(rf)

def arg_absmax_rfvector(
	rf,
	patch_dim,
	channel_dim
	):
	'''indices of abs max value of each rf vector component'''
	def arg_vec2mat(i):	
		return [i % patch_dim, i / patch_dim]
	a = N.argmax(N.abs(rf[0:channel_dim]))
	b = N.argmax(N.abs(rf[channel_dim:2*channel_dim]))
	c = N.argmax(N.abs(rf[2*channel_dim:3*channel_dim]))
	return N.array([arg_vec2mat(a), arg_vec2mat(b), arg_vec2mat(c)])

def value_of_rfvector_at(
	rf, 
	x, y, 
	patch_dim, 
	channel_dim
	):
	'''return value of rf vector by spatial x,y (matrix-) coordinates.'''
	x,y = N.round(x), N.round(y)
	if x < 0: x = 0
	if x > patch_dim-1: x = patch_dim-1
	if y < 0: y = 0
	if y > patch_dim-1: y = patch_dim-1
	idx = int(x)+int(y)*patch_dim
	return N.array([rf[idx], rf[idx+channel_dim], rf[idx+2*channel_dim]])

# def wrapper_value_of_rfvector_at(
# 	rf, 
# 	x, y, 
# 	patch_dim, 
# 	channel_dim
# 	):
# 	adj_values = []
# 	'''retrieve all adjacent neighbors 3x3
# 		0 1 2 
# 		3 4 5
# 		6 7 8'''
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x-1, y-1, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x, y-1, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x+1, y-1, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x-1, y, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x, y, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x+1, y, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x-1, y+1, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x, y+1, patch_dim, channel_dim))
# 	adj_values.append(
# 		value_of_rfvector_at(rf, x+1, y+1, patch_dim, channel_dim))

# 	largest = None
# 	largest_abs_max = 0
# 	for val in adj_values:
# 		val_abs_max = N.max(N.abs(val))
# 		if largest_abs_max < val_abs_max:
# 			largest_abs_max = val_abs_max
# 			largest = val

# 	return largest

def wrapper_value_of_rfvector_at(
	rf, 
	x, y, 
	patch_dim, 
	channel_dim,
	return_coord=False
	):
	'''retrieve all adjacent neighbors 3x3
		0 1 2 
		3 4 5
		6 7 8'''
	xy_perm = [
		(x-1,y-1), (x,  y-1), (x+1,y-1),
		(x-1,y),   (x,  y),   (x+1,y),
		(x-1,y+1), (x,  y+1), (x+1,y+1)]
	largest = None
	largest_coord = None
	largest_abs_max = 0
	for xy in xy_perm:
		val = value_of_rfvector_at(rf, xy[0], xy[1], patch_dim, channel_dim)
		val_abs_max = N.max(N.abs(val))
		if largest_abs_max < val_abs_max:
			largest_abs_max = val_abs_max
			largest = val
			largest_coord = xy
	if return_coord:
		return largest, largest_coord
	else:
		return largest


def is_close_to_zero(
	a,
	atol=1e-08,
	rtol=1e-05,
	verbose=False
	):
	allclose = N.allclose(a, N.zeros(a.shape), atol=atol)
	if verbose and allclose: 
		print '****Note: all RF weights are close to zero. atol =', atol
	return allclose


###### pickle helper f
import cPickle
def serialize_results(
	filepath,
	res_dict
	):
	w_file = open(filepath, 'wb')
	cPickle.dump(res_dict, w_file, -1)
	w_file.close()
	print 'pickled to', filepath

def deserialize_results(
	filepath
	):
	r_file = open(filepath)
	res_dict = cPickle.load(r_file)
	r_file.close()
	return res_dict

