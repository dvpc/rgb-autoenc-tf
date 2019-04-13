'''
A Shallow Constrained RGB Autoencoder
Some utility methods...
'''
import numpy as N
import tensorflow as tf
'''
Force matplotlib to not use any Xwindows backend.
see: http://stackoverflow.com/questions/29217543/why-does-this-solve-the-no-display-environment-issue-with-matplotlib
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_random_crop(
	image,
	patch_dim=(16,16),
	image_max_extent=(1024,768)#(288,288)#(756,1008)#(384,288)
	):
	border = max(patch_dim[1], patch_dim[0])
	bordermax_x = image_max_extent[1]-border
	bordermax_y = image_max_extent[0]-border
	r = tf.Variable(
		tf.random_uniform([1], border, bordermax_x, dtype=tf.int32))
	c = tf.Variable(
		tf.random_uniform([1], border, bordermax_y, dtype=tf.int32))
	return tf.image.crop_to_bounding_box(
		image, r[0], c[0], patch_dim[1], patch_dim[0])


def read_a_file(
	fname_queue, 
	subtract_mean=False,
	normalize=True,
	patch_dim=(16,16),
	chance_to_sawp_axes=50
	):
	reader = tf.WholeFileReader()
	key, value = reader.read(fname_queue)
	rimg = tf.image.decode_png(value)
	# preprocess
	crop = get_random_crop(rimg, patch_dim)
	# if N.random.randint(0, 100) > 100-chance_to_sawp_axes:
	# 	crop = tf.image.rot90(crop)
	# 	crop = tf.image.flip_left_right(crop)
	crop = tf.to_float(crop)
	if normalize:
		crop = crop / tf.abs(tf.reduce_max(crop)) 
		# crop = crop / 255.0 
	if subtract_mean:
		crop = crop - tf.reduce_mean(crop)
	# create flat tensor
	flat = tf.reshape(crop, [patch_dim[0]*patch_dim[1],3])
	flat = tf.reshape(tf.transpose(flat,[1,0]), [-1])
	return flat


def input_pipeline_multiple(
	fnames, 
	batch_size, 
	num_epochs, 
	read_threads,
	subtract_mean=False,
	normalize=True,
	patch_dim=(16,16)
	):
	fname_queue = tf.train.string_input_producer(
		fnames, 
		num_epochs=num_epochs, 
		shuffle=True, seed=None)
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	patch_list = [read_a_file(fname_queue, 
		subtract_mean=subtract_mean, 
		normalize=normalize, 
		patch_dim=patch_dim) for _ in range(read_threads)]
	batch = tf.train.shuffle_batch_join(
		[patch_list], 
		batch_size=batch_size,
		capacity=capacity,
		min_after_dequeue=min_after_dequeue)
	return batch


# below move!
# to scae.util

def pad_sequence_num(
	num,
	padlen=8,
	rev=False
	):
	strn = str(num)
	n_to_pad = padlen - len(strn)
	if rev:
		return strn+'0'*n_to_pad 
	else:
		return '0'*n_to_pad+strn



# TODO clean up!!!
def normalize_color2(
	a
	):
	amax = N.max(a)
	return a / amax


def transpose_color(
	a
	):
	if type(a) is list:
		a = N.array(a)
	amax = N.max(a)
	amin = N.min(a)
	print '_------'
	print a
	print (a + N.abs(amin))# / amax 
	return (a + N.abs(amin))
	# '''transpose color from [-1, 1] to [0, 1]'''
	# return (a + 1.)/2.


def normalize_color(
	a
	):
	highS = N.max(a)
	lowS = N.min(a)
	shiftS = lowS
	if shiftS > 0.0: lowS = 0.0
	a = (a - shiftS) / (highS - shiftS)
	return N.copy(a)

def transpose_color_zero_to_one(
	a
	):
	if type(a) is list:
		a = N.array(a)
	'''transpose color from [-1, 1] to [0, 1]'''
	return (a + 1.)/2.



def write_matrix_as_png(
	filename,
	outw,
	visible,
	hidden,
	infostr
	):
	'''reshape weight matrix; RGB = (h,w,3)'''
	m = N.zeros((hidden**2, visible**2, 3))
	for row in xrange(outw.shape[0]):
		channel_magn = outw[row].shape[0]/3
		r = outw[row][0:channel_magn:1]
		g = outw[row][channel_magn:2*channel_magn:1]
		b = outw[row][2*channel_magn::1]
		for c in xrange(0, channel_magn):
			m[row, c] = (r[c], g[c], b[c])
	'''write square tile map'''
	frame = 1
	margin = 2*hidden
	out = N.zeros( (
		frame+hidden*(frame+visible)+margin,
		frame+hidden*(frame+visible),
		3) ) + N.max(outw)/2.
	tile_num = 0
	for xx in range(hidden):
		for yy in range(hidden):
			start_h, start_w = \
				frame+xx*(frame+visible),\
				frame+yy*(frame+visible)
			tile = N.array([
				N.reshape(m[tile_num].T[0],(visible, visible)),
				N.reshape(m[tile_num].T[1],(visible, visible)),
				N.reshape(m[tile_num].T[2],(visible, visible))
				]).T
			out[start_h : start_h+visible, 
				start_w : start_w+visible] = tile
			tile_num += 1
	out = normalize_color(out)
	'''create and write matplotlib figure'''
	fig = plt.figure()
	fig.set_size_inches(1, 1)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(out, interpolation='nearest')
	ax.set_axis_bgcolor = 'black'
	ax.text(.0, .02, infostr, transform=ax.transAxes, 
		fontsize=2, color=(.9,.9,.9),
		bbox=dict(boxstyle='square', fc='black', ec='none'))	
	plt.savefig(filename, dpi=1024, facecolor='black')
	plt.close(fig)


