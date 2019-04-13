'''
import tensorflow as tf
sess = tf.Session()
sess.run(init)
a = tf.Variable( [ [[1,0,0],[2,0,0]], [[0,1,0],[0,2,0]], [[0,0,1],[0,0,2]]  ] )
flat = tf.reshape(a,[3*2,3])
init = tf.initialize_all_variables()
sess.run( init )
sess.run( flat )
sess.run( a )
sess.run( tf.transpose(flat,[1,0]) )
sess.run( tf.reshape(  tf.transpose(flat,[1,0]), [-1]  ) )
'''

import tensorflow as tf
import matplotlib.pyplot as plt 
IMG_PATH = '/home/bob/Work/Personal/article-code/dataoldpng/'
file_names = tf.train.match_filenames_once(IMG_PATH+"/*.png")

'''see: http://stackoverflow.com/questions/36838770/how-to-interpret-tensorflow-output'''
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction=0.3


import numpy as N
from rgbautoenc.cv.base.exptiles import exporttiles_rgb


def get_random_crop(
	image,
	patch_h, 
	patch_w
	):
	border = max(patch_h, patch_w)
	bordermax = 288-border # cant get real image dim here...
	r = tf.Variable(
		tf.random_uniform([1], border, bordermax, dtype=tf.int32))
	c = tf.Variable(
		tf.random_uniform([1], border, bordermax, dtype=tf.int32))
	return tf.image.crop_to_bounding_box(image, r[0], c[0], patch_h, patch_w)


def read_a_file(fname_queue):
	reader = tf.WholeFileReader()
	key, value = reader.read(fname_queue)
	rimg = tf.image.decode_png(value)
	# do some preprocessing
	crop = get_random_crop(rimg, 16, 16)
	crop = tf.to_float(crop)
	# normalize
	crop = crop / tf.abs(tf.reduce_max(crop)) 
	# subtract mean
	crop = crop - tf.reduce_mean(crop)
	# create flat tensor
	flat = tf.reshape(crop, [16*16,3])
	flat = tf.reshape(tf.transpose(flat,[1,0]), [-1])
	return flat


def input_pipeline(fnames, batch_size, num_epochs):
	fname_queue = tf.train.string_input_producer(
		fnames, 
		num_epochs=num_epochs, 
		shuffle=True, seed=None)
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	patch = read_a_file(fname_queue)
	batch = tf.train.shuffle_batch(
		[patch], 
		batch_size=batch_size,
		capacity=capacity,
		min_after_dequeue=min_after_dequeue)
	return batch


batch_pipeline = input_pipeline(
	file_names, batch_size=50, num_epochs=100)


learning_rate = 0.001

from rgcautoenc import rgcautoenc 
ae = rgcautoenc(dimensions=[768, 1256], lr=learning_rate, k=1e-04)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])



# Start a new session to show example output.
sess = tf.Session()

# Required !
init_op = tf.group(
	tf.initialize_all_variables(),
	tf.initialize_local_variables())
sess.run(init_op)

# Coordinate the loading of image files.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)



epochs = 0
import time
last_time = time.time()


try:
	while not coord.should_stop():

		# Run training steps or whatever
		batch = sess.run(batch_pipeline)

		for patch in batch:	
			sess.run(optimizer, feed_dict={ae['x']: [patch] })			

		if epochs % 10 == 0:
			outw0 = sess.run(ae['W'], feed_dict={ae['x']: [patch] })
			outw0 = N.swapaxes(outw0, 0, 1)
			print 'ep', epochs, 'dt:', time.time() - last_time
			print(epochs, sess.run(ae['cost'], feed_dict={ae['x']: [patch] }))
			exporttiles_rgb(outw0, 16, 16, './'+str(epochs)+'obs_W_0_1.pnm', 16, 16*3)

			last_time = time.time()
		
		epochs += 1
		



except tf.errors.OutOfRangeError:
	print('Done training -- epoch limit reached')
finally:
	coord.request_stop()

coord.request_stop()
coord.join(threads)

