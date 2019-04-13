'''
Read Tensor Flow record...

'''
import tensorflow as tf 

filename = './trainig_data.tfrecords'


def read_TFrecord(fname_queue, dim):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(fname_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'epoch': tf.FixedLenFeature([], tf.int64),
			'x': tf.FixedLenFeature([dim], tf.float32),
			# 'xb' : tf.FixedLenFeature([], tf.string)
		})	
	# epoch = tf.cast(features['epoch'], tf.int64)
	vec = tf.cast(features['x'], tf.float32)
	# return epoch, vec
	return vec


'''init model'''
from scae.autoenc import shallow_constrained_autoencoder
ae = shallow_constrained_autoencoder(
	dimensions=[768, 400],
	lr=0.0007, k=2.0, p=.5)
optimizer = tf.train.GradientDescentOptimizer(
	learning_rate=0.0007).minimize(ae['cost'])


import numpy as N

def get_all_rec(filename):
	with tf.Session() as sess:


		fname_queue = tf.train.string_input_producer(
			[filename], num_epochs=1)
		ep = read_TFrecord(fname_queue, dim=768)

		init_op = tf.group(
			tf.initialize_all_variables(),
			tf.initialize_local_variables())
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			while True:
				# import pdb; pdb.set_trace()
				example = sess.run(ep)
				# print example
				sess.run(optimizer, feed_dict={ae['x']: [example] })
				outw0 = sess.run(ae['W'])
				print N.max(outw0), N.min(outw0)

		except tf.errors.OutOfRangeError, e:
			coord.request_stop(e)
		finally:
			coord.request_stop()
		coord.join(threads)

get_all_rec(filename)







exit()

sess = tf.Session()
init_op = tf.group(
	tf.initialize_all_variables(),
	tf.initialize_local_variables())
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


try:
	while not coord.should_stop():

		fname_queue = tf.train.string_input_producer([filename], num_epochs=10)
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(fname_queue)
		features = tf.parse_single_example(
			serialized_example,
			features={
				'epoch': tf.FixedLenFeature([], tf.int64),
				'x': tf.FixedLenFeature([], tf.float32),
				'xb' : tf.FixedLenFeature([], tf.string)
			})	
		epoch = tf.cast(features['epoch'], tf.int64)
		print epoch
		
		import pdb; pdb.set_trace()



except tf.errors.OutOfRangeError:
	print('Done writing -- epoch limit reached')
finally:
	coord.request_stop()



coord.request_stop()
coord.join(threads)









