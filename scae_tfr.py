'''
Convert image patches to Standard Tensorflow Format.
TFR ecord

https://www.tensorflow.org/versions/r0.8/api_docs/python/python_io.html#tfrecords-format-details
https://www.tensorflow.org/versions/r0.8/how_tos/reading_data/index.html

'''
import numpy as N
import argparse, os
parser = argparse.ArgumentParser(prog='scae.tfr')
parser.add_argument('-path', '--path_to_data', type=str,
	help='Path to training data directory')
parser.add_argument('-bsize', '--batch_size', type=int, default=25,
	help='Num of training patches in batch')
parser.add_argument('-epoch', '--epochs_num', type=int, default=999,
	help='Num of training epochs')
parser.add_argument('-rthread', '--read_threads_num', type=int, default=2,
	help='Num of threads reading training data concurrently')
parser.add_argument('-subm', '--subtract_mean', action='store_true',
	help='Subtract mean of patch')
parser.add_argument('-vis', '--visible', type=int,
	help='Input patch size;-> num inputs  = vis^2 * 3', default=16)

args = parser.parse_args()

import tensorflow as tf

file_names = tf.train.match_filenames_once(args.path_to_data+"/*.png")

filename_queue = tf.train.string_input_producer(
		file_names, 
		num_epochs=args.epochs_num, 
		shuffle=True, 
		seed=None)
from scae import read_a_file
theimg = read_a_file(filename_queue,
	subtract_mean=args.subtract_mean, 
	normalize=True, 
	patch_dim=(args.visible, args.visible))


# from scae import input_pipeline_multiple
# batch_pipeline = input_pipeline_multiple(
# 	file_names, 
# 	batch_size=args.batch_size, 
# 	num_epochs=args.epochs_num, 
# 	read_threads=args.read_threads_num,
# 	subtract_mean=args.subtract_mean,
# 	normalize=True,
# 	patch_dim=(args.visible, args.visible))



sess = tf.Session()
init_op = tf.group(
	tf.initialize_all_variables(),
	tf.initialize_local_variables())
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

filename = './tfrecords/trainig_data.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

epochs = 0

import time
last_time = time.time()

try:
	while not coord.should_stop():
		image_tensor = sess.run(theimg)
		example = tf.train.Example(features=tf.train.Features(feature={
			# 'epoch': _int64_feature(epochs),
			'x': _float_feature(image_tensor.tolist()),
			# 'xb': _bytes_feature(image_tensor.tostring())
			}))
		writer.write(example.SerializeToString())

		# batch = sess.run(batch_pipeline)
		# for patch in batch:	
		# 	for subpatch in patch:
		# 		print patch.shape
		# 		example = ...
		# 		writer.writer(example.SerializeToString())

		print 'epochs:', epochs, 'dt:', time.time() - last_time
		epochs += 1

except tf.errors.OutOfRangeError:
	print('Done writing -- epoch limit reached')
finally:
	coord.request_stop()

writer.close()

coord.request_stop()
coord.join(threads)


