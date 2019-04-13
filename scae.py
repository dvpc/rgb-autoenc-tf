'''
A Shallow Constrained RGB Autoencoder

'''
import numpy as N
import argparse, os
parser = argparse.ArgumentParser(prog='scae.train')
parser.add_argument('-ckpt', '--checkpoint', default='None',
	help='Path to saved model checkpoint')
parser.add_argument('-epckpt', '--epoch_checkpoint', type=int)
parser.add_argument('-path', '--path_to_data', type=str,
	help='Path to training data directory')
parser.add_argument('-odir', '--output_dir', type=str,
	help='Path to output directory')
parser.add_argument('-bsize', '--batch_size', type=int,
	help='Num of training patches in batch')
parser.add_argument('-epoch', '--epochs_num', type=int,
	help='Num of training epochs')
parser.add_argument('-rthread', '--read_threads_num', type=int,
	help='Num of threads reading training data concurrently')
parser.add_argument('-wn', '--write_n', type=int,
	help='Write model snapshot every n epochs')
parser.add_argument('-subm', '--subtract_mean',
	help='Subtract mean of patch')
parser.add_argument('-vis', '--visible', type=int,
	help='Input patch size;-> num inputs  = vis^2 * 3')
parser.add_argument('-hid', '--hidden', type=int,
	help='Output patch size;-> num outputs = hid^2')
parser.add_argument('-lr', '--learning_rate', type=float,
	help='Model Parameter lr: Learning rate')
parser.add_argument('-k', type=float,
	help='Model Parameter k; Strength of weight constraint')
parser.add_argument('-p', type=float,
	help='Model Parameter p; L_p+1_Norm; Shape of weight constraint')

'''...parse command line to override possible later defaults'''
override_args = parser.parse_args()
'''load default values if any...'''
if os.path.exists('./scae_default.cfg'):
	import argparse_config
	argparse_config.read_config_file(parser, './scae_default.cfg')
	args = parser.parse_args([])
	'''override parsed defaults with args from cl'''
	for arg in override_args.__dict__:
		if override_args.__dict__[arg] != None:
			args.__dict__[arg] = override_args.__dict__[arg]
else:
	args = override_args
'''also check args if any param is None. if None complain and exit'''
try:
	for arg in args.__dict__:
		assert args.__dict__[arg] != None, \
			'Arg: ' + str(arg) + ' is None!'
except Exception, e:
	print e; exit()
'''check if model checkpoint is available'''
load_tensorflow_model = False
if args.checkpoint != None and args.checkpoint != 'None':
	import re
	if re.search('\.ckpt', args.checkpoint) != None:
		load_tensorflow_model = True
	else:
		print 'checkpoint file has no .ckpt suffix!'; exit()

'''bad bad... hand sanitizing oO'''
if args.subtract_mean == 'True':
	args.subtract_mean = True
else:
	args.subtract_mean = False

'''check if path to data is pointing a tfrecord'''
if args.path_to_data.find('tfrecords') != -1:
	use_input_pipeline = False
else:
	use_input_pipeline = True


print args


import tensorflow as tf
import math
from scae import pad_sequence_num, write_matrix_as_png







if use_input_pipeline:
	'''init training data pipeline'''
	from scae import input_pipeline_multiple
	file_names = tf.train.match_filenames_once(args.path_to_data+"/*.png")
	batch_pipeline = input_pipeline_multiple(
		file_names, 
		batch_size=args.batch_size, 
		num_epochs=args.epochs_num, 
		read_threads=args.read_threads_num,
		subtract_mean=args.subtract_mean,
		normalize=True,
		patch_dim=(args.visible, args.visible))
else:
	def read_TFrecord(fname_queue):
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(fname_queue)
		features = tf.parse_single_example(
			serialized_example,
			features={
				'x': tf.FixedLenFeature([args.visible**2*3], tf.float32)
			})
		vec = tf.cast(features['x'], tf.float32)
		return vec
	file_names = tf.train.match_filenames_once(args.path_to_data+"/*.tfrecords")
	fname_queue = tf.train.string_input_producer(
		#[args.path_to_data], 
		file_names,
		shuffle=True,
		num_epochs=args.epochs_num)
	tfrec = read_TFrecord(fname_queue)


'''init model'''
from scae.autoenc import shallow_constrained_autoencoder
ae = shallow_constrained_autoencoder(
	dimensions=[3*args.visible**2, args.hidden**2],
	lr=args.learning_rate, k=args.k, p=args.p)
optimizer = tf.train.GradientDescentOptimizer(
	learning_rate=args.learning_rate).minimize(ae['cost'])

'''init training session'''
sess = tf.Session()
saver = tf.train.Saver()
init_op = tf.group(
	tf.initialize_all_variables(),
	tf.initialize_local_variables())
sess.run(init_op)
if load_tensorflow_model:
	saver.restore(sess, args.checkpoint)
	print 'restoring checkpoint ', args.checkpoint
else:
	pass
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


epochs = args.epoch_checkpoint
import time
last_t = time.time()
from scae import pad_sequence_num, write_matrix_as_png

try:
	while not coord.should_stop():
		if use_input_pipeline:
			batch = sess.run(batch_pipeline)
			for patch in batch:	
				for subpatch in patch:
					sess.run(optimizer, feed_dict={ae['x']: [subpatch] })
		else:
			subpatch = sess.run(tfrec)
			sess.run(optimizer, feed_dict={ae['x']: [subpatch] })

		if epochs % args.write_n == 0:
			outw0 = sess.run(ae['W'])
			outw0 = N.swapaxes(outw0,0,1)
			
			info_patches = epochs*args.batch_size if use_input_pipeline else epochs
			infostr = \
				' epoch: '  +str(pad_sequence_num(epochs))+\
				' patches: '+str(pad_sequence_num(info_patches))+\
				' W min/max: (' +\
				str(pad_sequence_num(N.round(N.min(outw0),3),7,rev=True))+','+\
				str(pad_sequence_num(N.round(N.max(outw0),3),6,rev=True))+')'			
			
			outy0 = sess.run(ae['y'],feed_dict={ae['x']:[subpatch]})
			outx0 = subpatch
			print infostr + \
				' y min/max: (' +\
				str(N.round(N.min(outy0),3))+','+\
				str(N.round(N.max(outy0),3))+')'+\
				' x min/max: (' +\
				str(N.round(N.min(outx0),3))+','+\
				str(N.round(N.max(outx0),3))+')'
			
			print ('cost',sess.run(ae['cost'],feed_dict={ae['x']:[subpatch]}),
					'time',str(N.round(time.time()-last_t,4))+'s')
			infostr += '\n' +\
					' vis: ' + str(args.visible**2*3) +\
					' hid: ' + str(args.hidden**2) +\
					' submean: ' + str(args.subtract_mean) + \
					' k: ' + str(args.k) + \
					' p: ' + str(args.p)			
			last_t = time.time()
			'''write image file'''
			write_matrix_as_png(
				os.path.join(
					args.output_dir, 
					pad_sequence_num(epochs)+'_obs_W_0_1.png'),
				outw0, args.visible, args.hidden, infostr)
			'''write model checkpoint'''
			save_path = saver.save(sess, 
				os.path.join(args.output_dir,"scae_model.ckpt"))

		epochs += 1

except tf.errors.OutOfRangeError:
	print('Done training -- epoch limit reached')
finally:
	coord.request_stop()

coord.request_stop()
coord.join(threads)







