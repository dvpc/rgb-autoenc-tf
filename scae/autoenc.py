'''
A Shallow Constrained RGB Autoencoder
The model
'''
import tensorflow as tf
import numpy as np
import math


def shallow_constrained_autoencoder(
	dimensions=[768, 512], 
	lr=0.001, 
	k=4e-04, 
	p=0.5
	):

	'''placeholder'''
	x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
	W = tf.placeholder(tf.float32, [dimensions[1], dimensions[0]], name='W')
	
	'''init weights'''
	n_input = int(W.get_shape()[0])
	n_output = int(W.get_shape()[1])
	W = tf.Variable(tf.random_uniform([n_output, n_input], 
		-1.0 / math.sqrt(n_input),
		1.0 / math.sqrt(n_input)), name='W')

	'''reconstruct input'''
	y = tf.nn.relu(tf.matmul(x, W))
	z = tf.matmul(y, tf.transpose(W))

	'''constrain weights of W'''
	rows, indices = [], []
	for rowidx in xrange(n_output):
		rf = tf.gather(W, rowidx)
		rows.append( lr * k * tf.sign(rf) * tf.pow(tf.abs(rf),p) )
		indices.append(rowidx)
	W = tf.scatter_sub(W, indices, rows)

	'''compute cost'''
	cost = tf.reduce_sum(tf.square(z-x))


	return {'x':x, 'y':y, 'z':z, 'cost':cost, 'W':W}

