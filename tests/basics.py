
import tensorflow as tf
import matplotlib.pyplot as plt 

# a tenser
n_Values = 32
x = tf.linspace(-3.0, 3.0, n_Values)

# session
sess = tf.Session()
result = sess.run(x)

print result

sess.close()



