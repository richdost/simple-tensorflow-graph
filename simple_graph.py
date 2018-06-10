
# Author: Richard Dost
# Homework #2 in machine learning class

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

g = tf.Graph()
with g.as_default():
    
    with tf.name_scope('input'):
        input_placeholder = tf.placeholder(tf.float32, shape=None, name='input_placeholder')  #1
    
    with tf.name_scope('middle'):
        b_prod = tf.reduce_prod(input_placeholder, name='b_prod')
        c_mean = tf.reduce_mean(input_placeholder, name='c_mean')
        d_sum = tf.reduce_sum(input_placeholder, name='d_sum')
        e_add = tf.add(c_mean, b_prod, name='d_add')
    
    with tf.name_scope('final'):
        f_multiply = tf.multiply(e_add, d_sum, name='f_multiply')

# run the graph
data = np.random.normal(loc=1.0, scale=2.0, size=100)
session = tf.Session(graph=g)
result = session.run(f_multiply, feed_dict={input_placeholder: data})

# Writes graph to file for use with tensorboard. 
# Start tensorboard with 
#    >cd <wherever you ran this>
#    >tensorboard --logdir ./hw2_graph/
writer = tf.summary.FileWriter('./hw2_graph', graph = g)

# close things we no longer need
writer.close()
session.close()

# Plot the histogram of the input data
plt.hist(data, bins=10, facecolor='g', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Input Data')
plt.text(0.8, 2.0, 'Mean=1, StdDev=2')
plt.axis('tight')
plt.grid(True)
plt.show()

# output results
print('---')
print('result:',result)

# verify with simple numpy math
sanity_prod = np.prod(data)
sanity_sum = np.sum(data)
sanity_mean = np.mean(data)
sanity_add = sanity_mean + sanity_prod
sanity_multiply = sanity_add * sanity_sum
sanity_tolerance = abs(result * 0.01)

if math.isclose(sanity_multiply, result, abs_tol = sanity_tolerance):
    print('result verified')
else:
    print('Error: result does not match sanity check!')
    print('prod=',sanity_prod)
    print('sum=',sanity_sum)
    print('mean=',sanity_mean)
    print('add=',sanity_add)
    print('multiply=',sanity_multiply)

print('---')
print('data:',data)

