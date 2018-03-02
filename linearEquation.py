# Disable compatability warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import the tensorflow library
import tensorflow as tf

# Create a new tf session
sess = tf.Session()

# Store our mx + c equation variables
m = tf.Variable([.3], tf.float32)
c = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

# Store the placeholder for our y value
y = tf.placeholder(tf.float32)

# Create a model to represent a linear equation
linearModel = m * x + c

# Store the change in y
squared_deltas = tf.square(linearModel - y)

# Store the difference between given y and the recorded y
loss = tf.reduce_sum(squared_deltas)

# Initialize the variables
init = tf.global_variables_initializer()
sess.run(init) 

# Check the loss
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
