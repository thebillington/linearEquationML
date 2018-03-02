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

# Create an optimizer to optimize the values of m and c
optimizer = tf.train.GradientDescentOptimizer(0.01)

# Train the optimizer to minimize the value of loss
train = optimizer.minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()
sess.run(init) 

# Check the loss
#print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# Get values of x and y from user
xValues = []
yValues = []
r = "y"
while r == "y":
	xValues.append(float(input("Enter x value of coordinate: ")))
	yValues.append(float(input("Enter y value of coordinate: ")))
	r = input("Enter another coordinate? (y/n):")

# Run the linear model to minimize loss with specified x values and their corresponding, correct y values. Train on 100 iterations
for i in range(1000):
	sess.run(train, {x: xValues, y: yValues})

# Print the values of m and c
print("For input values of x = {0} and y = {1}, optimized linear equation is: y = {2} x + {3}".format(xValues, yValues, sess.run(m)[0], sess.run(c)[0]))
print("For input values of x = {0} and y = {1}, rounded optimized linear equation is: y = {2:.0f} x + {3:.0f}".format(xValues, yValues, sess.run(m)[0], sess.run(c)[0]))
