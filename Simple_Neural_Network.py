#This is a python program to build a simple neural network using tensorflow library
#Install tensorflow if not present

import tensorflow as tf
import numpy as np

#Defining the inputs , weights and bias
inputs = tf.constant([0.456 , 0.2345 , 0.7865], dtype = tf.float32)
weights = tf.constant([0.5 , -0.1 , 0.2], dtype = tf.float32)
bias = tf.constant([0.1],dtype=tf.float32)

#Calculating the weighted sum and output
weighted_sum = tf.reduce_sum(inputs*weights)+bias
output = tf.sigmoid(weighted_sum)

#Displaying the weighted sum and output
print("Weighted Sum is : ", np.round(weighted_sum.numpy(),4))
print("Output is : ", np.round(output.numpy(),4))


