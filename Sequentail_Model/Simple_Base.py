#Importing libraries
import tensorflow as tf
from keras import layers

#Build Model
model = tf.keras.Sequential()
model.add(layers.Input(shape = (2,)))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(3,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#Compile Model
model.compile(
  optimizer = 'adam',
  loss = 'binary_crossentrophy',
  metrics = ['accuracy']
)

#Model Summary
