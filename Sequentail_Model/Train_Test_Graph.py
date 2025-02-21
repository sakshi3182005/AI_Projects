#Import Libraries
import tensorflow as tf
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
import datetime
import os

#Data Aquisition
samples = 100000
X = np.random.rand(samples,3)
y = (X[:,0]+X[:,1]+X[:,2]>1).astype(int)

#Split train and test
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

#Build model
model = tf.keras.Sequential()
model.add(layers.Input(shape=(3,)))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(3,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#Compile model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

#Model Summary
model.summary()

#Create log directory
log_dir = os.path.join('logs','fit',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir,exist_ok = True)

#Create tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,
                                                      histogram_freq=2)

#Train the model
history = model.fit(X_train , y_train , epochs = 10 , batch_size=66,
                    validation_split=0.2,callbacks = [tensorboard_callback])

#Evaluate the model
test_loss , test_accuracy = model.evaluate(X_test , y_test)

#Print Analysis
print("Training Analysis")
print(f'Test Loss : {test_loss :.4f} \n Test Accuracy : {test_accuracy*100 :.2f}%')

#Load TensorBoard extension
%load_ext tensorboard

#Start TensorBoard
%tensorboard --logdir $log_dir --port 8754
