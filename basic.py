# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Math
import random
random.seed(6969)

print(tf.__version__)

#LOAD DATA HERE


image_arr = np.load('powerlines.npy')
image_arr = np.reshape(image_arr,(4000, 128, 128, 1))

classification = np.load('powerline_classification.npy')
classification = np.reshape(classification,(4000,1))

# Must investigate how the image arrays are sent in
train_images = np.zeros((1000,128,128,1))
train_labels = np.zeros((1000,1))

for i in range(1000):
	rand_ind = random.randint(0,4000)
	train_images[i] = image_arr[rand_ind]
	train_labels[i] = classification[rand_ind]

test_images = np.zeros((100,128,128,1))
test_labels = np.zeros((100,1))

for i in range(100):
	rand_ind = random.randint(0,4000)
	test_images[i] = image_arr[rand_ind]
	test_labels[i] = classification[rand_ind]

print(classification.shape)
print(image_arr.shape)

# train_images = image_arr[1500:2500]
# train_labels = classification[1500:2500]


# test_images = image_arr[3500:]
# test_labels = classification[3500:]

print(test_labels.shape)
print(test_images.shape)


model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(128,128,1)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(input_shape=(128, 128)),
    keras.layers.Dense(1, activation=tf.nn.softmax)])


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='hinge',
              metrics=['accuracy'])



model.fit(train_images, train_labels, epochs=3)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

img = test_images[0]

img = (np.expand_dims(img,0))

predictions = model.predict(test_images)

print(predictions)
print("-------------")
print(test_labels)




