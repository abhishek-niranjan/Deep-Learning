'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Abhishek Niranjan		
Roll No.:	13CS30003

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import cPickle as pickle
from keras.optimizers import RMSprop, SGD
import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn import preprocessing


def train(trainX, trainY):
	batch_size = 128
	num_classes = 10
	print(trainX.shape, 'train samples')
	trainX = trainX.reshape((60000,784)).astype(np.float)
	print(trainX.shape, 'train samples')
	trainX /= 255
	# print trainY[0]
	trainY = keras.utils.np_utils.to_categorical(trainY, num_classes)
	print(trainY.shape, 'train labels')
	# print trainY[0]
	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=((784,))))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
	model.fit(trainX, trainY, batch_size=batch_size, verbose=2)
	# weights = model.get_weights()
	# with open("weights.p", "w") as weights_file:
	# 	pickle.dump(weights, weights_file)
	model.save('MLP_Model.h5')



def test(testX, testY):
	batch_size = 128
	num_classes = 10
	epochs = 20
	testX = testX.reshape((10000,784)).astype(np.float)
	testX /= 255
	testY = keras.utils.np_utils.to_categorical(testY, num_classes)
	model = load_model('MLP_Model.h5')
	predLabels = model.predict(testX, batch_size=batch_size, verbose=1)
	# print predLabels
	labels = [np.argmax(item) for item in predLabels]
	# print labels
	return labels
	# score = model.evaluate(testX, testY, verbose=0)
	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])


