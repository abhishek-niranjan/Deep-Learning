""" Abhishek Niranjan
	13CS30003
	Programming Assignment 4
"""



import os
import sys
import scipy as sp
import numpy as np
import pandas as pd
import tensorflow as tf
from prepare_data import *
from ptb_reader import *
from keras.models import Sequential,  load_model
from keras.layers import Dense,  LSTM,  Dropout,  Embedding,  Reshape
from keras.optimizers import SGD
from keras.layers.wrappers import TimeDistributed
import keras.backend as kBack
import keras.losses
import keras.metrics
from scipy.stats import entropy


vocab_size = 10000

#################### FETCH DATA USING ptb_reader.py ##########################

def get_test_data(filename, max_length  =  82):
	data, vocab_size, eos_id = ptb_raw_data_file(filename)
	main_data, max_length = prepare_data(data, eos_id, vocab_size, max_length)
	return main_data


def get_data(folder):
	main_data = [0]*3
	train_data, val_data, test_data, vocab_size, eos_id = ptb_raw_data(folder)
	train_data, max_length = prepare_data(train_data, eos_id, vocab_size)
	val_data, max_length = prepare_data(val_data, eos_id, vocab_size, max_length)
	test_data, max_length = prepare_data(test_data, eos_id, vocab_size, max_length)
	return train_data, val_data, test_data, vocab_size


#################### DEFINE LOSS AND PERPLEXITY ############################

def perplexity(y_true,  y_pred):
	return kBack.mean(kBack.pow(kBack.categorical_crossentropy(y_pred,  y_pred), 2),  axis = -1)

def cross_entropy(y_true,  y_pred):
	vocab_size = 10000
	y_true = kBack.one_hot(kBack.batch_flatten(kBack.cast(y_true, 'int32')), vocab_size)
	return kBack.mean(kBack.categorical_crossentropy(y_pred,  y_true),  axis = -1)

def test_perplexity(y_true,  y_pred):
	y_true = np.ndarray.flatten(y_true)
	one_hot = np.zeros((len(y_true), y_pred.shape[-1]))
	one_hot[np.arange(len(y_true)), y_true] = 1
	y_true = np.reshape(one_hot, y_pred.shape)
	return np.mean(np.power(np.sum(np.multiply(y_true, -np.log2(y_pred)), axis = -1), 2))



#################### TRAIN MODEL ###########################################

def trainModel():

	################## FETCH THE DATA #################################
	train_data, val_data, test_data, vocab_size = get_data('./data')
	print train_data[0].shape
	# print train_data[1].shape
	print val_data[0].shape
	# print val_data[1].shape
	# print test_data[1].shape
	print test_data[0].shape
	
	#################  DEFINE ARCHITECTURE ############################
	lstm_size = 200
	max_length = train_data[0].shape[-2]
	word2vec_dim = 200
	

	model = Sequential()
	model.add(Reshape((max_length, ), input_shape = (max_length, 1)))
	model.add(Embedding(vocab_size, word2vec_dim, input_length = max_length))
	model.add(Reshape((max_length, word2vec_dim)))
	model.add(LSTM(lstm_size, return_sequences = True))
	model.add(TimeDistributed(Dense(vocab_size, activation = 'softmax')))
	optimizer = SGD(0.015, decay = 1e-5, momentum = 0.95)
	metrics = [perplexity]
	loss = cross_entropy
	model.compile(optimizer = optimizer, loss = loss, metrics = metrics)


	################## TRAN MODEL ##########################################
	model.fit(train_data[0], train_data[1], epochs = 1, batch_size = 32, validation_data = val_data)
	model.save('lstm_model')






def main():

	########################### TRAIN THE MODEL FIRST TO GET PARAMETERS ########################
	if 'lstm_model' not in os.listdir('.'):
		trainModel()
		exit()

	###################################### TESTING THE MODEL ###################################

	if len(sys.argv)!= 3:
		print 'Invalid Command'               # Type in command: python main.py --test test.txt
		exit()

	if sys.argv[1] == '--test':

		keras.losses.cross_entropy = cross_entropy
		keras.metrics.perplexity = perplexity
		model = load_model('lstm_model')

		test_filename = sys.argv[2]
		test_data = get_test_data(test_filename)
		
		
		
		batch_size = 100
		perplexities = []

		for i in range(0, test_data[0].shape[0], batch_size):
			testX = test_data[0][i:min(i+batch_size, test_data[0].shape[0])]
			y_true = test_data[1][i:min(i+batch_size, test_data[0].shape[0])]
			y_pred = model.predict(testX)
			perplex = test_perplexity(y_true, y_pred)
			# print perplex, len(testX)
			print "Perplexity:", perplex
			perplexities.append(perplex)
		average_perplexity = sum(perplexities)/(len(test_data[0])*len(test_data[0][0]))
		print 'average_perplexity', average_perplexity


main()