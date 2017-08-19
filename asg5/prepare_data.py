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
from ptb_reader import *


def prepare_data(data, eos_id, vocab_size, max_length = None):
	main_data = []
	temp_data = []
	for item in data:
		if item==eos_id:
			main_data.append(temp_data)
			temp_data = []
		else:
			temp_data.append(item)
	if len(temp_data)!= 0:
		main_data.append(temp_data)

	data = main_data

	length = 0
	for item in data:
		length = max(length, len(item))
	if max_length is not None:
		length = max(length, max_length)
	for i in range(len(data)):
		data[i] = data[i]+[eos_id]*(length+1-len(data[i]))

	max_length = length

	trainX = np.array(data)
	trainY = []

	for item in trainX:
		y = []
		for i in range(len(item)-1):
			y.append(item[i+1])
		y.append(eos_id)
		trainY.append(y)
	
	trainX = trainX.reshape(tuple(list(trainX.shape)+[1]))
	trainY = np.array(trainY)
	trainY = trainY.reshape(tuple(list(trainY.shape)+[1]))
	trainY = np.ndarray.astype(trainY, dtype = np.int32)
	return (trainX, trainY), max_length
