''''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Abhishek Niranjan
Roll No.: 13CS30003

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

def convertoOneHot(data) :
	out = np.zeros([len(data),10])
	for i in range(len(data)) :
		out[i][data[i]] = 1
	return np.array(out)  


shape1 = [5,5,1,32]
shape2 = [5,5,32,64]
shape3 = [7*7*64,1024]
shape4 = [1024,10]

with tf.Graph().as_default() :
	x = tf.placeholder(tf.float32, shape=[None, 28,28,1])


	W1 = tf.Variable(tf.random_uniform(shape1,-1/25**0.5,1/25**0.5,tf.float32))
	b1 = tf.Variable(tf.constant(0.1, shape=[shape1[-1]]))
	h1 = max_pool_2x2(tf.nn.relu(conv2d(x, W1) + b1))

	W2 = tf.Variable(tf.random_uniform(shape2,-1/800**0.5,1/800**0.5,tf.float32))
	b2 = tf.Variable(tf.constant(0.1, shape=[shape2[-1]]))
	h2 = max_pool_2x2(tf.nn.relu(conv2d(h1, W2) + b2))

	W_fc1 = tf.Variable(tf.random_uniform(shape3,-1/(7*7*64)**0.5,1/(7*7*64)**0.5,tf.float32))
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[shape3[-1]]))
	h2_flat = tf.reshape(h2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = tf.Variable(tf.random_uniform(shape4,-1/1024**0.5,1/1024**0.5,tf.float32))
	b_fc2 = tf.Variable(tf.constant(0.1, shape=[shape4[-1]]))
	y_out=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()

def train(trainX, trainY):
	'''
	Complete this function.
	'''
	
	trainY = convertoOneHot(trainY)
	batch_size = 50
	epochs = 1
	iterations = int(len(trainX)/batch_size)
	for i in range(epochs) :
		for j in range(iterations) :

			inp = trainX[j*batch_size:(j+1)*batch_size]
			out = trainY[j*batch_size:(j+1)*batch_size]
			
			if j%250 == 0 :
				train_accuracy = sess.run(accuracy,feed_dict={x:inp, y_: out, keep_prob: 1.0})
				print('%d steps reached and training accuracy is %g'%(i*iterations+j,train_accuracy))
			sess.run(train_step,feed_dict={x:inp,y_:out,keep_prob:0.5})   
			if j== 250 :
				break 
		perm = np.random.permutation(len(trainX))
		trainX = trainX[perm]
		trainY = trainY[perm]        

	saver.save(sess,'model_cnn')
	
	
def test(testX):
	'''
	Complete this function.
	This function must read the weight files and
	return the predicted labels.
	The returned object must be a 1-dimensional numpy array of
	length equal to the number of examples. The i-th element
	of the array should contain the label of the i-th test
	example.
	'''
	new_saver = tf.train.import_meta_graph('model_cnn.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	all_vars = tf.get_collection('vars')
	for v in all_vars:
		v_ = sess.run(v)
		print v_
	out = sess.run(y_out,feed_dict={x:testX,keep_prob:1.0})
	out = np.argmax(out,axis=1)
	return out
