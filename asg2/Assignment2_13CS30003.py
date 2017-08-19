""" Abhishek Niranjan
    13CS30003
    Assignment2 """



import os
import scipy as sp
import pandas as pd
import sklearn
import numpy as np
import math

def sigmoid(X,W,b):
    """
    Computes H = sigmoid(X . W + b) corresponding to the hidden unit
    activations of a one-hidden-layer MLP classifier
    Parameters
    ----------
    X : numpy.ndarray
        Batch of examples of shape (batch_size, num_vis)
    W : numpy.ndarray
        Weight matrix of shape (num_vis, num_hid)
    b : numpy.ndarray
        Bias vector of shape (num_hid, )
    """
    preActivation = np.dot(X, W) + b
    return (1.0)/(1.0 + np.exp(-preActivation))

################# TEST SIGMOID #############################

# H = np.random.randint(5, size=[2,4])
# print H
# V = np.random.randint(5, size=[4,10])
# print V
# d = np.random.randint(5, size=10)
# print d
# sigmoid(H,V,d)
    


def softmax(H, V, d):
    """
    Computes Y = softmax(H . V + d) corresponding to the output probabilities
    of a one-hidden-layer MLP classifier
    Parameters
    ----------
    H : numpy.ndarray
        Batch of hidden unit activations of shape (batch_size, num_hid)
    V : numpy.ndarray
        Weight matrix of shape (num_hid, num_classes)
    d : numpy.ndarray
        Bias vector of shape (num_classes, )
    """
    postActivation = np.dot(H,V) + d
    expVector = np.exp(postActivation)
    return expVector/(np.sum(expVector, axis=1)[:,np.newaxis])

######################## TEST SOFTMAX #################################

# a = np.random.rand(10)
# print a
# # a = a[:,np.newaxis]
# # print a
# print softmax(a)



def loss(Y, T):
    """
    Computes the binary cross-entropy loss of an MLP classifier
    Parameters
    ----------*/
    Y : numpy.ndarray
        Batch of output probabilities of shape (batch_size, num_classes)
    T : numpy.ndarray
        Batch of one-hot encoded targets of shape (batch_size, num_classes) / Target Function
    """
    lossFunction = -(T*np.log(Y)).sum(axis=1).mean(axis=0)
    return lossFunction

############### TEST LOSS #######################

# V = np.random.rand(4,10)
# print V
# k = np.random.randint(9, size=4)
# print k
# d = np.zeros((k.size,10), dtype=np.int)
# d[np.arange(k.size), k] = 1
# print d

# loss(V,d)

def forward(X,W,b,V,d):
    """
    Does the forward-prop on an MLP classifier
    Parameters
    ----------
    X : numpy.ndarray
        Batch of examples of shape (batch_size, num_vis)
    W : numpy.ndarray
        Weight matrix of shape (num_vis, num_hid)
    b : numpy.ndarray
        Bias vector of shape (num_hid, )
    V : numpy.ndarray
        Weight matrix of shape (num_hid, num_classes)
    d : numpy.ndarray
        Bias vector of shape (num_classes, )
    Returns
    -------
    H : numpy.ndarray
        Batch of activations in hidden layer shape (batch_size, num_hid)
    Y : numpy.ndarray
        Batch of probability vectors of shape (batch_size, num_classes)
    """
    H = sigmoid(X, W, b)
    Y = softmax(H, V, d)
    return H, Y

def calculateGradient(H, Y, T, V, X):
    VGrad = np.dot(H.T, Y-T)/H.shape[0]
    dGrad = (Y-T).mean(axis=0)
#     print Y-T
#     print dGrad
#     print "dGrad.shape", dGrad.shape
    WGrad = np.dot(X.T, np.dot(Y-T, V.T)*H*(1-H))/X.shape[0]
    bGrad = (np.dot((Y-T), V.T)) * H.T* ((1 - H))
#     print bGrad
#     print bGrad.shape
#     k = bGrad.mean(axis=0)
#     print "k",k.shape
    bGrad = bGrad.mean(axis=0)
    return [VGrad, dGrad, WGrad, bGrad]


################ TEST CalculateGradient #######################

# X = np.random.randint(5, size=[2,4])
# print X, "X"
# H = np.random.randint(5, size=[2,4])
# print "H", H
# Y = np.random.randint(5, size=[2,10])
# print "Y", Y
# k = np.random.randint(9, size=2)
# print "k", k
# T = np.zeros((k.size,10), dtype=np.int)
# T[np.arange(k.size), k] = 1
# print "T", T
# V = np.random.randint(5, size=[4,10])
# print "V", V
# print calculateGradient(H,Y,T)



def updateWeight(V, d, W, b, learningRate, gradList):
    V -= learningRate*gradList[0]
    d -= learningRate*gradList[1]
    W -= learningRate*gradList[2]
    b -= learningRate*gradList[3]
    return [V, d, W, b]

def train(W, b, V, d, dataX,dataY):
        
        ############# Model Parameters #########################
        
        epochs = 5
        batchSize = 100
        count = 0
        learningRate = 0.001
        noClasses = 10
                 
        
        ################### Model Training #########################
        
        for i in range(epochs) :
            for j in range(int(len(dataX)/batchSize)) :
                X = dataX[j*batchSize:(j+1)*batchSize]
                T = dataY[j*batchSize:(j+1)*batchSize]
                k = np.zeros((T.size, noClasses))
                k[np.arange(T.size), T] = 1
                T = k 
                [H, Y] = forward(X, W, b, V, d)
                lossValue = loss(Y, T)
                if count%100 == 0 :
                    print lossValue
                gradList = calculateGradient(H, Y, T, V, X)
#                 for i in range(len(gradList)):
#                     print gradList[i].shape
                [V, d, W, b] = updateWeight(V, d, W, b, learningRate, gradList)
                count+=1
        np.save('Weights_V.npy',V)
        np.save('Weights_d.npy', d)
        np.save('Weights_W.npy', W)
        np.save('Weights_b.npy', b)
        

def test(dataX, dataY):
    V = np.load('Weights_V.npy')
    d = np.load('Weights_d.npy')
    W = np.load('Weights_W.npy')
    b = np.load('Weights_b.npy')
    count = 0
    for i in range(len(dataX)):
        X = dataX[i:i+1]
        T = dataY[i:i+1]
        [H, Y] = forward(X, W, b, V, d)
#         print Y
        lossValue = loss(Y, T)
        Y = np.argmax(Y)
#         print Y, T
        if(Y==T):
            count += 1
    print "Test Accuracy is %g" %(float(count)/float(len(dataX)))
    return 



def load_mnist():
    data_dir = '../data/'
#     print os.path.join(data_dir, 'train-images.idx3-ubyte')
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY

def main():
    trainX, trainY, testX, testY = load_mnist()
    trainX = np.reshape(trainX,[-1,784])
    testX = np.reshape(testX,[-1,784])
    print trainX.shape
    print trainY.shape
    #print testX.shape
    
    ############## Model Weight Vectors #####################
        
    inputSize = 28*28
    hiddenSize = 100
    noClasses = 10

    ############## Training Data ###########################3
    
    W = -0.01*np.random.randn(inputSize,hiddenSize)
    b = np.zeros([hiddenSize])
    V = -0.01*np.random.randn(hiddenSize, noClasses)
    d = np.zeros([noClasses])
    
    train(W, b, V, d, trainX, trainY)
    test(testX, testY)   
    
    
if __name__ == '__main__':
    main()    
