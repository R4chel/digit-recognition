from __future__ import division
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def cost_function(self, training):
        return sum((x[1]- self.predict(x[0])**2) for x in training)/(2*len(training))

    def predict(self, x):
        #self.biases
        #self.weights
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w,x)+b)
        return x



def sigmoid(z):
    return 1/(1+np.exp(-z))
