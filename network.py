from __future__ import division
import numpy as np
import scipy.io

class Network(object):
    def __init__(self, sizes, X, y, iters, learning_rate):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.X = X
        self.y = y
        self.gradient_descent(iters, learning_rate)

    def cost_function(self, training):
        return sum((x[1]- self.predict(x[0])**2) for x in training)/(2*len(training))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def gradient_descent(self, iters, learning_rate):
        for i in xrange(iters):
            for x, y in zip(self.X, self.y):
                self.back_prop(x, y)

    def back_prop(self, x, y):
        pass

def sigmoid(z):
    return 1/(1+np.exp(-z))


def reshape_y(in_ys):
    ys = np.zeros((len(in_ys), 10))
    for i in xrange(len(in_ys)):
        ys[i, in_ys[i]%10] = 1
    return ys

# TODO split into test and train data
data = scipy.io.loadmat('data/ex4data1.mat')
X, y = data['X'], reshape_y(data['y'])
network = Network([X.shape[1], 25, 10])
