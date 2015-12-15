from __future__ import division
from operator import add
import numpy as np
import scipy.io

class Network(object):
    def __init__(self, sizes, X, y, iters=1000, learning_rate=1):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(yi,1) for yi in sizes[1:]]
        self.weights = [np.random.randn(yi,xi) for xi, yi in zip(sizes[:-1], sizes[1:])]
        self.X = X
        self.y = y
        self.gradient_descent(iters, learning_rate)

    def cost_function(self, test_data=None):
        if test_data is None:
            test_data = zip(self.X, self.y)
        return sum(sum(sum((y - self.predict(x))**2) for x, y in test_data))/(2*len(test_data))

    def predict(self, a):
        a = np.reshape(a, (len(a), 1))
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def evaluate(self, X, y):
        return sum(1 for x, y in zip(X, y) if np.argmax(self.predict(x)) == np.argmax(y))/len(y)

    def feedforward(self, a):
        zs = []
        activations = [a]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,a)+b
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)
        return zs, activations

    def gradient_descent(self, iters, learning_rate):
        for i in xrange(iters):
            if i % 20 == 0:
                print i, self.evaluate(self.X, self.y)
            delta_b = [np.zeros_like(bias) for bias in self.biases]
            delta_w = [np.zeros_like(weight) for weight in self.weights]
            for x, y in zip(self.X, self.y):
                nabla_b, nabla_w = self.backprop(x, y)
                delta_b = map(add, delta_b, nabla_b)
                delta_w = map(add, delta_w, nabla_w)
            self.biases = map(lambda b, d: b - d*learning_rate/len(self.y), self.biases, delta_b)
            self.weights = map(lambda w, d: w - d*learning_rate/len(self.y) , self.weights, delta_w)

    def backprop(self, x, y):
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))
        zs, activations = self.feedforward(x)
        nabla_w = [None]*(self.num_layers - 1)
        nabla_b = [None]*(self.num_layers - 1)
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].T)
        for i in xrange(self.num_layers - 2, 0, -1):
            delta = self.weights[i].T.dot(delta) * sigmoid_prime(zs[i-1])
            nabla_b[i-1] = delta
            nabla_w[i-1] = delta.dot(activations[i-1].T)
        return nabla_b, nabla_w

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

def reshape_y(in_ys):
    ys = np.zeros((len(in_ys), 10))
    for i in xrange(len(in_ys)):
        ys[i, in_ys[i]%10] = 1
    return ys


# TODO split into test and train data
data = scipy.io.loadmat('data/ex4data1.mat')
X, y = data['X'], reshape_y(data['y'])
network = Network([X.shape[1], 25, 10], X, y)
