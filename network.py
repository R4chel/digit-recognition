from __future__ import division
import numpy as np
import scipy.io

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def cost_function(self, training):
        return sum((x[1]- self.predict(x[0])**2) for x in training)/(2*len(training))

    def predict(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w,x)+b)
        return x

    def sigmoid(z):
        return 1/(1+np.exp(-z))

def reshape_y(in_ys):
    ys = np.zeros((len(in_ys), 10))
    for i in xrange(len(in_ys)):
        ys[i, in_ys[i]%10] = 1
    return ys

data = scipy.io.loadmat('data/ex4data1.mat')
#xys = zip(data['X'], data['y'])
#np.random.shuffle(xys)
#test, train = xys[:int(len(xys)/5)], xys[int(len(xys)/5):]
#train_x, train_y = zip(*train)

X, y = data['X'], reshape_y(data['y'])
network = Network([X.shape[1], 25, 10])
