'''
    Linear Regression Classifier Demo using gradient decent.
    usage:
        clf = LinearRegressionClassifier()
        clf.fit(X, y)
    For 2-D regression only :)
'''

import numpy as np 

class LinearRegressionClassifier:
    def __init__(self):
        self.X = None
        self.y = None
        self.weight = None
        self.bias = None
        self.learning_rate = None
        self.iters = None

    def fit(self, X, y, init_w=None, init_b=None, lr=0.001, iters=100):
        self.X = X
        self.y = y
        if not init_w:
            i, j = np.random.choice(len(X), 2)
            self.weight = (y[i] - y[j]) / (X[i] - X[j])
        else:
            self.weight = init_w
        if not init_b:
            self.bias = y[0] - (self.weight * X[0])
        else:
            self.bias = init_b
        self.learning_rate = lr
        self.iters = iters

        for i in range(self.iters): 
            iter_loss = 0         
            self.update_parameters()
            for l in self.total_loss():
                iter_loss += l
            mse = iter_loss / len(self.X)
            print(f'Iter: {i}, MSE: {mse}, weight: {self.weight}, bias: {self.bias}')

    def predict(self, X):
        return self.weight * X + self.bias
    
    def total_loss(self):
        return [(y[i] - (self.weight * x + self.bias)) ** 2 for i, x in enumerate(self.X)]
    
    def b_prime(self):
        return [(y[i] - (self.weight * x + self.bias)) * -2 for i, x in enumerate(self.X)]

    def w_prime(self):
        return [(y[i] - (self.weight * x + self.bias)) * -2 * x for i, x in enumerate(self.X)]

    def update_parameters(self):
        w_gradients = 0
        b_gradients = 0
        for w in self.w_prime():
            w_gradients += w
        for b in self.b_prime():
            b_gradients += b
        self.weight -= self.learning_rate * w_gradients / len(self.X)
        self.bias -= self.learning_rate * b_gradients / len(self.X)

