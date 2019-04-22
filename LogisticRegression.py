from math import exp
import numpy as np 


class LogisticRegressionClassifier:
    '''
    Demo of Logistic Regression Classifier
    usage:
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        clf.validate(X, y) or clf.predict(x)
    '''
    def __init__(self):
        self.weight = None
        self.bias = None
        self.X = None
        self.y = None
        self.iters = None
        self.learning_rate = None
        self.threshold = None
    
    def fit(self, X, y, iters=100, lr=0.001, threshold=0.5):
        self.X = X
        self.y = y
        self.weight = np.random.uniform(size=X.shape[1])
        self.bias = np.random.uniform()
        self.iters = iters
        self.learning_rate = lr
        self.threshold = threshold

        for i in range(iters):
            iter_loss =0
            self.update_parameters()
            for l in self.loss():
                iter_loss += l
            mse = iter_loss / len(self.y)
            print(f'Iter: {i}, MSE: {mse}, weight: {self.weight.mean()}, bias: {self.bias}')
    
    def _predict(self, x):
        return self.sigmoid(self.z(x))
    
    def predict(self, x):
        if self._predict(x) >= self.threshold:
            return 1
        else:
            return 0

    def validate(self, X, y):
        total_error = 0
        for i, x in enumerate(X):
            if y[i] != self.predict(x):
                total_error += 1
        accuracy = 1 - total_error / len(y)
        print('--->', f'Accuracy of {len(y)} validation is {accuracy}')

    def z(self, x):
        res = np.dot(self.weight, x) + self.bias
        #print(f'z of {x} is {res}')
        return res

    def sigmoid(self, z):
        res = 1 / (1 + exp(-z))
        #print(f'Sigmoid of {z} is {res}')
        return res
    
    def sigmoid_prime(self, z):
        res = self.sigmoid(z) * (1 - self.sigmoid(z))
        #print(f'Sigmoid prime of {z} is {res}')
        return res
    
    def weight_prime(self):
        return [(-2 * (self.y[i] - self.sigmoid(self.z(x))) * self.sigmoid_prime(self.z(x)) * x) for i, x in enumerate(self.X)]
    
    def bias_prime(self):
        return [(-2 * (self.y[i] - self.sigmoid(self.z(x))) * self.sigmoid_prime(self.z(x))) for i, x in enumerate(self.X)]
    
    def loss(self):
        return [((self.y[i] - self._predict(x)) ** 2) for i, x in enumerate(self.X)]
    
    def update_parameters(self):
        w_gradient = 0
        b_gradient = 0
        for i in self.weight_prime():
            w_gradient += i
        for i in self.bias_prime():
            b_gradient += i
        self.weight -= self.learning_rate * w_gradient / len(self.y)
        self.bias -= self.learning_rate * b_gradient / len(self.y)
    
