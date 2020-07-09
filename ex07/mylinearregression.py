import numpy as np

class MyLinearRegression:
    
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def predict_(self, x):
        return np.dot(np.c_[np.ones((len(x), 1)), x], self.thetas)



    def fit_(self, x, y):
        X = np.c_[np.ones((len(x), 1)), x]
        y = np.squeeze(y)
        i = 0
    
        while i < self.max_iter:
            gradient = (1 / len(x)) * (np.dot(np.transpose(X), (np.dot(X, self.thetas) - y)))
            self.thetas = self.thetas - self.alpha * gradient
            i += 1
        return self.thetas

    def cost_elem_(self, x, y):
        m = len(x)
        y_hat = self.predict_(x)
        print(y_hat)
        y = np.squeeze(y)
        return (((y_hat - y) ** 2) / (2 * m))

    def cost_(self, x, y):
        elem = self.cost_elem_(x, y)
        return sum(elem)
