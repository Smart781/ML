import numpy as np

# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return np.sum((y_true - y_predicted) ** 2) / y_true.shape[0]

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    u = np.sum((y_true - y_predicted) ** 2)
    y_mid = np.sum(y_true) / y_true.shape[0]
    v = np.sum((y_mid - y_true) ** 2)
    return 1 - u / v

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        ones = np.ones((1, X.shape[0]))
        X = X.T
        X = np.concatenate((ones, X))
        X = X.T
        self.weights = np.linalg.pinv((X.T).dot(X)).dot(X.T).dot(y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        ones = np.ones((1, X.shape[0]))
        X = X.T
        X = np.concatenate((ones, X))
        X = X.T
        return X.dot(self.weights)
    
# Task 3

class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.weights = None # Save weights here
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        self.bias = None
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        norm = NormalLR()
        norm.fit(X, y)
        self.weights = norm.weights
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        k = X.dot(self.weights) + self.bias - y
        for i in range(self.iterations):
            nabla_w = (2 * ((X.dot(self.weights) + self.bias - y).dot(X))) / X.shape[0] + self.l * np.sign(self.weights)
            nabla_b = (2 * (X.dot(self.weights) + self.bias - y)).sum() / X.shape[0] + self.l * np.sign(self.bias)
            self.weights -= self.alpha * nabla_w
            self.bias -= self.alpha * nabla_b
            
    def predict(self, X:np.ndarray):
        return X.dot(self.weights) + self.bias

# Task 4

def get_feature_importance(linear_regression):
    return abs(linear_regression.weights)

def get_most_important_features(linear_regression):
    weights = get_feature_importance(linear_regression)
    index = np.arange(0, len(weights)).tolist()
    index = sorted(index, key = lambda x: weights[x], reverse=True)
    return index