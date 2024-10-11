import numpy as np

class SimpleSVC:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization strength
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # No loss, so just regularize the weights
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Penalize the misclassification (Hinge loss)
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        # Predict the class based on the sign of the decision function
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
