import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # Regularization strength
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.coef_ = None  # Coefficients of the model
        self.intercept_ = None  # Intercept of the model

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Coordinate descent algorithm
        for iteration in range(self.max_iter):
            prev_coef = self.coef_.copy()

            # Update intercept
            self.intercept_ = np.mean(y - X.dot(self.coef_))

            # Update coefficients
            for j in range(n_features):

                # Calculate the partial residual
                residual = y - self.intercept_ - X.dot(self.coef_)
                rho = X[:, j].dot(residual)

                # Update coefficient with soft thresholding
                if rho < -self.alpha:
                    self.coef_[j] = (rho + self.alpha) / X[:, j].dot(X[:, j])
                elif rho > self.alpha:
                    self.coef_[j] = (rho - self.alpha) / X[:, j].dot(X[:, j])
                else:
                    self.coef_[j] = 0.0
            
            # Check for convergence
            if np.sum(np.abs(prev_coef - self.coef_)) < self.tol:
                break

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_
