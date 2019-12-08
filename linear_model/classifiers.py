""""
Author : Aditya Jain
Contact : https://adityajain.me
"""
import numpy as np
import matplotlib.pyplot as plt

class SGDClassifier():
    """
    SGD classifier model, that optimizes using gradient descent

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    learning_rate = float, default 0.01, learning rate while updating weights

    tol : float, default 0.01, stopping criteria
    
    seed : integer, random seed

    normalize : boolean, normalize X in fit method

    Attributes
    ----------
    coef_ : Estimated coefficients for the linear regression problem
    
    intercept_ : integer, bias for the linear regression problem

    """
    def __init__(self, learning_rate=0.01, tol=0.01, seed=None, normalize=False):
        np.random.seed(seed if seed is not None else np.random.randint(100))
        self.W = None
        self.b = None
        self.__lr = learning_rate
        self.__tol = tol
        self.__length = None
        self.__normalize = normalize
        self.__m = None
        self.__costs = []
        self.__iterations = []

    def __sigmoid(self,z): return 1/(1+np.exp(-z))

    def __initialize_weights_and_bais(self):
        self.W = np.random.randn(self.__length) #(n,1)
        self.b = 0
        
    def __computeCost(self,p,Y):
        loss = -( Y*np.log(p) + (1-Y)*np.log(1-p) )
        cost = np.sum(loss)/self.__m
        return cost

    def __optimize(self,X,y):
        p = self.__sigmoid( np.dot(X,self.W)+self.b )
        dW = np.dot( X.T, (p-y) )/self.__m # (4,1)
        db = np.sum(p-y)/self.__m
        self.W = self.W - self.__lr*dW
        self.b = self.b - self.__lr*db
    
    def __normalizeX(self,X): return (X-self.__mean) / (self.__std)

    def fit(self, X, y, verbose=False):
        """
        Fit X using y by optimizing weights and bias
        
        Parameters
        ----------
        X : 2D numpy array, independent variables

        y : 1D numpy array, dependent variable
        
        verbose : boolean, print out details while optimizing (Default : False) 
       
        """
        if self.__normalize:
            self.__mean, self.__std = X.mean(axis=0), X.std(axis=0)
            X = self.__normalizeX(X)
        self.__m,self.__length = X.shape
        self.__initialize_weights_and_bais()
        last_cost,i = float('inf'),0
        while True:
            p = self.__sigmoid( np.dot(X,self.W)+self.b )
            cost = self.__computeCost(p,y)
            if verbose: print(f"Iteration: {i}, Cost: {cost:.3f}")
            self.__optimize(X,y)
            if last_cost-cost < self.__tol: break
            else: last_cost,i = cost,i+1
            self.__costs.append(cost)
            self.__iterations.append(i)

    def predict(self,X):
        """
        Predict dependent variable

        Parameters
        ----------
        X : numpy array, independent variables

        Returns
        -------
        predicted classes    

        """
        return self.predict_proba(X)[:,1]>0.5

    def predict_proba(self,X):
        """
        Predict probability of all classes

        Parameters
        X : numpy array, independent variables

        Returns
        -------
        predicted probabilities

        """
        if self.__normalize: X = self.__normalizeX(X)
        ones = self.__sigmoid( np.dot(X,self.W)+self.b )
        return np.c_[1-ones,ones]

    def plot(self,figsize=(7,5)):
        """
        Plot a optimization graph
        """
        plt.figure(figsize=figsize)
        plt.plot(self.__iterations,self.__costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title("Iterations vs Cost")
        plt.show()

    def score(self,X,y):
        """
        Calculate accuracy from independent variables

        Parameters
        ----------
        X : numpy array, independent variables
        y : numpy array, dependent variable

        Returns
        -------
        accuracy score
        
        """
        return (self.predict(X) == y).sum() / len(y)

    @property
    def coef_(self): return self.W
    
    @property
    def intercept_(self): return self.b