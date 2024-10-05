import numpy as np
from numpy import linalg as la


class pca():
    
    """
    PCA is a dimendionality reduction technique.
    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space

    """

    def __init__(self):
        self.cov=cov
        self.eigen=eigen
        self.eigval=None
        self.eigvct=None
        self.covar=None
        self.princicomp=None

    def cov(self,x):
        
        """
        first we have to find the covariance matrix of the given array/dataframe

        """
        self.covar=np.cov(x)

        return self.covar

    def eigen(self,x):
        
        """
        Second step is to find eigen values and vectors of the given array  
        
        """
        self.eigval, self.eigvct = np.la.eig(self.covar)   
        
        return self.eigval,self.eigvct
    
    def ncomp(self,numcomponents):
        
        """
        Method to choose number of principal components

        """
        self.princicomp=self.eigvct[0:numcomponents]
        
        """
        For examples, if numcomponents=2 then this function returns first 2 rows of eigenvectors

        """

        return self.princicomp

        