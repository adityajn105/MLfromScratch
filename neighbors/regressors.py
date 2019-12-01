"""
Authors: Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np 

class KNeighborsRegressor():

	def __init__(self, n_neighbors=5, metric='minkowski', p=2, normalize=False):
		self.__n_neighbors = n_neighbors
		self.__X = None
		self.__y = None
		self.__normalize = normalize
		self.__metric = { 'minkowski': self.__minkowski, 'euclidean':self.__euclidean, 'manhatten':self.__manhatten }[metric]
		self.__p = p
		self.__n_classes = None
		self.__means = None
		self.__std = None

	def __euclidean(self,X1,X2):
		return np.sqrt(np.sum(np.square(X1-X2),axis=1))

	def __manhatten(self,X1,X2):
		return np.sum(np.abs(X1-X2),axis=1)

	def __minkowski(self,X1,X2):
		return np.power(np.sum(np.power(np.abs(X1-X2),self.__p),axis=1),1/self.__p)

	def __normalizeX(self,X):
		return (X-self.__means)/self.__std

	def fit(self,X,y):
		"""
		Fit X using y
		Parameters
		----------
		X : 2D numpy array, independent variables
		y : 1D numpy array, dependent variable
		"""
		self.__y, self.__n_classes = y, len(np.unique(y))
		if self.__normalize:
			self.__means, self.__std = X.mean(axis=0), X.std(axis=0)
			self.__X = self.__normalizeX(X)
		else:
			self.__X = X


	def predict(self,X):
		"""
		Predict dependent variable

		Parameters
		---------
		X : numpy array, independent variables

		Returns
		-------
		predicted classes    

		"""
		if self.__normalize: X = self.__normalizeX(X)
		probs = []
		for sample in X:
			x = np.expand_dims(sample,axis=0) 
			distances = self.__metric(self.__X, x)
			top_k_index = distances.argsort()[:self.__n_neighbors]
			probs.append(np.mean(self.__y[top_k_index]))
		return np.array(probs)

	def score(self,X,y):
		"""
		Computer Coefficient of Determination (rsquare)

		Parameters
		----------
		X : 2D numpy array, independent variables
		y : numpy array, dependent variables

		Output
		------
		r2 values
		
		"""
		return 1-(np.sum(((y-self.predict(X))**2))/np.sum((y-np.mean(y))**2))

