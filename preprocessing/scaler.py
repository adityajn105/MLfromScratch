"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np

class StandardScaler():
	"""
	Apply Standard Scaling on data
	
		scaled_data = (data - mean)/ std
	
	Attributes
	----------
	
	mean_ : mean of each column
	
	std_ : standard deviation of each columns
	
	"""
	def __init__(self):
		self.__mean = None
		self.__std = None
		
	def fit(self,X):
		"""
		Fit data to calucate mean and std
		
		Parameters
		----------
		X : numpy array
		
		"""
		self.__mean = X.mean(axis=0)
		self.__std = X.std(axis=0)
		
	def transform(self,X):
		"""
		Transform data using mean and std
		
		Parameters
		----------
		X : numpy array
		
		Returns
		-------
		transformed_X
		
		"""
		return (X-self.__mean)/self.__std
	
	def fit_transform(self,X):
		"""
		Fit and transform data at once
		
		Parameters
		----------
		X : numpy array
		
		Returns
		-------
		transformed_X
		
		"""
		self.fit(X)
		return self.transform(X)
	
	def inverse_transform(self,X):
		"""
		Convert transformed data into original data
		
		Parameter
		---------
		X : numpy array, transformed data
		
		Returns
		-------
		Original data
		
		"""
		return (X*self.__std)+self.__mean
	
	@property
	def mean_(self): return self.__mean
	
	@property
	def std_(self): return self.__std
	

class MinMaxScaler():
	"""
	Apply Min-Max scaling on data
	
		scaled_data = (data - min)/ (max - min)
	
	Attributes
	----------
	
	min_ : min of each column
	
	max_ : max of each columns
	
	"""
	def __init__(self):
		self.__min = None
		self.__max = None
		
	def fit(self,X):
		"""
		Fit data to calucate min and max
		
		Parameters
		----------
		X : numpy array
		
		"""
		self.__min = X.min(axis=0)
		self.__max = X.max(axis=0)
		
	def transform(self,X):
		"""
		Transform data using min and max
		
		Parameters
		----------
		X : numpy array
		
		Returns
		-------
		transformed_X
		
		"""
		return (X-self.__min)/(self.__max-self.__min)
	
	def fit_transform(self,X):
		"""
		Fit and transform data at once
		
		Parameters
		----------
		X : numpy array
		
		Returns
		-------
		transformed_X
		
		"""
		self.fit(X)
		return self.transform(X)
	
	def inverse_transform(self,X):
		"""
		Convert transformed data into original data
		
		Parameter
		---------
		X : numpy array, transformed data
		
		Returns
		-------
		Original data
		
		"""
		return (X*(self.__max-self.__min))+self.__min
	
	@property
	def min_(self): return self.__min
	
	@property
	def max_(self): return self.__max
	
	
class RobustScaler():
	"""
	Apply Robust scaling on data
	
		scaled_data = (data - median)/ (3rd and 1st quantile range))
	
	Attributes
	----------
	
	center_ : median for each column
	
	scale_ : inter quantile range for each column
	
	"""
	def __init__(self):
		self.__median = None
		self.__3rd_quantile = None
		self.__1st_quantile = None
		
	def fit(self,X):
		"""
		Fit data to calucate median and 1st and 3rd quartile range
		
		Parameters
		----------
		X : numpy array
		
		"""
		self.__median = np.median(X,axis=0)
		self.__3rd_quantile = np.percentile(X,75,axis=0)
		self.__1st_quantile = np.percentile(X,25,axis=0)
		
	def transform(self,X):
		"""
		Transform data using median, 1st and 3rd quartile range
		
		Parameters
		----------
		X : numpy array
		
		Returns
		-------
		transformed_X
		
		"""
		return (X-self.__median)/(self.__3rd_quantile-self.__1st_quantile)
	
	def fit_transform(self,X):
		"""
		Fit and transform data at once
		
		Parameters
		----------
		X : numpy array
		
		Returns
		-------
		transformed_X
		
		"""
		self.fit(X)
		return self.transform(X)
	
	def inverse_transform(self,X):
		"""
		Convert transformed data into original data
		
		Parameter
		---------
		X : numpy array, transformed data
		
		Returns
		-------
		Original data
		
		"""
		return (X*(self.__3rd_quantile-self.__1st_quantile))+self.__median
	
	@property
	def center_(self): return self.__median
	
	@property
	def scale_(self): return self.__3rd_quantile - self.__1st_quantile