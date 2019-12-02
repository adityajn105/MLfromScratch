"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np


class LabelEncoder():
	"""
	Perform Numerical encoding of categorical variables
	
	Attributes
	----------
	
	classes_ : Holds the label for each class.
	
	"""
	def __init__(self):
		self.__classes = None
	
	def fit(self,X):
		"""
		Fit encoder using labels
		
		Parameters
		----------
		X : numpy array, labels
		
		"""
		self.__classes = np.unique(X)
	
	def transform(self,X):
		"""
		Encode labels using fitted encoder
		
		Parameters
		----------
		X : numpy array, lables to be encoded
		
		Returns
		-------
		encoded labels
		
		"""
		X_new = np.zeros(X.shape, dtype=np.int)
		for i in range(0,len(self.__classes)):
			X_new[np.argwhere(X==self.__classes[i])] = i
		return X_new
	
	def fit_transform(self,X):
		"""
		Fit encoder and encode labels at once
		
		Parameters
		----------
		X : numpy array, labels to be encoded
		
		Returns
		-------
		encoded labels
		
		"""
		self.fit(X)
		return self.transform(X)
	
	def inverse_transform(self,X):
		"""
		Convert encoded labels into original labels
		
		Parameter
		---------
		X : numpy array, encoded labels
		
		Returns
		-------
		original labels
		
		"""
		return np.array([self.__classes[i] for i in X])
	
	@property
	def classes_(self): return self.__classes
	
class OneHotEncoder():
	"""
	Perform One Hot encoding of categorical variables
	
	
	"""
	def __init__(self):
		self.__classes = None
	
	def fit(self,X):
		"""
		Fit encoder using labels
		
		Parameters
		----------
		X : numpy array, labels
		
		"""
		self.__classes = np.unique(X)

	def transform(self,X):
		"""
		One Hot Encode labels using fitted encoder
		
		Parameters
		----------
		X : numpy array, lables to be encoded
		
		Returns
		-------
		one hot encoded labels
		
		"""
		encoding = np.identity( len(self.__classes), dtype=np.int)
		encoder_dict = {}
		for i in range(0,len(self.__classes)):
			encoder_dict[self.__classes[i]]  = encoding[i]
		arr = []
		for x in X:
			arr.append(encoder_dict[x])
		return np.array(arr)
	
	def fit_transform(self,X):
		"""
		Fit encoder and one hot encode labels at once
		
		Parameters
		----------
		X : numpy array, labels to be encoded
		
		Returns
		-------
		one hot encoded labels
		
		"""
		self.fit(X)
		return self.transform( X )
	
	def inverse_transform(self,X):
		"""
		Convert one hot encoded labels into original labels
		
		Parameter
		---------
		X : numpy array, one hot encoded labels
		
		Returns
		-------
		original labels
		
		"""
		args = np.argmax(X,axis=1)
		return np.array( [self.__classes[i] for i in args] )