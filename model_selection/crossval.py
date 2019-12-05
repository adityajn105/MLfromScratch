"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np 

class KFold():
	"""
	K-Folds cross-validator

	Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
	
	Parameters
	----------
	n_splits : int, default=3, Number of folds. Must be at least 2.

	shuffle : boolean, optional Whether to shuffle the data before splitting into batches.

	random_state : int, RandomState instance or None, optional, default=None
	
	"""
	def __init__(self, n_splits=3, shuffle=False, random_state=None):
		assert n_splits > 1, "Minimum splits must be greater that 1"
		self.__k = n_splits
		self.__shuffle = shuffle
		np.random.seed(random_state)
	
	def split(self,X):
		"""
		Generate indices to split data into training and test set.

		Parameters
		----------
		X : array-like, Training data
		
		Returns
		-------
		train array indices, test array indices
		
		"""
		length = len(X)
		indices = np.arange(0,length,1)
		if self.__shuffle: np.random.shuffle(indices)
		last_index, split_length = 0, int(np.ceil(length/self.__k))
		array = []
		while min(last_index,last_index+split_length) < length:
			array.append( indices[ last_index : min(last_index+split_length, length)] )
			last_index = last_index+split_length
		for i in range(len(array)):
			yield array[:i]+array[i+1:], array[i]
			
			
