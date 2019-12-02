"""
Authors : Aditya Jain
Contact : https://adityajain.me
"""


import numpy as np

def train_test_split(X,Y,test_size=None,random_state=5):
	"""
	Train Test split function
	
	Parameters
	----------

	X : numpy array, independent variables

	y : numpy array, dependent variables
	
	test_size : float, percent of test samples

	random_state : integer, random seed

	Returns
	-------
	X_train, X_test, Y_train, Y_test

	"""
	assert test_size!=None, "test_size cannot be None"
	np.random.seed(random_state)
	indexes = np.random.choice([False,True],size=len(X),p=[test_size,1-test_size])
	return X[indexes],X[~indexes],Y[indexes],Y[~indexes]