"""
Authors : Aditya Jain
Contact : https://adityajain.me
"""


import numpy as np

def train_test_split(X,Y,test_size=None,seed=5):
	"""
	Custom Train Test split function
	"""
	assert test_size!=None, "test_size cannot be None"
	np.random.seed(seed)
	indexes = np.random.choice([False,True],size=len(X),p=[test_size,1-test_size])
	return X[indexes],X[~indexes],Y[indexes],Y[~indexes]