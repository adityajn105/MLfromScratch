import pandas as pd
import numpy as np
import os

def train_test_split(X,Y,test_size=None,seed=5):
	"""
	Custom Train Test split function
	"""
	assert test_size!=None, "test_size cannot be None"
	np.random.seed(seed)
	indexes = np.random.choice([False,True],size=len(X),p=[test_size,1-test_size])
	return X[indexes],X[~indexes],Y[indexes],Y[~indexes]

def getRegData(path):
	reg = pd.read_csv(path)
	X = reg.drop('Y',axis=1).values
	y = reg.Y.values
	return train_test_split(X,y,test_size=0.3,seed=7)

def getClassiData(path):
	cls = pd.read_csv(path)
	X = cls.drop('Y',axis=1).values
	y = cls.Y.values
	return train_test_split(X,y,test_size=0.3,seed=7)