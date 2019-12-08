""""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np

def root_mean_squared_error(y_true,y_pred):
	"""
	Compute root mean square error

	Parameters
	----------
	y_pred : predicted values
	
	y_true : True values

	Returns
	-------
	rmse
	
	"""
	loss = np.square(y_pred-y_true)
	cost = np.sqrt( np.mean(loss) )
	return cost


def mean_squared_error(y_true,y_pred):
	"""
	Compute mean square error

	Parameters
	----------
	y_pred : predicted values
	
	y_true : true values
	
	Returns
	-------
	mse
	
	"""
	loss = np.square(y_pred-y_true)
	cost = np.mean(loss)
	return cost


def mean_absolute_error(y_true,y_pred):
	"""
	Compute mean absolute error

	Parameters
	----------
	y_pred : predicted values

	y_true : true values
	
	Returns
	-------
	mae

	"""
	loss = abs(y_pred-y_true)
	cost = np.mean(loss)
	return cost

def r2_score(y_true,y_pred):
	"""
	Compute Coefficient of Determinance, r2 score

	r2 score = ESS/TSS = 1 - RSS/TSS = 1 - (y_true - y_pred)^2 / (y_true - y_true.mean())

	Parameters
	----------
	y_pred : predicted values
	
	y_true : true values

	Returns
	-------
	r2_score
	"""
	return 1-(np.sum((( y_true - y_pred)**2))/np.sum((y_true-np.mean(y_true))**2))