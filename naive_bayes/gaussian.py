"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np 

class GaussianNB():
	"""
	Gaussian Naive Bayes

	Continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution.
	"""
	def __init__(self):
		self.__summary = None
		self.__total_cnt = None
	 
	def __summarize(self,X,y):
		summary = []
		for label in sorted(np.unique(y)):
			X_label = X[ y==label ]
			means, std, cnt = X_label.mean(axis=0), X_label.std(axis=0)*np.sqrt( len(X_label)/(len(X_label)-1) ), len(X_label)
			summary.append( (means, std, cnt) )
		return summary

	def fit(self,X,y):
		"""
		Fit X and y to calculate prior probabilities
		
		Parameters
		----------
		X : numpy array, independent continous variables
		y : numpy array, dependent variables
		"""
		self.__summary = self.__summarize(X,y)
		self.__total_cnt = len(y)
		
	def __predict_row(self,x):
		probs = []
		for means,stds,cnt in self.__summary:
			prob = cnt/self.__total_cnt
			for feature_index in range(len(x)):
				prob = prob*calculate_probability( x[feature_index], means[feature_index], stds[feature_index] )
			probs.append(prob)
		return np.array(probs)/sum(probs)
	
	def predict_proba(self,X):
		"""
		Predict probability associated with each class for all samples
		
		Parameters
		----------
		X : numpy array, independent variables
		
		Returns
		-------
		Probabilities for each class
		"""
		return np.array( [ self.__predict_row(x) for x in X ] )
	
	def predict(self,X):
		"""
		Predict class of each sample
		
		Parameters
		----------
		X : numpy array, independent variables
		
		Returns
		-------
		Class of each sample
		"""
		return np.argmax( self.predict_proba(X), axis=1 )
	
	def score(self,X,y):
		"""
		Accuracy Score
		
		Parameters
		----------
		X : numpy array, independent variables
		y : numpy array, dependent variable
		
		Returns
		-------
		accuracy score
		"""
		return (self.predict(X)==y).sum()/len(y)