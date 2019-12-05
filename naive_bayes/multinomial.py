"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np
from collections import defaultdict

class MultinomialNB():
	"""
	Naive Bayes classifier for multinomial models

	The multinomial Naive Bayes classifier is suitable for classification with discrete features 
	"""
	def __init__(self):
		self.__summary = None
		self.__total_cnt = None

	def __summarize(self,X,y):
		summary = []
		for label in sorted(np.unique(y)):
			X_label = X[y==label]
			array = []
			for feat_ind in range(X_label.shape[1]):
				probs = defaultdict(float)
				feats,cnts = np.unique(X_label[:,feat_ind], return_counts=True)
				for feat,cnt in zip(feats,cnts):
					probs[feat] = cnt/(y==label).sum()
				array.append(probs)
			summary.append(array)
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
		labels, cnts = np.unique( y, return_counts=True )
		for label, cnt in zip(labels,cnts):
			prob = cnt/self.__total_cnt
			for feat_ind in range(len(x)):
				prob *= self.__summary[int(label)][feat_ind][x[feat_ind]]
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