"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np
class VotingClassifier():
	"""
	Soft Voting/Majority Rule classifier for unfitted estimators.
	
	Parameters
	----------
	estimators : list of (str, estimator) tuples
	
	voting : 'hard' or 'soft', default to 'hard'
		If ‘hard’, uses predicted class labels for majority rule voting. 
		Else if ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities, 
		which is recommended for an ensemble of well-calibrated classifiers.
	
	weights: list of integers
		weight given to each estimator
		
	Attributes
	----------
	estimators_ : list of (str, fitted estimator)
	
	"""
	def __init__(self,estimators,voting='hard',weights=None):
		self.__estimators = estimators
		self.__n_estimators = len(estimators)
		self.__weights = weights if weights!=None else [1]*self.__n_estimators
		self.__voting = { 'hard':'hard','soft':'soft' }[voting]
		self.__n_classes = None
	
	def fit(self,X,y):
		"""
		Fit the observations into each model
		
		Parameters
		----------
		X : numpy array, training feature array
		
		y : training labels
		"""
		self.__n_classes = len(np.unique(y))
		for est,obj in self.__estimators:
			obj.fit(X,y)
			
	def predict(self,X):
		"""
		Perform prediction and find best class using voting
		
		Parameters
		----------
		X : numpy array, feature array
		
		Returns
		-------
		predicted labels
		
		"""
		if self.__voting=='hard':
			classes = np.empty( (len(X),0), dtype=np.int )
			for i in range(self.__n_estimators):
				predictions = self.__estimators[i][1].predict(X)
				for _ in range( self.__weights[i] ):
					classes = np.c_[ classes, predictions]
			return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=classes)
		else:
			probs = np.zeros( (len(X),self.__n_classes) )
			for i in range(self.__n_estimators):
				probabilities = self.__estimators[i][1].predict_proba(X)*self.__weights[i]
				probs += probabilities
			return np.argmax( probs, axis=1 )
		
	def score(self,X,y):
		"""
		Compute Accuracy Score
		
		Parameters
		----------
		X : numpy array, feature array
		
		y : numpy array, feature labels
		
		Returns
		-------
		accuracy score
		"""
		y_pred = self.predict(X)
		return (y_pred==y).sum()/len(y)
	
	@property
	def estimators_(self): return self.__estimators