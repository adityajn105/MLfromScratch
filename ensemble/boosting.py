"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np
from ..tree import DecisionTreeRegressor

class GradientBoostingRegressor():
	"""
	Gradient Boosting for regression.

	GB builds an additive model in a forward stage-wise fashion;
	it allows for the optimization of arbitrary differentiable loss functions.
	In each stage a regression tree is fit on the negative gradient of the
	given loss function.
	
	Parameters
	----------
	
	loss : 'str', default 'ls'
		Loss to be optimized. 'ls' refers to least squares
	regression.
		Currently only least squares regression is available
	
	learning_rate : float, default 0.1
		learning rate shrinks the contribution of each tree by `learning_rate`
	
	n_estimators : int, default 100
		Number of base estimators to train
	
	criterion : str,  Default 'mse',  ('mse', 'mae', 'std')
		The function to measure the quality of a split.
		'mse' is mean squared error
		'mae' is mean absolute error
		'std' is standard deviation
		
	max_depth : int, default : None i.e. 'inf'
		The maximum depth allowed for each base regressor
	
	min_samples_split : int, default 2
		The minimum number of samples required to split an internal node
		
	max_features : int, float, string or None, (default None)
		The number of features to consider when looking for the best split
		- If int, then consider `max_features` features at each split.
		- If float, then `max_features` is a percentage of n_features
		- If "auto", then `max_features=n_features`.
		- If "sqrt", then `max_features=sqrt(n_features)`.
		- If "log2", then `max_features=log2(n_features)`.
		- If None, then `max_features=n_features`.
		
	verbose : boolean, default False
		Enable verbose output, print loss once in a while.
	
	"""
	def __init__(self, loss='ls', learning_rate = 0.1, n_estimators=100, criterion='mse', 
				 max_depth=None, min_samples_split=2, max_features=None, verbose=False):
		self.__lr = learning_rate
		self.__n_estimators = n_estimators
		self.__criterion = criterion
		self.__max_depth = max_depth
		self.__min_samples_split = min_samples_split
		self.__max_features = None
		if isinstance(max_features,str):
			self.__max_features = { 
			'auto': lambda x: int(np.sqrt(x)), 'sqrt': lambda x: int(np.sqrt(x)), 
			'log2': lambda x: int(np.log2(x)), 'max_features': lambda x: x  }[max_features]
		elif isinstance(max_features, int):
			self.__max_features = lambda x: max_features
		elif isinstance(max_features, float):
			self.__max_features = lambda x: int(max_features*x)
		else:
			self.__max_features = lambda x: x
			
		self.__n_features = None
		self.__trees = []
		self.__verbose = verbose
		self.__f0 = None
	
	def __mse(self,y_pred,y_true):
		return np.sqrt( np.mean( (y_true-y_pred)**2 ) )
	
	def __negative_least_squares_gradient(self,y_pred,y_true):
		grad =  -(y_true - y_pred)
		return -1 * grad
	def __get_feature_index(self): 
		return np.random.choice( np.arange(0,self.__n_features,1), 
								size=self.__max_features(self.__n_features), replace=False)
	
	def fit(self, X, y):
		"""
		Fit decision trees to build GB model in additive fashion
		
		Parameters
		----------
		X : numpy array, feature observations
		
		y : numpy array, feature labels
		"""
		self.__n_features = X.shape[1]
		y_ = self.__f0 = y.mean()
		for i in range(0,self.__n_estimators):
			dt = DecisionTreeRegressor(criterion=self.__criterion, 
									   max_depth=self.__max_depth, 
									   min_samples_split=self.__min_samples_split)
			feature_index = self.__get_feature_index()
			h = self.__negative_least_squares_gradient(y_,y)
			dt.fit(X[:,feature_index], h)
			self.__trees.append( (dt.tree_,feature_index) )
			y_ = self.predict(X)
			if self.__verbose and i%5==0:
				print( f"MSE after trees {i+1} : {self.__mse(y_,y)}" )
			
	def predict(self, X):
		"""
		Predict labels for observations using GB model
		
		Parameters
		----------
		X : numpy array, features
		
		Returns
		-------
		y : numpy array, labels
		
		"""
		predictions = np.ones( len(X) ) * self.__f0
		for i in range(1,len(self.__trees)+1):
			root, features = self.__trees[i-1]
			predictions += self.__lr * np.array([ self.__predict_row(row,root) for row in X[:,features] ])
		return predictions
			
	def __predict_row(self,row,node):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict): return self.__predict_row(row,node['left'])
			else: return node['left']
		else:
			if isinstance(node['right'], dict): return self.__predict_row(row,node['right'])
			else: return node['right']
	
	def score(self,X,y):
		"""
		Compute Coefficient of Determinance, r2 score

		r2 score = ESS/TSS = 1 - RSS/TSS = 1 - (y_true - y_pred)^2 / (y_true - y_true.mean())

		Parameters
		----------
		X : numpy array, features

		y : numpy array, labels

		Returns
		-------
		r2_score
		"""
		y_pred = self.predict(X)
		return 1 - np.sum(np.square(y-y_pred))/np.sum(np.square(y-y.mean()))



class GradientBoostingClassifier():
	"""
	Gradient Boosting for regression.

	GB builds an additive model in a forward stage-wise fashion; 
	it allows for the optimization of arbitrary differentiable loss functions. 
	In each stage n_classes_ regression trees are fit on the negative gradient of the 
	binomial or multinomial deviance loss function.
	
	Parameters
	----------
	
	loss : 'str', default 'deviance'
		Loss to be optimized. 'deviance' refers to cross entropy
	regression.
		Currently only binomial deviance is available
	
	learning_rate : float, default 0.1
		learning rate shrinks the contribution of each tree by `learning_rate`
	
	n_estimators : int, default 100
		Number of base estimators to train
	
	criterion : str,  Default 'mse',  ('mse', 'mae', 'std')
		The function to measure the quality of a split.
		'mse' is mean squared error
		'mae' is mean absolute error
		'std' is standard deviation
		
	max_depth : int, default : None i.e. 'inf'
		The maximum depth allowed for each base regressor
	
	min_samples_split : int, default 2
		The minimum number of samples required to split an internal node
		
	max_features : int, float, string or None, (default None)
		The number of features to consider when looking for the best split
		- If int, then consider `max_features` features at each split.
		- If float, then `max_features` is a percentage of n_features
		- If "auto", then `max_features=n_features`.
		- If "sqrt", then `max_features=sqrt(n_features)`.
		- If "log2", then `max_features=log2(n_features)`.
		- If None, then `max_features=n_features`.
		
	verbose : boolean, default False
		Enable verbose output, print loss once in a while.
	
	"""
	def __init__(self, loss='deviance', learning_rate = 0.1, n_estimators=100, criterion='mse', 
				 max_depth=None, min_samples_split=2, max_features=None, verbose=False):
		self.__lr = learning_rate
		self.__n_estimators = n_estimators
		self.__criterion = criterion
		self.__max_depth = max_depth
		self.__min_samples_split = min_samples_split
		self.__max_features = None
		if isinstance(max_features,str):
			self.__max_features = { 
			'auto': lambda x: int(np.sqrt(x)), 'sqrt': lambda x: int(np.sqrt(x)), 
			'log2': lambda x: int(np.log2(x)), 'max_features': lambda x: x  }[max_features]
		elif isinstance(max_features, int):
			self.__max_features = lambda x: max_features
		elif isinstance(max_features, float):
			self.__max_features = lambda x: int(max_features*x)
		else:
			self.__max_features = lambda x: x
			
		self.__n_features = None
		self.__trees = []
		self.__verbose = verbose
		self.__f0 = None
	
	def __binomial_deviance(self,p_pred,y_true):
		return np.sum(-y_true*np.log(p_pred) - (1-y_true)*np.log(1-p_pred))
	
	def __negative_binomial_deviance_gradient(self,p_pred,y_true):
		grad =  -1 * (y_true - p_pred)
		return -1 * grad
	
	def __get_feature_index(self): 
		return np.random.choice( np.arange(0,self.__n_features,1), 
								size=self.__max_features(self.__n_features), replace=False)
	
	def fit(self, X, y):
		"""
		Fit the X and y to estimators
		
		Parameters
		----------
		X : numpy array, independent variables
		
		y : numpy array, target variable
		
		"""
		self.__n_features = X.shape[1]
		p = self.__f0 = max( (y==1).sum(), (y==0).sum()) / len(y)
		if self.__verbose:
			print( f"Binomial Deviance Loss, Accuracy after trees {0} : {self.__binomial_deviance(p,y)}, {self.score(X,y)}" )
		for i in range(0,self.__n_estimators):
			dt = DecisionTreeRegressor(criterion=self.__criterion, 
									   max_depth=self.__max_depth, 
									   min_samples_split=self.__min_samples_split)
			feature_index = self.__get_feature_index()
			h = self.__negative_binomial_deviance_gradient(p,y)
			dt.fit(X[:,feature_index], h)
			self.__trees.append( (dt.tree_,feature_index) )
			p = self.predict_proba(X)[:,1]
			if self.__verbose and (i+1)%5==0:
				print( f"Binomial Deviance Loss, Accuracy after trees {i+1} : {self.__binomial_deviance(p,y)}, {self.score(X,y)}" )
			
	def predict_proba(self,X):
		"""
		Predict probaibilty of each class using all estimators
		
		Parameters
		----------
		X : numpy array, independent variablesss
		
		Returns
		-------
		probability of each class [ n_samples, n_classes ] 
		
		"""
		predictions = np.ones( len(X) ) * self.__f0
		for i in range(1,len(self.__trees)+1):
			root, features = self.__trees[i-1]
			predictions += self.__lr * np.array([ self.__predict_row(row,root) for row in X[:,features] ])
		proba = np.zeros( (len(X),2) )
		proba[:,0] = (1-predictions)
		proba[:,1] = predictions
		return proba
		
		
	def predict(self, X):
		"""
		Predict labels using all estimators
		
		Parameters
		----------
		X : numpy array, independent variables
		
		Returns
		-------
		predicted labels
		
		"""
		proba = self.predict_proba(X)
		return (proba[:,1]>0.5)*1
			
	def __predict_row(self,row,node):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict): return self.__predict_row(row,node['left'])
			else: return node['left']
		else:
			if isinstance(node['right'], dict): return self.__predict_row(row,node['right'])
			else: return node['right']
	
	def score(self,X,y):
		"""
		Calculate accuracy from independent variables
		
		Parameters
		----------
		X : numpy array, independent variables
		
		y : numpy array, dependent variable
		
		Returns
		-------
		accuracy score
		
		"""
		y_pred = self.predict(X)
		return (y_pred==y).sum()/len(y)