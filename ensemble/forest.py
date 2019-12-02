"""
Authors : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np 
from ..tree import DecisionTreeClassifier
from ..tree import DecisionTreeRegressor

class RandomForestClassifier():
	"""
	Random Forest fits number of decision tree on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
	The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True.
	
	Parameter
	---------
	n_estimators : integer (Default 50), number of trees in forest
	
	max_depth : integer (Default 'inf'), maximum depth allowed for each tree
	
	min_samples_split : integer (Default 2), The minimum number of samples required to split an internal node
	
	max_features : ( 'auto', 'sqrt', 'log2', 'max_features' ) 
		The number of features to consider when looking for the best split:
		'auto' is same as 'sqrt'
		'sqrt' is sqrt(number of features)
		'log2' is log2(number of features)
		'max_features' is all features
	
	bootstrap : If False, the whole datset is used to build each tree.
	
	random_state : random seed
	
	"""
	def __init__(self, n_estimators=50, max_depth = None, min_samples_split=2, max_features="auto", 
				 bootstrap=True, random_state=None):
		np.random.seed( random_state if random_state!=None else np.random.randint(100)  )
		self.__n_estimators = n_estimators
		self.__max_depth = float('inf') if max_depth==None else max_depth
		self.__min_samples_split = min_samples_split
		self.__max_features = { 
			'auto': lambda x: int(np.sqrt(x)), 'sqrt': lambda x: int(np.sqrt(x)), 
			'log2': lambda x: int(np.log2(x)), 'max_features': lambda x: x  }[max_features]
		self.__bootstrap = bootstrap
		self.__n_samples = None
		self.__n_features = None
		self.__n_classes = None
		self.__trees = [  ]
		
	def __bootstrapX(self,X):
		indexes = np.random.choice( np.arange(0,len(X),1), size=self.__n_samples, replace=self.__bootstrap )
		return X[indexes,:]
	
	def __get_feature_index(self): 
		return np.random.choice( np.arange(0,self.__n_features,1), 
								size=self.__max_features(self.__n_features), replace=False)
	
	def fit(self,X,y):
		"""
		Fit the X and y to estimators
		
		Parameters
		----------
		X : numpy array, independent variables
		
		y : numpy array, target variable
		
		"""
		self.__n_samples, self.__n_features = X.shape
		self.__n_classes = len(np.unique(y))
		X_y = np.c_[X,y]
		for _ in range(self.__n_estimators):
			dt = DecisionTreeClassifier( max_depth=self.__max_depth, 
										min_samples_split=self.__min_samples_split, 
									   n_classes = self.__n_classes)
			data = self.__bootstrapX(X_y)
			features = self.__get_feature_index()
			dt.fit( data[:,features], data[:,-1] )
			self.__trees.append( (dt.tree_, features) )
	
	def predict(self,X):
		"""
		Predict labels using all estimators
		
		Parameters
		----------
		X : numpy array, independent variables
		
		Returns
		-------
		predicted labels
		
		"""
		return np.argmax( self.predict_proba(X), axis=1 )
	
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
		probs = np.zeros( (len(X),self.__n_classes) )
		for root, features in self.__trees:
			probs += np.array([ self.__predict_row(row,root)[1] for row in X[:,features] ])
		return probs/self.__n_estimators
		
	def __predict_row(self,row,node):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict):  return self.__predict_row(row,node['left'])
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
		return (y==self.predict(X)).sum()/len(y)


class RandomForestRegressor():
	"""
	Random Forest fits number of decision tree on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
	The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True.
	
	Parameter
	---------
	n_estimators : integer (Default 50), number of trees in forest
	
	criterion : ('mse', 'mae', 'std') ( Default 'mse' )
		The function to measure the quality of a split.
		'mse' is mean squared error
		'mae' is mean absolute error
		'std' is standard deviation
		
	max_depth : integer (Default 'inf'), maximum depth allowed for each tree
	
	min_samples_split : integer (Default 2), The minimum number of samples required to split an internal node
   
	max_features : ( 'auto', 'sqrt', 'log2', 'max_features' ) ( Default 'auto' )
		The number of features to consider when looking for the best split:
		'auto' is same as 'sqrt'
		'sqrt' is sqrt(number of features)
		'log2' is log2(number of features)
		'max_features' is all features
	
	bootstrap : If False, the whole datset is used to build each tree.
   
	random_state : random seed
	
	"""
	def __init__(self,  n_estimators=10, criterion='mse', max_depth = None, min_samples_split=2, max_features="auto", 
				 bootstrap=True, random_state=None):
		np.random.seed( random_state if random_state!=None else np.random.randint(100)  )
		self.__n_estimators = n_estimators
		self.__criterion = criterion
		self.__max_depth = float('inf') if max_depth==None else max_depth
		self.__min_samples_split = min_samples_split
		self.__max_features = { 
			'auto': lambda x: int(np.sqrt(x))+1, 'sqrt': lambda x: int(np.sqrt(x))+1, 
			'log2': lambda x: int(np.log2(x))+1, 'max_features': lambda x: x  }[max_features]
		self.__bootstrap = bootstrap
		self.__n_samples = None
		self.__n_features = None
		self.__trees = [  ]
		
	def __bootstrapX(self,X):
		indexes = np.random.choice( np.arange(0,len(X),1), size=self.__n_samples, replace=self.__bootstrap )
		return X[indexes,:]
	
	def __get_feature_index(self): 
		return np.random.choice( np.arange(0,self.__n_features,1), 
								size=self.__max_features(self.__n_features), replace=False)
	
	def fit(self,X,y):
		"""
		Fit the X and y to estimators
		
		Parameters
		----------
		
		X : numpy array, independent variables
		
		y : numpy array, target variable
		
		"""
		self.__n_samples, self.__n_features = X.shape
		X_y = np.c_[X,y]
		for _ in range(self.__n_estimators):
			dt = DecisionTreeRegressor( criterion=self.__criterion, max_depth=self.__max_depth, 
										min_samples_split=self.__min_samples_split)
			data = self.__bootstrapX(X_y)
			features = self.__get_feature_index()
			dt.fit( data[:,features], data[:,-1] )
			self.__trees.append( (dt.tree_, features) )
	
	def predict(self,X):
		"""
		Predict values using all estimators
		
		Parameters
		----------
		X : numpy array, independent variables
		
		Returns
		-------
		predicted values
		
		"""
		predictions = np.zeros( (len(X)) )
		for root, features in self.__trees:
			predictions += np.array([ self.__predict_row(row,root) for row in X[:,features] ])
		return predictions/self.__n_estimators
			
	def __predict_row(self,row,node):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict): return self.__predict_row(row,node['left'])
			else: return node['left']
		else:
			if isinstance(node['right'], dict): return self.__predict_row(row,node['right'])
			else: return node['right']
			
	def score(self,X,y):
		"""
		Computer Coefficient of Determination (rsquare)
		
		Parameters
		----------
		X : 2D numpy array, independent variables
		
		y : numpy array, dependent variables
	   
		Returns
		-------
		r2 values
		
		"""
		y_pred = self.predict(X)
		return 1-( np.sum( (y-y_pred)**2 )/np.sum( (y-y.mean())**2 ) )