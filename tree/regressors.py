"""
Authors : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np

class DecisionTreeRegressor():
	"""
	Decision Tree Regressor

	Parameters
	----------
	criterion : str, (mse, std, mae), criterion to be optimized for creating tree
	
	Attributes
	----------
	tree_ : dict, dictionary representation of tree
	depth_ : integer, current maximum depth of tree

	"""
	def __init__(self,criterion='mse'):
		self.__root = None
		self.__max_depth = 0
		self.__cost = { 'mse':self.__mse,'std':self.__std,'mae':self.__mae }[criterion]
		self.__X = None
		self.__y = None

	def __std(self,y):
		squared_error = (y-y.mean())**2
		return np.sqrt(  np.sum(squared_error)/len(y)  )

	def __mse(self,y):
		squared_error = (y-y.mean())**2
		return np.sum( squared_error/len(y) )

	def __mae(self,y): return np.sum(abs(y-y.mean())/len(y))

	def __computeCost(self,groups, y):
		n_instances = len(groups[0])+len(groups[1])  # count of all samples
		weighted_cost = 0.0 # sum weighted Gini index for each group
		for indexes in groups:
			size = len(indexes)
			# avoid divide by zero
			if size == 0: continue
			weighted_cost +=  self.__cost(y[indexes]) * (size/n_instances)
		return weighted_cost

	def __get_split(self,X,y):
		b_index, b_value, b_cost, b_groups = float('inf'), float('inf'), float('inf'), None
		for col_ind in range(X.shape[1]): #no of features
			for val in np.unique(X[:,col_ind]): #for each unique value in each of the features

				#left_index indexes lower than val for feature, right_index indexes greater that val for feature
				left_index = np.reshape( np.argwhere(X[:,col_ind]<val), (-1,) )
				right_index = np.reshape( np.argwhere(X[:,col_ind]>=val), (-1,) )

				#find gini index
				cost = self.__computeCost((left_index,right_index), y)
				if cost < b_cost:
					b_index, b_value, b_cost, b_groups = col_ind, val, cost, (left_index, right_index)
		return {'index':b_index, 'value':b_value, 'groups':b_groups}

	def __to_terminal(self,y): return y.mean()

	def __split(self,node, X, y, max_depth, min_samples_split, depth):
		self.__max_depth = max(depth,self.__max_depth)
		left, right = node.pop('groups')

		# check for a no split
		if len(left)==0 or len(right)==0:
			node['left'] = node['right'] = self.__to_terminal(y[np.append(left,right)])
			return

		# check for max depth
		if depth >= max_depth:
			node['left'], node['right'] = self.__to_terminal(y[left]), self.__to_terminal(y[right])
			return

		# process left child
		if len(left) <= min_samples_split:
			node['left'] = self.__to_terminal(y[left])
		else:
			node['left'] = self.__get_split(X[left],y[left])
			self.__split(node['left'], X[left], y[left], max_depth, min_samples_split, depth+1)

		# process right child
		if len(right) <= min_samples_split:
			node['right'] = self.__to_terminal(y[right])
		else:
			node['right'] = self.__get_split(X[right],y[right])
			self.__split(node['right'],X[right],y[right], max_depth, min_samples_split, depth+1)

	def fit(self, X, y, max_depth=None, min_samples_split=2):
		"""
		Fit X using y by optimizing splits costs using given criterion

		Parameters
		----------
		X : 2D numpy array, independent variables
		y : 1D numpy array, dependent variable
		max_depth : integer (Default 'inf'), maximum depth allowed in the decision tree
		min_samples_split : integer (Default 2), minimum nodes to consider before splitting
		
		"""
		self.__X, self.__y, max_depth = X, y, float('inf') if max_depth==None else max_depth
		self.__root = self.__get_split(X,y)
		self.__split(self.__root, X, y, max_depth, min_samples_split,1)

	def __predict_row(self,row,node):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict): return self.__predict_row(row,node['left'])
			else: return node['left']
		else:
			if isinstance(node['right'], dict): return self.__predict_row(row,node['right'])
			else: return node['right']

	def predict(self,rows): 
		"""
		Predict dependent variable
		
		Parameters
		----------
		X : numpy array, independent variables

		Output
		------
		precicted values
		"""
		return np.array( [self.__predict_row(row,self.__root) for row in rows] )

	def score(self,X,y):
		"""
		Computer Coefficient of Determination (rsquare)

		Parameters
		----------
		X : 2D numpy array, independent variables
		y : numpy array, dependent variables

		Output
		------
		r2 values
		"""
		y_pred = self.predict(X)
		return 1-( np.sum( (y-y_pred)**2 )/np.sum( (y-y.mean())**2 ) )

	@property
	def depth_(self): return self.__max_depth

	@property
	def tree_(self): return self.__root