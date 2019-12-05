"""
Author : Aditya Jain
Contact : https://adityajain.me
"""


import numpy as np
class AgglomerativeClustering():
	"""
	Agglomerative Clustering

	Recursively merges the pair of clusters that minimally increases a given linkage distance.
	
	Parameters
	----------
	n_clusters : int, default=2, 
		The number of clusters to find.

	affinity : string or callable, default: "euclidean", 
		Metric used to compute the linkage. Currently "euclidean" is available.
			
	linkage : {"ward", "complete", "average"}, optional, default: "ward"
		Which linkage criterion to use. The linkage criterion determines which
		distance to use between sets of observation. The algorithm will merge
		the pairs of cluster that minimize this criterion.
	
	Attributes
	----------
	labels_ : array [n_samples]
		cluster labels for each point
	
	"""
	def __init__(self, n_clusters=2, affinity='euclidean', linkage='ward'):
		self.__n_clusters = n_clusters
		self.__proximity = []
		self.__affinity = { 'euclidean': self.__euclidean }[affinity]
		self.__linkage = { 'complete': self.__max_distance, 'average': self.__average_distance, 
						  'ward':self.__ward_distance}[linkage]
		self.__clusters = None
		self.__labels_ = None
	
	def __euclidean(self, X1, X2 ): return np.sqrt( np.sum( (X1-X2)**2, axis=1 )  )
	
	def __max_distance(self,distances, c1 ,c2): return max(distances)
	def __average_distance(self,distances, c1, c2): return np.sum(distances)/(c1*c2)
	def __ward_distance(self,distances, c1, c2 ): return np.sum( np.square(distances) )/(c1*c2)
	
	def __distances(self, g1, g2 ):
		all_distances = np.array([])
		for row in g1:
			all_distances = np.append( all_distances, self.__affinity(g2,row) )
		return all_distances
	
	def __computeProximityMatrix(self,n_clusters):
		self.__proximity = np.ones( (n_clusters,n_clusters) )*float('inf')
		for i in range(n_clusters-1):
			for j in range(i+1,n_clusters):
				g1, g2 = self.__clusters[i], self.__clusters[j]
				proximity = self.__linkage( self.__distances(g1,g2), len(g1), len(g2) )
				self.__proximity[i][j] = self.__proximity[j][i] = proximity
	
	def __findMinIndex(self):
		minimum, index = float('inf'), (None,None)
		size = len(self.__proximity)
		for i in range(size):
			j = np.argmin( self.__proximity[i] )
			if minimum > self.__proximity[i][j]: 
				minimum = self.__proximity[i][j]
				index = (i,j)
		return index
	
	def __updateProximityMatrixAndClusters(self,i,j,n):
		i,j = min(i,j),max(i,j)
		new_cluster = np.append( self.__clusters[i], self.__clusters[j], axis=0)
		self.__proximity = np.delete(self.__proximity,[i,j],axis=0)
		self.__proximity = np.delete(self.__proximity,[i,j],axis=1)
		self.__clusters = self.__clusters[:i]+self.__clusters[i+1: j]+self.__clusters[j+1:]+[new_cluster]
		infs = np.ones( (n-2,) )*float('inf')
		self.__proximity = np.c_[ self.__proximity, infs ]
		infs = np.ones( (1,n-1) )*float('inf')
		self.__proximity = np.append(self.__proximity, infs, axis=0)
		for i in range(n-2):
			g = self.__clusters[i]
			proximity = self.__linkage( self.__distances(  new_cluster, g ), len(new_cluster), len(g) )
			self.__proximity[i][n-2] = self.__proximity[n-2][i] = proximity
	
	def fit(self,X):
		"""
		Fit the hierarchical clustering on the data

		Parameters
		----------
		X : array-like, observations
		
		"""
		self.__clusters = list( np.expand_dims( X_train, axis=1 ) )
		n_clusters = len(self.__clusters)
		self.__computeProximityMatrix(n_clusters)
		while n_clusters != self.__n_clusters:
			i,j = self.__findMinIndex()
			self.__updateProximityMatrixAndClusters(i,j,n_clusters)
			n_clusters = len(self.__clusters)
		self.__labels_ = self.predict(X)
	
	def predict(self,X):
		"""
		Predict cluster labels for observations

		Parameters
		----------
		X : array-like, observations
		"""
		distances = self.__predict_distances(X)
		return np.argmin(distances, axis=1)
	
	def __predict_distances(self,X):
		X = np.expand_dims( X, axis=1 )
		distances = []
		for sample in X:
			sample_dist = []
			for i in range(self.__n_clusters):
				proximity = self.__linkage( self.__distances(  sample, self.__clusters[i] ), 
										   1, len(self.__clusters[i]) )
				sample_dist.append(proximity)
			distances.append(sample_dist)
		return np.array(distances)
	
	def fit_predict(self,X):
		"""
		Fit the hierarchical clustering on the data and predict cluster labels

		Parameters
		----------
		X : array-like, observations
		
		"""
		self.fit(X)
		return self.__labels_
	
	@property
	def label_(self): return self.__labels_