"""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np 


class DBSCAN():
	"""
	DBSCAN - Density-Based Spatial Clustering of Applications with Noise. 
	Finds core samples of high density and expands clusters from them. 
	Good for data which contains clusters of similar density.
	
	Parameters
	----------
	eps : float, optional
		The maximum distance between two samples for them to be considered
		as in the same neighborhood.

	min_samples : int, optional
		The number of samples (or total weight) in a neighborhood for a point
		to be considered as a core point. This includes the point itself.

	metric : string, default : 'euclidean', ('euclidean','minkowski')
		The metric to use when calculating distance between instances in a
		feature array. 
		
	p : float, default : 2, 
		The power of the Minkowski metric to be used to calculate distance
		between points.

	Attributes
	----------
	labels_ : labels of cluster of feature array

	n_clusters_ : number of clusters found

	core_sample_indices_ : indices of core samples

	componenets_ : copy of core samples
	
	"""
	def __init__(self,eps=0.5,min_samples=5,metric='euclidean',p=2):
		self.__eps = eps
		self.__min_samples = min_samples
		self.__metric = { 'euclidean': self.__euclidean, 'minkowski':self.__minkowski }[metric]
		self.__p = p
		self.__clusters = []
		self.__labels = []
		self.__core_sample_indices = []
		self.__X = None
		
	def __euclidean(self,X1,X2): return  np.sqrt(np.sum((X1-X2)**2, axis=1))
	def __minkowski(self,X1,X2): return  np.power(np.sum(np.power(np.abs(X1-X2),self.__p),axis=1),1/self.__p)
	
	def __get_cores_index(self,X):
		cores = []
		for sample in X:
			cores.append((self.__metric(X,sample) < self.__eps).sum() >= self.__min_samples)
		return np.where(cores)[0]

	def __compute_core_labels(self,X):
		cluster_index = np.ones( (len(X),), dtype=np.int )*-1
		curr_index = 0
		for i in self.__core_sample_indices:
			if cluster_index[i]!=-1: continue
			self.__clusters.append( [i]  )
			cluster_index[i] = curr_index;
			new = { i }
			while len(new)>0:
				pt = X[new.pop()]
				indexes = np.where( ((self.__metric(X,pt) < self.__eps) & (cluster_index==-1)) )[0]
				for index in indexes:
					if index not in self.__core_sample_indices: continue
					new.add(index)
					self.__clusters [ curr_index ].append( index )
					cluster_index[index] = curr_index
			curr_index+=1
		self.__clusters = [ np.array([  X[index] for index in cluster ]) for cluster in self.__clusters  ]

	
	def fit(self,X):
		"""
		Perform DBSCAN clustering from features or distance matrix.

		Parameters
		----------
		X : array, feature array
		
		"""
		self.__X = X
		self.__core_sample_indices = self.__get_cores_index(X)
		self.__compute_core_labels(X)
		self.__labels = self.predict(X)
		
	def predict(self,X):
		"""
		Predict cluster labels of new samples using cores
		
		Parameters
		----------
		X : numpy array, feature array
		
		Returns
		-------
		cluster labels
		"""
		labels_ = []
		for index in range( len(X) ):
			dists = []
			for cluster in self.__clusters:
				dists.append( (self.__metric(cluster, X[index])< self.__eps).sum() )
			cls = np.argmax(dists)
			labels_.append( cls if dists[cls]!=0 else -1 )
		return np.array(labels_)
	
	def fit_predict(self,X):
		"""
		Perform DBSCAN clustering and predict cluster labels
		
		Parameters
		----------
		X : numpy array, feature array
		
		Returns
		-------
		cluster labels
		"""
		self.fit(X)
		return self.__labels
	
	@property
	def labels_(self): return self.__labels
	
	@property
	def n_clusters_(self): return len(self.__clusters)
	
	@property
	def core_sample_indices_(self): return self.__core_sample_indices
	
	@property
	def componenets_(self): return self.__X[self.core_sample_indices_]
