"""
Author : Aditya Jain
Contact: https://adityajain.me
"""

import numpy as np

class KMeans():
	"""
	Kmeans Clustering using euclidean distance

	Parameters
	----------
	n_clusters : interger (Default 8), Number of clusters

	Attributes
	----------
	clusters_centers_ : array, [n_clusters, n_features],  Coordinates of cluster centers

	inertia_ : float, Sum of squared distances of samples to their closest cluster center.

	"""
	def __init__(self, n_clusters=8):
		self.__n_clusters = n_clusters
		self.__centers = None
		self.__n_features = None
		self.__inertia = None

	def __distance(self,X1,X2 ):
		return np.sqrt( np.sum( (X1-X2)**2, axis=1) )

	def __initialize_centers(self,X):
		centers = np.empty( (self.__n_clusters,0) )
		for i in range(self.__n_features):
			centers = np.c_[ centers, np.random.randint( X[:,i].min(), X[:,i].max(), size = self.__n_clusters ) ]
		return [ list(center) for center in centers[:,1:] ]

	def __areCentersChanged(self, old):
		for n,o in zip(self.__centers, old):
			if n!=o: return True
		return False

	def __getNewMeans(self,X):
		newCluster = np.empty((len(X),0))
		#find distance from each cluster mean
		for i in range(self.__n_clusters):
			newCluster = np.c_[ newCluster, self.__distance( X, np.array(self.__centers[i]) ) ]
		#get cluster name
		newCluster = np.argmin(newCluster,axis=1)
		new_means = []
		#get new means
		for i in range(self.__n_clusters):
			newPoints = X_train[newCluster==i]
			if len(newPoints)==0: new_means.append( self.__centers[i] )
			else: new_means.append( tuple( newPoints.mean(axis=0) )  )
		return new_means

	def fit(self, X, verbose=False):
		"""
		Fit the data(X) to get the best clusters centers
		
		Parameters
		----------
		X : numpy array of features

		verbose : boolean, print details while finding clusters center

		"""
		self.__n_features = X.shape[1]
		self.__centers = self.__initialize_centers(X)
		old_centers = [None]*len(self.__centers)
		i=1
		while self.__areCentersChanged( old_centers ):
			if verbose: print(f'Iteration {i}');i+=1
			old_centers = self.__centers
			self.__centers = self.__getNewMeans(X)

		distances = np.empty((len(X),0))
		for i in range(self.__n_clusters):
			distances = np.c_[ distances, self.__distance( X, np.array(self.__centers[i]) )]
		self.__inertia = np.sum( np.min(distances, axis=1)**2 )

	def predict(self,X):
		"""
		Predict cluster labels

		Parameters
		----------
		X : numpy array of features
		
		Returns
		-------
		cluster labels

		"""
		distances = np.empty((len(X),0))
		for i in range(self.__n_clusters):
			distances = np.c_[ distances, self.__distance( X, np.array(self.__centers[i]) )]
		return np.argmin(distances,axis=1)

	@property
	def clusters_centers_(self): return self.__centers

	@property
	def inertia_(self): return self.__inertia