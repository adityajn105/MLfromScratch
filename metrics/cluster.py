"""
Author : Aditya Jain
Contact: https://adityajain.me
"""

import numpy as np 

def adjusted_rand_score(labels_true, labels_pred):
	"""
	Rand index adjusted for chance.

	The Rand Index computes a similarity measure between two clusterings
	by considering all pairs of samples and counting pairs that are
	assigned in the same or different clusters in the predicted and
	true clusterings.

	ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

	Note : https://davetang.org/muse/2017/09/21/adjusted-rand-index/
	
	Parameters
	----------

	labels_true : numpy array,  true cluster labels
	label_pred : numpy array, predicted cluster label
	
	Returns
	-------
	adjusted rand index


	"""
	def factorail(n):
		if n<=1: return 1
		return n*factorail(n-1)
	def comb(n,r):
		if n<r: return 0
		return factorail(n)/(factorail(r)*factorail(n-r))

	n_2 = comb( len(labels_true), 2 )
	contigency_matrix = np.zeros( (len(np.unique(labels_true)),len(np.unique(labels_pred))) )
	for i in range(len(contigency_matrix)):
		for j in range(len(contigency_matrix[0])):
			contigency_matrix[i][j] = ((labels_true==i) & (labels_pred==j)).sum()
	rows_sum = np.sum(contigency_matrix, axis=1)
	cols_sum = np.sum(contigency_matrix, axis=0)

	ri = sum([ comb(n,2) for n in contigency_matrix.ravel()])
	rows_comb = sum([ comb(n,2) for n in rows_sum])
	cols_comb = sum([ comb(n,2) for n in cols_sum])

	numerator = (ri - (rows_comb*cols_comb)/n_2)
	denominator = (0.5*(rows_comb+cols_comb)) - ((rows_comb*cols_comb)/n_2)
	return numerator/denominator
