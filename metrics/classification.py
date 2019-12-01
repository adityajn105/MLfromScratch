""""
Author : Aditya Jain
Contact : https://adityajain.me
"""

import numpy as np 
import matplotlib.pyplot as plt

def accuracy_score(y_true,y_pred):
	"""
	Calculate Accuracy

	Parameters
	----------
	y_true : true labels
	y_pred : predicted labels

	Returns
	-------
	accuracy
	"""
	return (y_true==y_pred).sum()/len(y_true)


def confusion_matrix(y_true,y_pred):
	"""
	Calculate confusion matrix
	tn | fp
	-------
	fn | tp
	
	Note: this implementation is restricted to the binary classification task.

	Parameters
	----------
	y_true : true labels
	y_pred : predicted labels

	Returns
	-------
	confusion matrix
	"""
	trues,falses,pos,neg  = y_true==y_pred, y_true!=y_pred, y_true==1, y_true==0
	tp, tn = (trues & pos).sum(), (trues & neg).sum()
	fp, fn = (falses & neg).sum(), (falses & pos).sum()
	return np.array( [[tn,fp],[fn,tp]] )

def roc_curve(y_true,y_score,plot=False):
	"""
	Compute Receiver operating characteristic (ROC) and return fpr,tpr or plot curve
	
	Note: this implementation is restricted & neg).sum()
	return np.array( [[tn,fn],[fp,tp]] )to the binary classification task.

	Parameters
	----------
	y_true : true labels
	y_score : predicted probabilities
	plot: boolean (Default False), plot a matplotlib roc curve

	Returns
	-------
	fpr,tpr,thresholds

	"""
	thres = np.sort(np.unique(y_score))[::-1]
	fpr,tpr = [],[]
	for th in thres:
		y_pred = y_score>=th
		positives, falses, trues = y_pred==1, y_pred!=y_true, y_pred==y_true
		tpr.append((trues & positives).sum() /  y_true.sum())
		fpr.append((falses & positives).sum() / (y_true==0).sum())
	if plot:
		plt.plot(fpr,tpr)
		plt.plot([0,1],[0,1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
	return np.array(fpr), np.array(tpr), thres


def roc_auc_score(y_true, y_score):
	"""
	Compute Area under Receiver operating characteristic (ROC)
	
	Note: this implementation is restricted to the binary classification task.

	Parameters
	----------
	y_true : true labels
	y_score : predicted probabilities

	Returns
	-------
	area under curve

	"""
	fpr,tpr,_ = roc_curve(y_true,y_score)
	fpr_last,tpr_last, area = fpr[0],tpr[0],0
	for i in range(1,len(fpr)):
		fpr_curr, tpr_curr = fpr[i],tpr[i]
		tri_area = 0.5 * (fpr_curr-fpr_last) * (tpr_curr-tpr_last)
		rect_area = (fpr_curr-fpr_last) * (tpr_last-0)
		area, fpr_last,tpr_last = area+tri_area+rect_area, fpr_curr, tpr_curr
	return area


def precision_score(y_true,y_pred):
	"""
	Compute Precision

	Precision = tp/(tp+fp)

	Parameters
	----------
	y_true : true labels
	y_pred : predicted labels

	Returns
	-------
	precision score
	"""
	tn,fp,fn,tp = tuple(confusion_matrix(y_true,y_pred).ravel())
	return tp/(tp+fp)

def recall_score(y_true,y_pred):
	"""
	Compute Recall

	Recall = tp/(tp+fn)

	Parameters
	----------
	y_true : true labels
	y_pred : predicted labels

	Returns
	-------
	recall score
	"""
	tn,fp,fn,tp = tuple(confusion_matrix(y_true,y_pred).ravel())
	return tp/(tp+fn)

def sensitivity_score(y_true,y_pred):
	"""
	Compute Sensitivity

	Sensitivity = tp/(tp+fn)

	Parameters
	----------
	y_true : true labels
	y_pred : predicted labels

	Returns
	-------
	sensitivity score
	"""
	tn,fp,fn,tp = tuple(confusion_matrix(y_true,y_pred).ravel())
	return tp/(tp+fn)

def specificity_score(y_true,y_pred):
	"""
	Compute Specificity

	Specificity = tn/(tn+fn)

	Parameters
	----------
	y_true : true labels
	y_pred : predicted labels

	Returns
	-------
	specificity score
	"""
	tn,fp,fn,tp = tuple(confusion_matrix(y_true,y_pred).ravel())
	return tn/(tn+fp)


def f1_score(y_true,y_pred):
	"""
	Compute f1_score

	f1_score = harmonic mean of precision and recall
	f1_score = 2 / ( 1/precision + 1/recall ) = (2 * precision * recall) / ( precision + recall )

	Note: this implementation is restricted to the binary classification task.

	Parameters
	----------
	y_true : true labels
	y_score : predicted probabilities

	Returns
	-------
	f1 score
	"""
	precision = precision_score(y_true,y_pred)
	recall = recall_score(y_true,y_pred)
	f1_score = 2*(precision*recall)/(precision+recall)
	return f1_score