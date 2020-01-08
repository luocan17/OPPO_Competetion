import numpy as np
import sklearn.metrics as metrics

def gini(actual, pred):
	assert (len(actual) == len(pred))
	all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
	all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))] # np.lexsort((b,a)), sort by a then by b, descending
	totalLosses = all[:, 0].sum()
	giniSum = all[:, 0].cumsum().sum() / totalLosses

	giniSum -= (len(actual) + 1) / 2.
	return giniSum / len(actual)

def gini_norm(actual, pred):
	return gini(actual, pred) / gini(actual, actual)

def accuracy_score(actual, pred):
	assert (len(actual) == len(pred))
	return metrics.accuracy_score(actual, pred)
def precision_score(actual, pred):
	assert (len(actual) == len(pred))
	return metrics.precision_score(actual, pred)
def recall_score(actual, pred):
	assert (len(actual) == len(pred))
	return metrics.recall_score(actual, pred)
def f1_score(actual, pred):
	assert (len(actual) == len(pred))
	return metrics.f1_score(actual, pred)

