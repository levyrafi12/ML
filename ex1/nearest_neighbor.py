from sklearn.datasets import fetch_mldata
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

mnist = fetch_mldata('MNIST original', data_home='.')
data = mnist['data'] # shape = 2 dimensional mat: num examples * 784
labels = mnist['target']

# training and test set of images as follows:
import numpy.random
# sampling 11,000 indices in range[0, 70,000)
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int) # 10,000 training data
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int) # 1,000 test data
test_labels = labels[idx[10000:]]

# Write a function that accepts as input: (i) a set of images; (ii) a vector of labels,
# corresponding to the images (iii) a query image; and (iv) a number k. The function will
# implement the k-NN algorithm to return a prediction of the query image, given the given
# label set of images.  The function will use the k nearest neighbors, using the Euclidean
# L2 metric. In case of a tie between the k labels of neighbors, it will choose an arbitrary
# option.

def knn(mat, labels, vec, k):
	# euclid_mat = np.apply_along_axis(lambda x: np.sqrt(np.dot((x - vec),(x - vec))), 1, mat)
	euclid_mat = np.sum((mat - vec) * (mat - vec), axis=1)
	indices = np.argsort(euclid_mat)
	nearest_idx = indices[:k]
	nearest_labels = [labels[i] for i in nearest_idx]

	c = Counter(nearest_labels) # a dictionary of keys and their counts
	return c.most_common()[0][0]

def calc_loss_given_k(k):
	print("k {}".format(k))
	loss = 0
	n = len(test[:100])
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(train[:1000], train_labels[:1000])
	for i in range(n):
		# pred_label = knn(train[:1000], train_labels[:1000], test[i], k)
		pred_label = neigh.predict(test[i])
		# if pred_label != test_labels[i]:
		# print("k {}, test label {}, predicted label {}".format(k, test_labels[i], pred_label))
		loss += (pred_label != test_labels[i])
	return loss / n

def visualize_data(train_data, labels):
	X = train_data
	y = labels

	pca = PCA(n_components=2)

	pca_result = pca.fit_transform(X)

	plt.scatter(x=pca_result[:, 0], y = pca_result[:, 1], c=y, cmap=plt.cm.get_cmap('Paired'))

	plt.show()

if __name__ == '__main__':
	# loss = calc_loss_given_k(10)
	# print("loss {}".format(loss))
	visualize_data(train[:1000], train_labels[:1000])
	acc_vals = []
	k_vals = []

	for k in range(100):
		loss = calc_loss_given_k(k + 1)
		acc_vals.append(1 - loss)
		k_vals.append(k)

	title = 'acc_as_func_of_k'
	plt.title(title)
	plt.plot(k_vals, acc_vals, 'r--')
	plt.ylim(ymax = 1)
	plt.ylabel('accuracy')
	plt.xlabel('k')
	plt.savefig(title)
	plt.show()


	