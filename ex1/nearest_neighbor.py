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

def knn(mat, labels, vecs, k):
	n = vecs.shape[0] # num test images
	pred_labels = np.zeros(n)
	for i in range(n):
		vec = vecs[i]
		# euclid_mat = np.apply_along_axis(lambda x: np.sqrt(np.dot((x - vec),(x - vec))), 1, mat)
		euclid_mat = np.sum((mat - vec) * (mat - vec), axis=1) # sum is performed along each row
		indices = np.argsort(euclid_mat)
		nearest_idx = indices[:k]
		nearest_labels = [labels[i] for i in nearest_idx]

		c = Counter(nearest_labels) # a dictionary of keys and their counts
		pred_labels[i] = c.most_common(1)[0][0]

	return pred_labels

def calc_loss_given_k(X, y, k):
	loss = 0
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(X, y)
	# pred_labels = knn(X, y, test, k)
	pred_labels = neigh.predict(test)
	loss += np.sum(pred_labels != test_labels)
	return loss / test.shape[0]

def visualize_data(X, y):
	pca = PCA(n_components=2)
	pca_result = pca.fit_transform(X)

	plt.scatter(x=pca_result[:, 0], y = pca_result[:, 1], c=y, cmap=plt.cm.get_cmap('Paired'))
	plt.show()
	plt.close()

def iterate_over_k(from_k, to_k, step):
	acc_vals = []
	k_vals = []

	for k in range(from_k, to_k + step):
		# print("k {}".format(k))
		loss = calc_loss_given_k(train[:1000], train_labels[:1000], k)
		acc_vals.append(1 - loss)
		k_vals.append(k)

	best_acc, best_k = sorted(zip(acc_vals, k_vals), reverse=True)[0]
	print("Best K is {} with accuracy {}".format(best_k, best_acc))

	title = 'MNIST - K Nearest Neighbor'
	plt.text(0.5, 0.9, 'accuracy as a functioben of k', transform=plt.gca().transAxes, ha='center')
	plt.title(title)
	plt.plot(k_vals, acc_vals, 'r-')
	plt.ylim(ymax = max(acc_vals))
	plt.ylabel('Accuracy')
	plt.xlabel('k-NN')
	plt.savefig(title)
	# plt.show()
	plt.close()

def iterate_over_n(from_n, to_n, step):
	acc_vals = []
	n_vals = []
	
	for n in range(from_n, to_n + step, step):
		# print("n {}".format(n))
		loss = calc_loss_given_k(train[:n], train_labels[:n], 1)
		acc_vals.append(1 - loss)
		n_vals.append(n)

	best_acc, best_n = sorted(zip(acc_vals, n_vals), reverse=True)[0]
	print("Best n is {} with accuracy {}".format(best_n, best_acc))

	title = 'MNIST - 1 Nearest Neighbor'
	plt.text(0.5, 0.5, 'accuracy as a function of n', transform=plt.gca().transAxes, ha='center')
	plt.title(title)
	plt.plot(n_vals, acc_vals, 'b-')
	plt.ylim(ymax = max(acc_vals))
	plt.ylabel('Accuracy')
	plt.xlabel('samples (n)')
	plt.savefig(title)
	# plt.show()
	plt.close()

if __name__ == '__main__':
	# loss = calc_loss_given_k(10)
	# print("loss {} for k 10".format(loss))
	# visualize_data(train[:1000], train_labels[:1000])
	iterate_over_k(1, 100, 1)
	iterate_over_n(100, 5000, 100)

	


	