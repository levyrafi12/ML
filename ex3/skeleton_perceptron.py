#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from skeleton_sgd import helper_unscale
from skeleton_sgd import score

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def perceptron(data, labels, t=0):
	"""
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
	"""
	n = data.shape[0]
	d = data.shape[1]
	w = np.zeros(d)

	# clf = Perceptron(random_state=t)
	# clf.fit(data, labels)

	idx = np.random.RandomState(t).permutation(n)
	for j in idx:
		y_pred = np.sign(np.dot(w, data[j, :]) * 2 - 1)
		if y_pred != labels[j]:
			w += labels[j] * data[j, :]	
	return w

def iter_multiple_samples(train_data, train_labels, test_data, test_labels):
	multiple_samples = [5,50,100,500,1000,5000]
	T = 100

	for n in multiple_samples:
		acc = np.zeros(T)
		for t in range(T):
			w = perceptron(train_data[:n,:], train_labels[:n], t)
			acc[t] = score(w, test_data, test_labels)

		acc = np.sort(acc)
	
		print("average test accuracy {}, n {}".format(np.sum(acc) / T, n))
		print("5% of test accuracy {}, n {}".format(acc[5], n))
		print("95% of test accuracy {}, n {}".format(acc[95], n))

def normalize(mat):
	return np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1,mat)

def helper_normalize():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_unscale()
	train_data_normalized = normalize(train_data)
	validation_data_normalized = normalize(validation_data)
	test_data_normalized = normalize(test_data)

	return train_data_normalized, train_labels, validation_data_normalized, \
		validation_labels, test_data_normalized, test_labels

#################################

# Place for additional code

#################################

if __name__ == '__main__':
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_normalize()
	iter_multiple_samples(train_data, train_labels, test_data, test_labels)

	w = perceptron(train_data, train_labels)
	plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
	plt.show()
