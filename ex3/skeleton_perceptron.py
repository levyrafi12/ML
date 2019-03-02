#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from skeleton_sgd import helper
from skeleton_sgd import score

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def perceptron(data, labels, T):
	"""
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
	"""
	n = data.shape[0]
	d = data.shape[1]
	w = np.zeros(d)

	for t in range(T):
		idx = np.random.RandomState(t).permutation(n)
		for j in idx:
			y_pred = np.sign(np.dot(w, data[j]) * 2 - 1)
			if y_pred != labels[j]:
				w += labels[j] * data[j]	
	return w

#################################

# Place for additional code

#################################

if __name__ == '__main__':
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

	w = perceptron(train_data, train_labels, 50)
	acc = score(w, train_data, train_labels)
	print("train accuracy {}".format(acc))

	acc = score(w, validation_data, validation_labels)
	print("validation accuracy {}".format(acc))

	acc = score(w, test_data, test_labels)
	print("test accuracy {}".format(acc))

