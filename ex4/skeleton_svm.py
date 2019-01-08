#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
	X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
	return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
	plt.clf()

	# plot the data points
	plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

	# plot the decision function
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# create grid to evaluate model
	xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
	yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
	YY, XX = np.meshgrid(yy, xx)
	xy = np.vstack([XX.ravel(), YY.ravel()]).T
	Z = clf.decision_function(xy).reshape(XX.shape)

	# plot decision boundary and margins
	ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
			linestyles=['--', '-', '--'])


def train_kernel(X_train, y_train, kernel_f):
	clf = svm.SVC(C=1000, kernel=kernel_f)
	if kernel_f == 'poly':
		clf.set_params(degree=2)

	clf.fit(X_train, y_train)
	create_plot(X_train, y_train, clf)
	plt.show()
	return clf

def train_three_kernels(X_train, y_train, X_val, y_val):
	"""
	Returns: np.ndarray of shape (3,2) :
    	A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
	"""
	n_sv = np.zeros((3,2))

	lin_clf = train_kernel(x_train, y_train, 'linear')
	poly_clf = train_kernel(x_train, y_train, 'poly')
	rbf_clf = train_kernel(x_train, y_train, 'rbf')
	clf_arr = [lin_clf, poly_clf, rbf_clf]

	for i in range(3):
		for j in range(2):
			n_sv[i, j] = clf_arr[i].n_support_[j]

	return n_sv

def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
	"""
		Returns: np.ndarray of shape (11,) :
			An array that contains the accuracy of the resulting model on the VALIDATION set.
	"""
	# TODO: add your code here


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
	"""
		Returns: np.ndarray of shape (11,) :
			An array that contains the accuracy of the resulting model on the VALIDATION set.
	"""
	# TODO: add your code here

if __name__ == "__main__":
	x_train, y_train, x_val, y_val = get_points()
	# for i in range(len(x_train)):
	# print("{} {}".format(x_train[i], y_train[i]))
	n_sv = train_three_kernels(x_train, y_train, x_val, y_val)
	print(n_sv)