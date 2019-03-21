import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# warnings.warn(CV_WARNING, FutureWarning)

def plot_vector_as_image(name, image, h, w, i, k=0):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pixels
	"""	
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	title = name
	title += '_'
	if k > 0:
		title += str(k) + '_'
	title += str(i)
	plt.title(title)

def build_dataset(lfw_people):
	"""
		Create a dataset of all images and their labels
	"""

	n_samples, h, w = lfw_people.images.shape

	X = np.zeros((n_samples, h * w))
	y = np.zeros(n_samples)

	i = 0
	for image, target in zip(lfw_people.images, lfw_people.target):
		X[i] = image.reshape(h*w)
		y[i] = target
		i += 1

	X_perm = np.zeros((n_samples, h * w))
	y_perm = np.zeros(n_samples)	

	idx = np.random.permutation(n_samples)
	j = 0
	for i in idx:
		X_perm[j] = X[i]
		y_perm[j] = y[i]
		j += 1

	return X_perm, y_perm

def get_pictures_by_name(lfw_people, name):
	"""
	Given list of names, returns all the pictures of persons with these specific names.
	"""
	_, h, w = lfw_people.images.shape

	selected_images = []
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)

	mat = np.asarray(selected_images).reshape(len(selected_images), h * w)

	return mat, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  V - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors 
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""

	mu = np.average(X, axis=0) / X.shape[0]
	X_bar = np.apply_along_axis(lambda row: row - mu, 1, X)
	U, S, V_t = np.linalg.svd(X_bar)
	return V_t[:k,:], S[:k] ** 2

def iterate_k(name):
	ks = [1, 5, 10, 30, 50, 100]
	lfw_people = load_data()
	selected_images, h, w = get_pictures_by_name(lfw_people, name)
	n = selected_images.shape[0]
	L2_losses = []

	for k in ks:
		V_t, eigenvalues = PCA(selected_images, k)
		A = np.matmul(V_t, selected_images.T)
		images_tag = np.matmul(V_t.T, A).T
		# print("eigenvalues {}".format(eigenvalues))
		idx = np.random.choice(n, 5)
		for i in idx:
			plt.subplot(121)
			plot_vector_as_image(name, images_tag[i], h, w, i, k)
			plt.subplot(122)
			plot_vector_as_image(name, selected_images[i], h, w, i)
			plt.show()

		L2_losses.append(np.linalg.norm(selected_images - images_tag))

	plt.clf()
	plt.plot(ks, L2_losses, 'r-', linewidth=2)
	title = name 
	title += ' - L2 loss versus k'
	plt.title(title)
	plt.xlabel('K')
	plt.ylabel('L2 loss')
	plt.savefig(title)
	plt.show()

def plot_eigen_vectors_as_images(name):
	lfw_people = load_data()
	selected_images, h, w = get_pictures_by_name(lfw_people, name)
	V, _ = PCA(selected_images, 10)
	for i in range(V.shape[0]):
		plt.subplot(1, 2, (i % 2) + 1)
		plot_vector_as_image(name, V[i], h, w, i + 1, 10)
		if (i  + 1) % 2 == 0:
			plt.show()

def classify_images():
	lfw_people = load_data()
	X, y = build_dataset(lfw_people)
	n_samples = len(y)

	n_train = int(0.75 * n_samples) # number of training examples
	y_train = y[:n_train]
	y_test = y[n_train:]

	ks = [1, 5, 10, 30, 50, 100, 150, 300]
	acc = []

	# clf = svm.SVC(C=1000, kernel='rbf', gamma=1e-7)
	param_grid = {'C':[1e3,1e4], 'gamma':[1e-6,1e-7]}
	clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)

	for k in ks:
		# print("k {}".format(k))
		V_t, eigenvalues = PCA(X, k)
		# print("eigenvalues {}".format(eigenvalues))
		A = np.matmul(V_t, X.T).T
		X_train = A[:n_train, :]
		X_test = A[n_train:, :]
		clf.fit(X_train, y_train)
		# print(clf.best_estimator_)
		y_pred = clf.predict(X_test)
		acc.append(100 * (1 - np.sum(y_pred != y_test) / len(y_test)))
		# print("acc {}".format(acc[-1]))

	plt.clf()
	plt.plot(ks, acc, 'b-', linewidth=2)
	title = name 
	title += ' - Accuracy versus k'
	plt.title(title)
	plt.xlabel('K')
	plt.ylabel('Accuracy in percentage')
	plt.savefig(title)
	plt.show()

if __name__ == '__main__':
	name = 'Ariel Sharon'
	# plot_eigen_vectors_as_images(name)
	# iterate_k(name)
	classify_images()