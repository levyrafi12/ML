import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
#from numpy.linalg import inv
from numpy.linalg import inv

import warnings
warnings.filterwarnings("ignore")

def eigen_face_title(name, i):
	title = name
	title += " eigen_face_"
	title += str(i)
	return title

def decoded_face_title(k):
	title = "decoded_face_"
	title += str(k)
	return title

def plot_vector_as_image(title, image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pixels
	"""	
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title(title)

def get_pictures_by_names(lfw_people, names):
	"""
	Given list of names, returns all the pictures of persons with these specific names.
	"""
	persons = []
	n_samples, h, w = lfw_people.images.shape

	for name in names:
		selected_images = []
		target_label = list(lfw_people.target_names).index(name)
		for image, target in zip(lfw_people.images, lfw_people.target):
			if (target == target_label):
				image_vector = image.reshape((h*w, 1))
				selected_images.append(image_vector)

		n = len(selected_images)
		d = selected_images[0].shape[0]
		persons.append(np.asarray(selected_images).reshape(n, d))

	return persons, h, w, n_samples

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

	mean_cols = np.average(X, axis=0)
	X_bar = X - mean_cols
	U, S, V_t = np.linalg.svd(X_bar)
	return V_t[:k,:], S[:k] ** 2

def encoding_decoding(eigen_vecs_T, image):
	encoded_image = np.matmul(eigen_vecs_T, image.T)
	decoded_image = np.matmul(eigen_vecs_T.T, encoded_image)
	return encoded_image.T, decoded_image

def plot_eigen_faces(name, k=10): # question ii
	lfw_people = load_data()
	[selected_images], h, w, _ = get_pictures_by_names(lfw_people, [name])
	eigen_vecs_T, _ = PCA(selected_images, k)
	for i in range(k):
		plt.subplot(1, 2, (i % 2) + 1)
		plot_vector_as_image(eigen_face_title(name, i), eigen_vecs_T[i], h, w)
		if (i  + 1) % 2 == 0:
			plt.show()

def iterate_k(name): # question iii
	ks = [1, 5, 10, 30, 50, 100, 150]
	lfw_people = load_data()
	[selected_images], h, w, _ = get_pictures_by_names(lfw_people, [name])
	n = selected_images.shape[0]
	L2_vals = []

	for k in ks:
		# print("k {}".format(k))
		eigen_vecs_T, eigen_values = PCA(selected_images, k)
		idx = np.random.choice(n, 5)
		L2 = 0
		for i in idx:
			_, decoded_image = encoding_decoding(eigen_vecs_T, selected_images[i])
			plt.subplot(121)
			plot_vector_as_image(decoded_face_title(k), decoded_image, h, w)
			plt.subplot(122)
			plot_vector_as_image(name, selected_images[i], h, w)
			plt.show()
			L2 += np.linalg.norm(selected_images[i] - decoded_image) 
		L2_vals.append(L2 / 5)
	plt.close()
	plt.plot(ks, L2_vals, 'r-', linewidth=2)
	title = name 
	title += ' - L2 distance versus k'
	plt.title(title)
	plt.xlabel('K')
	plt.ylabel('sum of L2 distances')
	plt.savefig(title)
	plt.show()

def classify_images():
	x_train, y_train, x_test, y_test = train_test_split()

	X = np.concatenate((x_train, x_test), axis=0)
	n_samples = X.shape[0]
	n_train = x_train.shape[0]
	n_test = n_samples - n_train

	ks = [1, 5, 10, 30, 50, 100, 150, 300]
	acc = []

	clf = svm.SVC(C=1000, kernel='rbf', gamma=1e-7)
	# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	# 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	# clf = GridSearchCV(svc, param_grid)

	for k in ks:
		print("k {}".format(k))
		eigen_vecs_T, _ = PCA(X, k)
		encoded_images, _ = encoding_decoding(eigen_vecs_T, X)
		X_train_reduced = encoded_images[:n_train, :]
		clf = clf.fit(X_train_reduced, y_train)
		X_test_reduced = encoded_images[n_train:, :]
		y_pred = clf.predict(X_test_reduced)
		acc.append(100 * (1 - np.sum(y_pred != y_test) / len(y_test)))
		# print("acc {}".format(acc[-1]))

	plt.plot(ks, acc, 'b-', linewidth=2)
	title = 'Accuracy testset versus k'
	plt.title(title)
	plt.xlabel('K')
	plt.ylabel('testset accuracy')
	plt.savefig(title)
	plt.show()

def train_test_split():
	lfw_people = load_data()
	index_to_target_name = []
	i = 0
	for target_name in lfw_people.target_names:
		index_to_target_name.append(target_name)
		i += 1

	images_per_targ, h, w, n_samples = get_pictures_by_names(lfw_people, lfw_people.target_names)
	images_targ_pairs = []
	targ_ind = 0
	for images in images_per_targ:
		n_images = images.shape[0] # num images of a person
		for j in range(n_images):
			images_targ_pairs.append((images[j], targ_ind))
		print("{} {}".format(index_to_target_name[targ_ind], n_images))
		targ_ind = targ_ind + 1

	X = np.zeros((n_samples, h * w))
	y = np.zeros(n_samples)
	idx = np.random.permutation(n_samples)
	j = 0
	for i in idx:
		X[j] = images_targ_pairs[i][0]
		y[j] = images_targ_pairs[i][1]
		j += 1

	n_train = int(0.75 * n_samples)

	x_train = X[:n_train, :]
	y_train = y[:n_train]
	x_test = X[n_train:, :]
	y_test = y[n_train:]

	return x_train, y_train, x_test, y_test

if __name__ == '__main__':
	# name = 'Tony Blair'
	name = 'George W Bush'
	# plot_eigen_faces(name)
	# iterate_k(name)
	classify_images()