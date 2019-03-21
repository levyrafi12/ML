import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
#from numpy.linalg import inv
from numpy.linalg import inv

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

	mu = np.average(X, axis=0) / X.shape[0]
	X_bar = np.apply_along_axis(lambda row: row - mu, 1, X)
	U, S, V_t = np.linalg.svd(X_bar)
	return V_t[:k,:], S[:k] ** 2

def iterate_k(name):
	ks = [1, 5, 10, 30, 50, 100]
	lfw_people = load_data()
	[selected_images], h, w, _ = get_pictures_by_names(lfw_people, [name])
	n = selected_images.shape[0]
	L2_vals = []

	for k in ks:
		V_t, _ = PCA(selected_images, k)
		idx = np.random.choice(n, 5)
		L2 = 0
		for i in idx:
			a = np.matmul(V_t, selected_images[i].T)
			image_tag = np.matmul(V_t.T, a)
			plt.subplot(121)
			plot_vector_as_image(name, image_tag, h, w, i, k)
			plt.subplot(122)
			plot_vector_as_image(name, selected_images[i], h, w, i)
			plt.show()
			L2 += np.linalg.norm(selected_images[i] - image_tag) 
		L2_vals.append(L2 / 5)
	plt.close()
	plt.plot(ks, L2_vals, 'r-', linewidth=2)
	title = name 
	title += ' - L2 distance versus k'
	plt.title(title)
	plt.xlabel('K')
	plt.ylabel('sum of L2 distances')
	plt.show()

def plot_eigen_vectors_as_images(name):
	lfw_people = load_data()
	[selected_images], h, w, _ = get_pictures_by_names(lfw_people, [name])
	V, _ = PCA(selected_images, 10)
	for i in range(V.shape[0]):
		plt.subplot(1, 2, (i % 2) + 1)
		plot_vector_as_image(name, V[i], h, w, i + 1, 10)
		if (i  + 1) % 2 == 0:
			plt.show()

def classify_images():
	lfw_people = load_data()
	index_to_target_name = {}
	i = 0
	for target_name in lfw_people.target_names:
		index_to_target_name[str(i)] = target_name
		i += 1

	images_per_targ, h, w, n_samples = get_pictures_by_names(lfw_people, lfw_people.target_names)
	images_targ_pairs = []
	targ_ind = 0
	for images in images_per_targ:
		for j in range(images.shape[0]):
			images_targ_pairs.append((images[j], targ_ind))
		targ_ind += 1

	X = np.zeros((n_samples, h * w))
	y = np.zeros(n_samples)
	idx = np.random.permutation(n_samples)
	j = 0
	for i in idx:
		X[j] = images_targ_pairs[i][0]
		y[j] = images_targ_pairs[i][1]
		j += 1

	n_train = int(0.75 * n_samples)
	n_test = n_samples - n_train
	y_train = y[:n_train]
	y_test = y[n_train:]

	ks = [1, 5, 10, 30, 50, 100, 150, 300]
	acc = []

	clf = svm.SVC(C=100, kernel='rbf', gamma=0.28)
	# params = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[0.2,0.3]}
	params = {'kernel':('linear','rbf'), 'C':[10], 'gamma':[0.22]}
	# clf = GridSearchCV(svc, params)

	# print(y_test)
	for k in ks:
		X_train_reduced = np.zeros((n_train, k))
		X_test_reduced = np.zeros((n_test, k))
		print("k {}".format(k))
		V_t, _ = PCA(X, k)
		for i in range(n_samples):
			a = np.matmul(V_t, X[i].T)
			if i < n_train:
				X_train_reduced[i] = a
			else:
				X_test_reduced[i - n_train] = a

		clf.fit(X_train_reduced, y_train)
		y_pred = clf.predict(X_test_reduced)
		# print(y_pred)
		acc.append(100 * (1 - np.sum(y_pred != y_test) / len(y_test)))
		print("acc {}".format(acc[-1]))

if __name__ == '__main__':
	name = 'Ariel Sharon'
	# plot_eigen_vectors_as_images(name)
	# iterate_k(name)
	classify_images()