import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np
#from numpy.linalg import inv
from numpy.linalg import inv

def plot_vector_as_image(image, h, w, i):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""	
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	#plt.title(title, size=12)
	plt.savefig('qB_Hugo_Chavez_' + str(i) + '.png')
	plt.show()


def get_pictures_by_name(name='Hugo Chavez'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)

	n = len(selected_images)
	d = selected_images[0].shape[0]
	return np.asarray(selected_images).reshape(n, d), h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	print(X.shape)
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors 
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""

	mu = np.average(X, axis=0) / X.shape[0]
	X_bar = np.apply_along_axis(lambda row: row - mu, 1, X)

	# Sigma = np.matmal(X_bar.T, X_bar) # covariance matrix
	V, S, U = np.linalg.svd(X_bar)
	print(X_bar.shape)
	print(V.shape)
	print(S.shape)
	print(U.shape)

	return U[:k,:], S[:k]

def main():
	selected_images, h, w = get_pictures_by_name('Hugo Chavez')
	print("{} {} {}".format(len(selected_images), selected_images[0].shape, h, w))
	# plot_vector_as_image(selected_images[0], h, w, 0)
	U, S = PCA(selected_images, 10)
	print(U.shape)
	for i in range(U.shape[0]):
		plot_vector_as_image(U[i], h, w, i)

if __name__ == '__main__':
    main()