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
	return selected_images, h, w

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
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors 
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	#set X matrix means to zero
	col_means = np.average(X, axis=0)
	print(X.shape)
	print(col_means.shape)
	X_norm = X - col_means
	#print(X)
	#print(X_norm.shape)
	#n = dim(X)[0]
	n = len(X)

	#Perform SVD
	u, s, vh = np.linalg.svd(X_norm.transpose(), full_matrices=True)
	print(u.shape)
	print(s.shape)
	print(vh.shape)

	eigen_vals = s*s #np.dot(s,s)/(n-1)
	#print(eigen_vals)
	#eigen_ranks = np.rankdata(eigen_vals, method='ordinal')
	#eigen_order = eigen_vals.argsort()
	#print(eigen_order)
	#eigen_ranks = eigen_order.argsort()
	#print(eigen_ranks)
	#eigen_ranks_k = eigen_ranks[eigen_ranks <= k]
	#print(eigen_ranks_k)
	#keep_v_col = eigen_ranks <= k
	#print(keep_v_col)

	eigen_vec_cols = u[:,range(k)]
	print(eigen_vec_cols.shape)
	eigen_vec = eigen_vec_cols #.transpose()
	#print(eigen_vec.shape)
	#sort the eigenvector matrix by adding a column with eigenvalues, sort by eignvalues then remove eigenvalues.
	#eigen_vec = np.insert(eigen_vec, [1], eigen_vals, axis = 1)
	#sort_ind = np.argsort(eigen_vec[:: -1])
	#eigen_vec = eigen_vec[sort_ind,:]
	#print(eigen_vec)

	U = eigen_vec #eigen_vec.sort(eigen_ranks_k, axis = 0)#???????????
	S = eigen_vals[k-1]

	#U = None
	#S = None
	return U, S


#data_loaded = load_data()
picture_data = get_pictures_by_name()
#print(picture_data)
image = picture_data[0][1]
image_h = picture_data[1]
image_w = picture_data[2]

#print(picture_data[0][0])
print(image.shape)
#print(image_h)
#print(image_w)

#plot_vector_as_image(image, image_h, image_w)


#picture_matrix_list = picture_data[0]
#picture_matrix = np.squeeze(np.stack( picture_matrix_list, axis=0 ), axis = 2)
#picture_matrix = picture_matrix
#print(picture_matrix)
#PC_Chavez = PCA(picture_matrix,10)

#print(PC_Chavez[0][0])


def question_b():
	picture_data = get_pictures_by_name()
	image_h = picture_data[1]
	image_w = picture_data[2]
	picture_matrix_list = picture_data[0]
	picture_matrix = np.squeeze(np.stack(picture_matrix_list, axis=0), axis=2)
	picture_matrix = picture_matrix
	PC_Chavez = PCA(picture_matrix, 10)
	for i in range(10):
		plot_vector_as_image(PC_Chavez[0][:,i], image_h, image_w, i)


question_b()

def question_c():
	picture_data = get_pictures_by_name()
	image_h = picture_data[1]
	image_w = picture_data[2]
	picture_matrix_list = picture_data[0]
	picture_matrix = np.squeeze(np.stack(picture_matrix_list, axis=0), axis=2)
	picture_matrix = picture_matrix
	PC_Chavez = PCA(picture_matrix, 71)
	print(PC_Chavez[0].shape)
	transform_mat = np.matmul(PC_Chavez[0],PC_Chavez[0].transpose())

	for i in range(15,20):
		picture = picture_matrix[i,:]
		#print(picture.shape)
		#print(transform_mat.shape)
		transformed_picture = np.matmul(transform_mat, picture)
		#for i in range(20,21):
		plot_vector_as_image(picture, image_h, image_w, 100+ i)
		plot_vector_as_image(transformed_picture, image_h, image_w, 200 + i)

question_c()