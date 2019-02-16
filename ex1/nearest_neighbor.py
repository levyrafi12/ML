from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data = mnist['data']
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

def knn(train_points, labels, test_point, k):
	print(train_points.shape)

if __name__ == '__main__':
	loss = 0
	n = len(test)
	for i in range(n):
		pred_label = knn(train[:1000], train_labels[:1000], test[i], 10)
		loss += (pred_label != test_labels[i])

	print("loss ".format(loss / n))