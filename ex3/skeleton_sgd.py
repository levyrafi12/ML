#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper_unscale():
  mnist = fetch_mldata('MNIST original', data_home='.')
  data = mnist['data']
  labels = mnist['target']

  # extract indices of digits 0 and 8
  neg, pos = 0,8
  train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
  test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

  # print("{} {}".format(train_idx.shape, test_idx.shape))

  train_data_unscaled = data[train_idx[:6000], :].astype(float)
  train_labels = (labels[train_idx[:6000]] == pos)*2-1 # zero is mapped to -1 , 8 to +1

  validation_data_unscaled = data[train_idx[6000:], :].astype(float) 
  validation_labels = (labels[train_idx[6000:]] == pos)*2-1

  test_data_unscaled = data[60000+test_idx, :].astype(float)
  test_labels = (labels[60000+test_idx] == pos)*2-1
 
  print("train size {} validation size {} test size {}".\
    format(len(train_labels), len(validation_labels), len(test_labels)))

  return train_data_unscaled, train_labels, validation_data_unscaled, \
    validation_labels, test_data_unscaled, test_labels

def helper():
    train_data_unscaled, train_labels, validation_data_unscaled, \
    validation_labels, test_data_unscaled, test_labels = helper_unscale()

    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD(data, labels, C, eta_0, T):
  """
  Implements Hinge loss using SGD.
  returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
  """
  d = data.shape[1]
  n = data.shape[0]
  w = np.zeros(d)
 
  for t in range(T):
    eta = eta_0 / (t + 1)
    i = np.random.choice(n)
    if labels[i] * np.dot(w, data[i, :]) < 1: 
      w = (1 - eta) * w + C * eta * labels[i] * data[i, :]
    else:
      w = (1 - eta) * w
  return w
	
def score(w, X, y):
  return 1 - (np.sum(np.sign((np.matmul(X, w.T) * 2 - 1)) != y)) / X.shape[0]

def iterate_eta_0():
  vals = [math.pow(10, i) for i in range(-5, 6)]
  n = len(validation_labels)
  acc_vals = []

  for eta_0 in vals:
    acc = 0
    for i in range(10):
      w = SGD(train_data, train_labels, 1, eta_0, 1000)
      acc += score(w, validation_data, validation_labels)
    acc_vals.append(acc / 10)

  title = 'acc_vs_eta_0'
  plt.plot(np.log10(vals), acc_vals, 'g-', linewidth=2)
  plt.xlabel('log10(eta_0)')
  plt.ylabel('Accuracy')
  plt.title(title)
  plt.xticks(np.arange(-5, 6))
  plt.savefig(title)
  # plt.show()
  plt.close()

  best_eta_0 = math.pow(10,-5) * math.pow(10, np.argmax(acc_vals))
  print("Best eta_0 {}".format(best_eta_0))

def iterate_C():
  vals = [math.pow(10, i) for i in range(-5, 6)]
  n = len(validation_labels)
  acc_vals = []

  for C in vals:
    acc = 0
    for i in range(10):
      w = SGD(train_data, train_labels, C, 1, 1000)
      acc += score(w, validation_data, validation_labels)
    acc_vals.append(acc / 10)

  title = 'acc_vs_C'
  plt.plot(np.log10(vals), acc_vals, 'b-', linewidth=2)
  plt.xlabel('log10(C)')
  plt.ylabel('Accuracy')
  plt.title(title)
  plt.xticks(np.arange(-5, 6))
  plt.savefig(title)
  # plt.show()
  plt.close()

  best_C = math.pow(10,-5) * math.pow(10, np.argmax(acc_vals))
  print("Best C {}".format(best_C))

#################################

# Place for additional code

#################################

if __name__ == '__main__':
  train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
  # iterate_eta_0()
  # iterate_C()

  w = SGD(train_data, train_labels, 0.0001, 1, 20000)
  # plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
  # plt.show()
  acc = score(w, test_data, test_labels)
  print("SGD accuracy on test data {}".format(acc))