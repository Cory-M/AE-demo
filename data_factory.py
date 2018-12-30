import os
import numpy as np
#import mxnet as mx
from math import sin,cos,sqrt
from sklearn.datasets import fetch_mldata


def gaussian(batch_size, n_dim, mean=0, var=1):
	z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
	return z

def gaussian_mixture_axis(batch_size, label_indices=None, n_dim=10, n_labels=10, var_1=0.5, var=0.1, shift=5):
	'''
	  put 10 manifolds into a high-dimentianal space,
	  each of the manifolds occupies one axis.
	  Attention: the number of manifolds is fixed! Equal to 10! (for Cifar10 task)
	'''
	if n_dim < 10:
		raise Exception('n_dim must be greater or equal to 10 !')
	def sample(x, label, n_labels, shift):
		if label == 0:
			new_x = np.concatenate(((x[0]+shift).reshape(1,),x[1:]))
		else:
			new_x = np.concatenate((x[label].reshape(1,),x[1:label].reshape(label-1,),(x[0]+shift).reshape(1,),x[label+1:].reshape(n_dim-label-1,)))
		return new_x
	cov_matrix = np.zeros((n_dim, n_dim))
	cov_matrix[0][0] = var_1
	for i in range(1,n_dim):
		cov_matrix[i][i] = var
	mean = [0] * n_dim
	x = np.random.multivariate_normal(mean, cov_matrix, batch_size)
	z = np.empty((batch_size, n_dim),dtype=np.float32)
	for batch in range(batch_size):
		if label_indices is not None:
			z[batch] = sample(x[batch], label_indices[batch], n_labels, shift)
		else:
			z[batch] = sample(x[batch], np.random.randint(0,n_labels), n_labels, shift)
	return z

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
	if n_dim % 2 != 0:
		raise Exception("n_dim must be a multiple of 2.")

	def sample(x, y, label, n_labels):
		shift = 1.4
		r = 2.0 * np.pi / float(n_labels) * float(label)
		new_x = x * cos(r) - y * sin(r)
		new_y = x * sin(r) + y * cos(r)
		new_x += shift * cos(r)
		new_y += shift * sin(r)
		return np.array([new_x, new_y]).reshape((2,))
	x = np.random.normal(0, x_var, (batch_size, int(n_dim / 2)))
	y = np.random.normal(0, y_var, (batch_size, int(n_dim / 2)))
	z = np.empty((batch_size, n_dim), dtype=np.float32)
	for batch in range(batch_size):
		for zi in range(int(n_dim / 2)):
			if label_indices is not None:
				z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
			else:
				z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

	return z


