#!/usr/bin/env python3
""" Principal Component Analysis """
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    X: numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    ndim: new dimensionality of the transformed X
    Returns: the weight matrix W that maintains var fraction of X's original
        variance
    """
    x_mean = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(x_mean)
    W = vh[:ndim].T
    return x_mean @ W
