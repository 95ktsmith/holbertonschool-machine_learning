#!/usr/bin/env python3
""" Principal Component Analysis """
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    X: numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    var: fraction of the variance that the PCA transformation should maintain
    Returns: the weight matrix W that maintains var fraction of X's original
        variance
    """
    u, s, vh = np.linalg.svd(X)
    var_cumsum = np.cumsum(s)
    var_threshold = var_cumsum[-1] * var
    nd = 0
    while var_cumsum[nd] < var_threshold:
        nd += 1
    W = vh[:nd + 1].T
    return W
