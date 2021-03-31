#!/usr/bin/env python3
""" Initialize GMM """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Guassian Mixture Model
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    Returns pi, m, S, or None, None, None onl failure
        pi is a numpy.ndarray of shape (k,) containing the priors for each
            cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster, initialized as identity matrices
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    n, d = X.shape
    if type(k) is not int or k < 1 or k >= n:
        return None, None, None
    pi = np.ones((k)) / k
    m, _ = kmeans(X, k)
    S = np.ones((k, d, d))
    S[:, :] = np.identity(2)

    return pi, m, S
