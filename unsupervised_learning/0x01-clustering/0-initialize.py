#!/usr/bin/env python3
""" Initialize K-means """
import numpy as np


def initialize(X, k):
    """
    Inializes cluster centroids for K-means
    X is a numpy.ndarray of shape (n, d) containing the data set
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    Returns a numpy.ndarray of shape (k, d) containing the centroids for each
        cluster, or None on failure
    """
    try:
        maxes = np.amax(X, axis=0)
        mins = np.amin(X, axis=0)
        centroids = np.random.uniform(mins, maxes, (k, X.shape[1]))
        return centroids
    except Exception as e:
        return None
