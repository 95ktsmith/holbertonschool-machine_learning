#!/usr/bin/env python3
""" Intra-cluster Variance """
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    Returns var, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    n, d = X.shape
    k, dk = C.shape
    if k > n or d != dk:
        return None

    points = np.repeat(X, k, axis=0).reshape((n, k, d))
    distances = ((points - C) ** 2).sum(axis=2)
    min_distance = np.amin(distances, axis=1)
    var = np.sum(min_distance)
    return var
