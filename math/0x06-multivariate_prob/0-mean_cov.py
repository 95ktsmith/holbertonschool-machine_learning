#!/usr/bin/env python3
""" Mean and Covariance """
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    X is a numpy.ndarray of shape(n, d) containing the data set
        n is the number of data points
        d is the number of dimensions in each data point
    Returns mean, cov:
        mean is a numpy.ndarray of shape (1, d) containing the data set mean
        cov is a numpy.ndarray of shape (d, d) containing the data set
            covariance matrix
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.average(X, axis=0).reshape((1, X.shape[1]))
    x = X[:, :] - mean.T[0, :]
    cov = x.T @ x / (x.shape[0] - 1)
    return mean, cov
