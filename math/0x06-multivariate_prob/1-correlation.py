#!/usr/bin/env python3
""" Correlation """
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    C is a numpy.ndarray of shape (d, d) containing a covariance matrix
        d is the number of dimensions
    Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    D = np.diag(1 / np.sqrt(np.diag(C)))
    return D @ C @ D
