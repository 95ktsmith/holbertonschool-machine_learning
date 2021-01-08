#!/usr/bin/env python3
""" One Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ Converts a numeric label vector into a one-hot matrix
        Y is a numpy.ndarray with shape (m,) containing numeric class labels
        classes is the maximum number of classes found in Y
    """
    if classes < 1 or len(Y) < 1:
        return None

    mat = np.zeros((classes, len(Y)))
    for i in range(0, len(Y)):
        try:
            mat[Y[i]][i] = 1
        except IndexError:
            return None

    return mat
