#!/usr/bin/env python3
""" Precision """
import numpy as np


def precision(confusion):
    """ Calculuates the precision for each class in a confusion matrix
        confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
        classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the precision
            of each class
    """
    true = np.array([confusion[i][i] for i in range(confusion.shape[0])])
    false = np.sum(confusion, axis=0)
    for i in range(confusion.shape[0]):
        false[i] -= confusion[i][i]
    precision = true / (true + false)
    return precision
