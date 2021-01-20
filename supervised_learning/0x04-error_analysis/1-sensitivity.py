#!/usr/bin/env python3
""" Sensitivity """
import numpy as np


def sensitivity(confusion):
    """ Calculates the sensitivity of each class in a confusion matrix
        confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
                classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the sensitivity
            of each class
    """
    sens = np.array([confusion[i][i] for i in range(confusion.shape[0])])
    sens /= np.sum(confusion, axis=1)
    return sens
