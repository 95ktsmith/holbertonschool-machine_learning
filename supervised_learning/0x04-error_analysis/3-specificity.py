#!/usr/bin/env python3
""" Specificity """
import numpy as np


def specificity(confusion):
    """ Calculates the specificity for each class in a confusion matrix
        confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the specificity
            of each class
    """
    spec = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        true_negatives = np.sum(confusion) - np.sum(confusion[i])
        false_positives = np.sum(confusion, axis=0)[i] - confusion[i][i]
        true_negatives -= false_positives
        spec[i] = true_negatives / (true_negatives + false_positives)
    return spec
