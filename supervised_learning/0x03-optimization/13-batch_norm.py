#!/usr/bin/env python3
""" Batch normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ Normalizes an unactivated ouput of a neural network using batch
            normalization
        Z is a numpy.ndarray of shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in Z
        gamma is a numpy.ndarray of shape (1, n) containing the scales used
            for batch normalization
        beta is a numpy.ndarray of shape (1, n) containing the offsets used
            for batch normalization
        epsilon is a small number used to avoid division by zero
    """
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    Znorm = (Z - mean) / ((std ** 2) + epsilon) ** 0.5
    Zt = (gamma * Znorm) + beta
    return Zt
