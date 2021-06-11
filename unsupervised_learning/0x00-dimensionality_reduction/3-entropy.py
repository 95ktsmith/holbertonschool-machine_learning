#!/usr/bin/env python3
""" Shannon Entropy and P Affinities """
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    Di: numpy.ndarray of shape(n - 1,) containing the pairwise distances
        between a data point and all other points except itself
    beta: numpy.ndarray of shape (1,) containing the beta value for the
        Gaussian distribution
    Returns: Hi, Pi:
        Hi: Shannon entropy of points
        Pi: numpy.ndarray of shape (n - 1,) containing the P affinities of the
            points
    """

    Pi = np.exp(-1 * Di * beta)
    Pi /= np.sum(Pi)

    Hi = -1 * np.sum(Pi * np.log2(Pi))
    return Hi, Pi
