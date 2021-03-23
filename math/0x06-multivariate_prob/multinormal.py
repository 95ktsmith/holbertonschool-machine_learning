#!/usr/bin/env python3
""" Multinormal """
import numpy as np


class MultiNormal:
    """ Multiviate Normal Distribution """

    def __init__(self, data):
        """
        data must be a 2D numpy.ndarray of shape (d, n) where n is >= 2
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.average(data, axis=1).reshape((data.shape[0], 1))
        x = data.T[:, :] - self.mean.T[:]
        self.cov = x.T @ x / (x.shape[0] - 1)
