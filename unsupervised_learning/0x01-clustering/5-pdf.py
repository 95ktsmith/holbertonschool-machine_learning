#!/usr/bin/env python3
""" PDF """
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for each
            data point
        Values below 1e-300 are raised to 1e-300
    """
    # Slow but works
    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
        x_m = X - m
        num = np.exp(-0.5 * (x_m @ inv @ x_m.T))
        den = (2 * np.pi) ** (len(m) / 2) * (det ** 0.5)
        P = num / den
        P = P.reshape(len(P) * len(P))[::len(P) + 1]
        return np.where(P < 1e-300, 1e-300, P)
    except Exception as e:
        return None
