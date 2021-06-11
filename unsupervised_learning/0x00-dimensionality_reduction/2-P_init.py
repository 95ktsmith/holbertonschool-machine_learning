#!/usr/bin/env python3
""" t-SNE Initialization """
import numpy as np


def P_init(X, perplexity):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
    perplexity: peplexity that all Guassian distributions should have
    Returns: D, P, betas, H
    """
    n, d = X.shape

    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1) ** 2
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
