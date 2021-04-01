#!/usr/bin/env python3
""" Maximization """
import numpy as np


def maximization(X, g):
    """
    Calculcates the maximization step in the EM algorithm for a GMM:
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    return None, None, None
