#!/usr/bin/env python3
""" Expectation """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expecation step in the EM algorithm for a GMM
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    k = pi.shape[0]
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    return None, None
