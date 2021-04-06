#!/usr/bin/env python3
""" Regular steady state probabilities """
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    """
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    P2 = P @ P
    if np.any(P2 == 0):
        return None

    Sprev = np.ones(P.shape[0]) / P.shape[0]
    S = Sprev @ P
    while not np.array_equal(S, Sprev):
        Sprev = S
        S = S @ P
    return S
