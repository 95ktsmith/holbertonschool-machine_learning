#!/usr/bin/env python3
""" Markov Chain after t Iterations """
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular state
    after a specififed number of iterations
    """
    if type(P) is not numpy.ndarray or P.ndim != 2:
        return None
    if type(s) is not numpy.ndarray or s.ndim != 1:
        return None
    if P.shape[0] != P.shape[1] or s.shape[0] != P.shape[0]:
        return None
    S = s @ P
    for i in range(t):
        S = S @ P
    return m
