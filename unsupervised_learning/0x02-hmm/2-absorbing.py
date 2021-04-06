#!/usr/bin/env python3
""" Absorbing Markov Chain """
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    """
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    # Make a list of absorbing states
    A = []
    for i in range(P.shape[0]):
        if P[i, i] == 1 and np.sum(P[i]) == 1:
            A.append(i)

    # Make a 2d array of each state's ability to get to each absorbing state
    reachable = [[] for n in range(len(A))]
    for a in range(len(A)):
        for s in range(len(P)):
            reachable[a].append(can_reach(P, s, A[a]))

    # Return whether or not all states can reach at least one absorbing state
    return np.all(np.any(reachable, axis=0))


def can_reach(P, s, a, prev=[]):
    """
    Determines whether or not state s can reach state a in matrix P
    """
    if s == a:
        return True

    moves = []
    for i in range(len(P[s])):
        if P[s][i] > 0 and i not in prev:
            moves.append(i)

    reachable = []
    for move in moves:
        reachable.append(can_reach(P, move, a, prev + [s]))

    return True in reachable
