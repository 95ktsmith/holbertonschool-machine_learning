#!/usr/bin/env python3
""" epsilon-greedy """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsion-greedy to determine the next action
    Q: a numpy.ndarray containing the q-table
    state: the current state
    epsilon: the epsilon value to use for the calculation
    Returns: The next action index
    """
    p = np.random.uniform()

    if p > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(Q.shape[1])

    return action
