#!/usr/bin/env python3
""" Compute policy gradient """
import numpy as np


def policy(matrix, weight):
    """ Computes to policy with a weight of a matrix """
    P = np.exp(matrix @ weight)
    return P / np.sum(P)


def softmax_gradient(softmax):
    """ Compute the gradient of a softmax output """
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - s @ s.T


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and weight matrix
    state: matrix representing the current observation of the environment
    weight: matrix of random weight
    Returns the action and the gradient
    """
    P = policy(state, weight)
    action = np.random.choice(P.shape[1], p=P[0])

    d_P = softmax_gradient(P)[action, :]
    d_log = d_P / P[0, action]
    grad = state.T @ d_log[None, :]
    return action, grad
