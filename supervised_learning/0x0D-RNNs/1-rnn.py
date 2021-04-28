#!/usr/bin/env python3
""" RNN Forward Propagation """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    Returns H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t+1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        H[step+1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
