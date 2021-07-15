#!/usr/bin/env python3
""" Deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    Returns H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy ndarray containing all of the outputs
    """
    layers = len(rnn_cells)
    t, m, _ = X.shape
    h = h_0.shape[2]
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t+1, layers, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for cell in range(layers):
        for step in range(t):
            H[step+1, cell], Y[step] = rnn_cells[cell].forward(
                H[step, cell],
                X[step] if cell == 0 else H[step+1, cell-1]
            )
    return H, Y
