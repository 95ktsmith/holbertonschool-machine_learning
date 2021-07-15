#!/usr/bin/env python3
""" Bidirectional RNN Forward Propagation """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    X is a numpy.ndarray of shape(t, m, i) containing the input data
    h_0 is a numpy.ndarray of shape (m, h) containing the initial hidden state
        of the forward direction
    h_t is a numpy.ndarray of shape (m, h) containing the initial hidden state
        of the backward direction
    Returns H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    H = np.zeros((t, m, h * 2))

    # Forwards
    for step in range(t):
        H[step, :, :h] = bi_cell.forward(
            h_0 if step == 0 else H[step-1, :, :h],
            X[step]
        )

    # Backwards
    for step in range(t-1, -1, -1):
        H[step, :, h:] = bi_cell.backward(
            h_t if step == t-1 else H[step+1, :, h:],
            X[step]
        )

    Y = bi_cell.output(H)
    return H, Y
