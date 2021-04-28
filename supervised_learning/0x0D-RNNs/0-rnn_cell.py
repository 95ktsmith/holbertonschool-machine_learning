#!/usr/bin/env python3
""" RNN Cell """
import numpy as np


class RNNCell:
    """ Representation of a cell of a simple RNN """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        Returns h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state
        h_next = np.tanh(concat @ self.Wh + self.bh)

        # Softmax activation
        x = h_next @ self.Wy + self.by
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return h_next, y
