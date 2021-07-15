#!/usr/bin/env python3
""" GRU Cell """
import numpy as np


class GRUCell:
    """ Class represenation of a gated recurrent unit """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
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

        # Reset gate
        x = concat @ self.Wr + self.br
        reset = 1 / (1 + np.exp(-x))
        reset_state = h_prev * reset

        # Update gate
        x = concat @ self.Wz + self.bz
        update = 1 / (1 + np.exp(-x))
        updated_state = h_prev * (1 - update)

        # Next hidden state
        reset_concat = np.concatenate((reset_state, x_t), axis=1)
        h_next = np.tanh(reset_concat @ self.Wh + self.bh)
        h_next = h_next * update
        h_next = h_next + updated_state

        # Softmax activation
        x = h_next @ self.Wy + self.by
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        return h_next, y
