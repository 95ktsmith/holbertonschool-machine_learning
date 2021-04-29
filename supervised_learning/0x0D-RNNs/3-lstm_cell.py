#!/usr/bin/env python3
""" Long Short Term Memory Cell """
import numpy as np


class LSTMCell:
    """ Class representation of an LSTM cell """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        Returns h_next, c_next, y
            h_next i the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        x = concat @ self.Wf + self.bf
        forget = 1 / (1 + np.exp(-x))

        # Update gate
        x = concat @ self.Wu + self.bu
        update = 1 / (1 + np.exp(-x))

        # Intermediate cell state
        ics = np.tanh(concat @ self.Wc + self.bc)

        # Output gate
        x = concat @ self.Wo + self.bo
        output = 1 / (1 + np.exp(-x))

        # Next cell state
        c_next = c_prev * forget
        c_next = c_next + (update * ics)

        # Next hidden state
        h_next = np.tanh(c_next) * output

        # Softmax activation
        x = h_next @ self.Wy + self.by
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        return h_next, c_next, y
