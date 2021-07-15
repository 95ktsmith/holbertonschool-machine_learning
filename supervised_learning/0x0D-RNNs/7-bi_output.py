#!/usr/bin/env python3
""" Bidirectional Cell """
import numpy as np


class BidirectionalCell:
    """ Class representation of a bidirectional cell of an RNN """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h * 2, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step
        Returns h_next
            h_next is the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step
        Returns h_prev
            h_prev is the previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN
        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions, excluding their
            initialized states
        Returns Y
            Y is the outputs
        """
        # Softmax activation
        x = H[:, :] @ self.Wy + self.by
        Y = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        return Y
