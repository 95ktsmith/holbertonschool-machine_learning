#!/usr/bin/env python3
""" Forward Propagation with dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Conducts forward propagation using Dropout
        X is a numpy.ndarray of shape (nx, m) containing the input data for the
            network
        nx is the number of input features
        m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        L the number of layers in the network
        keep_prob is the probability that a node will be kept
        All layers except the last should use the tanh activation function
        The last layer should use the softmax activation function
        Returns: a dictionary containing the outputs of each layer and the
            dropout mask used on each layer (see example for format)
    """
    activations = {"A0": X}
    for i in range(1, L + 1):
        w = "W" + str(i)
        a = "A" + str(i - 1)
        b = "b" + str(i)
        Y = weights[w] @ activations[a] + weights[b]
        if i < L:
            A = (2 / (1 + np.exp(Y * -2))) - 1
            d = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            d = np.where(d, 1, 0)
            A *= d / keep_prob
            activations["d" + str(i)] = d
        else:
            A = np.exp(Y) / np.sum(np.exp(Y), axis=0, keepdims=True)
        activations["A" + str(i)] = A

    return activations
