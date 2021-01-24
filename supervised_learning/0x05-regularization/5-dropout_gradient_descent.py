#!/usr/bin/env python3
""" Gradient Descent with Dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Updates the weights of a neural network with Dropout regularization
            using gradient descent
        Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
            correct labels for the data
        classes is the number of classes
        m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs and dropout masks of each layer of
            the neural network
        alpha is the learning rate
        keep_prob is the probability that a node will be kept
        L is the number of layers of the network
        All layers use thetanh activation function except the last, which uses
            the softmax activation function
        The weights of the network should be updated in place
    """
    m = len(Y[0])
    dz2 = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        dz1 = (W.T @ dz2) * (1 - (A * A))
        if i > 1:
            dz1 *= cache["D" + str(i - 1)] / keep_prob
        dw = dz2 @ A.T / m
        db = np.sum(dz2, axis=1, keepdims=True) / m
        dz2 = dz1
        weights["W" + str(i)] -= alpha * dw
        weights["b" + str(i)] -= alpha * db
