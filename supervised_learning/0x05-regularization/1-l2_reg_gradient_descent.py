#!/usr/bin/env python3
""" L2 regularization in gradient descent """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Updates the weights and biases of a neural network using gradient
            descent with L2 regularization
        Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
            correct labels for the data
        classes is the number of classes
        m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of outputs of each layer of the neural network
        alpha is the learning rate
        lambtha is the L2 regularization parameter
        L is the number of layers of the network
        The neural network uses tanh activations on each layer except the last,
            which uses a softmax activation
        The weights and biases of the network should be updated in place
    """
    m = len(Y[0])
    dz2 = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        dz1 = (W.T @ dz2) * (A * (1 - A))
        dw = dz2 @ A.T / m
        dw += (lambtha / m) * W
        db = np.sum(dz2, axis=1, keepdims=True) / m
        dz2 = dz1
        weights["W" + str(i)] -= alpha * dw
        weights["b" + str(i)] -= alpha * db
