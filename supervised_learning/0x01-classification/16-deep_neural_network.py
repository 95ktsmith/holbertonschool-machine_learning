#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


class DeepNeuralNetwork:
    """ Class representing a deep neural network """
    def __init__(self, nx, layers):
        """ Init
            nx is the number of input features
                nx must be a positive integer
            layers is a list representing the number of nodes in each layer
                layers must be a list of positive integers
            L: number of layers in the network
            cache: dictionary to hold all intermediary values of the network
            weights: dictionary to hold all weights and biases of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(0, self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights['W1'] = w
                self.weights['b1'] = np.zeros((layers[i], 1))
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w *= np.sqrt(2 / layers[i - 1])
                self.weights['W{}'.format(i + 1)] = w
                self.weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))
