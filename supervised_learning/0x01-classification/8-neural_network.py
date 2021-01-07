#!/usr/bin/env python3
""" Nerual Network """
import numpy as np


class NeuralNetwork:
    """ Class defining a neural network with one hidden layer performing
        binary classification
    """
    def __init__(self, nx, nodes):
        """ Init
            nx is the numpy of input features
                nx must be a positive integer > 1
            nodes is the number of nodes found in the hidden layer
                nodes must be a positive integer > 1
            W1: The weights vector for the hidden layer
            b1: The bias for the hidden layer
            A1: The activated output for the hidden layer
            W2: The weights vector for the output neuron
            b2: The bias for the output neuron
            A2: The activated output for the output neuron (prediction)
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.array([[float(0)] for i in range(nodes)])
        self.A1 = 0

        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
