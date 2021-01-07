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

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1 getter """
        return self.__W1

    @property
    def b1(self):
        """ b1 getter """
        return self.__b1

    @property
    def A1(self):
        """ A1 getter """
        return self.__A1

    @property
    def W2(self):
        """ W2 getter """
        return self.__W2

    @property
    def b2(self):
        """ b2 getter """
        return self.__b2

    @property
    def A2(self):
        """ A2 getter """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propogation of the neural network
            X is a numpy.ndarry with shape(nx, m) that contains input data
            m is the number of examples
            Updates private attributes __A! and __A2
            Returns __A1 and __A2 respectively
        """
        self.__A1 = 1 / (1 + np.exp((np.matmul(self.W1, X) + self.__b1) * -1))
        self.__A2 = (1 +
                     np.exp((np.matmul(self.W2, self.A1) + self.__b2) * -1))
        self.__A2 = 1 / self.A2
        return self.A1, self.A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression
            Y is a numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data
            A is a numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each example
            Returns calculated cost
        """
        # Using L(Y, A) = -(Ylog(A) + (1 - Y)log(1 - A)) loss function
        cost = np.matmul(Y, np.log(A).T)
        cost += np.matmul((1 - Y), np.log(1.0000001 - A).T)
        cost /= len(Y[0]) * -1
        return cost[0][0]
