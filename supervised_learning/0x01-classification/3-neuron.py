#!/usr/bin/env python3
""" Neuron """
import numpy as np


class Neuron:
    """ Class to represent a neuron performing binary classification """
    def __init__(self, nx):
        """ nx is the number of input features to the neuron """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W Getter """
        return self.__W

    @property
    def b(self):
        """ b Getter """
        return self.__b

    @property
    def A(self):
        """ A Getter """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propogation of the neuron
            X is a numpy.ndarry with shape(nx, m) that contains input data
            m is the number of examples
        """
        self.__A = 1 / (1 + np.exp((np.matmul(self.W, X) + self.__b) * -1))
        return self.__A

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
