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
