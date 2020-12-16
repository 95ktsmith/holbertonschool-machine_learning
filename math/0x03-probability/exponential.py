#!/usr/bin/env python3
""" Exponential Distribution """


class Exponential:
    """ Class representing an exponential distribution """

    def __init__(self, data=None, lambtha=1):
        """ data is a list of the data to be sued to estimate the distribution
            lambtha is the expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = 1 / (sum(data) / len(data))