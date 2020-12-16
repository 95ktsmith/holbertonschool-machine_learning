#!/usr/bin/env python3
""" Exponential Distribution """


class Exponential:
    """ Class representing an exponential distribution """

    def __init__(self, data=None, lambtha=1):
        """ data is a list of the data to be sued to estimate the distribution
            lambtha is the expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            avg = 0
            for i in range(0, len(data) - 1):
                time = data[i + 1] - data[i]
                if time < 0:
                    time *= -1
                avg += time
            self.lambtha = round(float(avg / len(data) * 10), 2)
