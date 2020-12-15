#!/usr/bin/env python3
""" Poisson """


class Poisson:
    """ Class representing a poisson distribution """

    def __init__(self, data=None, lambtha=1):
        """ data is a list of the data used to estimate the distribution
            lambtha is the expected number of occurences in a given time frame
        """

        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes,
            where k is the number of successes
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        return ((e ** (self.lambtha * -1)) * (self.lambtha ** k)) / fact(k)

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes,
            where k is the number of successes
        """
        cdf = 0
        e = 2.7182818285
        for i in range(0, k + 1):
            cdf += ((e ** (self.lambtha * -1) * self.lambtha ** i) / fact(i))
        return cdf


def fact(n):
    """ Returns the factorial of an integer """
    if n == 0 or n == 1:
        return 1
    return n * fact(n - 1)
