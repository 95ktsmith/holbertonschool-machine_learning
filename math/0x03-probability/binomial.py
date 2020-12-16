#!/usr/bin/env python3
""" Binomial """


class Binomial:
    """ Class representing a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ data is a list of the data to be used to estimate the distribution
            n is the number of Bernoulli trials
            p is the probability of "success"
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)

            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            var = 0
            for n in data:
                var += (n - mean) ** 2
            var /= len(data)
            p = 1 - (var / mean)
            self.n = round(mean / p)
            self.p = mean / self.n
