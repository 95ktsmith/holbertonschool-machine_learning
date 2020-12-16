#!/usr/bin/env python3
""" Normal distribution """


class Normal:
    """ Class representing a normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ data is a list of the data to be used to estimate the distribution
            mean is the mean of the distribution
            stddev is the standard deviation of the distribution
        """
        if stddev <= 0:
            raise ValueError("stddev must be a positive value")
        self.stddev = float(stddev)

        if data is None:
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))

            sum_diff = 0
            for n in data:
                sum_diff += (n - self.mean) ** 2
            self.stddev = float((sum_diff / len(data)) ** 0.5)

    def z_score(self, x):
        """ Calculates the z-score of a given x-value """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculates the value of the PDF for a given x-value """
        e = 2.7182818285
        pi = 3.1415926536
        a = e ** ((((x - self.mean) / self.stddev) ** 2) / -2)
        b = self.stddev * ((2 * pi) ** 0.5)
        return a / b
