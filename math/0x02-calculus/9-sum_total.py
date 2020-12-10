#!/usr/bin/env python3
""" Sigma summation """


def summation_i_squared(n):
    """ Returns the sum of i^2 for i in range from 1 through n, or None
        if n is not a valid number
    """
    if type(n) is not int or n < 1:
        return None
    return (n * (n + 1) * ((2 * n) + 1) / 6)
