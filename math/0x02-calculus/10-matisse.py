#!/usr/bin/env python3
""" Derivative """


def poly_derivative(poly):
    """ Returns the derivative of a polynomial
        poly is a list of integers that represent the coefficients of the
        power of their index
        Returns None if poly is not valid
    """
    if type(poly) is not list:
        return None
    for item in poly:
        if type(item) is not int:
            return None

    derivative = []
    for power in range(len(poly) - 1, 0, -1):
        derivative = [poly[power] * power] + derivative

    return derivative


if __name__ == "__main__":
    print(poly_derivative([1, -5, 0, 3, 1]))
