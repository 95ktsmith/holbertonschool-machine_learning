#!/usr/bin/env python3
""" Integrate """


def poly_integral(poly, C=0):
    """ Returns the integral of a polynomial
        Returns None if poly or C are not valid
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    for item in poly:
        if type(item) is not int:
            return None
    if type(C) is not int:
        return None

    integral = [C]
    for i in range(0, len(poly)):
        coefficient = poly[i] / (i + 1)
        if coefficient % 1 == 0:
            integral.append(int(coefficient))
        else:
            integral.append(coefficient)

    return integral
