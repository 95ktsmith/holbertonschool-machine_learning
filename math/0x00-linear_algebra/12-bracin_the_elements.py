#!/usr/bin/env python3
""" Matrix operations """


def np_elementwise(mat1, mat2):
    """ Performs element-wise operations with mat1 and mat2, and returns
        a tuple of (sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
