#!/usr/bin/env python3
""" Size me please """


def matrix_shape(matrix):
    """ Returns the shape of a matrix as a list of integers """
    if len(matrix) == 0:
        return [0]
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + matrix_shape(matrix[0])
