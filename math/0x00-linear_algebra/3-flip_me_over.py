#!/usr/bin/env python3
""" Transpose """


def matrix_transpose(matrix):
    """ Return a transposed version of a 2D matrix """
    new_matrix = [[] for i in range(0, len(matrix[0]))]
    for row in matrix:
        for i in range(0, len(row)):
            new_matrix[i].append(row[i])
    return new_matrix
