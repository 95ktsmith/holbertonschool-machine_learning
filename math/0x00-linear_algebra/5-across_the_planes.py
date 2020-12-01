#!/usr/bin/env python3
""" Add 2D matrices """


def add_matrices2D(mat1, mat2):
    """ Returns a new matrix of mat1 and mat2 added together, or
        None if the matrices are of different shapes
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    sum_mat = [[] for i in range(0, len(mat1))]
    for r in range(0, len(mat1)):
        for c in range(0, len(mat1[r])):
            sum_mat[r].append(mat1[r][c] + mat2[r][c])
    return sum_mat


def matrix_shape(matrix):
    """ Returns the shape of a matrix as a list of integers """
    if len(matrix) == 0:
        return [0]
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + matrix_shape(matrix[0])
