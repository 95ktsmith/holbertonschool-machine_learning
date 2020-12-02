#!/usr/bin/env python3
""" Matrix multiplication """


def mat_mul(mat1, mat2):
    """ Mutliplies two 2D matrices together and returns the result as a new
        matrix, or returns None if they matrices cannot be multiplied
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for mat1_row in range(0, len(mat1)):
        result.append([])
        for mat2_col in range(0, len(mat2[0])):
            sum = 0
            for mat1_col in range(0, len(mat1[mat1_row])):
                sum += mat1[mat1_row][mat1_col] * mat2[mat1_col][mat2_col]
            result[mat1_row].append(sum)

    return result
